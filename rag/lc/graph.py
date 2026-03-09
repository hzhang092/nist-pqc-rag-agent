"""
LangGraph agent graph definition and execution.

What this module is for:
- Defines a bounded retrieve-assess-refine-answer controller for PQC RAG.
- Enforces explicit step/tool/round budgets and deterministic stop rules.
- Refuses when evidence is insufficient instead of calling the answer LLM.

How it is used:
- Imported by `rag.agent.ask` to execute the agent graph for a user question.
- Nodes call helper functions from `state_utils` to update `AgentState`.
- Trace events record loop decisions for debugging and evaluation artifacts.

CLI flags:
- None. This is a library module (non-CLI) and is not executed via command-line flags.
"""

# rag/lc/graph.py
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph

# Allow direct script execution: `python rag/lc/graph.py`
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rag.config import SETTINGS
from rag.text_normalize import normalize_identifier_like_spans

try:
    from . import tools as lc_tools
    from .state import AgentState, Citation, CompareTopics, EvidenceItem, Plan, QueryAnalysis
    from .state_utils import (
        add_trace,
        init_state,
        set_answer,
        set_evidence,
        set_final_answer,
        set_plan,
        set_query_analysis,
    )
except ImportError:
    from rag.lc import tools as lc_tools
    from rag.lc.state import AgentState, Citation, CompareTopics, EvidenceItem, Plan, QueryAnalysis
    from rag.lc.state_utils import (
        add_trace,
        init_state,
        set_answer,
        set_evidence,
        set_final_answer,
        set_plan,
        set_query_analysis,
    )


MAX_STEPS = SETTINGS.AGENT_MAX_STEPS
MAX_TOOL_CALLS = SETTINGS.AGENT_MAX_TOOL_CALLS
MAX_RETRIEVAL_ROUNDS = SETTINGS.AGENT_MAX_RETRIEVAL_ROUNDS
MIN_EVIDENCE_HITS = SETTINGS.AGENT_MIN_EVIDENCE_HITS
DEFAULT_TOP_K = SETTINGS.TOP_K
ANALYSIS_MODE_VALUES = {"general", "definition", "algorithm", "compare"}
ALLOWED_DOC_IDS = (
    "NIST.FIPS.203",
    "NIST.FIPS.204",
    "NIST.FIPS.205",
    "NIST.SP.800-227",
    "NIST.IR.8545",
    "NIST.IR.8547.ipd",
)

_ANCHOR_PATTERNS = (
    re.compile(r"\bAlgorithm\s+\d+\b", flags=re.IGNORECASE),
    re.compile(r"\bTable\s+\d+\b", flags=re.IGNORECASE),
    re.compile(r"\bSection\s+\d+(?:\.\d+)*\b", flags=re.IGNORECASE),
    re.compile(r"\b[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)+\b"),
)
_ANCHOR_KEYWORDS = ("keygen", "encaps", "decaps", "shake128", "shake256", "xof")
_DOC_PATTERNS = (
    (re.compile(r"\b(?:fips\s*203|ml-kem)\b", flags=re.IGNORECASE), "NIST.FIPS.203", "FIPS 203"),
    (re.compile(r"\b(?:fips\s*204|ml-dsa)\b", flags=re.IGNORECASE), "NIST.FIPS.204", "FIPS 204"),
    (re.compile(r"\b(?:fips\s*205|slh-dsa)\b", flags=re.IGNORECASE), "NIST.FIPS.205", "FIPS 205"),
    (re.compile(r"\b(?:sp\s*800-227|sp800-227)\b", flags=re.IGNORECASE), "NIST.SP.800-227", "SP 800-227"),
    (re.compile(r"\b(?:ir\s*8545|nist\s*ir\s*8545)\b", flags=re.IGNORECASE), "NIST.IR.8545", "IR 8545"),
    (
        re.compile(r"\b(?:ir\s*8547|nist\s*ir\s*8547)\b", flags=re.IGNORECASE),
        "NIST.IR.8547.ipd",
        "IR 8547",
    ),
)
_DEFN_RE = re.compile(
    r"(?:what\s+is|what(?:'s|\s+is)|define|explain|definition\s+of)\s+"
    r"(?P<term>[A-Za-z0-9][A-Za-z0-9._-]*)",
    flags=re.IGNORECASE,
)
_WHAT_DOES_MEAN_RE = re.compile(
    r"what\s+does\s+(?P<term>[A-Za-z0-9][A-Za-z0-9._-]*)\s+(?:mean|stand\s+for)\b",
    flags=re.IGNORECASE,
)


# -----------------------------
# Helpers
# -----------------------------

def _bump_step(state: AgentState, node: str) -> None:
    state["steps"] = int(state.get("steps", 0)) + 1
    add_trace(
        state,
        {
            "type": "step",
            "node": node,
            "steps": state["steps"],
            "tool_calls": state.get("tool_calls", 0),
            "retrieval_round": state.get("retrieval_round", 0),
        },
    )


def _step_limit_hit(state: AgentState) -> bool:
    return int(state.get("steps", 0)) >= MAX_STEPS


def _tool_limit_hit(state: AgentState) -> bool:
    return int(state.get("tool_calls", 0)) >= MAX_TOOL_CALLS


def _round_limit_hit(state: AgentState) -> bool:
    return int(state.get("retrieval_round", 0)) >= MAX_RETRIEVAL_ROUNDS


def _budget_limit_reason(state: AgentState) -> Optional[str]:
    if _step_limit_hit(state):
        return "step_budget_exhausted"
    if _tool_limit_hit(state):
        return "tool_budget_exhausted"
    if _round_limit_hit(state):
        return "retrieval_round_budget_exhausted"
    return None


def _clean_topic_text(text: str) -> str:
    return text.strip().strip(" .,:;\"'`[](){}")


def _extract_compare_topics(question: str) -> tuple[Optional[str], Optional[str]]:
    q = question.strip().rstrip("?").strip()
    if not q:
        return None, None

    patterns = [
        re.compile(r"(?:differences?|difference)\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+)$", flags=re.IGNORECASE),
        re.compile(r"(?:compare|comparison\s+of)\s+(?P<a>.+?)\s+(?:and|vs|versus)\s+(?P<b>.+)$", flags=re.IGNORECASE),
        re.compile(r"(?P<a>.+?)\s+(?:vs|versus)\s+(?P<b>.+)$", flags=re.IGNORECASE),
    ]

    for pattern in patterns:
        m = pattern.search(q)
        if not m:
            continue
        topic_a = _clean_topic_text(m.group("a"))
        topic_b = _clean_topic_text(m.group("b"))
        if topic_a and topic_b and topic_a.lower() != topic_b.lower():
            return topic_a, topic_b

    return None, None


def _dedupe_strs(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = str(item or "").strip()
        lowered = key.lower()
        if not key or lowered in seen:
            continue
        seen.add(lowered)
        out.append(key)
    return out


def _record_timing(state: AgentState, key: str, started_at: float) -> None:
    elapsed_ms = (perf_counter() - started_at) * 1000.0
    timing = state.setdefault("timing_ms", {})
    timing[key] = round(float(timing.get(key, 0.0)) + elapsed_ms, 3)


def _normalize_doc_ids(doc_ids: Any) -> List[str]:
    if doc_ids is None:
        return []
    if isinstance(doc_ids, str):
        raw_items = [doc_ids]
    else:
        raw_items = list(doc_ids)

    normalized: List[str] = []
    for raw in raw_items:
        text = str(raw or "").strip()
        if not text:
            continue
        for _pattern, doc_id, _family in _DOC_PATTERNS:
            if text.lower() == doc_id.lower():
                normalized.append(doc_id)
                break
        else:
            if text in ALLOWED_DOC_IDS:
                normalized.append(text)
    return [doc_id for doc_id in ALLOWED_DOC_IDS if doc_id in {d for d in normalized}]


def _detect_doc_scope(text: str) -> tuple[List[str], Optional[str]]:
    matches: List[str] = []
    families: List[str] = []
    for pattern, doc_id, family in _DOC_PATTERNS:
        if pattern.search(text):
            matches.append(doc_id)
            families.append(family)
    doc_ids = _normalize_doc_ids(matches)
    doc_family = families[0] if len(set(families)) == 1 and families else None
    return doc_ids, doc_family


def _extract_definition_term(question: str) -> Optional[str]:
    what_does_match = _WHAT_DOES_MEAN_RE.search(question.strip())
    if what_does_match:
        term = str(what_does_match.group("term") or "").strip(" .,:;?()[]{}")
        return term or None
    match = _DEFN_RE.search(question.strip())
    if not match:
        return None
    term = str(match.group("term") or "").strip(" .,:;?()[]{}")
    return term or None


def _to_evidence_items(evidence_dicts: List[Dict[str, Any]]) -> List[EvidenceItem]:
    out: List[EvidenceItem] = []
    for e in evidence_dicts:
        out.append(
            EvidenceItem(
                score=float(e.get("score", 0.0)),
                chunk_id=str(e["chunk_id"]),
                doc_id=str(e["doc_id"]),
                start_page=int(e["start_page"]),
                end_page=int(e["end_page"]),
                text=str(e.get("text", "")),
            )
        )
    return out


def _merge_evidence(existing: List[EvidenceItem], incoming: List[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    merged: List[EvidenceItem] = []
    for item in existing + incoming:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        merged.append(item)
    return merged


def _extract_anchor_terms(question: str) -> List[str]:
    terms: List[str] = []
    ql = question.lower()

    for pattern in _ANCHOR_PATTERNS:
        for m in pattern.finditer(question):
            token = m.group(0).strip()
            if not token:
                continue
            terms.append(token)

    for keyword in _ANCHOR_KEYWORDS:
        if keyword in ql:
            terms.append(keyword)

    return _dedupe_strs(terms)


def _evidence_contains_any_anchor(evidence: List[EvidenceItem], anchors: List[str]) -> bool:
    if not anchors:
        return True
    lowered = [(e.text or "").lower() for e in evidence]
    for anchor in anchors:
        a = anchor.lower()
        if any(a in txt for txt in lowered):
            return True
    return False


def _doc_diversity(evidence: List[EvidenceItem]) -> int:
    return len({e.doc_id for e in evidence})


def _topic_doc_bias_tokens(topic: str) -> List[str]:
    tl = topic.lower()
    if "ml-kem" in tl:
        return ["FIPS 203", "ML-KEM"]
    if "ml-dsa" in tl:
        return ["FIPS 204", "ML-DSA"]
    if "slh-dsa" in tl:
        return ["FIPS 205", "SLH-DSA"]
    return []


def _append_terms(base_query: str, terms: List[str]) -> str:
    base = base_query.strip()
    existing = base.lower()
    extras = [t for t in terms if t and t.lower() not in existing]
    if not extras:
        return base
    return f"{base} {' '.join(extras)}".strip()


def _infer_mode_hint(
    question: str,
    *,
    compare_topics: Optional[CompareTopics],
    anchors: List[str],
) -> str:
    ql = question.strip().lower()
    if compare_topics is not None:
        return "compare"
    if anchors or "algorithm" in ql or "table" in ql or "shake" in ql:
        return "algorithm"
    if ql.startswith(("what is", "what's", "define", "explain")) or _WHAT_DOES_MEAN_RE.search(question):
        return "definition"
    return "general"


def _build_canonical_query(
    question: str,
    *,
    mode_hint: str,
    compare_topics: Optional[CompareTopics],
    definition_term: Optional[str],
) -> str:
    normalized = normalize_identifier_like_spans(question.strip())
    if mode_hint == "compare" and compare_topics is not None:
        return f"{compare_topics.topic_a} vs {compare_topics.topic_b} intended use-cases definition key properties"
    if mode_hint == "definition" and definition_term:
        return definition_term
    return normalized


def _deterministic_query_analysis(question: str) -> QueryAnalysis:
    """Performs a deterministic analysis of the input question to extract structured hints for retrieval.
    This function uses regex patterns and heuristics to analyze the question without any LLM calls.
    The analysis includes:
    - Extracting comparison topics if the question is a comparison.
    - Extracting anchor terms that indicate specific sections, algorithms, or tables.
    - Detecting any explicit document mentions to scope retrieval.
    - Inferring a mode hint (definition, algorithm, compare, general) based on question patterns.
    - Building a canonical query that normalizes the input for retrieval."""
    
    original_query = question.strip()
    topic_a, topic_b = _extract_compare_topics(original_query)
    compare_topics = (
        CompareTopics(topic_a=topic_a, topic_b=topic_b)
        if topic_a and topic_b
        else None
    )
    required_anchors = _extract_anchor_terms(original_query)
    doc_ids, doc_family = _detect_doc_scope(original_query)
    mode_hint = _infer_mode_hint(
        original_query,
        compare_topics=compare_topics,
        anchors=required_anchors,
    )
    definition_term = _extract_definition_term(original_query)
    canonical_query = _build_canonical_query(
        original_query,
        mode_hint=mode_hint,
        compare_topics=compare_topics,
        definition_term=definition_term,
    )
    analysis_notes = "deterministic-fallback"
    return QueryAnalysis(
        original_query=original_query,
        canonical_query=canonical_query,
        mode_hint=mode_hint,  # type: ignore[arg-type]
        required_anchors=required_anchors,
        compare_topics=compare_topics,
        doc_ids=doc_ids,
        doc_family=doc_family,
        analysis_notes=analysis_notes,
    )


def _analysis_system_prompt() -> str:
    allowed_doc_ids = ", ".join(ALLOWED_DOC_IDS)
    return (
        "You analyze PQC RAG queries for a bounded retrieval controller.\n"
        "Return JSON only, no markdown, no prose.\n"
        "Schema keys:\n"
        '- canonical_query: string\n'
        '- mode_hint: one of "definition", "algorithm", "compare", "general"\n'
        '- required_anchors: array of strings\n'
        '- compare_topics: null or {"topic_a": string, "topic_b": string}\n'
        f'- doc_ids: array of strings chosen only from [{allowed_doc_ids}]\n'
        '- doc_family: null or short string\n'
        '- analysis_notes: null or short string\n'
        "Prefer short bounded values. Do not invent anchors or doc_ids."
    )


def _analysis_user_prompt(question: str, deterministic: QueryAnalysis) -> str:
    return (
        f"Question: {question.strip()}\n"
        f"Deterministic hints: {json.dumps(deterministic.to_dict(), ensure_ascii=False)}\n"
        "Return a corrected JSON object that follows the schema exactly."
    )


def _extract_json_object(text: str) -> str:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in analysis response.")
    return candidate[start : end + 1]


def _sanitize_analysis_notes(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:240]


def _normalize_compare_topics(value: Any) -> Optional[CompareTopics]:
    if not isinstance(value, dict):
        return None
    topic_a = _clean_topic_text(str(value.get("topic_a") or ""))
    topic_b = _clean_topic_text(str(value.get("topic_b") or ""))
    if not topic_a or not topic_b or topic_a.lower() == topic_b.lower():
        return None
    return CompareTopics(topic_a=topic_a, topic_b=topic_b)


def _validate_query_analysis(raw: Dict[str, Any], deterministic: QueryAnalysis) -> QueryAnalysis:
    compare_topics = _normalize_compare_topics(raw.get("compare_topics")) or deterministic.compare_topics
    mode_hint = str(raw.get("mode_hint") or deterministic.mode_hint).strip().lower()
    if compare_topics is not None:
        mode_hint = "compare"
    if mode_hint not in ANALYSIS_MODE_VALUES:
        mode_hint = deterministic.mode_hint

    canonical_query = normalize_identifier_like_spans(
        str(raw.get("canonical_query") or deterministic.canonical_query).strip()
    )
    if not canonical_query:
        canonical_query = deterministic.canonical_query

    required_anchors = _dedupe_strs(
        [str(item).strip() for item in (raw.get("required_anchors") or deterministic.required_anchors or [])]
    )
    doc_ids = _normalize_doc_ids(raw.get("doc_ids")) or list(deterministic.doc_ids or [])
    doc_family = str(raw.get("doc_family") or deterministic.doc_family or "").strip() or None
    analysis_notes = _sanitize_analysis_notes(raw.get("analysis_notes")) or deterministic.analysis_notes

    return QueryAnalysis(
        original_query=deterministic.original_query,
        canonical_query=canonical_query,
        mode_hint=mode_hint,  # type: ignore[arg-type]
        required_anchors=required_anchors,
        compare_topics=compare_topics,
        doc_ids=doc_ids,
        doc_family=doc_family,
        analysis_notes=analysis_notes,
    )


def _call_llm_query_analysis(question: str, deterministic: QueryAnalysis) -> QueryAnalysis:
    from rag.llm.factory import get_backend

    backend = get_backend()
    response = backend.generate(
        _analysis_user_prompt(question, deterministic),
        system=_analysis_system_prompt(),
        temperature=0.0,
    )
    payload = json.loads(_extract_json_object(response))
    if not isinstance(payload, dict):
        raise ValueError("Analysis response must be a JSON object.")
    return _validate_query_analysis(payload, deterministic)


def _build_query_analysis(question: str) -> QueryAnalysis:
    deterministic = _deterministic_query_analysis(question)
    try:
        analysis = _call_llm_query_analysis(question, deterministic)
        if analysis.analysis_notes:
            notes = analysis.analysis_notes
        else:
            notes = "llm-validated"
        return QueryAnalysis(
            original_query=analysis.original_query,
            canonical_query=analysis.canonical_query,
            mode_hint=analysis.mode_hint,
            required_anchors=analysis.required_anchors,
            compare_topics=analysis.compare_topics,
            doc_ids=analysis.doc_ids,
            doc_family=analysis.doc_family,
            analysis_notes=notes,
        )
    except Exception:
        return deterministic


def _build_refined_query(state: AgentState) -> Tuple[str, str]:
    plan = state.get("plan") or {}
    base_query = str(plan.get("query") or state.get("canonical_query") or state["question"]).strip()
    reason = str(state.get("stop_reason", "")).lower()
    anchors = list(state.get("required_anchors") or [])
    compare_topics = state.get("compare_topics") or {}

    if "anchor_missing" in reason and anchors:
        return _append_terms(base_query, anchors), "anchor_token_bias"

    topic_a = str(compare_topics.get("topic_a") or "").strip()
    topic_b = str(compare_topics.get("topic_b") or "").strip()
    if "compare_doc_diversity_missing" in reason and topic_a and topic_b:
        doc_tokens = _topic_doc_bias_tokens(topic_a) + _topic_doc_bias_tokens(topic_b)
        terms = [topic_a, topic_b, "comparison", "intended use-cases", *doc_tokens]
        return _append_terms(base_query, terms), "compare_doc_bias"

    if "insufficient_hits" in reason:
        if str(plan.get("action")) == "resolve_definition":
            args = plan.get("args", {}) or {}
            term = str(args.get("term") or state.get("canonical_query") or state["question"]).strip()
            return f"definition of {term}; notation; section", "definition_bias"
        return _append_terms(base_query, ["section", "algorithm", "definition"]), "coverage_bias"

    return base_query, "no_change"


def _refusal_message(refusal_reason: str) -> str:
    rr = (refusal_reason or "").lower()
    if "anchor_missing" in rr:
        return (
            "I could not find citable evidence for the specific algorithm/table/section anchor "
            "in the indexed NIST documents."
        )
    if "compare_doc_diversity_missing" in rr:
        return (
            "I could not find enough citable evidence across both topics to produce a reliable comparison "
            "from the indexed NIST documents."
        )
    if rr == "missing_citations":
        return "I could not produce reliable citations for the drafted answer, so I cannot return it."
    if rr == "empty_draft_answer":
        return "I could not produce a citable grounded answer from the indexed NIST documents."
    return "I do not have enough citable evidence in the indexed NIST documents to answer that reliably."


def _derive_refusal_reason(
    state: AgentState,
    *,
    sufficient: bool,
    draft: str,
    evidence: List[EvidenceItem],
    citations: List[Dict[str, Any]],
) -> str:
    if not sufficient:
        return str(state.get("stop_reason", "") or "insufficient_evidence")
    if not draft:
        return "empty_draft_answer"
    if not evidence:
        return "empty_evidence"
    if len(citations) == 0:
        return "missing_citations"
    return ""


def _call_rag_answer(question: str, evidence: list[EvidenceItem]) -> dict:
    """
    Adapter for this repo: uses rag.rag_answer.build_cited_answer(question, hits, generate_fn).
    """
    from rag.llm.factory import get_backend
    from rag.rag_answer import build_cited_answer
    from rag.retriever.base import ChunkHit

    hits = [
        ChunkHit(
            score=float(e.score),
            chunk_id=str(e.chunk_id),
            doc_id=str(e.doc_id),
            start_page=int(e.start_page),
            end_page=int(e.end_page),
            text=str(e.text),
        )
        for e in evidence
    ]

    llm_backend = get_backend()
    generate_fn = lambda prompt: llm_backend.generate(prompt, temperature=SETTINGS.LLM_TEMPERATURE)
    result = build_cited_answer(question=question, hits=hits, generate_fn=generate_fn)

    citations = []
    for c in result.citations:
        citations.append(
            {
                "doc_id": c.doc_id,
                "start_page": int(c.start_page),
                "end_page": int(c.end_page),
                "chunk_id": c.chunk_id,
                "key": getattr(c, "key", None),
            }
        )

    return {"answer": result.answer, "citations": citations}


# -----------------------------
# LangGraph nodes
# -----------------------------

def node_analyze_query(state: AgentState) -> AgentState:
    _bump_step(state, "analyze_query")
    started_at = perf_counter()

    analysis = _build_query_analysis(state["question"])
    set_query_analysis(state, analysis)
    _record_timing(state, "analyze", started_at)
    return state


def node_route(state: AgentState) -> AgentState:
    _bump_step(state, "route")

    if _step_limit_hit(state):
        state["stop_reason"] = "step_budget_exhausted"
        set_plan(state, Plan(action="refuse", reason="Step budget exhausted before routing."))
        add_trace(state, {"type": "loop_stop", "reason": state["stop_reason"]})
        return state

    compare_topics = state.get("compare_topics") or {}
    canonical_query = str(state.get("canonical_query") or state["question"]).strip()
    mode_hint = str(state.get("mode_hint") or "general").strip() or "general"

    if compare_topics.get("topic_a") and compare_topics.get("topic_b"):
        plan = Plan(
            action="compare",
            reason="Comparison action selected from analyzed compare_topics.",
            query=canonical_query,
            args={
                "topic_a": str(compare_topics["topic_a"]),
                "topic_b": str(compare_topics["topic_b"]),
            },
            mode_hint="compare",
        )
    elif mode_hint == "definition":
        plan = Plan(
            action="resolve_definition",
            reason="Definition action selected from analyzed mode_hint.",
            query=canonical_query,
            args={"term": canonical_query},
            mode_hint="definition",
        )
    else:
        plan = Plan(
            action="retrieve",
            reason="Retrieve action selected from analyzed query state.",
            query=canonical_query,
            mode_hint=mode_hint,
        )
    set_plan(state, plan)
    return state


def node_retrieve(state: AgentState) -> AgentState:
    _bump_step(state, "retrieve")
    started_at = perf_counter()

    # Only retrieval increments tool_calls; this keeps budget accounting in one place.
    if _step_limit_hit(state):
        state["stop_reason"] = "step_budget_exhausted"
        add_trace(state, {"type": "loop_stop", "reason": state["stop_reason"]})
        return state
    if _tool_limit_hit(state):
        state["stop_reason"] = "tool_budget_exhausted"
        add_trace(state, {"type": "loop_stop", "reason": state["stop_reason"]})
        return state
    if _round_limit_hit(state):
        state["stop_reason"] = "retrieval_round_budget_exhausted"
        add_trace(state, {"type": "loop_stop", "reason": state["stop_reason"]})
        return state

    plan = state.get("plan") or {}
    action = str(plan.get("action", "retrieve"))
    mode_hint = str(plan.get("mode_hint") or state.get("mode_hint") or "general")
    requested_k = int(state.get("request_k") or DEFAULT_TOP_K)
    doc_ids = list(state.get("doc_ids") or [])

    state["tool_calls"] = int(state.get("tool_calls", 0)) + 1
    state["retrieval_round"] = int(state.get("retrieval_round", 0)) + 1
    round_no = int(state.get("retrieval_round", 0))
    add_trace(
        state,
        {
            "type": "retrieval_round_started",
            "round": round_no,
            "action": action,
            "tool_calls": state["tool_calls"],
        },
    )

    tool_out: Dict[str, Any]
    if action == "retrieve":
        query = str(plan.get("query") or state.get("canonical_query") or state["question"])
        tool_out = lc_tools.retrieve.invoke(
            {
                "query": query,
                "k": requested_k,
                "mode_hint": mode_hint,
                "doc_ids": doc_ids or None,
                "use_query_fusion": False,
                "enable_mode_variants": False,
            }
        )
    elif action == "resolve_definition":
        args = plan.get("args", {}) or {}
        tool_out = lc_tools.resolve_definition.invoke(
            {
                "term": args.get("term", state.get("canonical_query") or state["question"]),
                "query": plan.get("query") or state.get("canonical_query") or state["question"],
                "k": requested_k,
                "doc_ids": doc_ids or None,
                "use_query_fusion": False,
                "enable_mode_variants": False,
            }
        )
    elif action == "compare":
        args = plan.get("args", {}) or {}
        tool_out = lc_tools.compare.invoke(
            {
                "topic_a": args.get("topic_a", state["question"]),
                "topic_b": args.get("topic_b", state["question"]),
                "k": max(2, requested_k),
                "doc_ids": doc_ids or None,
                "use_query_fusion": False,
                "enable_mode_variants": False,
            }
        )
    elif action == "summarize":
        args = plan.get("args", {}) or {}
        tool_out = lc_tools.summarize.invoke(
            {
                "doc_id": args["doc_id"],
                "start_page": int(args["start_page"]),
                "end_page": int(args["end_page"]),
                "k": int(args.get("k", 30)),
            }
        )
    else:
        state["stop_reason"] = f"unsupported_action:{action}"
        add_trace(state, {"type": "tool_skip", "reason": state["stop_reason"]})
        return state

    incoming = _to_evidence_items(tool_out.get("evidence", []))
    existing = _to_evidence_items(state.get("evidence", []))
    merged = _merge_evidence(existing, incoming)
    set_evidence(state, merged)
    _record_timing(state, "retrieve", started_at)

    stats = {
        "round": round_no,
        "action": action,
        "new_hits": len(incoming),
        "total_hits": len(merged),
        "tool_stats": tool_out.get("stats", {}),
        "mode_hint": tool_out.get("mode_hint") or mode_hint,
        "doc_ids": doc_ids,
    }
    state["last_retrieval_stats"] = stats
    add_trace(state, {"type": "retrieval_round_result", **stats})
    return state


def node_assess_evidence(state: AgentState) -> AgentState:
    _bump_step(state, "assess_evidence")

    evidence = _to_evidence_items(state.get("evidence", []))
    anchors = list(state.get("required_anchors") or [])
    anchor_match = _evidence_contains_any_anchor(evidence, anchors)
    compare_topics = state.get("compare_topics") or {}
    compare_required = bool(compare_topics.get("topic_a") and compare_topics.get("topic_b"))
    doc_diversity = _doc_diversity(evidence)

    reasons: List[str] = []
    if len(evidence) < MIN_EVIDENCE_HITS:
        reasons.append("insufficient_hits")
    if anchors and not anchor_match:
        reasons.append("anchor_missing")
    if compare_required and doc_diversity < 2:
        reasons.append("compare_doc_diversity_missing")

    sufficient = len(reasons) == 0
    state["evidence_sufficient"] = sufficient

    budget_reason = None
    if not sufficient:
        budget_reason = _budget_limit_reason(state)
        if budget_reason:
            state["stop_reason"] = budget_reason
        else:
            state["stop_reason"] = reasons[0]
    else:
        state["stop_reason"] = "sufficient_evidence"

    add_trace(
        state,
        {
            "type": "assessment_decision",
            "sufficient": sufficient,
            "reasons": reasons,
            "budget_reason": budget_reason,
            "evidence_hits": len(evidence),
            "doc_diversity": doc_diversity,
            "anchors": anchors,
            "anchor_match": anchor_match,
            "tool_calls": state.get("tool_calls", 0),
            "steps": state.get("steps", 0),
            "retrieval_round": state.get("retrieval_round", 0),
        },
    )
    return state


def node_refine_query(state: AgentState) -> AgentState:
    _bump_step(state, "refine_query")

    if _step_limit_hit(state):
        state["stop_reason"] = "step_budget_exhausted"
        add_trace(state, {"type": "loop_stop", "reason": state["stop_reason"]})
        return state

    previous_plan = state.get("plan") or {}
    previous_query = str(previous_plan.get("query") or state.get("canonical_query") or state["question"])
    refined_query, strategy = _build_refined_query(state)
    mode_hint = str(state.get("mode_hint") or previous_plan.get("mode_hint") or "general")
    next_action = "resolve_definition" if str(previous_plan.get("action")) == "resolve_definition" else "retrieve"
    next_args = {"term": state.get("canonical_query") or refined_query} if next_action == "resolve_definition" else {}

    set_plan(
        state,
        Plan(
            action=next_action,
            reason=f"Refined retrieval query via {strategy}.",
            query=refined_query,
            args=next_args,
            mode_hint=mode_hint,
        ),
    )
    add_trace(
        state,
        {
            "type": "query_refined",
            "strategy": strategy,
            "previous_query": previous_query,
            "refined_query": refined_query,
        },
    )
    return state


def node_answer(state: AgentState) -> AgentState:
    _bump_step(state, "answer")
    started_at = perf_counter()

    if not bool(state.get("evidence_sufficient", False)):
        add_trace(state, {"type": "answer_skip", "reason": "insufficient_evidence"})
        return state

    evidence = _to_evidence_items(state.get("evidence", []))
    if not evidence:
        add_trace(state, {"type": "answer_skip", "reason": "no_evidence"})
        return state

    out = _call_rag_answer(str(state.get("answer_prompt_question") or state["question"]), evidence)
    answer_text = out.get("answer", "").strip()

    cits: list[Citation] = []
    for c in out.get("citations", []) or []:
        key = c.get("key")
        cits.append(
            Citation(
                doc_id=str(c["doc_id"]),
                start_page=int(c["start_page"]),
                end_page=int(c["end_page"]),
                chunk_id=str(c.get("chunk_id", "")),
                key=(str(key) if key is not None and str(key).strip() else None),
            )
        )

    set_answer(state, answer_text, cits)
    _record_timing(state, "generate", started_at)
    return state


def node_verify_or_refuse(state: AgentState) -> AgentState:
    _bump_step(state, "verify_or_refuse")

    evidence = _to_evidence_items(state.get("evidence", []))
    citations = state.get("citations", []) or []
    draft = (state.get("draft_answer") or "").strip()
    sufficient = bool(state.get("evidence_sufficient", False))

    should_refuse = (not sufficient) or (not draft) or (not evidence) or (len(citations) == 0)
    if should_refuse:
        refusal_reason = _derive_refusal_reason(
            state,
            sufficient=sufficient,
            draft=draft,
            evidence=evidence,
            citations=citations,
        )
        state["refusal_reason"] = refusal_reason
        msg = _refusal_message(refusal_reason)
        state["citations"] = []
        set_final_answer(state, msg)
        add_trace(
            state,
            {
                "type": "verify",
                "result": "refuse",
                "stop_reason": state.get("stop_reason", ""),
                "refusal_reason": refusal_reason,
                "citations": 0,
            },
        )
        return state

    state["refusal_reason"] = ""
    set_final_answer(state, draft)
    add_trace(
        state,
        {
            "type": "verify",
            "result": "ok",
            "stop_reason": state.get("stop_reason", ""),
            "refusal_reason": "",
            "citations": len(citations),
        },
    )
    return state


# -----------------------------
# Graph assembly
# -----------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("analyze_query", node_analyze_query)
    g.add_node("route", node_route)
    g.add_node("retrieve", node_retrieve)
    g.add_node("assess_evidence", node_assess_evidence)
    g.add_node("refine_query", node_refine_query)
    g.add_node("answer", node_answer)
    g.add_node("verify_or_refuse", node_verify_or_refuse)

    g.set_entry_point("analyze_query")

    def _route_edge(state: AgentState) -> str:
        action = str((state.get("plan") or {}).get("action", "retrieve"))
        if action in {"retrieve", "resolve_definition", "compare", "summarize"}:
            return "retrieve"
        if action == "answer":
            return "answer"
        return "verify_or_refuse"

    def _assess_edge(state: AgentState) -> str:
        if bool(state.get("evidence_sufficient", False)):
            return "answer"
        if _budget_limit_reason(state) is not None:
            return "verify_or_refuse"
        return "refine_query"

    def _refine_edge(state: AgentState) -> str:
        if _budget_limit_reason(state) is not None:
            return "verify_or_refuse"
        return "retrieve"

    g.add_conditional_edges(
        "route",
        _route_edge,
        {
            "retrieve": "retrieve",
            "answer": "answer",
            "verify_or_refuse": "verify_or_refuse",
        },
    )

    g.add_edge("analyze_query", "route")
    g.add_edge("retrieve", "assess_evidence")
    g.add_conditional_edges(
        "assess_evidence",
        _assess_edge,
        {
            "answer": "answer",
            "refine_query": "refine_query",
            "verify_or_refuse": "verify_or_refuse",
        },
    )
    g.add_conditional_edges(
        "refine_query",
        _refine_edge,
        {
            "retrieve": "retrieve",
            "verify_or_refuse": "verify_or_refuse",
        },
    )

    g.add_edge("answer", "verify_or_refuse")
    g.add_edge("verify_or_refuse", END)

    return g.compile()


GRAPH = build_graph()


def run_agent(question: str, *, k: int | None = None) -> AgentState:
    started_at = perf_counter()
    state = init_state(question, k=k)
    recursion_limit = max(20, MAX_STEPS * 4)
    out = GRAPH.invoke(state, config={"recursion_limit": recursion_limit})
    out.setdefault("timing_ms", {})["total"] = round((perf_counter() - started_at) * 1000.0, 3)
    return out

if __name__ == "__main__":
    # Render graph image for terminal usage.
    try:
        png_bytes = GRAPH.get_graph().draw_mermaid_png()
        out_path = Path.cwd() / "runs" / "agent" / "graph.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png_bytes)
        print(f"Graph image saved to: {out_path}")

        try:
            if sys.platform.startswith("win"):
                import os

                os.startfile(str(out_path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(out_path)], check=False)
            else:
                subprocess.run(["xdg-open", str(out_path)], check=False)
        except Exception as open_err:
            print(f"Saved image but could not auto-open it: {open_err}")
    except Exception:
        print(GRAPH.get_graph().draw_mermaid())
