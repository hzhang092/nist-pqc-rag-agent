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


def _extract_doc_reference_spans(question: str) -> List[str]:
    spans: List[str] = []
    for pattern, _doc_id, _family in _DOC_PATTERNS:
        for match in pattern.finditer(question):
            token = re.sub(r"\s+", " ", match.group(0).strip())
            if token:
                spans.append(token)
    return _dedupe_strs(spans)


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


def _planner_query_text(value: Any, *, fallback: str, limit: int = 240) -> str:
    text = normalize_identifier_like_spans(str(value or "").strip())
    if not text:
        text = normalize_identifier_like_spans(fallback.strip())
    return text[:limit].strip()


def _sanitize_float(value: Any, *, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    return max(0.0, min(1.0, round(number, 3)))


def _extract_protected_spans(question: str) -> List[str]:
    anchor_terms = _extract_anchor_terms(question)
    doc_terms = _extract_doc_reference_spans(question)
    return _dedupe_strs(anchor_terms + doc_terms)


def _is_doc_reference_span(span: str) -> bool:
    doc_ids, _doc_family = _detect_doc_scope(span)
    return bool(doc_ids)


def _assessable_protected_spans(protected_spans: List[str]) -> List[str]:
    return [span for span in _dedupe_strs(protected_spans) if not _is_doc_reference_span(span)]


def _topic_expected_doc_id(topic: str) -> Optional[str]:
    doc_ids, _doc_family = _detect_doc_scope(topic)
    return doc_ids[0] if doc_ids else None


def _base_compare_query(compare_topics: CompareTopics) -> str:
    return f"{compare_topics.topic_a} {compare_topics.topic_b} intended use-cases comparison"


def _default_compare_subqueries(compare_topics: CompareTopics) -> List[str]:
    return [
        f"{compare_topics.topic_a} intended use-cases and deployment context",
        f"{compare_topics.topic_b} intended use-cases and deployment context",
    ]


def _compute_planner_queries(
    *,
    original_query: str,
    canonical_query: str,
    mode_hint: str,
    protected_spans: List[str],
    compare_topics: Optional[CompareTopics],
    definition_term: Optional[str],
    doc_family: Optional[str],
) -> tuple[bool, str, str, List[str], str]:
    normalized_question = normalize_identifier_like_spans(original_query.strip())
    protected_tokens = [span for span in protected_spans if span.lower() not in normalized_question.lower()]
    answer_prompt_question = original_query.strip() or canonical_query

    if mode_hint == "compare" and compare_topics is not None:
        sparse_query = _append_terms(
            _base_compare_query(compare_topics),
            protected_spans + _topic_doc_bias_tokens(compare_topics.topic_a) + _topic_doc_bias_tokens(compare_topics.topic_b),
        )
        dense_query = (
            f"compare intended use-cases and deployment differences between "
            f"{compare_topics.topic_a} and {compare_topics.topic_b}"
        )
        return True, sparse_query, dense_query, _default_compare_subqueries(compare_topics), answer_prompt_question

    if mode_hint == "definition":
        target = definition_term or canonical_query or normalized_question
        sparse_query = _append_terms(target, ["definition", *protected_tokens])
        dense_suffix = f" in {doc_family}" if doc_family else " in NIST PQC standards"
        dense_query = f"definition and notation for {target}{dense_suffix}"
        rewrite_needed = normalize_identifier_like_spans(target).lower() != normalized_question.lower()
        return rewrite_needed, sparse_query, dense_query, [], answer_prompt_question

    if mode_hint == "algorithm":
        anchor_text = " ".join(_assessable_protected_spans(protected_spans)) or canonical_query or normalized_question
        sparse_query = _append_terms(normalized_question or canonical_query, protected_tokens)
        dense_query = f"steps and procedure for {anchor_text}"
        if doc_family:
            dense_query = f"{dense_query} in {doc_family}"
        return False, sparse_query, dense_query, [], answer_prompt_question

    sparse_query = _append_terms(normalized_question or canonical_query, protected_tokens)
    dense_query = normalized_question or canonical_query
    return False, sparse_query, dense_query, [], answer_prompt_question


def _deterministic_confidence(
    *,
    compare_topics: Optional[CompareTopics],
    protected_spans: List[str],
    doc_ids: List[str],
    definition_term: Optional[str],
) -> float:
    score = 0.52
    if compare_topics is not None:
        score += 0.16
    if protected_spans:
        score += 0.12
    if doc_ids:
        score += 0.08
    if definition_term:
        score += 0.05
    return min(0.95, round(score, 3))


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
    """Builds a bounded deterministic retrieval plan used as the planner guardrail and fallback."""

    original_query = question.strip()
    topic_a, topic_b = _extract_compare_topics(original_query)
    compare_topics = (
        CompareTopics(topic_a=topic_a, topic_b=topic_b)
        if topic_a and topic_b
        else None
    )
    protected_spans = _extract_protected_spans(original_query)
    required_anchors = _assessable_protected_spans(protected_spans)
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
    rewrite_needed, sparse_query, dense_query, subqueries, answer_prompt_question = _compute_planner_queries(
        original_query=original_query,
        canonical_query=canonical_query,
        mode_hint=mode_hint,
        protected_spans=protected_spans,
        compare_topics=compare_topics,
        definition_term=definition_term,
        doc_family=doc_family,
    )
    analysis_notes = "deterministic-fallback"
    return QueryAnalysis(
        original_query=original_query,
        canonical_query=canonical_query,
        mode_hint=mode_hint,  # type: ignore[arg-type]
        rewrite_needed=rewrite_needed,
        protected_spans=protected_spans,
        required_anchors=required_anchors,
        sparse_query=sparse_query,
        dense_query=dense_query,
        subqueries=subqueries,
        confidence=_deterministic_confidence(
            compare_topics=compare_topics,
            protected_spans=protected_spans,
            doc_ids=doc_ids,
            definition_term=definition_term,
        ),
        compare_topics=compare_topics,
        doc_ids=doc_ids,
        doc_family=doc_family,
        analysis_notes=analysis_notes,
        answer_prompt_question=answer_prompt_question,
    )


def _analysis_system_prompt() -> str:
    allowed_doc_ids = ", ".join(ALLOWED_DOC_IDS)
    return (
        "You are a retrieval planner for a bounded hybrid PQC standards retriever.\n"
        "Return JSON only, no markdown, no prose.\n"
        "Schema keys:\n"
        '- canonical_query: string\n'
        '- mode_hint: one of "definition", "algorithm", "compare", "general"\n'
        '- rewrite_needed: boolean\n'
        '- protected_spans: array of strings\n'
        '- required_anchors: array of strings\n'
        '- sparse_query: string\n'
        '- dense_query: string\n'
        '- subqueries: array of strings\n'
        '- compare_topics: null or {"topic_a": string, "topic_b": string}\n'
        f'- doc_ids: array of strings chosen only from [{allowed_doc_ids}]\n'
        '- doc_family: null or short string\n'
        '- answer_prompt_question: string\n'
        '- analysis_notes: null or short string\n'
        '- confidence: number between 0 and 1\n'
        "Rules:\n"
        "- Preserve every protected span exactly.\n"
        "- Do not invent doc_ids.\n"
        "- Emit subqueries only for compare mode; at most 2 subqueries.\n"
        "- Keep sparse_query lexical and dense_query semantic.\n"
        "- Prefer rewrite_needed=false when the user query is already precise."
    )


def _analysis_user_prompt(question: str, deterministic: QueryAnalysis) -> str:
    return (
        f"Question: {question.strip()}\n"
        f"Deterministic guardrails: {json.dumps(deterministic.to_dict(), ensure_ascii=False)}\n"
        "Return a retrieval plan JSON object that follows the schema exactly."
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


def _normalize_planner_doc_ids(raw_doc_ids: Any, deterministic: QueryAnalysis) -> List[str]:
    if not deterministic.doc_ids:
        return []
    normalized = _normalize_doc_ids(raw_doc_ids)
    allowed = set(deterministic.doc_ids)
    normalized = [doc_id for doc_id in normalized if doc_id in allowed]
    return normalized or list(deterministic.doc_ids or [])


def _normalize_protected_spans(value: Any, deterministic: QueryAnalysis) -> List[str]:
    raw_items = value if isinstance(value, list) else []
    candidate_spans = _dedupe_strs([str(item).strip() for item in raw_items])
    protected_spans = list(deterministic.protected_spans or [])
    for span in candidate_spans:
        if span.lower() not in {item.lower() for item in protected_spans}:
            protected_spans.append(span)
    return _dedupe_strs(protected_spans)


def _normalize_subqueries(value: Any, *, mode_hint: str, deterministic: QueryAnalysis) -> List[str]:
    if mode_hint != "compare":
        return []
    raw_items = value if isinstance(value, list) else list(deterministic.subqueries or [])
    cleaned = [
        _planner_query_text(item, fallback="")
        for item in raw_items
        if str(item or "").strip()
    ]
    out = _dedupe_strs([item for item in cleaned if item])
    return out[:2] if out else list(deterministic.subqueries or [])[:2]


def _validate_query_analysis(raw: Dict[str, Any], deterministic: QueryAnalysis) -> QueryAnalysis:
    compare_topics = _normalize_compare_topics(raw.get("compare_topics")) or deterministic.compare_topics
    mode_hint = str(raw.get("mode_hint") or deterministic.mode_hint).strip().lower()
    if compare_topics is not None:
        mode_hint = "compare"
    if mode_hint not in ANALYSIS_MODE_VALUES:
        mode_hint = deterministic.mode_hint

    canonical_query = _planner_query_text(raw.get("canonical_query"), fallback=deterministic.canonical_query)
    rewrite_needed = bool(raw.get("rewrite_needed", deterministic.rewrite_needed))
    protected_spans = _normalize_protected_spans(raw.get("protected_spans"), deterministic)
    required_anchors = _assessable_protected_spans(protected_spans)
    sparse_query = _planner_query_text(raw.get("sparse_query"), fallback=deterministic.sparse_query)
    dense_query = _planner_query_text(raw.get("dense_query"), fallback=deterministic.dense_query)
    subqueries = _normalize_subqueries(raw.get("subqueries"), mode_hint=mode_hint, deterministic=deterministic)
    doc_ids = _normalize_planner_doc_ids(raw.get("doc_ids"), deterministic)
    doc_family = deterministic.doc_family or None
    answer_prompt_question = _planner_query_text(
        raw.get("answer_prompt_question"),
        fallback=deterministic.answer_prompt_question or deterministic.original_query,
        limit=320,
    )
    analysis_notes = _sanitize_analysis_notes(raw.get("analysis_notes")) or deterministic.analysis_notes
    confidence = _sanitize_float(raw.get("confidence"), fallback=deterministic.confidence)

    return QueryAnalysis(
        original_query=deterministic.original_query,
        canonical_query=canonical_query,
        mode_hint=mode_hint,  # type: ignore[arg-type]
        rewrite_needed=rewrite_needed,
        protected_spans=protected_spans,
        required_anchors=required_anchors,
        sparse_query=sparse_query,
        dense_query=dense_query,
        subqueries=subqueries,
        confidence=confidence,
        compare_topics=compare_topics,
        doc_ids=doc_ids,
        doc_family=doc_family,
        analysis_notes=analysis_notes,
        answer_prompt_question=answer_prompt_question,
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
            notes = "llm-planned"
        return QueryAnalysis(
            original_query=analysis.original_query,
            canonical_query=analysis.canonical_query,
            mode_hint=analysis.mode_hint,
            rewrite_needed=analysis.rewrite_needed,
            protected_spans=analysis.protected_spans,
            required_anchors=analysis.required_anchors,
            sparse_query=analysis.sparse_query,
            dense_query=analysis.dense_query,
            subqueries=analysis.subqueries,
            confidence=analysis.confidence,
            compare_topics=analysis.compare_topics,
            doc_ids=analysis.doc_ids,
            doc_family=analysis.doc_family,
            analysis_notes=notes,
            answer_prompt_question=analysis.answer_prompt_question,
        )
    except Exception:
        return deterministic


def _text_contains_span(text: str, span: str) -> bool:
    normalized_text = normalize_identifier_like_spans(text or "").lower()
    normalized_span = normalize_identifier_like_spans(span or "").lower()
    if not normalized_span:
        return False
    if re.search(r"[._-]|\d|\s", normalized_span):
        return normalized_span in normalized_text
    return re.search(rf"\b{re.escape(normalized_span)}\b", normalized_text) is not None


def _missing_protected_spans(evidence: List[EvidenceItem], protected_spans: List[str]) -> List[str]:
    assessable = _assessable_protected_spans(protected_spans)
    if not assessable:
        return []
    missing: List[str] = []
    for span in assessable:
        if any(_text_contains_span(item.text, span) for item in evidence):
            continue
        missing.append(span)
    return missing


def _comparison_missing_sides(
    evidence: List[EvidenceItem],
    *,
    compare_topics: Dict[str, str],
    doc_ids: List[str],
) -> List[str]:
    topic_a = str(compare_topics.get("topic_a") or "").strip()
    topic_b = str(compare_topics.get("topic_b") or "").strip()
    topics = [topic for topic in (topic_a, topic_b) if topic]
    if len(topics) < 2:
        return []

    present_docs = {item.doc_id for item in evidence}
    expected_doc_pairs = [(topic, _topic_expected_doc_id(topic)) for topic in topics]
    if len(doc_ids) >= 2 and all(doc_id for _topic, doc_id in expected_doc_pairs):
        missing_topics = [topic for topic, doc_id in expected_doc_pairs if doc_id not in present_docs]
        if missing_topics:
            return missing_topics

    missing_topics = []
    for topic in topics:
        if any(_text_contains_span(item.text, topic) for item in evidence):
            continue
        missing_topics.append(topic)
    return missing_topics


def _build_refined_queries(state: AgentState) -> tuple[str, str, List[str], str]:
    plan = state.get("plan") or {}
    plan_args = plan.get("args", {}) or {}
    sparse_query = str(plan_args.get("sparse_query") or state.get("sparse_query") or state["question"]).strip()
    dense_query = str(plan_args.get("dense_query") or state.get("dense_query") or state["question"]).strip()
    subqueries = list(plan_args.get("subqueries") or state.get("subqueries") or [])
    reason = str(state.get("stop_reason", "")).lower()

    if reason == "missing_protected_span":
        missing_spans = _missing_protected_spans(
            _to_evidence_items(state.get("evidence", [])),
            list(state.get("protected_spans") or []),
        )
        bias_terms = missing_spans or list(state.get("required_anchors") or [])
        return (
            _append_terms(sparse_query, bias_terms),
            _append_terms(dense_query, bias_terms[:2]),
            subqueries,
            "protected_span_bias",
        )

    if reason == "wrong_doc_scope":
        return sparse_query, dense_query, subqueries, "doc_scope_retry"

    if reason == "one_sided_comparison":
        compare_topics = state.get("compare_topics") or {}
        missing_topics = _comparison_missing_sides(
            _to_evidence_items(state.get("evidence", [])),
            compare_topics=compare_topics,
            doc_ids=list(state.get("doc_ids") or []),
        )
        bias_terms: List[str] = []
        for topic in missing_topics:
            bias_terms.extend([topic, *_topic_doc_bias_tokens(topic)])
        return (
            _append_terms(sparse_query, bias_terms),
            _append_terms(dense_query, bias_terms[:4]),
            subqueries,
            "comparison_balance_bias",
        )

    if reason == "insufficient_hits":
        mode_hint = str(state.get("mode_hint") or plan.get("mode_hint") or "general")
        if mode_hint == "definition":
            return (
                _append_terms(sparse_query, ["definition", "notation", "term"]),
                _append_terms(dense_query, ["definition", "notation"]),
                subqueries,
                "definition_bias",
            )
        return (
            _append_terms(sparse_query, ["section", "algorithm", "definition"]),
            _append_terms(dense_query, ["standard", "guidance"]),
            subqueries,
            "coverage_bias",
        )

    return sparse_query, dense_query, subqueries, "no_change"


def _refusal_message(refusal_reason: str) -> str:
    rr = (refusal_reason or "").lower()
    if rr == "missing_protected_span":
        return (
            "I could not find citable evidence for the specific algorithm/table/section anchor "
            "in the indexed NIST documents."
        )
    if rr == "wrong_doc_scope":
        return "I could not find enough in-scope citable evidence in the requested NIST document set."
    if rr == "one_sided_comparison":
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

    canonical_query = str(state.get("canonical_query") or state["question"]).strip()
    mode_hint = str(state.get("mode_hint") or "general").strip() or "general"
    set_plan(
        state,
        Plan(
            action="retrieve",
            reason="Retrieve action selected from analyzed planner state.",
            query=canonical_query,
            mode_hint=mode_hint,
        ),
    )
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
    plan_args = plan.get("args", {}) or {}
    mode_hint = str(plan.get("mode_hint") or state.get("mode_hint") or "general")
    requested_k = int(state.get("request_k") or DEFAULT_TOP_K)
    doc_ids = list(plan_args.get("doc_ids") or state.get("doc_ids") or [])
    sparse_query = str(
        plan_args.get("sparse_query")
        or state.get("sparse_query")
        or plan.get("query")
        or state.get("canonical_query")
        or state["question"]
    ).strip()
    dense_query = str(
        plan_args.get("dense_query")
        or state.get("dense_query")
        or plan.get("query")
        or state.get("canonical_query")
        or state["question"]
    ).strip()
    subqueries = list(plan_args.get("subqueries") or state.get("subqueries") or [])
    protected_spans = list(plan_args.get("protected_spans") or state.get("protected_spans") or [])
    canonical_query = str(plan.get("query") or state.get("canonical_query") or state["question"]).strip()

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
        tool_out = lc_tools.retrieve.invoke(
            {
                "query": canonical_query,
                "k": requested_k,
                "mode_hint": mode_hint,
                "doc_ids": doc_ids or None,
                "canonical_query": canonical_query,
                "sparse_query": sparse_query,
                "dense_query": dense_query,
                "subqueries": subqueries,
                "protected_spans": protected_spans,
                "use_query_fusion": False,
                "enable_mode_variants": False,
            }
        )
    elif action == "summarize":
        tool_out = lc_tools.summarize.invoke(
            {
                "doc_id": plan_args["doc_id"],
                "start_page": int(plan_args["start_page"]),
                "end_page": int(plan_args["end_page"]),
                "k": int(plan_args.get("k", 30)),
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
        "sparse_query": sparse_query,
        "dense_query": dense_query,
        "subqueries": subqueries,
        "protected_spans": protected_spans,
    }
    state["last_retrieval_stats"] = stats
    add_trace(state, {"type": "retrieval_round_result", **stats})
    return state


def node_assess_evidence(state: AgentState) -> AgentState:
    _bump_step(state, "assess_evidence")

    evidence = _to_evidence_items(state.get("evidence", []))
    protected_spans = list(state.get("protected_spans") or [])
    missing_protected_spans = _missing_protected_spans(evidence, protected_spans)
    compare_topics = state.get("compare_topics") or {}
    compare_required = bool(compare_topics.get("topic_a") and compare_topics.get("topic_b"))
    doc_ids = list(state.get("doc_ids") or [])
    missing_compare_topics = _comparison_missing_sides(
        evidence,
        compare_topics=compare_topics,
        doc_ids=doc_ids,
    )
    doc_diversity = _doc_diversity(evidence)

    reasons: List[str] = []
    if doc_ids and not evidence:
        reasons.append("wrong_doc_scope")
    if missing_protected_spans:
        reasons.append("missing_protected_span")
    if compare_required and missing_compare_topics:
        reasons.append("one_sided_comparison")
    if len(evidence) < MIN_EVIDENCE_HITS:
        reasons.append("insufficient_hits")

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
            "doc_ids": doc_ids,
            "protected_spans": protected_spans,
            "missing_protected_spans": missing_protected_spans,
            "missing_compare_topics": missing_compare_topics,
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
    previous_query = str(previous_plan.get("query") or state.get("canonical_query") or state["question"]).strip()
    previous_args = previous_plan.get("args", {}) or {}
    previous_sparse = str(previous_args.get("sparse_query") or state.get("sparse_query") or previous_query).strip()
    previous_dense = str(previous_args.get("dense_query") or state.get("dense_query") or previous_query).strip()
    refined_sparse, refined_dense, refined_subqueries, strategy = _build_refined_queries(state)
    mode_hint = str(state.get("mode_hint") or previous_plan.get("mode_hint") or "general")

    set_plan(
        state,
        Plan(
            action="retrieve",
            reason=f"Refined retrieval query via {strategy}.",
            query=refined_sparse,
            args={
                "sparse_query": refined_sparse,
                "dense_query": refined_dense,
                "subqueries": refined_subqueries,
                "protected_spans": list(state.get("protected_spans") or []),
                "doc_ids": list(state.get("doc_ids") or []),
            },
            mode_hint=mode_hint,
        ),
    )
    add_trace(
        state,
        {
            "type": "query_refined",
            "strategy": strategy,
            "previous_query": previous_query,
            "refined_query": refined_sparse,
            "previous_sparse_query": previous_sparse,
            "previous_dense_query": previous_dense,
            "refined_sparse_query": refined_sparse,
            "refined_dense_query": refined_dense,
            "refined_subqueries": refined_subqueries,
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
