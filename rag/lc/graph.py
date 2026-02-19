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

import re
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from rag.config import SETTINGS

from . import tools as lc_tools
from .state import AgentState, Citation, EvidenceItem, Plan
from .state_utils import add_trace, init_state, set_answer, set_evidence, set_final_answer, set_plan


MAX_STEPS = SETTINGS.AGENT_MAX_STEPS
MAX_TOOL_CALLS = SETTINGS.AGENT_MAX_TOOL_CALLS
MAX_RETRIEVAL_ROUNDS = SETTINGS.AGENT_MAX_RETRIEVAL_ROUNDS
MIN_EVIDENCE_HITS = SETTINGS.AGENT_MIN_EVIDENCE_HITS

_ANCHOR_PATTERNS = (
    re.compile(r"\bAlgorithm\s+\d+\b", flags=re.IGNORECASE),
    re.compile(r"\bTable\s+\d+\b", flags=re.IGNORECASE),
    re.compile(r"\bSection\s+\d+(?:\.\d+)*\b", flags=re.IGNORECASE),
)
_ANCHOR_KEYWORDS = ("keygen", "encaps", "decaps", "shake128", "shake256", "xof")


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


def _heuristic_route(question: str) -> Plan:
    q = question.strip()
    ql = q.lower()

    if any(x in ql for x in ["compare", "difference between", "differences between", "vs", "versus"]):
        topic_a, topic_b = _extract_compare_topics(q)
        if topic_a and topic_b:
            return Plan(
                action="compare",
                reason="Comparison intent detected with parsed topics.",
                args={"topic_a": topic_a, "topic_b": topic_b},
                mode_hint="general",
            )
        return Plan(
            action="retrieve",
            reason="Comparison intent detected but topics were ambiguous; using broad retrieval.",
            query=q,
            mode_hint="general",
        )

    if "algorithm" in ql or "shake" in ql:
        return Plan(action="retrieve", reason="Algorithm-like query detected; retrieve evidence.", query=q, mode_hint="algorithm")

    if ql.startswith(("what is", "what's", "define", "explain")):
        term = q.split(" ", 2)[-1].strip(" ?")
        return Plan(action="resolve_definition", reason="Definition intent detected.", args={"term": term}, mode_hint="definition")

    return Plan(action="retrieve", reason="Default to retrieval.", query=q, mode_hint="general")


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

    seen_terms = set()
    for pattern in _ANCHOR_PATTERNS:
        for m in pattern.finditer(question):
            token = m.group(0).strip()
            if not token:
                continue
            key = token.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            terms.append(token)

    for keyword in _ANCHOR_KEYWORDS:
        if keyword in ql and keyword not in seen_terms:
            seen_terms.add(keyword)
            terms.append(keyword)

    return terms


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


def _plan_mode_hint(plan: Dict[str, Any], fallback_query: str) -> Optional[str]:
    mode_hint = plan.get("mode_hint")
    if mode_hint:
        return str(mode_hint)
    ql = fallback_query.lower()
    if "algorithm" in ql or "shake" in ql:
        return "algorithm"
    if ql.startswith(("define", "what is", "what's", "explain")):
        return "definition"
    return "general"


def _build_refined_query(state: AgentState) -> Tuple[str, str]:
    plan = state.get("plan") or {}
    base_query = str(plan.get("query") or state["question"]).strip()
    reason = str(state.get("stop_reason", "")).lower()
    anchors = _extract_anchor_terms(state["question"])

    if "anchor_missing" in reason and anchors:
        return _append_terms(base_query, anchors), "anchor_token_bias"

    topic_a, topic_b = _extract_compare_topics(state["question"])
    if "compare_doc_diversity_missing" in reason and topic_a and topic_b:
        doc_tokens = _topic_doc_bias_tokens(topic_a) + _topic_doc_bias_tokens(topic_b)
        terms = [topic_a, topic_b, "comparison", "intended use-cases", *doc_tokens]
        return _append_terms(base_query, terms), "compare_doc_bias"

    if "insufficient_hits" in reason:
        if str(plan.get("action")) == "resolve_definition":
            args = plan.get("args", {}) or {}
            term = str(args.get("term") or state["question"]).strip()
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
    from rag.llm.gemini import make_generate_fn
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

    generate_fn = make_generate_fn()
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

def node_route(state: AgentState) -> AgentState:
    _bump_step(state, "route")

    if _step_limit_hit(state):
        state["stop_reason"] = "step_budget_exhausted"
        set_plan(state, Plan(action="refuse", reason="Step budget exhausted before routing."))
        add_trace(state, {"type": "loop_stop", "reason": state["stop_reason"]})
        return state

    plan = _heuristic_route(state["question"])
    set_plan(state, plan)
    return state


def node_retrieve(state: AgentState) -> AgentState:
    _bump_step(state, "retrieve")

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
        query = str(plan.get("query") or state["question"])
        tool_out = lc_tools.retrieve.invoke({"query": query, "k": 8})
    elif action == "resolve_definition":
        args = plan.get("args", {}) or {}
        tool_out = lc_tools.resolve_definition.invoke({"term": args.get("term", state["question"]), "k": 8})
    elif action == "compare":
        args = plan.get("args", {}) or {}
        tool_out = lc_tools.compare.invoke(
            {
                "topic_a": args.get("topic_a", state["question"]),
                "topic_b": args.get("topic_b", state["question"]),
                "k": 6,
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

    stats = {
        "round": round_no,
        "action": action,
        "new_hits": len(incoming),
        "total_hits": len(merged),
        "tool_stats": tool_out.get("stats", {}),
        "mode_hint": tool_out.get("mode_hint") or _plan_mode_hint(plan, state["question"]),
    }
    state["last_retrieval_stats"] = stats
    add_trace(state, {"type": "retrieval_round_result", **stats})
    return state


def node_assess_evidence(state: AgentState) -> AgentState:
    _bump_step(state, "assess_evidence")

    evidence = _to_evidence_items(state.get("evidence", []))
    anchors = _extract_anchor_terms(state["question"])
    anchor_match = _evidence_contains_any_anchor(evidence, anchors)
    compare_required = _extract_compare_topics(state["question"])[0] is not None
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
    previous_query = str(previous_plan.get("query") or state["question"])
    refined_query, strategy = _build_refined_query(state)
    mode_hint = _plan_mode_hint(previous_plan, refined_query)

    set_plan(
        state,
        Plan(
            action="retrieve",
            reason=f"Refined retrieval query via {strategy}.",
            query=refined_query,
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

    if not bool(state.get("evidence_sufficient", False)):
        add_trace(state, {"type": "answer_skip", "reason": "insufficient_evidence"})
        return state

    evidence = _to_evidence_items(state.get("evidence", []))
    if not evidence:
        add_trace(state, {"type": "answer_skip", "reason": "no_evidence"})
        return state

    out = _call_rag_answer(state["question"], evidence)
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

    g.add_node("route", node_route)
    g.add_node("retrieve", node_retrieve)
    g.add_node("assess_evidence", node_assess_evidence)
    g.add_node("refine_query", node_refine_query)
    g.add_node("answer", node_answer)
    g.add_node("verify_or_refuse", node_verify_or_refuse)

    g.set_entry_point("route")

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


def run_agent(question: str) -> AgentState:
    state = init_state(question)
    recursion_limit = max(20, MAX_STEPS * 4)
    return GRAPH.invoke(state, config={"recursion_limit": recursion_limit})
