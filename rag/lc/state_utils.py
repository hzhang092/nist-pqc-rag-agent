"""
LangGraph AgentState helper utilities.

What this module is for:
- Initializes the agent state used by the LangGraph workflow.
- Provides small, deterministic state mutation helpers for query analysis, plan, evidence, and answer updates.
- Appends trace events consistently so downstream trace JSON is easy to inspect.

How it is used:
- Called by graph nodes in rag.lc.graph to update AgentState in one place.
- Keeps state-shape logic centralized and avoids duplicated mutation code.
- Functions like set_query_analysis() unpack structured analysis into flat state fields for easier access.

Key functions:
- init_state(): Creates a new AgentState with all required fields initialized.
- set_query_analysis(): Unpacks QueryAnalysis into state fields (canonical_query, mode_hint, doc_ids, etc.).
- set_evidence(): Stores retrieved evidence items and logs trace event.
- set_answer(): Stores draft answer with citations.
- set_final_answer(): Stores the final answer after any post-processing.
- add_trace(): Appends structured events to the trace log for debugging and observability.

CLI flags:
- None. This is a library module (non-CLI) and is not executed via command-line flags.
"""
# rag/lc/state_utils.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .state import AgentState, Citation, EvidenceItem, Plan, QueryAnalysis


def _preview_text(text: str, *, limit: int = 240) -> str:
    preview = str(text or "").strip().replace("\n", " ")
    if len(preview) > limit:
        return preview[:limit] + "..."
    return preview


def _top_doc_ids(evidence: List[EvidenceItem], *, limit: int = 5) -> List[str]:
    doc_ids: List[str] = []
    seen = set()
    for item in evidence:
        if item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        doc_ids.append(item.doc_id)
        if len(doc_ids) >= limit:
            break
    return doc_ids


def _plan_args_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_ids": list(args.get("doc_ids") or [])[:5],
        "protected_spans": list(args.get("protected_spans") or [])[:5],
        "subqueries": list(args.get("subqueries") or [])[:5],
        "sparse_query": _preview_text(str(args.get("sparse_query") or "")),
        "dense_query": _preview_text(str(args.get("dense_query") or "")),
    }

def init_state(
    question: str,
    *,
    k: Optional[int] = None,
    use_graph_lookup: bool = True,
) -> AgentState:
    return {
        "question": question,
        "original_query": question,
        "canonical_query": question,
        "mode_hint": "general",
        "rewrite_needed": False,
        "protected_spans": [],
        "required_anchors": [],
        "sparse_query": question,
        "dense_query": question,
        "subqueries": [],
        "confidence": 0.0,
        "compare_topics": None,
        "doc_ids": [],
        "doc_family": "",
        "analysis_notes": "",
        "answer_prompt_question": question,
        "query_analysis": {},
        "graph_lookup_enabled": bool(use_graph_lookup),
        "graph_lookup": {},
        "request_k": int(k) if k is not None else 0,
        "evidence": [],
        "citations": [],
        "draft_answer": "",
        "final_answer": "",
        "tool_calls": 0,
        "steps": 0,
        "retrieval_round": 0,
        "evidence_sufficient": False,
        "stop_reason": "",
        "refusal_reason": "",
        "last_retrieval_stats": {},
        "timing_ms": {
            "analyze": 0.0,
            "retrieve": 0.0,
            "generate": 0.0,
            "total": 0.0,
        },
        "trace": [],
        "errors": [],
        "_trace_active_node": "",
    }

def add_trace(state: AgentState, event: Dict[str, Any]) -> None:
    payload = dict(event)
    if not payload.get("node"):
        active_node = str(state.get("_trace_active_node") or "")
        if active_node:
            payload["node"] = active_node
    state.setdefault("trace", []).append(payload)

def set_plan(state: AgentState, plan: Plan) -> None:
    state["plan"] = plan.to_dict()
    plan_payload = state["plan"]
    add_trace(
        state,
        {
            "type": "plan_applied",
            "action": plan_payload.get("action"),
            "reason": _preview_text(str(plan_payload.get("reason") or "")),
            "query": _preview_text(str(plan_payload.get("query") or "")),
            "mode_hint": plan_payload.get("mode_hint"),
            "args_summary": _plan_args_summary(plan_payload.get("args") or {}),
        },
    )


def set_query_analysis(state: AgentState, analysis: QueryAnalysis) -> None:
    payload = analysis.to_dict()
    state["original_query"] = analysis.original_query
    state["canonical_query"] = analysis.canonical_query
    state["mode_hint"] = analysis.mode_hint
    state["rewrite_needed"] = bool(analysis.rewrite_needed)
    state["protected_spans"] = list(analysis.protected_spans)
    state["required_anchors"] = list(analysis.required_anchors)
    state["sparse_query"] = analysis.sparse_query
    state["dense_query"] = analysis.dense_query
    state["subqueries"] = list(analysis.subqueries)
    state["confidence"] = float(analysis.confidence)
    state["compare_topics"] = (
        analysis.compare_topics.to_dict() if analysis.compare_topics is not None else None
    )
    state["doc_ids"] = list(analysis.doc_ids or [])
    state["doc_family"] = analysis.doc_family or ""
    state["analysis_notes"] = analysis.analysis_notes or ""
    state["query_analysis"] = payload
    state["answer_prompt_question"] = analysis.answer_prompt_question or analysis.original_query
    add_trace(
        state,
        {
            "type": "analysis_applied",
            "mode_hint": analysis.mode_hint,
            "rewrite_needed": bool(analysis.rewrite_needed),
            "protected_spans": list(analysis.protected_spans or [])[:5],
            "doc_ids": list(analysis.doc_ids or [])[:5],
            "sparse_query": _preview_text(analysis.sparse_query),
            "dense_query": _preview_text(analysis.dense_query),
            "subqueries_count": len(analysis.subqueries or []),
            "analysis_notes": _preview_text(analysis.analysis_notes or ""),
        },
    )

def set_evidence(state: AgentState, evidence: List[EvidenceItem]) -> None:
    state["evidence"] = [e.to_dict() for e in evidence]
    add_trace(
        state,
        {
            "type": "evidence_updated",
            "round": int(state.get("retrieval_round", 0)),
            "total_hits": len(evidence),
            "top_chunk_ids": [item.chunk_id for item in evidence[:5]],
            "top_doc_ids": _top_doc_ids(evidence),
        },
    )

def set_answer(
    state: AgentState,
    answer: str,
    citations: List[Citation],
    *,
    timing_ms_generate: float | None = None,
) -> None:
    state["draft_answer"] = answer
    state["citations"] = [c.to_dict() for c in citations]
    citation_keys = [c.key for c in citations if c.key][:5]
    add_trace(
        state,
        {
            "type": "answer_drafted",
            "answer_prompt_question": _preview_text(str(state.get("answer_prompt_question") or state.get("question") or "")),
            "draft_len": len(answer.strip()),
            "citations_count": len(citations),
            "citation_keys": citation_keys,
            "timing_ms_generate": round(float(timing_ms_generate or 0.0), 3),
        },
    )


def set_final_answer(
    state: AgentState,
    answer: str,
    *,
    result: str,
    used_refusal_template: bool,
) -> None:
    state["final_answer"] = answer
    add_trace(
        state,
        {
            "type": "final_answer_set",
            "result": result,
            "final_len": len(answer.strip()),
            "used_refusal_template": bool(used_refusal_template),
            "stop_reason": str(state.get("stop_reason") or ""),
            "refusal_reason": str(state.get("refusal_reason") or ""),
        },
    )
