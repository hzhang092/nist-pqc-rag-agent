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

def init_state(question: str, *, k: Optional[int] = None) -> AgentState:
    return {
        "question": question,
        "original_query": question,
        "canonical_query": question,
        "mode_hint": "general",
        "required_anchors": [],
        "compare_topics": None,
        "doc_ids": [],
        "doc_family": "",
        "analysis_notes": "",
        "answer_prompt_question": question,
        "query_analysis": {},
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
    }

def add_trace(state: AgentState, event: Dict[str, Any]) -> None:
    state.setdefault("trace", []).append(event)

def set_plan(state: AgentState, plan: Plan) -> None:
    state["plan"] = plan.to_dict()
    add_trace(state, {"type": "plan", **state["plan"]})


def set_query_analysis(state: AgentState, analysis: QueryAnalysis) -> None:
    payload = analysis.to_dict()
    state["original_query"] = analysis.original_query
    state["canonical_query"] = analysis.canonical_query
    state["mode_hint"] = analysis.mode_hint
    state["required_anchors"] = list(analysis.required_anchors)
    state["compare_topics"] = (
        analysis.compare_topics.to_dict() if analysis.compare_topics is not None else None
    )
    state["doc_ids"] = list(analysis.doc_ids or [])
    state["doc_family"] = analysis.doc_family or ""
    state["analysis_notes"] = analysis.analysis_notes or ""
    state["query_analysis"] = payload
    state["answer_prompt_question"] = analysis.original_query
    add_trace(state, {"type": "analysis", **payload})

def set_evidence(state: AgentState, evidence: List[EvidenceItem]) -> None:
    state["evidence"] = [e.to_dict() for e in evidence]
    add_trace(state, {"type": "evidence", "n": len(evidence)})

def set_answer(state: AgentState, answer: str, citations: List[Citation]) -> None:
    state["draft_answer"] = answer
    state["citations"] = [c.to_dict() for c in citations]
    add_trace(state, {"type": "answer", "citations": len(citations)})


def set_final_answer(state: AgentState, answer: str) -> None:
    state["final_answer"] = answer
    add_trace(state, {"type": "final_answer"})
