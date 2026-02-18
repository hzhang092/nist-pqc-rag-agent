"""
LangGraph AgentState helper utilities.

What this module is for:
- Initializes the agent state used by the LangGraph workflow.
- Provides small, deterministic state mutation helpers for plan/evidence/answer updates.
- Appends trace events consistently so downstream trace JSON is easy to inspect.

How it is used:
- Called by graph nodes in rag.lc.graph to update AgentState in one place.
- Keeps state-shape logic centralized and avoids duplicated mutation code.

CLI flags:
- None. This is a library module (non-CLI) and is not executed via command-line flags.
"""
# rag/lc/state_utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from .state import AgentState, EvidenceItem, Citation, Plan

def init_state(question: str) -> AgentState:
    return {
        "question": question,
        "evidence": [],
        "citations": [],
        "final_answer": "",
        "tool_calls": 0,
        "steps": 0,
        "trace": [],
        "errors": [],
    }

def add_trace(state: AgentState, event: Dict[str, Any]) -> None:
    state.setdefault("trace", []).append(event)

def set_plan(state: AgentState, plan: Plan) -> None:
    state["plan"] = plan.to_dict()
    add_trace(state, {"type": "plan", **state["plan"]})

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
