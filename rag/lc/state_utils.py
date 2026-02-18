# rag/lc/state_utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from .state import AgentState, EvidenceItem, Citation, Plan

def init_state(question: str) -> AgentState:
    return {
        "question": question,
        "evidence": [],
        "citations": [],
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
