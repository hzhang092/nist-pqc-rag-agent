# rag/lc/graph.py
from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from .state import AgentState, Plan, EvidenceItem, Citation
from .state_utils import init_state, add_trace, set_plan, set_evidence, set_answer
from . import tools as lc_tools


MAX_STEPS = 4
MAX_TOOL_CALLS = 2


# -----------------------------
# Helpers
# -----------------------------

def _bump_step(state: AgentState, node: str) -> None:
    state["steps"] = int(state.get("steps", 0)) + 1
    add_trace(state, {"type": "step", "node": node, "steps": state["steps"], "tool_calls": state.get("tool_calls", 0)})


def _budget_exceeded(state: AgentState) -> bool:
    return int(state.get("steps", 0)) >= MAX_STEPS or int(state.get("tool_calls", 0)) >= MAX_TOOL_CALLS


def _heuristic_route(question: str) -> Plan:
    q = question.strip()
    ql = q.lower()

    # Compare intent
    if any(x in ql for x in ["compare", "difference between", "vs", "versus"]):
        # Try to split "A vs B" crudely; graph can evolve later
        return Plan(action="compare", reason="Comparison intent detected.", args={"topic_a": q, "topic_b": q}, mode_hint="general")

    # Algorithm intent (your SHAKE128 questions)
    if "algorithm" in ql or "shake" in ql:
        return Plan(action="retrieve", reason="Algorithm-like query detected; retrieve evidence.", query=q, mode_hint="algorithm")

    # Definition intent
    if ql.startswith(("what is", "what's", "define", "explain")):
        # extract term roughly: "what is X"
        term = q.split(" ", 2)[-1].strip(" ?")
        return Plan(action="resolve_definition", reason="Definition intent detected.", args={"term": term}, mode_hint="definition")

    # Default: retrieve
    return Plan(action="retrieve", reason="Default to retrieval.", query=q, mode_hint="general")


def _to_evidence_items(evidence_dicts: List[Dict[str, Any]]) -> List[EvidenceItem]:
    out: List[EvidenceItem] = []
    for e in evidence_dicts:
        out.append(EvidenceItem(
            score=float(e.get("score", 0.0)),
            chunk_id=str(e["chunk_id"]),
            doc_id=str(e["doc_id"]),
            start_page=int(e["start_page"]),
            end_page=int(e["end_page"]),
            text=str(e.get("text", "")),
        ))
    return out


def _default_citations_from_evidence(evidence: List[EvidenceItem], n: int = 3) -> List[Citation]:
    cits: List[Citation] = []
    for e in evidence[:n]:
        cits.append(Citation(doc_id=e.doc_id, start_page=e.start_page, end_page=e.end_page, chunk_id=e.chunk_id))
    return cits


def _call_rag_answer(question: str, evidence: list[EvidenceItem]) -> dict:
    """
    Adapter for your repo: uses rag.rag_answer.build_cited_answer(question, hits, generate_fn).
    """
    from rag.rag_answer import build_cited_answer
    from rag.retriever.base import ChunkHit
    from rag.llm.gemini import make_generate_fn  # matches rag/ask.py pattern

    # Convert EvidenceItem -> ChunkHit (what build_cited_answer expects)
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

    # Normalize citations to dicts for your AgentState
    citations = []
    for c in result.citations:
        citations.append({
            "doc_id": c.doc_id,
            "start_page": int(c.start_page),
            "end_page": int(c.end_page),
            "chunk_id": c.chunk_id,
            # optional: keep key if you want it later
            "key": getattr(c, "key", None),
        })

    return {"answer": result.answer, "citations": citations}


# -----------------------------
# LangGraph nodes
# -----------------------------

def node_route(state: AgentState) -> AgentState:
    _bump_step(state, "route")

    # If budgets exceeded, force answer/refuse
    if _budget_exceeded(state):
        p = Plan(action="answer", reason="Budget exceeded; forcing answer/refuse.")
        set_plan(state, p)
        return state

    q = state["question"]
    plan = _heuristic_route(q)
    set_plan(state, plan)
    return state


def node_do_tool(state: AgentState) -> AgentState:
    _bump_step(state, "do_tool")

    if _budget_exceeded(state):
        add_trace(state, {"type": "budget", "note": "Skipped tool call due to budget."})
        return state

    plan = state.get("plan") or {}
    action = plan.get("action", "retrieve")

    # increment tool_calls once per tool node execution
    state["tool_calls"] = int(state.get("tool_calls", 0)) + 1

    tool_out: Dict[str, Any]
    if action == "retrieve":
        tool_out = lc_tools.retrieve.invoke({"query": plan.get("query", state["question"]), "k": 8})
    elif action == "resolve_definition":
        args = plan.get("args", {}) or {}
        tool_out = lc_tools.resolve_definition.invoke({"term": args.get("term", state["question"]), "k": 8})
    elif action == "compare":
        args = plan.get("args", {}) or {}
        # If router didn’t extract topics, pass question as both; you can improve later
        tool_out = lc_tools.compare.invoke({
            "topic_a": args.get("topic_a", state["question"]),
            "topic_b": args.get("topic_b", state["question"]),
            "k": 6,
        })
    elif action == "summarize":
        args = plan.get("args", {}) or {}
        tool_out = lc_tools.summarize.invoke({
            "doc_id": args["doc_id"],
            "start_page": int(args["start_page"]),
            "end_page": int(args["end_page"]),
            "k": int(args.get("k", 30)),
        })
    else:
        add_trace(state, {"type": "tool_skip", "reason": f"Unknown or non-tool action: {action}"})
        return state

    ev_dicts = tool_out.get("evidence", [])
    evidence = _to_evidence_items(ev_dicts)
    set_evidence(state, evidence)

    add_trace(state, {"type": "tool", "action": action, "stats": tool_out.get("stats", {}), "mode_hint": tool_out.get("mode_hint")})
    return state


def node_answer(state: AgentState) -> AgentState:
    _bump_step(state, "answer")

    evidence = _to_evidence_items(state.get("evidence", []))
    if not evidence:
        # No evidence: refuse now
        msg = "I couldn’t find supporting evidence in the indexed NIST documents for that question. Try asking with a specific standard (e.g., FIPS 203/204/205) and a section/algorithm name."
        set_answer(state, msg, [])
        return state

    out = _call_rag_answer(state["question"], evidence)
    answer_text = out.get("answer", "").strip()

    # Convert to your lc/state.Citation objects
    cits: list[Citation] = []
    for c in out.get("citations", []) or []:
        cits.append(Citation(
            doc_id=str(c["doc_id"]),
            start_page=int(c["start_page"]),
            end_page=int(c["end_page"]),
            chunk_id=str(c.get("chunk_id", "")),
        ))

    set_answer(state, answer_text, cits)
    return state


def node_verify_or_refuse(state: AgentState) -> AgentState:
    _bump_step(state, "verify_or_refuse")

    evidence = _to_evidence_items(state.get("evidence", []))
    citations = state.get("citations", []) or []

    # Hard rule: any non-refusal answer must have citations
    draft = (state.get("draft_answer") or "").strip()
    is_refusal = "couldn’t find supporting evidence" in draft.lower() or "could not find" in draft.lower()

    if not is_refusal and (not evidence or len(citations) == 0):
        msg = "I don’t have enough citable evidence in the indexed NIST documents to answer that reliably."
        state["draft_answer"] = msg
        state["citations"] = []
        add_trace(state, {"type": "verify", "result": "refuse_no_citations"})
        return state

    add_trace(state, {"type": "verify", "result": "ok", "citations": len(citations)})
    return state


# -----------------------------
# Graph assembly
# -----------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route", node_route)
    g.add_node("do_tool", node_do_tool)
    g.add_node("answer", node_answer)
    g.add_node("verify_or_refuse", node_verify_or_refuse)

    g.set_entry_point("route")

    # route -> tool or answer
    def _route_edge(state: AgentState) -> str:
        action = (state.get("plan") or {}).get("action", "retrieve")
        if action in {"retrieve", "resolve_definition", "compare", "summarize"}:
            return "do_tool"
        if action in {"answer", "refuse"}:
            return "answer"
        return "do_tool"

    g.add_conditional_edges("route", _route_edge, {
        "do_tool": "do_tool",
        "answer": "answer",
    })

    g.add_edge("do_tool", "answer")
    g.add_edge("answer", "verify_or_refuse")
    g.add_edge("verify_or_refuse", END)

    return g.compile()


GRAPH = build_graph()


def run_agent(question: str) -> AgentState:
    state = init_state(question)
    return GRAPH.invoke(state)
