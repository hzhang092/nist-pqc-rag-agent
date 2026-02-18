"""
This module contains unit tests for the `graph` module in the `rag.lc` package.

Tests:
- `test_heuristic_route_parses_differences_between`: Verifies that the `_heuristic_route` function correctly identifies a comparison query and extracts the topics for comparison.
- `test_heuristic_route_ambiguous_compare_falls_back_to_retrieve`: Ensures that ambiguous comparison queries default to the `retrieve` action.
- `test_node_answer_preserves_citation_key`: Tests that the `node_answer` function preserves the citation key in the output. Uses monkeypatching to mock the `_call_rag_answer` function.

Notes:
- The `monkeypatch` fixture is used to override the `_call_rag_answer` function for testing purposes.
- The `init_state` function from `state_utils` is used to initialize the agent state for the `node_answer` test.
- These tests ensure the correctness of routing logic and citation handling in the `graph` module.
"""

from rag.lc import graph as g
from rag.lc.state_utils import init_state


def test_heuristic_route_parses_differences_between():
    plan = g._heuristic_route("What are the differences between ML-KEM and ML-DSA?")
    assert plan.action == "compare"
    assert plan.args["topic_a"] == "ML-KEM"
    assert plan.args["topic_b"] == "ML-DSA"


def test_heuristic_route_ambiguous_compare_falls_back_to_retrieve():
    plan = g._heuristic_route("Compare these schemes")
    assert plan.action == "retrieve"


def test_node_answer_preserves_citation_key(monkeypatch):
    state = init_state("What is ML-KEM?")
    state["evidence"] = [
        {
            "score": 1.0,
            "chunk_id": "NIST.FIPS.203::p0010::c000",
            "doc_id": "NIST.FIPS.203",
            "start_page": 10,
            "end_page": 10,
            "text": "ML-KEM is a key encapsulation mechanism.",
        }
    ]

    monkeypatch.setattr(
        g,
        "_call_rag_answer",
        lambda question, evidence: {
            "answer": "ML-KEM is a KEM [c1].",
            "citations": [
                {
                    "doc_id": "NIST.FIPS.203",
                    "start_page": 10,
                    "end_page": 10,
                    "chunk_id": "NIST.FIPS.203::p0010::c000",
                    "key": "c1",
                }
            ],
        },
    )

    out = g.node_answer(state)
    assert out["citations"][0]["key"] == "c1"


def test_verify_sets_final_answer_and_keeps_draft_when_refusing():
    state = init_state("Q")
    state["draft_answer"] = "Uncited claim from model."
    state["citations"] = []
    state["evidence"] = []

    out = g.node_verify_or_refuse(state)
    assert out["draft_answer"] == "Uncited claim from model."
    assert out["final_answer"] == "I don’t have enough citable evidence in the indexed NIST documents to answer that reliably."


def test_verify_sets_final_answer_to_draft_on_ok():
    state = init_state("Q")
    state["draft_answer"] = "ML-KEM is a KEM [c1]."
    state["citations"] = [
        {
            "doc_id": "NIST.FIPS.203",
            "start_page": 10,
            "end_page": 10,
            "chunk_id": "NIST.FIPS.203::p0010::c000",
            "key": "c1",
        }
    ]
    state["evidence"] = [
        {
            "score": 1.0,
            "chunk_id": "NIST.FIPS.203::p0010::c000",
            "doc_id": "NIST.FIPS.203",
            "start_page": 10,
            "end_page": 10,
            "text": "ML-KEM is a key encapsulation mechanism.",
        }
    ]

    out = g.node_verify_or_refuse(state)
    assert out["final_answer"] == "ML-KEM is a KEM [c1]."

def test_budget_skips_tool_call(monkeypatch):
    from rag.lc import graph as g

    class BoomTool:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("should not call")

    # Replace the tool object entirely
    monkeypatch.setattr(g.lc_tools, "retrieve", BoomTool())

    state = {
        "question": "Q",
        "steps": g.MAX_STEPS,   # budget exceeded
        "tool_calls": 0,
        "trace": [],
        "evidence": [],
        "citations": [],
        "errors": [],
        "plan": {"action": "retrieve", "query": "Q"},
    }
    out = g.node_do_tool(state)

    # Should skip tool call due to budget
    assert out["tool_calls"] == 0
    assert any(ev.get("type") == "budget" for ev in out["trace"])

    
def test_do_tool_writes_evidence(monkeypatch):
    from rag.lc import graph as g

    fake = {
        "tool": "retrieve",
        "evidence": [{
            "score": 0.9,
            "chunk_id": "X",
            "doc_id": "DOC",
            "start_page": 1,
            "end_page": 1,
            "text": "hello",
        }],
        "stats": {"n": 1},
        "mode_hint": "general",
    }

    class FakeTool:
        def invoke(self, *_args, **_kwargs):
            return fake

    monkeypatch.setattr(g.lc_tools, "retrieve", FakeTool())

    state = {
        "question": "Q",
        "steps": 0,
        "tool_calls": 0,
        "trace": [],
        "evidence": [],
        "citations": [],
        "errors": [],
        "plan": {"action": "retrieve", "query": "Q"},
    }
    out = g.node_do_tool(state)

    assert out["tool_calls"] == 1
    assert len(out["evidence"]) == 1
    assert out["evidence"][0]["chunk_id"] == "X"

