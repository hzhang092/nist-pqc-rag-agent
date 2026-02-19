"""
Unit tests for the `graph` module in the `rag.lc` package.

These tests validate the behavior of the bounded retrieve-assess-refine agent, ensuring that routing, evidence handling, and stopping conditions work as expected.

Tests:
- `test_heuristic_route_parses_differences_between`: Verifies that the `_heuristic_route` function correctly identifies a comparison query and extracts the topics for comparison.
- `test_heuristic_route_ambiguous_compare_falls_back_to_retrieve`: Ensures that ambiguous comparison queries default to the `retrieve` action.
- `test_node_answer_preserves_citation_key`: Tests that the `node_answer` function preserves the citation key in the output. Uses monkeypatching to mock the `_call_rag_answer` function.
- `test_loop_refines_then_answers`: Simulates a multi-step refinement loop where evidence is retrieved in multiple rounds before generating an answer. Verifies that the agent stops after sufficient evidence is gathered.
- `test_assess_compare_requires_doc_diversity`: Ensures that the `node_assess_evidence` function requires evidence from diverse documents for comparison tasks.
- `test_budget_stop_refuses_without_calling_answer_llm`: Verifies that the agent stops and refuses to proceed when the tool call budget is exhausted, without invoking the answer generation LLM.
- `test_round_limit_stops_and_refuses`: Ensures that the agent stops and refuses to proceed when the retrieval round limit is reached.
- `test_verify_uses_refusal_reason_when_stop_reason_is_sufficient`: Tests that the `node_verify_or_refuse` function uses the refusal reason when citations are missing, even if evidence is sufficient.

Notes:
- The `monkeypatch` fixture is used extensively to mock retrieval and answer generation functions.
- These tests ensure that the agent adheres to its constraints, such as tool call budgets, retrieval round limits, and evidence sufficiency checks.
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
    state["evidence_sufficient"] = True
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


def test_loop_refines_then_answers(monkeypatch):
    class SeqRetrieve:
        def __init__(self):
            self.calls = 0

        def invoke(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "tool": "retrieve",
                    "evidence": [
                        {
                            "score": 0.9,
                            "chunk_id": "C1",
                            "doc_id": "NIST.FIPS.203",
                            "start_page": 10,
                            "end_page": 10,
                            "text": "ML-KEM key generation overview.",
                        }
                    ],
                    "stats": {"n": 1},
                    "mode_hint": "general",
                }
            return {
                "tool": "retrieve",
                "evidence": [
                    {
                        "score": 0.8,
                        "chunk_id": "C2",
                        "doc_id": "NIST.FIPS.203",
                        "start_page": 11,
                        "end_page": 11,
                        "text": "Additional key generation details with parameters.",
                    }
                ],
                "stats": {"n": 1},
                "mode_hint": "general",
            }

    seq = SeqRetrieve()
    monkeypatch.setattr(g.lc_tools, "retrieve", seq)
    monkeypatch.setattr(
        g,
        "_call_rag_answer",
        lambda question, evidence: {
            "answer": "ML-KEM key generation is specified in FIPS 203 [c1].",
            "citations": [
                {
                    "doc_id": "NIST.FIPS.203",
                    "start_page": 10,
                    "end_page": 10,
                    "chunk_id": "C1",
                    "key": "c1",
                }
            ],
        },
    )

    state = g.run_agent("ML-KEM key generation")

    assert seq.calls == 2
    assert state["tool_calls"] == 2
    assert state["retrieval_round"] == 2
    assert state["evidence_sufficient"] is True
    assert "[c1]" in state["final_answer"]
    assert state["refusal_reason"] == ""


def test_assess_compare_requires_doc_diversity():
    state = init_state("What are the differences between ML-KEM and ML-DSA?")
    state["plan"] = {"action": "compare", "args": {"topic_a": "ML-KEM", "topic_b": "ML-DSA"}}
    state["evidence"] = [
        {
            "score": 0.9,
            "chunk_id": "C1",
            "doc_id": "NIST.FIPS.203",
            "start_page": 10,
            "end_page": 10,
            "text": "ML-KEM use-case text.",
        },
        {
            "score": 0.8,
            "chunk_id": "C2",
            "doc_id": "NIST.FIPS.203",
            "start_page": 11,
            "end_page": 11,
            "text": "More ML-KEM details.",
        },
    ]

    out = g.node_assess_evidence(state)
    assert out["evidence_sufficient"] is False
    assert out["stop_reason"] == "compare_doc_diversity_missing"


def test_budget_stop_refuses_without_calling_answer_llm(monkeypatch):
    called = {"answer": False}

    class SparseRetrieve:
        def invoke(self, *_args, **_kwargs):
            return {
                "tool": "retrieve",
                "evidence": [
                    {
                        "score": 0.9,
                        "chunk_id": "C1",
                        "doc_id": "NIST.FIPS.203",
                        "start_page": 10,
                        "end_page": 10,
                        "text": "single sparse hit",
                    }
                ],
                "stats": {"n": 1},
                "mode_hint": "general",
            }

    def boom_answer(_question, _evidence):
        called["answer"] = True
        raise RuntimeError("answer LLM should not be called")

    monkeypatch.setattr(g.lc_tools, "retrieve", SparseRetrieve())
    monkeypatch.setattr(g, "_call_rag_answer", boom_answer)
    monkeypatch.setattr(g, "MAX_TOOL_CALLS", 1)
    monkeypatch.setattr(g, "MAX_RETRIEVAL_ROUNDS", 3)
    monkeypatch.setattr(g, "MAX_STEPS", 20)
    monkeypatch.setattr(g, "MIN_EVIDENCE_HITS", 2)

    state = g.run_agent("ML-KEM key generation")

    assert called["answer"] is False
    assert state["tool_calls"] == 1
    assert state["stop_reason"] == "tool_budget_exhausted"
    assert state["refusal_reason"] == "tool_budget_exhausted"
    assert state["citations"] == []
    assert "citable evidence" in state["final_answer"].lower()


def test_round_limit_stops_and_refuses(monkeypatch):
    class SparseRetrieve:
        def invoke(self, *_args, **_kwargs):
            return {
                "tool": "retrieve",
                "evidence": [
                    {
                        "score": 0.9,
                        "chunk_id": "C1",
                        "doc_id": "NIST.FIPS.203",
                        "start_page": 10,
                        "end_page": 10,
                        "text": "single sparse hit",
                    }
                ],
                "stats": {"n": 1},
                "mode_hint": "general",
            }

    monkeypatch.setattr(g.lc_tools, "retrieve", SparseRetrieve())
    monkeypatch.setattr(g, "MAX_TOOL_CALLS", 5)
    monkeypatch.setattr(g, "MAX_RETRIEVAL_ROUNDS", 1)
    monkeypatch.setattr(g, "MAX_STEPS", 20)
    monkeypatch.setattr(g, "MIN_EVIDENCE_HITS", 2)

    state = g.run_agent("ML-KEM key generation")

    assert state["tool_calls"] == 1
    assert state["retrieval_round"] == 1
    assert state["stop_reason"] == "retrieval_round_budget_exhausted"
    assert state["refusal_reason"] == "retrieval_round_budget_exhausted"
    assert "citable evidence" in state["final_answer"].lower()


def test_verify_uses_refusal_reason_when_stop_reason_is_sufficient():
    state = init_state("What is ML-KEM?")
    state["evidence_sufficient"] = True
    state["stop_reason"] = "sufficient_evidence"
    state["draft_answer"] = "ML-KEM is a KEM."
    state["citations"] = []
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
    assert out["stop_reason"] == "sufficient_evidence"
    assert out["refusal_reason"] == "missing_citations"
    assert out["final_answer"] == "I could not produce reliable citations for the drafted answer, so I cannot return it."
