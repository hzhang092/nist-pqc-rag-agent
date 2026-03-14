from __future__ import annotations

import pytest

from rag.lc import graph as g
from rag.lc.state import CompareTopics, QueryAnalysis
from rag.lc.state_utils import init_state, set_query_analysis


def _patch_analysis(monkeypatch):
    monkeypatch.setattr(g, "_call_llm_query_analysis", lambda question, deterministic: deterministic)


def _set_analysis(
    state,
    *,
    original_query: str | None = None,
    canonical_query: str,
    mode_hint: str,
    rewrite_needed: bool = False,
    protected_spans: list[str] | None = None,
    required_anchors: list[str] | None = None,
    sparse_query: str | None = None,
    dense_query: str | None = None,
    subqueries: list[str] | None = None,
    confidence: float = 0.81,
    compare_topics: CompareTopics | None = None,
    doc_ids: list[str] | None = None,
    answer_prompt_question: str | None = None,
):
    resolved_protected_spans = protected_spans or required_anchors or []
    set_query_analysis(
        state,
        QueryAnalysis(
            original_query=original_query or state["question"],
            canonical_query=canonical_query,
            mode_hint=mode_hint,  # type: ignore[arg-type]
            rewrite_needed=rewrite_needed,
            protected_spans=resolved_protected_spans,
            required_anchors=required_anchors or [],
            sparse_query=sparse_query or canonical_query,
            dense_query=dense_query or canonical_query,
            subqueries=subqueries or [],
            confidence=confidence,
            compare_topics=compare_topics,
            doc_ids=doc_ids or [],
            doc_family=None,
            analysis_notes="test",
            answer_prompt_question=answer_prompt_question or (original_query or state["question"]),
        ),
    )


@pytest.mark.parametrize(
    ("question", "expected_mode", "expected_doc_ids", "expected_compare_topics"),
    [
        ("What is ML-KEM?", "definition", ["NIST.FIPS.203"], None),
        ("What are the steps in Algorithm 2 SHAKE128?", "algorithm", [], None),
        (
            "What are the differences between ML-KEM and ML-DSA?",
            "compare",
            ["NIST.FIPS.203", "NIST.FIPS.204"],
            ("ML-KEM", "ML-DSA"),
        ),
        ("What does NIST say about PQC for WiFi 9?", "general", [], None),
    ],
)
def test_build_query_analysis_modes(
    monkeypatch,
    question,
    expected_mode,
    expected_doc_ids,
    expected_compare_topics,
):
    _patch_analysis(monkeypatch)

    analysis = g._build_query_analysis(question)

    assert analysis.mode_hint == expected_mode
    assert analysis.doc_ids == expected_doc_ids
    assert analysis.sparse_query
    assert analysis.dense_query
    assert analysis.subqueries == ([] if expected_mode != "compare" else analysis.subqueries)
    assert analysis.confidence > 0.0
    if expected_compare_topics is None:
        assert analysis.compare_topics is None
        assert analysis.subqueries == []
    else:
        assert analysis.compare_topics is not None
        assert analysis.compare_topics.topic_a == expected_compare_topics[0]
        assert analysis.compare_topics.topic_b == expected_compare_topics[1]
        assert len(analysis.subqueries) == 2


def test_build_query_analysis_falls_back_on_invalid_llm_output(monkeypatch):
    monkeypatch.setattr(g, "_call_llm_query_analysis", lambda question, deterministic: (_ for _ in ()).throw(ValueError("bad json")))

    analysis = g._build_query_analysis("What is ML-KEM?")

    assert analysis.mode_hint == "definition"
    assert analysis.canonical_query == "ML-KEM"
    assert analysis.doc_ids == ["NIST.FIPS.203"]
    assert analysis.protected_spans == ["ML-KEM"]
    assert analysis.sparse_query == "ML-KEM definition"
    assert "definition and notation for ML-KEM" in analysis.dense_query
    assert analysis.analysis_notes == "deterministic-fallback"


def test_node_analyze_query_applies_definition_graph_lookup(monkeypatch):
    monkeypatch.setattr(
        g,
        "_build_query_analysis",
        lambda question: QueryAnalysis(
            original_query=question,
            canonical_query="encapsulation",
            mode_hint="definition",
            rewrite_needed=True,
            protected_spans=["encapsulation"],
            required_anchors=[],
            sparse_query="encapsulation definition",
            dense_query="definition and notation for encapsulation",
            subqueries=[],
            confidence=0.88,
            compare_topics=None,
            doc_ids=[],
            doc_family=None,
            analysis_notes="test",
            answer_prompt_question=question,
        ),
    )
    monkeypatch.setattr(
        g,
        "lookup_term",
        lambda term, doc_ids=None: {
            "matched_entities": [
                {
                    "node_id": "term::encapsulation",
                    "display_name": "encapsulation",
                    "normalized_term": "encapsulation",
                    "term_type": "operation",
                    "definition_strength": "heuristic_definition_section",
                }
            ],
            "candidate_doc_ids": ["NIST.FIPS.203"],
            "candidate_section_ids": ["section::NIST.FIPS.203::2. Terms and Definitions"],
            "required_anchors": ["encapsulation"],
            "match_reason": "exact_normalized_term",
        },
    )

    state = init_state("What is encapsulation?", use_graph_lookup=True)
    out = g.node_analyze_query(state)

    assert out["doc_ids"] == ["NIST.FIPS.203"]
    assert out["required_anchors"] == ["encapsulation"]
    graph_event = next(event for event in out["trace"] if event.get("type") == "graph_lookup_applied")
    assert graph_event["candidate_section_ids"] == ["section::NIST.FIPS.203::2. Terms and Definitions"]
    assert graph_event["applied_doc_ids"] == ["NIST.FIPS.203"]


def test_node_analyze_query_skips_graph_lookup_when_disabled(monkeypatch):
    monkeypatch.setattr(
        g,
        "_build_query_analysis",
        lambda question: QueryAnalysis(
            original_query=question,
            canonical_query="encapsulation",
            mode_hint="definition",
            rewrite_needed=True,
            protected_spans=["encapsulation"],
            required_anchors=[],
            sparse_query="encapsulation definition",
            dense_query="definition and notation for encapsulation",
            subqueries=[],
            confidence=0.88,
            compare_topics=None,
            doc_ids=[],
            doc_family=None,
            analysis_notes="test",
            answer_prompt_question=question,
        ),
    )
    monkeypatch.setattr(
        g,
        "lookup_term",
        lambda term, doc_ids=None: (_ for _ in ()).throw(AssertionError("graph lookup should be disabled")),
    )

    state = init_state("What is encapsulation?", use_graph_lookup=False)
    out = g.node_analyze_query(state)

    assert out["doc_ids"] == []
    assert out["graph_lookup"] == {}
    assert all(event.get("type") != "graph_lookup_applied" for event in out["trace"])


def test_route_uses_single_retrieve_action_without_reparsing(monkeypatch):
    state = init_state("Compare these schemes")
    _set_analysis(
        state,
        canonical_query="ML-KEM vs ML-DSA intended use-cases definition key properties",
        mode_hint="compare",
        rewrite_needed=True,
        protected_spans=["ML-KEM", "ML-DSA"],
        compare_topics=CompareTopics(topic_a="ML-KEM", topic_b="ML-DSA"),
        sparse_query="ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
        dense_query="compare intended use-cases and deployment differences between ML-KEM and ML-DSA",
        subqueries=[
            "ML-KEM intended use-cases and deployment context",
            "ML-DSA intended use-cases and deployment context",
        ],
        doc_ids=["NIST.FIPS.203", "NIST.FIPS.204"],
    )

    monkeypatch.setattr(g, "_extract_compare_topics", lambda _question: (_ for _ in ()).throw(RuntimeError("should not parse")))

    out = g.node_route(state)

    assert out["plan"]["action"] == "retrieve"
    assert out["plan"]["mode_hint"] == "compare"
    assert out["compare_topics"]["topic_a"] == "ML-KEM"
    assert out["compare_topics"]["topic_b"] == "ML-DSA"


def test_route_keeps_definition_as_retrieve_action():
    state = init_state("What is ML-KEM?")
    _set_analysis(
        state,
        canonical_query="ML-KEM",
        mode_hint="definition",
        rewrite_needed=True,
        protected_spans=["ML-KEM"],
        required_anchors=[],
        sparse_query="ML-KEM definition",
        dense_query="definition and notation for ML-KEM in FIPS 203",
        doc_ids=["NIST.FIPS.203"],
    )

    out = g.node_route(state)

    assert out["plan"]["action"] == "retrieve"
    assert out["plan"]["mode_hint"] == "definition"
    assert out["doc_ids"] == ["NIST.FIPS.203"]


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


def test_node_retrieve_passes_analyzed_args(monkeypatch):
    captured = {}

    class CaptureRetrieve:
        def invoke(self, payload):
            captured.update(payload)
            return {
                "tool": "retrieve",
                "evidence": [
                    {
                        "score": 0.9,
                        "chunk_id": "C1",
                        "doc_id": "NIST.FIPS.203",
                        "start_page": 10,
                        "end_page": 10,
                        "text": "Algorithm 2 SHAKE128 details.",
                    }
                ],
                "stats": {"n": 1},
                "mode_hint": payload["mode_hint"],
            }

    monkeypatch.setattr(g.lc_tools, "retrieve", CaptureRetrieve())

    state = init_state("What are the steps in Algorithm 2 SHAKE128?", k=3)
    _set_analysis(
        state,
        canonical_query="What are the steps in Algorithm 2 SHAKE128?",
        mode_hint="algorithm",
        protected_spans=["Algorithm 2", "SHAKE128"],
        required_anchors=["Algorithm 2", "SHAKE128"],
        sparse_query="Algorithm 2 SHAKE128 steps FIPS 203",
        dense_query="steps of Algorithm 2 SHAKE128 in ML-KEM standard",
        doc_ids=["NIST.FIPS.205"],
    )
    state["plan"] = {
        "action": "retrieve",
        "reason": "test",
        "query": state["canonical_query"],
        "args": {},
        "mode_hint": "algorithm",
    }

    out = g.node_retrieve(state)

    assert captured["query"] == "What are the steps in Algorithm 2 SHAKE128?"
    assert captured["k"] == 3
    assert captured["mode_hint"] == "algorithm"
    assert captured["doc_ids"] == ["NIST.FIPS.205"]
    assert captured["canonical_query"] == "What are the steps in Algorithm 2 SHAKE128?"
    assert captured["sparse_query"] == "Algorithm 2 SHAKE128 steps FIPS 203"
    assert captured["dense_query"] == "steps of Algorithm 2 SHAKE128 in ML-KEM standard"
    assert captured["subqueries"] == []
    assert captured["protected_spans"] == ["Algorithm 2", "SHAKE128"]
    assert captured["use_query_fusion"] is False
    assert captured["enable_mode_variants"] is False
    assert out["last_retrieval_stats"]["doc_ids"] == ["NIST.FIPS.205"]


def test_node_refine_query_uses_analyzed_state_not_raw_question():
    state = init_state("Compare these schemes")
    _set_analysis(
        state,
        canonical_query="ML-KEM vs ML-DSA intended use-cases definition key properties",
        mode_hint="compare",
        rewrite_needed=True,
        protected_spans=["ML-KEM", "ML-DSA"],
        compare_topics=CompareTopics(topic_a="ML-KEM", topic_b="ML-DSA"),
        sparse_query="ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
        dense_query="compare intended use-cases and deployment differences between ML-KEM and ML-DSA",
        subqueries=[
            "ML-KEM intended use-cases and deployment context",
            "ML-DSA intended use-cases and deployment context",
        ],
        doc_ids=["NIST.FIPS.203", "NIST.FIPS.204"],
    )
    state["plan"] = {
        "action": "retrieve",
        "reason": "test",
        "query": state["canonical_query"],
        "args": {},
        "mode_hint": "compare",
    }
    state["stop_reason"] = "one_sided_comparison"

    out = g.node_refine_query(state)

    assert out["plan"]["action"] == "retrieve"
    assert "ML-KEM" in out["plan"]["args"]["sparse_query"]
    assert "ML-DSA" in out["plan"]["args"]["sparse_query"]
    assert "Compare these schemes" not in out["plan"]["args"]["sparse_query"]


def test_loop_starts_with_analyze_query_and_then_answers(monkeypatch):
    class SeqRetrieve:
        def __init__(self):
            self.calls = 0

        def invoke(self, _payload):
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

    _patch_analysis(monkeypatch)
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

    step_nodes = [event["node"] for event in state["trace"] if event.get("type") == "step"]
    assert step_nodes[0] == "analyze_query"
    assert "route" in step_nodes
    assert seq.calls == 2
    assert state["tool_calls"] == 2
    assert state["retrieval_round"] == 2
    assert state["evidence_sufficient"] is True
    assert "[c1]" in state["final_answer"]
    assert state["refusal_reason"] == ""


def test_assess_compare_requires_doc_diversity():
    state = init_state("What are the differences between ML-KEM and ML-DSA?")
    _set_analysis(
        state,
        canonical_query="ML-KEM vs ML-DSA intended use-cases definition key properties",
        mode_hint="compare",
        rewrite_needed=True,
        protected_spans=["ML-KEM", "ML-DSA"],
        compare_topics=CompareTopics(topic_a="ML-KEM", topic_b="ML-DSA"),
        doc_ids=["NIST.FIPS.203", "NIST.FIPS.204"],
    )
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
    assert out["stop_reason"] == "one_sided_comparison"


def test_assess_flags_missing_protected_span():
    state = init_state("What are the steps in Algorithm 2 SHAKE128?")
    _set_analysis(
        state,
        canonical_query="What are the steps in Algorithm 2 SHAKE128?",
        mode_hint="algorithm",
        protected_spans=["Algorithm 2", "SHAKE128"],
        required_anchors=["Algorithm 2", "SHAKE128"],
        sparse_query="Algorithm 2 SHAKE128 steps",
        dense_query="steps of Algorithm 2 SHAKE128",
    )
    state["evidence"] = [
        {
            "score": 0.9,
            "chunk_id": "C1",
            "doc_id": "NIST.FIPS.203",
            "start_page": 10,
            "end_page": 10,
            "text": "ML-KEM procedure text without the requested anchor.",
        }
    ]

    out = g.node_assess_evidence(state)

    assert out["stop_reason"] == "missing_protected_span"


def test_assess_flags_wrong_doc_scope():
    state = init_state("What does FIPS 203 say about ML-KEM?")
    _set_analysis(
        state,
        canonical_query="ML-KEM",
        mode_hint="definition",
        protected_spans=["FIPS 203", "ML-KEM"],
        required_anchors=[],
        sparse_query="ML-KEM definition",
        dense_query="definition and notation for ML-KEM in FIPS 203",
        doc_ids=["NIST.FIPS.203"],
    )
    state["evidence"] = []

    out = g.node_assess_evidence(state)

    assert out["stop_reason"] == "wrong_doc_scope"


def test_budget_stop_refuses_without_calling_answer_llm(monkeypatch):
    called = {"answer": False}

    class SparseRetrieve:
        def invoke(self, _payload):
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

    _patch_analysis(monkeypatch)
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
        def invoke(self, _payload):
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

    _patch_analysis(monkeypatch)
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
