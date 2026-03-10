from __future__ import annotations

import json

from rag.lc import graph as g
from rag.lc.state import Citation, CompareTopics, EvidenceItem, Plan, QueryAnalysis
from rag.lc.state_utils import init_state, set_answer, set_evidence, set_final_answer, set_plan, set_query_analysis
from rag.lc.trace import summarize_trace, write_trace


def _rich_compare_state() -> dict:
    long_text = (
        "ML-KEM provides key establishment while ML-DSA provides digital signatures. "
        "This evidence is intentionally long to exercise preview truncation in the trace writer. "
        "It also gives the summary enough content to show meaningful top chunks and document scope. "
        "The summary should keep a shorter preview here while the raw trace preserves the full evidence text "
        "for debugging, citation checking, and later replay against the retrieval pipeline."
    )
    return {
        "question": "What are the differences between ML-KEM and ML-DSA?",
        "original_query": "What are the differences between ML-KEM and ML-DSA?",
        "canonical_query": "ML-KEM vs ML-DSA intended use-cases definition key properties",
        "mode_hint": "compare",
        "rewrite_needed": True,
        "protected_spans": ["ML-KEM", "ML-DSA"],
        "required_anchors": [],
        "sparse_query": "ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
        "dense_query": "compare intended use-cases and deployment differences between ML-KEM and ML-DSA",
        "subqueries": [
            "ML-KEM intended use-cases and deployment context",
            "ML-DSA intended use-cases and deployment context",
        ],
        "confidence": 0.88,
        "compare_topics": {"topic_a": "ML-KEM", "topic_b": "ML-DSA"},
        "doc_ids": ["NIST.FIPS.203", "NIST.FIPS.204"],
        "doc_family": "",
        "analysis_notes": "deterministic-fallback",
        "answer_prompt_question": "What are the differences between ML-KEM and ML-DSA?",
        "query_analysis": {},
        "plan": {"action": "retrieve", "reason": "test", "query": "ML-KEM vs ML-DSA", "args": {}, "mode_hint": "compare"},
        "evidence": [
            {
                "score": 0.91,
                "chunk_id": "NIST.FIPS.203::p0003::c000",
                "doc_id": "NIST.FIPS.203",
                "start_page": 3,
                "end_page": 3,
                "text": long_text,
            },
            {
                "score": 0.87,
                "chunk_id": "NIST.FIPS.204::p0005::c002",
                "doc_id": "NIST.FIPS.204",
                "start_page": 5,
                "end_page": 5,
                "text": long_text,
            },
        ],
        "draft_answer": "ML-KEM is for key establishment, while ML-DSA is for digital signatures [c1][c2].",
        "final_answer": "ML-KEM is for key establishment, while ML-DSA is for digital signatures [c1][c2].",
        "citations": [
            {
                "key": "c1",
                "doc_id": "NIST.FIPS.203",
                "start_page": 3,
                "end_page": 3,
                "chunk_id": "NIST.FIPS.203::p0003::c000",
            },
            {
                "key": "c2",
                "doc_id": "NIST.FIPS.204",
                "start_page": 5,
                "end_page": 5,
                "chunk_id": "NIST.FIPS.204::p0005::c002",
            },
        ],
        "tool_calls": 2,
        "steps": 9,
        "retrieval_round": 2,
        "trace": [
            {"type": "step", "node": "analyze_query", "steps": 1, "tool_calls": 0, "retrieval_round": 0},
            {
                "type": "analysis",
                "mode_hint": "compare",
                "rewrite_needed": True,
                "protected_spans": ["ML-KEM", "ML-DSA"],
                "doc_ids": ["NIST.FIPS.203", "NIST.FIPS.204"],
                "sparse_query": "ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
                "dense_query": "compare intended use-cases and deployment differences between ML-KEM and ML-DSA",
                "subqueries": [
                    "ML-KEM intended use-cases and deployment context",
                    "ML-DSA intended use-cases and deployment context",
                ],
                "analysis_notes": "deterministic-fallback",
            },
            {"type": "step", "node": "route", "steps": 2, "tool_calls": 0, "retrieval_round": 0},
            {"type": "plan", "action": "retrieve", "reason": "test", "query": "ML-KEM vs ML-DSA", "args": {}, "mode_hint": "compare"},
            {"type": "step", "node": "retrieve", "steps": 3, "tool_calls": 1, "retrieval_round": 1},
            {"type": "retrieval_round_started", "round": 1, "action": "retrieve", "tool_calls": 1},
            {"type": "evidence", "n": 1},
            {
                "type": "retrieval_round_result",
                "round": 1,
                "action": "retrieve",
                "new_hits": 1,
                "total_hits": 1,
                "mode_hint": "compare",
                "doc_ids": ["NIST.FIPS.203", "NIST.FIPS.204"],
                "sparse_query": "ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
                "dense_query": "compare intended use-cases and deployment differences between ML-KEM and ML-DSA",
                "subqueries": [
                    "ML-KEM intended use-cases and deployment context",
                    "ML-DSA intended use-cases and deployment context",
                ],
                "protected_spans": ["ML-KEM", "ML-DSA"],
                "timing_ms_retrieve": 5.2,
            },
            {"type": "step", "node": "assess_evidence", "steps": 4, "tool_calls": 1, "retrieval_round": 1},
            {
                "type": "assessment_decision",
                "sufficient": False,
                "reasons": ["one_sided_comparison"],
                "budget_reason": "",
                "evidence_hits": 1,
                "doc_diversity": 1,
                "missing_protected_spans": [],
                "missing_compare_topics": ["ML-DSA"],
            },
            {"type": "step", "node": "refine_query", "steps": 5, "tool_calls": 1, "retrieval_round": 1},
            {
                "type": "query_refined",
                "strategy": "comparison-bias",
                "previous_query": "ML-KEM vs ML-DSA",
                "refined_query": "ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
                "refined_sparse_query": "ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
                "refined_dense_query": "compare intended use-cases and deployment differences between ML-KEM and ML-DSA",
                "refined_subqueries": [
                    "ML-KEM intended use-cases and deployment context",
                    "ML-DSA intended use-cases and deployment context",
                ],
            },
            {"type": "step", "node": "retrieve", "steps": 6, "tool_calls": 2, "retrieval_round": 2},
            {"type": "retrieval_round_started", "round": 2, "action": "retrieve", "tool_calls": 2},
            {
                "type": "evidence_updated",
                "round": 2,
                "total_hits": 2,
                "top_chunk_ids": ["NIST.FIPS.203::p0003::c000", "NIST.FIPS.204::p0005::c002"],
                "top_doc_ids": ["NIST.FIPS.203", "NIST.FIPS.204"],
            },
            {
                "type": "retrieval_round_result",
                "round": 2,
                "action": "retrieve",
                "new_hits": 1,
                "total_hits": 2,
                "mode_hint": "compare",
                "doc_ids": ["NIST.FIPS.203", "NIST.FIPS.204"],
                "sparse_query": "ML-KEM ML-DSA intended use-cases comparison FIPS 203 FIPS 204",
                "dense_query": "compare intended use-cases and deployment differences between ML-KEM and ML-DSA",
                "subqueries": [
                    "ML-KEM intended use-cases and deployment context",
                    "ML-DSA intended use-cases and deployment context",
                ],
                "protected_spans": ["ML-KEM", "ML-DSA"],
                "timing_ms_retrieve": 4.8,
            },
            {"type": "step", "node": "assess_evidence", "steps": 7, "tool_calls": 2, "retrieval_round": 2},
            {
                "type": "assessment_decision",
                "sufficient": True,
                "reasons": [],
                "budget_reason": "",
                "evidence_hits": 2,
                "doc_diversity": 2,
                "missing_protected_spans": [],
                "missing_compare_topics": {},
            },
            {"type": "step", "node": "answer", "steps": 8, "tool_calls": 2, "retrieval_round": 2},
            {"type": "answer", "citations": 2},
            {"type": "step", "node": "verify_or_refuse", "steps": 9, "tool_calls": 2, "retrieval_round": 2},
            {"type": "final_answer"},
            {"type": "verify", "result": "ok", "stop_reason": "sufficient_evidence", "refusal_reason": "", "citations": 2},
        ],
        "errors": [],
        "evidence_sufficient": True,
        "stop_reason": "sufficient_evidence",
        "refusal_reason": "",
        "last_retrieval_stats": {},
        "timing_ms": {"analyze": 2.1, "retrieve": 10.0, "generate": 7.3, "total": 19.8},
    }


def test_mutators_emit_compact_events_with_node_attribution():
    state = init_state("What is ML-KEM?")
    state["_trace_active_node"] = "analyze_query"
    set_query_analysis(
        state,
        QueryAnalysis(
            original_query="What is ML-KEM?",
            canonical_query="ML-KEM",
            mode_hint="definition",
            rewrite_needed=True,
            protected_spans=["ML-KEM"],
            required_anchors=[],
            sparse_query="ML-KEM definition",
            dense_query="definition and notation for ML-KEM in FIPS 203",
            subqueries=[],
            confidence=0.91,
            compare_topics=None,
            doc_ids=["NIST.FIPS.203"],
            doc_family="FIPS 203",
            analysis_notes="test",
            answer_prompt_question="What is ML-KEM?",
        ),
    )

    state["_trace_active_node"] = "route"
    set_plan(state, Plan(action="retrieve", reason="Use analyzed retrieve path.", query="ML-KEM", mode_hint="definition"))

    state["retrieval_round"] = 1
    state["_trace_active_node"] = "retrieve"
    set_evidence(
        state,
        [
            EvidenceItem(
                score=0.9,
                chunk_id="NIST.FIPS.203::p0003::c000",
                doc_id="NIST.FIPS.203",
                start_page=3,
                end_page=3,
                text="ML-KEM is a key-encapsulation mechanism.",
            )
        ],
    )

    state["_trace_active_node"] = "answer"
    set_answer(
        state,
        "ML-KEM is a key-encapsulation mechanism [c1].",
        [
            Citation(
                key="c1",
                doc_id="NIST.FIPS.203",
                start_page=3,
                end_page=3,
                chunk_id="NIST.FIPS.203::p0003::c000",
            )
        ],
        timing_ms_generate=12.4,
    )

    state["_trace_active_node"] = "verify_or_refuse"
    state["stop_reason"] = "sufficient_evidence"
    set_final_answer(state, state["draft_answer"], result="ok", used_refusal_template=False)

    trace = state["trace"]
    assert [event["type"] for event in trace] == [
        "analysis_applied",
        "plan_applied",
        "evidence_updated",
        "answer_drafted",
        "final_answer_set",
    ]
    assert [event["node"] for event in trace] == [
        "analyze_query",
        "route",
        "retrieve",
        "answer",
        "verify_or_refuse",
    ]
    assert trace[0]["subqueries_count"] == 0
    assert trace[1]["args_summary"]["doc_ids"] == []
    assert trace[2]["top_doc_ids"] == ["NIST.FIPS.203"]
    assert trace[3]["citations_count"] == 1
    assert trace[4]["result"] == "ok"


def test_summarize_trace_normalizes_legacy_verify_and_groups_rounds():
    summary = summarize_trace(_rich_compare_state())

    assert summary["run"]["entry_node"] == "analyze_query"
    assert summary["run"]["retrieval_rounds"] == 2
    assert summary["run"]["result"] == "ok"
    assert summary["timeline"][-1]["type"] == "verification_decision"
    retrieve_visits = [item for item in summary["trace_by_node"] if item["node"] == "retrieve"]
    assert [item["visit"] for item in retrieve_visits] == [1, 2]
    assert summary["retrieval_summary"][0]["assessment_reasons"] == ["one_sided_comparison"]
    assert summary["retrieval_summary"][0]["refinement"]["strategy"] == "comparison-bias"
    assert summary["retrieval_summary"][1]["sufficient"] is True
    assert summary["answer_summary"]["result"] == "ok"
    assert summary["answer_summary"]["citation_keys"] == ["c1", "c2"]


def test_summarize_trace_accepts_legacy_dict_compare_gap_payload():
    state = _rich_compare_state()
    for event in state["trace"]:
        if event.get("type") == "assessment_decision" and not event.get("sufficient"):
            event["missing_compare_topics"] = {"topic_b": "ML-DSA"}
            break

    summary = summarize_trace(state)
    first_assessment = next(
        event for event in summary["timeline"] if event["type"] == "assessment_decision" and not event["sufficient"]
    )

    assert first_assessment["missing_compare_topics"] == {"topic_b": "ML-DSA"}


def test_write_trace_writes_summary_and_raw_artifacts(tmp_path):
    state = _rich_compare_state()
    summary_path = write_trace(state, out_dir=str(tmp_path))
    summary_file = tmp_path / summary_path.split("/")[-1]
    raw_file = tmp_path / summary_file.name.replace(".json", ".raw.json")

    assert summary_file.exists()
    assert raw_file.exists()

    summary_payload = json.loads(summary_file.read_text(encoding="utf-8"))
    raw_payload = json.loads(raw_file.read_text(encoding="utf-8"))

    assert sorted(summary_payload.keys()) == [
        "analysis",
        "answer_summary",
        "evidence_preview",
        "retrieval_summary",
        "run",
        "timeline",
        "trace_by_node",
    ]
    assert summary_payload["run"]["result"] == "ok"
    assert summary_payload["retrieval_summary"][0]["timing_ms_retrieve"] == 5.2
    assert "full final state" not in summary_payload
    assert len(summary_payload["evidence_preview"][0]["preview_text"]) < len(raw_payload["evidence"][0]["text"])
    assert raw_payload["evidence"][0]["text"].startswith("ML-KEM provides key establishment")


def test_trace_active_node_is_observability_only_for_routing():
    state = init_state("What is ML-KEM?")
    state["_trace_active_node"] = "verify_or_refuse"
    set_query_analysis(
        state,
        QueryAnalysis(
            original_query="What is ML-KEM?",
            canonical_query="ML-KEM",
            mode_hint="definition",
            rewrite_needed=True,
            protected_spans=["ML-KEM"],
            required_anchors=[],
            sparse_query="ML-KEM definition",
            dense_query="definition and notation for ML-KEM in FIPS 203",
            subqueries=[],
            confidence=0.9,
            compare_topics=None,
            doc_ids=["NIST.FIPS.203"],
            doc_family="FIPS 203",
            analysis_notes="test",
            answer_prompt_question="What is ML-KEM?",
        ),
    )

    out = g.node_route(state)

    assert out["plan"]["action"] == "retrieve"
    assert out["plan"]["mode_hint"] == "definition"


def test_verify_node_emits_new_trace_events_for_success_and_refusal():
    success_state = init_state("What is ML-KEM?")
    success_state["evidence_sufficient"] = True
    success_state["draft_answer"] = "ML-KEM is a key-encapsulation mechanism [c1]."
    success_state["citations"] = [
        {
            "key": "c1",
            "doc_id": "NIST.FIPS.203",
            "start_page": 3,
            "end_page": 3,
            "chunk_id": "NIST.FIPS.203::p0003::c000",
        }
    ]
    success_state["evidence"] = [
        {
            "score": 0.9,
            "chunk_id": "NIST.FIPS.203::p0003::c000",
            "doc_id": "NIST.FIPS.203",
            "start_page": 3,
            "end_page": 3,
            "text": "ML-KEM is a key-encapsulation mechanism.",
        }
    ]
    success_state["stop_reason"] = "sufficient_evidence"

    g.node_verify_or_refuse(success_state)

    success_tail = [event["type"] for event in success_state["trace"][-2:]]
    assert success_tail == ["final_answer_set", "verification_decision"]
    assert success_state["trace"][-1]["result"] == "ok"

    refusal_state = init_state("What is ML-KEM?")
    refusal_state["evidence_sufficient"] = False
    refusal_state["stop_reason"] = "insufficient_hits"

    g.node_verify_or_refuse(refusal_state)

    refusal_tail = [event["type"] for event in refusal_state["trace"][-2:]]
    assert refusal_tail == ["final_answer_set", "verification_decision"]
    assert refusal_state["trace"][-1]["result"] == "refuse"
    assert refusal_state["trace"][-1]["used_refusal_template"] is True
