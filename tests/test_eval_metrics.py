import pytest

from eval.metrics import (
    compute_retrieval_metrics,
    compute_retrieval_metrics_by_ks,
    evaluate_answer_payload,
    hit_matches_gold,
    hit_matches_gold_doc_only,
    hit_matches_gold_with_tolerance,
)
from rag.types import REFUSAL_TEXT


def test_retrieval_metrics_expected_values():
    hits = [
        {"doc_id": "NIST.FIPS.203", "start_page": 1, "end_page": 1},
        {"doc_id": "NIST.FIPS.203", "start_page": 9, "end_page": 9},
        {"doc_id": "NIST.FIPS.204", "start_page": 3, "end_page": 3},
        {"doc_id": "NIST.FIPS.203", "start_page": 5, "end_page": 5},
    ]
    gold = [
        {"doc_id": "NIST.FIPS.203", "start_page": 5, "end_page": 5},
        {"doc_id": "NIST.FIPS.203", "start_page": 9, "end_page": 9},
    ]

    metrics = compute_retrieval_metrics(hits, gold, k=3)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["mrr_at_k"] == pytest.approx(0.5)
    assert metrics["ndcg_at_k"] == pytest.approx(0.3868528, rel=1e-5)


def test_answer_metrics_non_refusal_and_refusal_paths():
    payload = {
        "answer": "ML-KEM is a KEM [c1]. It uses encapsulation [c2].",
        "citations": [{"key": "c1"}, {"key": "c2"}],
    }
    metrics = evaluate_answer_payload(payload, answerable=True)
    assert metrics["is_refusal"] is False
    assert metrics["citation_presence_ok"] is True
    assert metrics["inline_citation_sentence_rate"] == pytest.approx(1.0)
    assert metrics["refusal_accuracy"] == pytest.approx(1.0)

    refusal_payload = {"answer": REFUSAL_TEXT, "citations": []}
    refusal_metrics = evaluate_answer_payload(refusal_payload, answerable=False)
    assert refusal_metrics["is_refusal"] is True
    assert refusal_metrics["citation_presence_ok"] is True
    assert refusal_metrics["inline_citation_sentence_rate"] is None
    assert refusal_metrics["refusal_accuracy"] == pytest.approx(1.0)


def test_hit_matches_gold_uses_doc_and_page_overlap():
    gold = {"doc_id": "NIST.FIPS.203", "start_page": 12, "end_page": 12}
    overlap_hit = {"doc_id": "NIST.FIPS.203", "start_page": 10, "end_page": 12}
    wrong_doc_hit = {"doc_id": "NIST.FIPS.204", "start_page": 12, "end_page": 12}
    no_overlap_hit = {"doc_id": "NIST.FIPS.203", "start_page": 13, "end_page": 14}

    assert hit_matches_gold(overlap_hit, gold) is True
    assert hit_matches_gold(wrong_doc_hit, gold) is False
    assert hit_matches_gold(no_overlap_hit, gold) is False


def test_ndcg_not_inflated_by_duplicate_hits_on_same_gold():
    hits = [
        {"doc_id": "NIST.FIPS.203", "start_page": 44, "end_page": 44},
        {"doc_id": "NIST.FIPS.203", "start_page": 44, "end_page": 44},
        {"doc_id": "NIST.FIPS.204", "start_page": 7, "end_page": 7},
    ]
    gold = [{"doc_id": "NIST.FIPS.203", "start_page": 44, "end_page": 44}]

    metrics = compute_retrieval_metrics(hits, gold, k=3)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr_at_k"] == pytest.approx(1.0)
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)


def test_compute_retrieval_metrics_by_ks_returns_all_requested_ks():
    hits = [
        {"doc_id": "NIST.FIPS.203", "start_page": 8, "end_page": 8},
        {"doc_id": "NIST.FIPS.203", "start_page": 9, "end_page": 9},
    ]
    gold = [{"doc_id": "NIST.FIPS.203", "start_page": 9, "end_page": 9}]

    by_k = compute_retrieval_metrics_by_ks(hits, gold, ks=[1, 2, 2, 3])
    assert set(by_k.keys()) == {"k1", "k2", "k3"}
    assert by_k["k1"]["mrr_at_k"] == pytest.approx(0.0)
    assert by_k["k2"]["mrr_at_k"] == pytest.approx(0.5)


def test_hit_matches_gold_doc_only_and_tolerance():
    gold = {"doc_id": "NIST.FIPS.203", "start_page": 20, "end_page": 20}
    same_doc_far_page = {"doc_id": "NIST.FIPS.203", "start_page": 30, "end_page": 30}
    same_doc_near_page = {"doc_id": "NIST.FIPS.203", "start_page": 21, "end_page": 21}
    wrong_doc_near_page = {"doc_id": "NIST.FIPS.204", "start_page": 21, "end_page": 21}

    assert hit_matches_gold_doc_only(same_doc_far_page, gold) is True
    assert hit_matches_gold_doc_only(wrong_doc_near_page, gold) is False

    assert hit_matches_gold_with_tolerance(same_doc_near_page, gold, page_tolerance=1) is True
    assert hit_matches_gold_with_tolerance(same_doc_far_page, gold, page_tolerance=1) is False
    assert hit_matches_gold_with_tolerance(wrong_doc_near_page, gold, page_tolerance=1) is False

    with pytest.raises(ValueError, match="page_tolerance"):
        hit_matches_gold_with_tolerance(same_doc_near_page, gold, page_tolerance=-1)
