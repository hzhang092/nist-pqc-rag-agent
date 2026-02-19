"""
Unit tests for the `retrieve_for_eval` function in the `rag.retrieve` module.

These tests validate the behavior of the retrieval evaluation API, ensuring that it correctly maps evaluation knobs to the hybrid search function and returns results in the expected format.

Tests:
- `test_retrieve_for_eval_returns_dicts_and_maps_eval_knobs`: Verifies that the `retrieve_for_eval` function:
  - Maps evaluation parameters (e.g., `k`, `k0`, `candidate_multiplier`, etc.) to the hybrid search function correctly.
  - Returns a list of dictionaries with the expected keys and values for each retrieved chunk.
  - Uses monkeypatching to mock the `hybrid_search` function and capture the parameters passed to it.

Notes:
- The `monkeypatch` fixture is used to override the `hybrid_search` function for testing purposes.
- The test ensures that the retrieval results include metadata such as `chunk_id`, `doc_id`, `start_page`, `end_page`, `score`, `text`, `rank`, and `mode`.
- This test ensures that the retrieval evaluation API behaves deterministically and adheres to the expected interface.
"""

from rag.retrieve import retrieve_for_eval
from rag.retriever.base import ChunkHit


def test_retrieve_for_eval_returns_dicts_and_maps_eval_knobs(monkeypatch):
    captured = {}

    def _fake_hybrid_search(
        query: str,
        top_k: int,
        candidate_multiplier: int,
        k0: int,
        use_query_fusion: bool,
        enable_rerank: bool,
        rerank_pool: int,
    ):
        captured.update(
            {
                "query": query,
                "top_k": top_k,
                "candidate_multiplier": candidate_multiplier,
                "k0": k0,
                "use_query_fusion": use_query_fusion,
                "enable_rerank": enable_rerank,
                "rerank_pool": rerank_pool,
            }
        )
        return [
            ChunkHit(0.42, "c-1", "NIST.FIPS.203", 18, 19, "first chunk"),
            ChunkHit(0.30, "c-2", "NIST.FIPS.204", 7, 7, "second chunk"),
        ]

    monkeypatch.setattr("rag.retrieve.hybrid_search", _fake_hybrid_search)

    rows = retrieve_for_eval(
        query="ML-KEM key generation",
        mode="hybrid",
        k=2,
        k0=50,
        candidate_multiplier=6,
        fusion=True,
        evidence_window=True,
        cheap_rerank=False,
        debug=True,
    )

    assert captured["query"] == "ML-KEM key generation"
    assert captured["top_k"] == 2
    assert captured["candidate_multiplier"] == 6
    assert captured["k0"] == 50
    assert captured["use_query_fusion"] is True
    assert captured["enable_rerank"] is False

    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        "chunk_id",
        "doc_id",
        "start_page",
        "end_page",
        "score",
        "text",
        "rank",
        "mode",
    }
    assert rows[0]["chunk_id"] == "c-1"
    assert rows[0]["doc_id"] == "NIST.FIPS.203"
    assert rows[0]["start_page"] == 18
    assert rows[0]["end_page"] == 19
    assert rows[0]["mode"] == "hybrid"


def test_retrieve_for_eval_applies_stable_tie_break_sort(monkeypatch):
    def _fake_hybrid_search(
        query: str,
        top_k: int,
        candidate_multiplier: int,
        k0: int,
        use_query_fusion: bool,
        enable_rerank: bool,
        rerank_pool: int,
    ):
        _ = (query, top_k, candidate_multiplier, k0, use_query_fusion, enable_rerank, rerank_pool)
        return [
            ChunkHit(0.50, "chunk-b", "DOC", 2, 2, "b"),
            ChunkHit(0.50, "chunk-a", "DOC", 1, 1, "a"),
            ChunkHit(0.50, "chunk-c", "DOC", 1, 2, "c"),
        ]

    monkeypatch.setattr("rag.retrieve.hybrid_search", _fake_hybrid_search)

    rows = retrieve_for_eval(query="tie case", mode="hybrid", k=3)
    assert [r["chunk_id"] for r in rows] == ["chunk-a", "chunk-c", "chunk-b"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
