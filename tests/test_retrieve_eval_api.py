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

from rag.retrieve import retrieve_for_eval, retrieve_for_eval_with_stages
from rag.retriever.base import ChunkHit


def test_retrieve_for_eval_returns_dicts_and_maps_eval_knobs(monkeypatch):
    captured = {}

    def _fake_build_hybrid_stage_outputs(
        query: str,
        top_k: int,
        candidate_multiplier: int,
        k0: int,
        use_query_fusion: bool,
        enable_rerank: bool,
        rerank_pool: int,
        diagnostic_pre_rerank_depth: int,
        mode_hint=None,
        enable_mode_variants=True,
        faiss=None,
        bm25=None,
    ):
        _ = (faiss, bm25)
        captured.update(
            {
                "query": query,
                "top_k": top_k,
                "candidate_multiplier": candidate_multiplier,
                "k0": k0,
                "use_query_fusion": use_query_fusion,
                "enable_rerank": enable_rerank,
                "rerank_pool": rerank_pool,
                "diagnostic_pre_rerank_depth": diagnostic_pre_rerank_depth,
                "mode_hint": mode_hint,
                "enable_mode_variants": enable_mode_variants,
            }
        )
        final = [
            ChunkHit(0.42, "c-1", "NIST.FIPS.203", 18, 19, "first chunk"),
            ChunkHit(0.30, "c-2", "NIST.FIPS.204", 7, 7, "second chunk"),
        ]
        return {
            "final_hits": final,
            "pre_rerank_fused_hits": final,
            "post_rerank_hits": final,
            "rerank_pool": rerank_pool,
        }

    monkeypatch.setattr("rag.retrieve._build_hybrid_stage_outputs", _fake_build_hybrid_stage_outputs)

    rows = retrieve_for_eval(
        query="ML-KEM key generation",
        mode="hybrid",
        k=2,
        k0=50,
        candidate_multiplier=6,
        fusion=True,
        evidence_window=True,
        cheap_rerank=False,
        mode_hint="definition",
        debug=True,
    )

    assert captured["query"] == "ML-KEM key generation"
    assert captured["top_k"] == 2
    assert captured["candidate_multiplier"] == 6
    assert captured["k0"] == 50
    assert captured["use_query_fusion"] is True
    assert captured["enable_rerank"] is False
    assert captured["diagnostic_pre_rerank_depth"] >= 60
    assert captured["mode_hint"] == "definition"
    assert captured["enable_mode_variants"] is True

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


def test_retrieve_for_eval_preserves_retrieval_order(monkeypatch):
    def _fake_build_hybrid_stage_outputs(
        query: str,
        top_k: int,
        candidate_multiplier: int,
        k0: int,
        use_query_fusion: bool,
        enable_rerank: bool,
        rerank_pool: int,
        diagnostic_pre_rerank_depth: int,
        mode_hint=None,
        enable_mode_variants=True,
        faiss=None,
        bm25=None,
    ):
        _ = (
            query,
            top_k,
            candidate_multiplier,
            k0,
            use_query_fusion,
            enable_rerank,
            rerank_pool,
            diagnostic_pre_rerank_depth,
            mode_hint,
            enable_mode_variants,
            faiss,
            bm25,
        )
        final = [
            ChunkHit(0.50, "chunk-b", "DOC", 2, 2, "b"),
            ChunkHit(0.50, "chunk-a", "DOC", 1, 1, "a"),
            ChunkHit(0.50, "chunk-c", "DOC", 1, 2, "c"),
        ]
        return {
            "final_hits": final,
            "pre_rerank_fused_hits": final,
            "post_rerank_hits": final,
            "rerank_pool": rerank_pool,
        }

    monkeypatch.setattr("rag.retrieve._build_hybrid_stage_outputs", _fake_build_hybrid_stage_outputs)

    rows = retrieve_for_eval(query="tie case", mode="hybrid", k=3)
    assert [r["chunk_id"] for r in rows] == ["chunk-b", "chunk-a", "chunk-c"]
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_retrieve_for_eval_with_stages_returns_stage_rows(monkeypatch):
    def _fake_build_hybrid_stage_outputs(
        query: str,
        top_k: int,
        candidate_multiplier: int,
        k0: int,
        use_query_fusion: bool,
        enable_rerank: bool,
        rerank_pool: int,
        diagnostic_pre_rerank_depth: int,
        mode_hint=None,
        enable_mode_variants=True,
        faiss=None,
        bm25=None,
    ):
        _ = (
            query,
            top_k,
            candidate_multiplier,
            k0,
            use_query_fusion,
            enable_rerank,
            rerank_pool,
            diagnostic_pre_rerank_depth,
            mode_hint,
            enable_mode_variants,
            faiss,
            bm25,
        )
        return {
            "final_hits": [ChunkHit(0.9, "f1", "DOC", 4, 4, "final")],
            "pre_rerank_fused_hits": [ChunkHit(0.8, "p1", "DOC", 5, 5, "pre")],
            "post_rerank_hits": [ChunkHit(0.7, "r1", "DOC", 6, 6, "post")],
            "rerank_pool": 40,
        }

    monkeypatch.setattr("rag.retrieve._build_hybrid_stage_outputs", _fake_build_hybrid_stage_outputs)
    rows = retrieve_for_eval_with_stages(query="diag", mode="hybrid", k=1)

    assert rows["rerank_pool"] == 40
    assert [r["chunk_id"] for r in rows["hits"]] == ["f1"]
    assert [r["chunk_id"] for r in rows["pre_rerank_fused_hits"]] == ["p1"]
    assert [r["chunk_id"] for r in rows["post_rerank_hits"]] == ["r1"]
