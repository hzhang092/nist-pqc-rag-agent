"""
Unit tests for retrieval adapters in the `rag.retrieve` module.

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

from rag.retrieve import (
    _build_planner_hybrid_stage_outputs,
    retrieve_for_eval,
    retrieve_for_eval_with_stages,
)
from rag.retriever.base import ChunkHit


class _RecordingRetriever:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5):
        self.queries.append(query)
        base = query.replace(" ", "_")
        return [
            ChunkHit(1.0, f"{self.prefix}-{base}-1", "DOC", 1, 1, query),
            ChunkHit(0.8, f"{self.prefix}-{base}-2", "DOC", 2, 2, query),
        ][:k]

    def score_text(self, query: str, text: str) -> float:
        _ = query, text
        return 0.0


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
        doc_ids=None,
        enable_mode_variants=True,
        faiss=None,
        bm25=None,
    ):
        _ = (faiss, bm25, doc_ids)
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


def test_planner_hybrid_uses_sparse_and_dense_queries_without_subqueries_for_non_compare():
    faiss = _RecordingRetriever("dense")
    bm25 = _RecordingRetriever("sparse")

    outputs = _build_planner_hybrid_stage_outputs(
        query="What is ML-KEM?",
        canonical_query="ML-KEM",
        sparse_query="ML-KEM definition",
        dense_query="definition and notation for ML-KEM in FIPS 203",
        subqueries=["should not run"],
        top_k=2,
        enable_rerank=False,
        mode_hint="definition",
        faiss=faiss,
        bm25=bm25,
    )

    assert faiss.queries == ["definition and notation for ML-KEM in FIPS 203"]
    assert bm25.queries == ["ML-KEM definition"]
    assert len(outputs["final_hits"]) == 2


def test_planner_hybrid_caps_compare_subqueries_and_is_deterministic():
    faiss_a = _RecordingRetriever("dense")
    bm25_a = _RecordingRetriever("sparse")
    out_a = _build_planner_hybrid_stage_outputs(
        query="Compare ML-DSA and SLH-DSA",
        canonical_query="ML-DSA vs SLH-DSA",
        sparse_query="ML-DSA SLH-DSA intended use-cases comparison",
        dense_query="compare intended use-cases and deployment differences between ML-DSA and SLH-DSA",
        subqueries=[
            "ML-DSA intended use-cases and deployment context",
            "SLH-DSA intended use-cases and deployment context",
            "extra compare branch should be ignored",
        ],
        top_k=4,
        enable_rerank=False,
        mode_hint="compare",
        faiss=faiss_a,
        bm25=bm25_a,
    )

    faiss_b = _RecordingRetriever("dense")
    bm25_b = _RecordingRetriever("sparse")
    out_b = _build_planner_hybrid_stage_outputs(
        query="Compare ML-DSA and SLH-DSA",
        canonical_query="ML-DSA vs SLH-DSA",
        sparse_query="ML-DSA SLH-DSA intended use-cases comparison",
        dense_query="compare intended use-cases and deployment differences between ML-DSA and SLH-DSA",
        subqueries=[
            "ML-DSA intended use-cases and deployment context",
            "SLH-DSA intended use-cases and deployment context",
            "extra compare branch should be ignored",
        ],
        top_k=4,
        enable_rerank=False,
        mode_hint="compare",
        faiss=faiss_b,
        bm25=bm25_b,
    )

    assert faiss_a.queries == [
        "compare intended use-cases and deployment differences between ML-DSA and SLH-DSA",
        "ML-DSA intended use-cases and deployment context",
        "SLH-DSA intended use-cases and deployment context",
    ]
    assert bm25_a.queries == [
        "ML-DSA SLH-DSA intended use-cases comparison",
        "ML-DSA intended use-cases and deployment context",
        "SLH-DSA intended use-cases and deployment context",
    ]
    assert [hit.chunk_id for hit in out_a["final_hits"]] == [hit.chunk_id for hit in out_b["final_hits"]]


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
        doc_ids=None,
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
            doc_ids,
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
        doc_ids=None,
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
            doc_ids,
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
