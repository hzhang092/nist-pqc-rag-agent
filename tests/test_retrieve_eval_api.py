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
