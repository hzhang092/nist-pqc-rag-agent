"""
Unit tests for the retrieval fusion and reranking logic.

These tests verify the core components of the hybrid retrieval strategy:
- `rrf_fuse`: Tests the Reciprocal Rank Fusion implementation to ensure it
  correctly combines multiple ranked lists and handles tie-breaking.
- `rerank_fused_hits`: Tests the final reranking step, which applies a
  BM25-like score to the fused results to fine-tune the ranking based on
  term frequency.
"""
from rag.retrieve import rerank_fused_hits, rrf_fuse
from rag.retriever.base import ChunkHit


def _hit(score: float, chunk_id: str, doc_id: str, page: int) -> ChunkHit:
    """Helper function to create a mock `ChunkHit` for testing."""
    return ChunkHit(
        score=score,
        chunk_id=chunk_id,
        doc_id=doc_id,
        start_page=page,
        end_page=page,
        text="",
    )


def test_rrf_fuses_and_prefers_shared_ranked_chunks():
    """
    Verifies that Reciprocal Rank Fusion (RRF) correctly combines hits from
    different retrievers and prioritizes chunks that appear in multiple lists.
    """
    vector_hits = [
        _hit(0.9, "x", "NIST.FIPS.203", 35),
        _hit(0.8, "y", "NIST.FIPS.203", 9),
        _hit(0.7, "z", "NIST.FIPS.204", 12),
    ]
    bm25_hits = [
        _hit(10.0, "x", "NIST.FIPS.203", 35),
        _hit(9.0, "y", "NIST.FIPS.203", 9),
        _hit(8.0, "w", "NIST.FIPS.203", 44),
    ]

    fused = rrf_fuse([vector_hits, bm25_hits], top_k=3, k0=60)
    assert [h.chunk_id for h in fused] == ["x", "y", "w"]


def test_rrf_stable_tie_break_doc_page_chunk():
    """
    Verifies that the RRF implementation has a stable tie-breaking mechanism
    (based on doc_id, page, and chunk_id) when RRF scores are identical.
    """
    a = _hit(1.0, "chunk-b", "B_DOC", 5)
    b = _hit(1.0, "chunk-a", "A_DOC", 5)

    fused = rrf_fuse([[a], [b]], top_k=2, k0=60)
    assert [h.chunk_id for h in fused] == ["chunk-a", "chunk-b"]


class _FakeBM25:
    """A mock BM25 scorer that returns predefined scores for specific texts."""
    def __init__(self, by_chunk_id: dict[str, float]):
        self.by_chunk_id = by_chunk_id

    def score_text(self, query: str, text: str) -> float:
        _ = query
        return float(self.by_chunk_id.get(text, 0.0))


def test_rerank_prefers_exact_technical_token_then_bm25():
    """
    Verifies that the final reranking step correctly prioritizes hits.
    It should prefer chunks containing exact technical tokens from the query,
    and then use the BM25 score as a secondary ranking criterion.
    """
    hits = [
        ChunkHit(0.4, "a", "DOC", 1, 1, "chunk-a"),
        ChunkHit(0.3, "b", "DOC", 2, 2, "chunk-b contains ml-kem.keygen"),
        ChunkHit(0.2, "c", "DOC", 3, 3, "chunk-c"),
    ]

    bm25 = _FakeBM25({"chunk-a": 100.0, "chunk-b contains ml-kem.keygen": 5.0, "chunk-c": 50.0})
    ranked = rerank_fused_hits(query="Explain ML-KEM.KeyGen", hits=hits, top_k=3, bm25=bm25)

    assert [h.chunk_id for h in ranked][:2] == ["b", "a"]
