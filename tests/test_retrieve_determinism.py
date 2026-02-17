"""
Tests for ensuring the deterministic behavior of the retrieval pipeline.

These tests are crucial for reproducibility and stable evaluation. They verify
that different components of the retrieval process produce the same output given
the same input, even when internal orderings might vary (e.g., due to
asynchronous operations or hash seed randomization).

This includes:
- `query_variants`: Ensuring query variations are generated in a stable order.
- `rrf_fuse`: Ensuring Reciprocal Rank Fusion is stable, especially when scores are tied.
- `hybrid_search`: Ensuring the final hybrid search result is stable regardless
  of the order of results from underlying retrievers.
"""
# tests/test_retrieve_determinism.py
from rag.retrieve import query_variants

def test_query_variants_deterministic_and_deduped():
    """
    Verifies that `query_variants` produces the exact same list of variants
    (in the same order) for the same input query, and that the list contains
    no duplicates.
    """
    # Deterministic query variants (order + de-dup)
    q = "What does FIPS 203 specify for ML-KEM key generation? Algorithm 19 ML-KEM.KeyGen"
    v1 = query_variants(q)
    v2 = query_variants(q)
    assert v1 == v2
    assert len(v1) == len(set(v1))  # no duplicates
    assert v1[0] == q.strip()       # original first
    
from rag.retrieve import rrf_fuse
from rag.retriever.base import ChunkHit

def test_rrf_fuse_deterministic_with_ties():
    """
    Verifies that `rrf_fuse` produces a stable ranking even when different
    input rankings would result in tied RRF scores. The tie-breaking logic
    (based on doc_id, start_page, etc.) should ensure a consistent final order.
    """
    # Deterministic RRF fusion under ties
    # Two rankings that will induce equal RRF scores for A and B
    a = ChunkHit(score=1.0, chunk_id="A", doc_id="DOC1", start_page=2, end_page=2, text="a")
    b = ChunkHit(score=1.0, chunk_id="B", doc_id="DOC1", start_page=1, end_page=1, text="b")

    rankings1 = [[a, b], [b, a]]
    rankings2 = [[b, a], [a, b]]  # same content, different order of rankings

    out1 = rrf_fuse(rankings1, top_k=2, k0=60)
    out2 = rrf_fuse(rankings2, top_k=2, k0=60)

    # must be stable across ranking order
    assert [h.chunk_id for h in out1] == [h.chunk_id for h in out2]

    # and must match your tie-break: doc_id, start_page, chunk_id
    # here b has earlier start_page so it should come first when tied
    assert [h.chunk_id for h in out1] == ["B", "A"]

from rag.retrieve import hybrid_search
from rag.retriever.base import ChunkHit

class FakeRetriever:
    """A mock retriever that returns a predefined list of hits."""
    def __init__(self, hits):
        self._hits = hits

    def search(self, _q: str, k: int = 5):
        return list(self._hits)[:k]

class FakeBM25Retriever(FakeRetriever):
    """Fake BM25 with a deterministic lexical scorer."""
    def score_text(self, query: str, text: str) -> float:
        # Deterministic toy score: count how many "technical tokens" appear.
        # Good enough for tests; doesn’t need to reflect real BM25.
        q = query.lower()
        t = (text or "").lower()
        return float(int(q in t))

def test_hybrid_search_deterministic_across_backend_order():
    h1 = ChunkHit(score=9, chunk_id="X", doc_id="D", start_page=2, end_page=2, text="x")
    h2 = ChunkHit(score=8, chunk_id="Y", doc_id="D", start_page=1, end_page=1, text="y")
    h3 = ChunkHit(score=7, chunk_id="Z", doc_id="D", start_page=3, end_page=3, text="z")

    faiss_a = FakeRetriever([h1, h2, h3])
    bm25_a  = FakeBM25Retriever([h2, h1, h3])

    faiss_b = FakeRetriever([h3, h2, h1])  # reversed
    bm25_b  = FakeBM25Retriever([h1, h3, h2])

    out_a = hybrid_search("ML-KEM.KeyGen", top_k=3, use_query_fusion=False, faiss=faiss_a, bm25=bm25_a)
    out_b = hybrid_search("ML-KEM.KeyGen", top_k=3, use_query_fusion=False, faiss=faiss_b, bm25=bm25_b)

    assert [h.chunk_id for h in out_a] == [h.chunk_id for h in out_b]
