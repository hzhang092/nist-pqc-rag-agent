from rag.retrieve import rrf_fuse
from rag.retriever.base import ChunkHit


def _hit(score: float, chunk_id: str, doc_id: str, page: int) -> ChunkHit:
    return ChunkHit(
        score=score,
        chunk_id=chunk_id,
        doc_id=doc_id,
        start_page=page,
        end_page=page,
        text="",
    )


def test_rrf_fuses_and_prefers_shared_ranked_chunks():
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
    a = _hit(1.0, "chunk-b", "B_DOC", 5)
    b = _hit(1.0, "chunk-a", "A_DOC", 5)

    fused = rrf_fuse([[a], [b]], top_k=2, k0=60)
    assert [h.chunk_id for h in fused] == ["chunk-a", "chunk-b"]
