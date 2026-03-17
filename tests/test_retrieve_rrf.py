"""
Unit tests for the retrieval fusion and reranking logic.

These tests verify the core components of the hybrid retrieval strategy:
- `rrf_fuse`: Tests the Reciprocal Rank Fusion implementation to ensure it
  correctly combines multiple ranked lists and handles tie-breaking.
- `rerank_fused_hits`: Tests the final reranking step, which applies a
  BM25-like score to the fused results to fine-tune the ranking based on
  term frequency.
"""
import rag.retrieve as retrieve_module
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


def test_rrf_tie_break_is_stable_across_ranking_order():
    a = _hit(1.0, "chunk-z", "DOC", 2)
    b = _hit(1.0, "chunk-y", "DOC", 2)
    c = _hit(1.0, "chunk-x", "DOC", 1)

    out1 = rrf_fuse([[a, b, c], [b, c, a]], top_k=3, k0=80)
    out2 = rrf_fuse([[b, c, a], [a, b, c]], top_k=3, k0=80)
    assert [h.chunk_id for h in out1] == [h.chunk_id for h in out2]


class _FakeBM25:
    """A mock BM25 scorer that returns predefined scores for specific texts."""
    def __init__(self, by_chunk_id: dict[str, float]):
        self.by_chunk_id = by_chunk_id

    def score_text(self, query: str, text: str) -> float:
        _ = query
        return float(self.by_chunk_id.get(text, 0.0))


def test_rerank_prefers_exact_technical_token_then_bm25():
    """Do-no-harm: when no candidate clears gates, fused order is preserved."""
    hits = [
        ChunkHit(0.4, "a", "DOC", 1, 1, "chunk-a"),
        ChunkHit(0.3, "b", "DOC", 2, 2, "chunk-b"),
        ChunkHit(0.2, "c", "DOC", 3, 3, "chunk-c"),
    ]

    bm25 = _FakeBM25({"chunk-a": 1.0, "chunk-b": 1.0, "chunk-c": 1.0})
    ranked = rerank_fused_hits(
        query="Explain broad concept",
        hits=hits,
        top_k=3,
        bm25=bm25,
        mode_hint="general",
    )

    assert [h.chunk_id for h in ranked] == ["a", "b", "c"]


def test_rerank_promotes_when_gate_and_signal_are_strong():
    hits = [
        ChunkHit(0.4, "a", "DOC", 1, 1, "chunk-a"),
        ChunkHit(0.3, "b", "DOC", 2, 2, "chunk-b contains ml-kem.keygen details"),
        ChunkHit(0.2, "c", "DOC", 3, 3, "chunk-c"),
    ]
    bm25 = _FakeBM25({"chunk-a": 0.5, "chunk-b contains ml-kem.keygen details": 10.0, "chunk-c": 0.1})
    ranked = rerank_fused_hits(
        query="Algorithm 19 ML-KEM.KeyGen",
        hits=hits,
        top_k=3,
        bm25=bm25,
        mode_hint="algorithm",
    )
    assert [h.chunk_id for h in ranked][0] == "b"


def test_rerank_matches_normalized_identifier_tokens():
    hits = [
        ChunkHit(0.4, "a", "DOC", 1, 1, r"contains MAC\_Data fields"),
        ChunkHit(0.3, "b", "DOC", 2, 2, "contains unrelated text"),
    ]
    bm25 = _FakeBM25({r"contains MAC\_Data fields": 1.0, "contains unrelated text": 100.0})
    ranked = rerank_fused_hits(
        query="Which components form MAC_Data?",
        hits=hits,
        top_k=2,
        bm25=bm25,
        mode_hint="definition",
    )
    assert [h.chunk_id for h in ranked] == ["a", "b"]


def test_rerank_definition_acronym_anchor_mlwe():
    hits = [
        ChunkHit(0.3, "a", "DOC", 1, 1, "MLWE stands for Module Learning with Errors"),
        ChunkHit(0.4, "b", "DOC", 2, 2, "unrelated chunk"),
    ]
    bm25 = _FakeBM25({"MLWE stands for Module Learning with Errors": 1.0, "unrelated chunk": 0.1})
    ranked = rerank_fused_hits(
        query="What does MLWE mean?",
        hits=hits,
        top_k=2,
        bm25=bm25,
        mode_hint="definition",
    )
    assert [h.chunk_id for h in ranked] == ["a", "b"]


def test_rerank_section_prior_promotes_matching_section(monkeypatch):
    hits = [
        ChunkHit(0.4, "a", "DOC", 1, 1, "chunk-a"),
        ChunkHit(0.3, "b", "DOC", 2, 2, "chunk-b"),
    ]
    bm25 = _FakeBM25({"chunk-a": 1.0, "chunk-b": 1.0})
    monkeypatch.setattr(
        retrieve_module,
        "_chunk_metadata_map",
        lambda: {
            "a": {"doc_id": "DOC", "section_path": "1. Intro"},
            "b": {
                "doc_id": "DOC",
                "section_path": "7.1 ML-KEM Key Generation > Algorithm 19 ML-KEM.KeyGen ()",
            },
        },
    )

    debug = {}
    ranked = rerank_fused_hits(
        query="Algorithm 19",
        hits=hits,
        top_k=2,
        bm25=bm25,
        mode_hint="algorithm",
        candidate_section_ids=["section::DOC::7.1 ML-KEM Key Generation > Algorithm 19 ML-KEM.KeyGen ()"],
        debug_out=debug,
    )

    assert [h.chunk_id for h in ranked] == ["b", "a"]
    assert debug["section_prior_applied"] is True
