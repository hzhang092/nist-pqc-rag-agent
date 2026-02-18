import pytest

from rag.rag_answer import build_cited_answer
from rag.retriever.base import ChunkHit
from rag.types import REFUSAL_TEXT
from rag.config import SETTINGS


def _pad_hits_to_min(hits):
    """Pad with low-score dummy hits so len(select_evidence(hits)) >= ASK_MIN_EVIDENCE_HITS."""
    # Ensure unique chunk_ids; select_evidence dedupes by chunk_id.
    min_hits = max(int(getattr(SETTINGS, "ASK_MIN_EVIDENCE_HITS", 1)), 1)

    # Count unique chunk_ids we already have
    seen = {h.chunk_id for h in hits}
    needed = max(0, min_hits - len(seen))

    for i in range(needed):
        hits.append(
            ChunkHit(
                score=0.0001 * (i + 1),   # keep these after the real evidence
                chunk_id=f"PAD_{i}",
                doc_id="PAD",
                start_page=1,
                end_page=1,
                text=f"Padding evidence {i}",
            )
        )
    return hits


def fake_hits_with_algorithm_steps():
    # Real algorithm chunk (should become [c1])
    txt = (
        "Algorithm 2: SHAKE128\n"
        "1: Init ctx\n"
        "2: Absorb str1\n"
        "3: Squeeze out\n"
    )
    hits = [
        ChunkHit(
            score=1.0,
            chunk_id="ALG::p0001::c000",
            doc_id="NIST.FIPS.205",
            start_page=1,
            end_page=1,
            text=txt,
        )
    ]
    return _pad_hits_to_min(hits)


def test_build_cited_answer_accepts_valid_cited_output():
    hits = _pad_hits_to_min([
        ChunkHit(score=1.0, chunk_id="A0", doc_id="D", start_page=1, end_page=1, text="Evidence text")
    ])

    def gen(_prompt: str) -> str:
        # 3 bullets, each one sentence, each includes a valid citation marker.
        return "- Statement one [c1].\n- Statement two [c1].\n- Statement three [c1]."

    result = build_cited_answer("Q", hits, gen)
    assert result.answer != REFUSAL_TEXT
    assert len(result.citations) == 1
    assert result.citations[0].key == "c1"


def test_build_cited_answer_rejects_uncited_output():
    hits = _pad_hits_to_min([
        ChunkHit(score=1.0, chunk_id="A0", doc_id="D", start_page=1, end_page=1, text="Evidence text")
    ])

    def gen(_prompt: str) -> str:
        return "This has no citations."

    result = build_cited_answer("Q", hits, gen)
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []


def test_algorithm_fallback_kicks_in_when_model_refuses():
    hits = fake_hits_with_algorithm_steps()

    def gen(_prompt: str) -> str:
        # Force refusal to trigger deterministic fallback
        return REFUSAL_TEXT

    result = build_cited_answer("What are the steps in Algorithm 2 SHAKE128?", hits, gen)

    # Should succeed due to deterministic fallback extraction
    assert result.answer != REFUSAL_TEXT
    assert "[c1]" in result.answer
    assert len(result.citations) >= 1
