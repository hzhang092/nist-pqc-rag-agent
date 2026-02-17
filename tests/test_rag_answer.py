"""
Unit tests for the `rag.rag_answer` module.

These tests verify the behavior of the `build_cited_answer` function, ensuring
it correctly handles various scenarios, including:
- Refusing to answer when evidence is insufficient.
- Enforcing strict citation requirements on the language model's output.
- Correctly formatting successful, cited answers.

The tests use `monkeypatch` to modify the application settings in-memory and
a fake generation function (`fake_gen`) to simulate the language model's
output, allowing for controlled testing of the validation logic.

To run the tests, use the command:
    pytest tests/test_rag_answer.py
"""
from __future__ import annotations

from dataclasses import replace

from rag.rag_answer import build_cited_answer
from rag.retriever.base import ChunkHit
from rag.types import REFUSAL_TEXT
import rag.rag_answer as rag_answer_module


def _make_hits(n: int) -> list[ChunkHit]:
    """Helper function to create a list of mock `ChunkHit` objects for testing."""
    hits: list[ChunkHit] = []
    for i in range(1, n + 1):
        hits.append(
            ChunkHit(
                score=1.0 / i,
                chunk_id=f"chunk-{i}",
                doc_id="NIST.FIPS.203",
                start_page=i,
                end_page=i,
                text=f"Evidence text {i} about ML-KEM parameters and definitions.",
            )
        )
    return hits


def test_refusal_on_empty_hits(monkeypatch):
    """
    Verifies that `build_cited_answer` refuses to answer if the initial
    list of hits is empty, and that the LLM is not called.
    """
    monkeypatch.setattr(
        rag_answer_module,
        "SETTINGS",
        replace(rag_answer_module.SETTINGS, ASK_MIN_EVIDENCE_HITS=1),
    )

    called = {"value": False}

    def fake_gen(_prompt: str) -> str:
        called["value"] = True
        return "This is a supported sentence [c1]."

    result = build_cited_answer(question="What is ML-KEM?", hits=[], generate_fn=fake_gen)

    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert called["value"] is False


def test_refusal_when_hits_below_minimum(monkeypatch):
    """
    Verifies that `build_cited_answer` refuses to answer if the number of
    evidence chunks is below the configured `ASK_MIN_EVIDENCE_HITS` threshold.
    """
    monkeypatch.setattr(
        rag_answer_module,
        "SETTINGS",
        replace(rag_answer_module.SETTINGS, ASK_MIN_EVIDENCE_HITS=2),
    )

    def fake_gen(_prompt: str) -> str:
        return "This is a supported sentence [c1]."

    result = build_cited_answer(
        question="What is ML-KEM?",
        hits=_make_hits(1),
        generate_fn=fake_gen,
    )

    assert result.answer == REFUSAL_TEXT
    assert result.citations == []


def test_refusal_when_any_sentence_lacks_citation(monkeypatch):
    """

    Verifies that the system refuses to answer if the LLM's generated text
    contains any sentence that does not have an inline citation marker.
    """
    monkeypatch.setattr(
        rag_answer_module,
        "SETTINGS",
        replace(rag_answer_module.SETTINGS, ASK_MIN_EVIDENCE_HITS=2),
    )

    def fake_gen(_prompt: str) -> str:
        return "First supported statement [c1]. Second statement has no marker."

    result = build_cited_answer(
        question="What is ML-KEM?",
        hits=_make_hits(2),
        generate_fn=fake_gen,
    )

    assert result.answer == REFUSAL_TEXT
    assert result.citations == []


def test_refusal_when_answer_uses_unknown_citation_key(monkeypatch):
    """
    Verifies that the system refuses to answer if the LLM's generated text
    uses a citation key (e.g., `[c99]`) that was not provided in the evidence context.
    """
    monkeypatch.setattr(
        rag_answer_module,
        "SETTINGS",
        replace(rag_answer_module.SETTINGS, ASK_MIN_EVIDENCE_HITS=2),
    )

    def fake_gen(_prompt: str) -> str:
        return "Supported statement with unknown key [c99]."

    result = build_cited_answer(
        question="What is ML-KEM?",
        hits=_make_hits(2),
        generate_fn=fake_gen,
    )

    assert result.answer == REFUSAL_TEXT
    assert result.citations == []


def test_citations_list_matches_used_keys(monkeypatch):
    """
    Verifies that for a successful, well-formed answer, the final `AnswerResult`
    contains a list of `Citation` objects that correctly corresponds to the
    inline citation keys used in the answer text.
    """
    monkeypatch.setattr(
        rag_answer_module,
        "SETTINGS",
        replace(rag_answer_module.SETTINGS, ASK_MIN_EVIDENCE_HITS=2),
    )

    def fake_gen(_prompt: str) -> str:
        return "This is a supported sentence [c1]. Another supported sentence [c2]."

    result = build_cited_answer(
        question="What is ML-KEM?",
        hits=_make_hits(2),
        generate_fn=fake_gen,
    )

    assert "[c1]" in result.answer
    assert result.citations[0].key == "c1"
    assert [c.key for c in result.citations] == ["c1", "c2"]


def test_deterministic_citation_key_assignment_across_hit_order(monkeypatch):
    """
    Determinism test:
    - Same logical hit set, different input order
    - Evidence selection + stable sorting must lead to identical c1/c2 mapping
    - tests dedup (duplicate chunk-a)
    - tests stable sorting tie-break (two chunks with equal score)
    - tests key assignment stability (same c1/c2 mapping even when input hit order flips)
    """
    monkeypatch.setattr(
        rag_answer_module,
        "SETTINGS",
        replace(
            rag_answer_module.SETTINGS,
            ASK_MIN_EVIDENCE_HITS=2,
            ASK_MAX_CONTEXT_CHUNKS=2,   # keep evidence size fixed
            ASK_MAX_CONTEXT_CHARS=10_000,
        ),
    )

    # Two top-scoring chunks tie on score to exercise tie-breaks.
    # Tie-break order should be deterministic by (doc_id, start_page, end_page, chunk_id).
    hits_a = [
        ChunkHit(score=0.90, chunk_id="chunk-b", doc_id="B_DOC", start_page=5, end_page=5, text="B evidence."),
        ChunkHit(score=0.90, chunk_id="chunk-a", doc_id="A_DOC", start_page=5, end_page=5, text="A evidence."),
        # Duplicate chunk-a with worse score: dedup should keep the best one (0.90)
        ChunkHit(score=0.10, chunk_id="chunk-a", doc_id="A_DOC", start_page=5, end_page=5, text="A dup worse."),
        # Lower score: should be dropped because we cap to 2 chunks
        ChunkHit(score=0.80, chunk_id="chunk-c", doc_id="A_DOC", start_page=6, end_page=6, text="C evidence."),
    ]
    hits_b = list(reversed(hits_a))  # different incoming order

    def fake_gen(_prompt: str) -> str:
        # Always cite c1 then c2 so we can compare what those map to.
        return "First grounded sentence [c1]. Second grounded sentence [c2]."

    r1 = build_cited_answer("Q", hits=hits_a, generate_fn=fake_gen)
    r2 = build_cited_answer("Q", hits=hits_b, generate_fn=fake_gen)

    # Same mapping and same ordering across different input order.
    assert [c.key for c in r1.citations] == ["c1", "c2"]
    assert [c.key for c in r2.citations] == ["c1", "c2"]

    assert [c.chunk_id for c in r1.citations] == [c.chunk_id for c in r2.citations]

    # Optional stronger assertion: with the tie, A_DOC should come before B_DOC
    assert [c.chunk_id for c in r1.citations] == ["chunk-a", "chunk-b"]
