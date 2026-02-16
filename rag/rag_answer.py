"""
Core logic for generating a cited answer from retrieved document chunks.

This module orchestrates the process of taking a list of retrieved `ChunkHit`
objects, selecting the most relevant ones as evidence, constructing a prompt
for a language model, and then validating the generated answer to ensure it is
properly grounded in the provided citations.

The main entry point is `build_cited_answer`, which enforces a strict contract:
- Evidence is selected and budgeted.
- A detailed prompt instructs the model on how to use inline citations.
- The model's raw output is rigorously validated by `enforce_inline_citations`
  to ensure every sentence is cited and no external information is used.
- If validation fails, the system defaults to a safe, "I can't answer" response.
"""
# rag/rag_answer.py
from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

from rag.config import SETTINGS
from rag.retriever.base import ChunkHit
from rag.types import AnswerResult, Citation, REFUSAL_TEXT, validate_answer


_CITE_RE = re.compile(r"\[(c\d+)\]")

def _stable_sort_key(h: ChunkHit) -> tuple:
    """
    Creates a sort key for `ChunkHit` objects for deterministic ordering.

    Sorts primarily by score (descending), then uses document/chunk IDs and
    page numbers as tie-breakers to ensure stable results across runs.
    """
    # Deterministic tie-breaks matter for stable reruns/tests.
    return (-h.score, h.doc_id, h.start_page, h.end_page, h.chunk_id)


def select_evidence(hits: List[ChunkHit]) -> List[ChunkHit]:
    """
    Filters, de-duplicates, and budgets raw search hits to form the final evidence set.

    This function performs several steps to refine the list of retrieved chunks:
    1.  De-duplicates hits based on `chunk_id`, keeping only the highest-scoring
        instance of each chunk.
    2.  Sorts the unique chunks deterministically using `_stable_sort_key`.
    3.  Truncates the list to `ASK_MAX_CONTEXT_CHUNKS`.
    4.  Further filters the list to stay within the `ASK_MAX_CONTEXT_CHARS` budget,
        preventing overly long prompts.

    Args:
        hits: The raw list of `ChunkHit` objects from the retriever.

    Returns:
        A de-duplicated, sorted, and budgeted list of `ChunkHit` objects.
    """
    # Dedup by chunk_id, keep best-scoring instance
    best: Dict[str, ChunkHit] = {}
    for h in hits:
        prev = best.get(h.chunk_id)
        if prev is None or h.score > prev.score:
            best[h.chunk_id] = h

    ordered = sorted(best.values(), key=_stable_sort_key)

    # Apply chunk count limit
    ordered = ordered[: SETTINGS.ASK_MAX_CONTEXT_CHUNKS]

    # Apply char budget (avoid huge algorithm blocks blowing context)
    budgeted: List[ChunkHit] = []
    total = 0
    for h in ordered:
        t = h.text or ""
        if not t.strip():
            continue
        if total + len(t) > SETTINGS.ASK_MAX_CONTEXT_CHARS and budgeted:
            break
        budgeted.append(h)
        total += len(t)

    return budgeted


def build_context_and_citations(evidence: List[ChunkHit]) -> Tuple[str, Dict[str, Citation]]:
    """
    Constructs the context string for the LLM prompt and a citation map.

    It assigns stable citation keys (e.g., `[c1]`, `[c2]`) to each piece of
    evidence and formats them into a single string to be passed to the language
    model. It also returns a dictionary mapping these keys to their full
    `Citation` objects.

    Args:
        evidence: The final list of evidence chunks to be used.

    Returns:
        A tuple containing:
        - The formatted context string for the prompt.
        - A dictionary mapping citation keys (e.g., 'c1') to `Citation` objects.
    """
    key_to_cit: Dict[str, Citation] = {}
    blocks: List[str] = []

    for i, h in enumerate(evidence, start=1):
        key = f"c{i}"
        key_to_cit[key] = Citation(
            key=key,
            doc_id=h.doc_id,
            start_page=h.start_page,
            end_page=h.end_page,
            chunk_id=h.chunk_id,
        )
        blocks.append(
            f"[{key}] {h.doc_id} p{h.start_page}-p{h.end_page} chunk_id={h.chunk_id}\n"
            f"{h.text.strip()}"
        )

    context = "\n\n---\n\n".join(blocks)
    return context, key_to_cit


def _sentences(text: str) -> List[str]:
    """
    Splits text into sentences using a simple regex heuristic.

    Note: This is a basic implementation sufficient for compliance checks.
    It could be replaced with a more sophisticated sentence splitter if needed.
    """
    # Simple heuristic: good enough for Day 2 compliance checks.
    # You can replace later with a better splitter.
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p for p in parts if p.strip()]


def enforce_inline_citations(answer_text: str, key_to_cit: Dict[str, Citation]) -> AnswerResult:
    """
    Validates the raw LLM output to enforce strict citation grounding.

    This function enforces several critical rules:
    - The answer must contain inline citation markers (e.g., `[c1]`).
    - Every sentence in the answer must have at least one citation marker.
    - All used citation markers must correspond to the provided evidence.

    If any of these rules are violated, the function returns a standard
    "refusal to answer" result, ensuring that no un-cited claims are ever
    shown to the user.

    Args:
        answer_text: The raw text output from the language model.
        key_to_cit: The dictionary mapping valid citation keys to `Citation` objects.

    Returns:
        A validated `AnswerResult` object. If validation fails, it will be a
        standard refusal.
    """
    ans = answer_text.strip()

    # If model tries to refuse in a different string, normalize to our refusal contract.
    if ans.lower() in {"not found", "not found in documents", REFUSAL_TEXT}:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    used = set(_CITE_RE.findall(ans))
    if not used:
        # No inline markers → fail hard into refusal for Day 2 (safe behavior).
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    unknown = used - set(key_to_cit.keys())
    if unknown:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    # Every sentence must have ≥1 citation marker
    for s in _sentences(ans):
        if not _CITE_RE.search(s):
            result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
            validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
            return result

    # Build citations list in stable order c1..cN but keep only those actually used
    citations = [key_to_cit[k] for k in sorted(used, key=lambda x: int(x[1:]))]

    result = AnswerResult(answer=ans, citations=citations)
    validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
    return result


def build_cited_answer(
    question: str,
    hits: List[ChunkHit],
    generate_fn: Callable[[str], str],
) -> AnswerResult:
    """
    The main entry point for generating a citation-grounded answer.

    This function orchestrates the entire process:
    1.  Selects and budgets evidence from the initial retrieval hits.
    2.  Refuses to answer if insufficient evidence is found.
    3.  Builds a detailed prompt with instructions for the LLM on how to format
        the answer and use citations.
    4.  Calls the provided `generate_fn` to get a raw answer from the LLM.
    5.  Passes the raw answer through `enforce_inline_citations` to ensure it
        is fully grounded and correctly formatted.

    Args:
        question: The user's original question.
        hits: The list of `ChunkHit` objects from the retriever.
        generate_fn: A callable that takes a prompt string and returns the
            language model's text response.

    Returns:
        A validated `AnswerResult` object.
    """
    evidence = select_evidence(hits)
    if len(evidence) < SETTINGS.ASK_MIN_EVIDENCE_HITS:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    context, key_to_cit = build_context_and_citations(evidence)

    prompt = (
        "You are a citation-grounded assistant. Answer ONLY using the evidence below.\n"
        "Rules:\n"
        "1) Every sentence MUST end with at least one inline citation marker like [c1].\n"
        "2) You may ONLY use citation markers that appear in the evidence headers.\n"
        f"3) If the evidence is insufficient, reply exactly: {REFUSAL_TEXT}\n"
        "4) Be concise and factual.\n\n"
        "5) NEVER hallucinate or use information not in the provided evidence.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{context}\n"
    )

    raw_answer = generate_fn(prompt)
    return enforce_inline_citations(raw_answer, key_to_cit)
