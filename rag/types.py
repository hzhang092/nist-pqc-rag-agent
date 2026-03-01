"""
Core data types and validation for the question-answering component.

This module defines the data structures (`dataclasses`) that form the contract
for inputs and outputs of the answering agent. It ensures that answers,
citations, and refusals are handled in a consistent and verifiable way.

Key components:
- `Citation`: Represents a single reference to a source document chunk.
- `AnswerResult`: Encapsulates the final generated answer, its associated
  citations, and any additional notes. It defines the structure for both
  successful answers and refusals.
- `validate_answer`: A crucial function that enforces the data contract,
  checking for invariants like valid page numbers, required citations for
  non-refusal answers, and correct inline citation markers.
"""
# rag/types.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Set
import re

# The specific string used to indicate that the model could not find an answer.
REFUSAL_TEXT = "not found in provided docs"


@dataclass(frozen=True)
class Citation:
    """
    Represents a single piece of evidence backing a claim in the answer.

    It links a stable key (like 'c1'), used for inline references in the answer
    text, to a specific chunk within a source document.
    """
    key: str          # e.g., "c1"
    doc_id: str       # e.g., "NIST.FIPS.203"
    start_page: int
    end_page: int
    chunk_id: str


@dataclass(frozen=True)
class AnswerResult:
    """
    Encapsulates the final output of the question-answering process.

    This structure holds the generated text answer, a list of all citations
    that support the answer, and optional notes for debugging or metadata.
    It provides a consistent format for both successful answers and refusals.
    """
    answer: str
    citations: List[Citation]
    notes: Optional[str] = None

    def is_refusal(self) -> bool:
        """Checks if the answer is a refusal to answer."""
        return self.answer.strip().lower() == REFUSAL_TEXT

    def to_dict(self) -> dict:
        """Converts the AnswerResult to a dictionary."""
        return asdict(self)


_CITE_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_CITE_TOKEN_RE = re.compile(r"\bc\d+\b", flags=re.IGNORECASE)

def extract_citation_keys(answer_text: str) -> Set[str]:
    """
    Finds all unique inline citation markers (e.g., `[c1]`, `[c2]`) in the text.
    """
    keys: Set[str] = set()
    for bracket_content in _CITE_BRACKET_RE.findall(answer_text):
        for token in _CITE_TOKEN_RE.findall(bracket_content):
            keys.add(token.lower())
    return keys


def validate_answer(
    result: AnswerResult,
    require_citations: bool = True,
    require_inline_markers: bool = False,
) -> None:
    """
    Enforces data contract invariants on an AnswerResult.

    This function is critical for ensuring that the output of the answering
    agent is well-formed and can be trusted by downstream components like a
    CLI, evaluation harness, or another agent. It raises a `ValueError` if
    any of the specified rules are violated.

    Args:
        result: The AnswerResult to validate.
        require_citations: If True, a non-refusal answer must have at least one citation.
        require_inline_markers: If True, the answer text must contain citation
            markers (e.g., `[c1]`) that correspond to the provided citations.

    Raises:
        ValueError: If the AnswerResult violates any of the validation rules.
    """
    # Basic page sanity
    for c in result.citations:
        if c.start_page <= 0 or c.end_page <= 0:
            raise ValueError(f"Invalid page numbers in citation {c}")
        if c.start_page > c.end_page:
            raise ValueError(f"start_page > end_page in citation {c}")

    if result.is_refusal():
        if len(result.citations) != 0:
            raise ValueError("Refusal must return empty citations.")
        return

    # Non-refusal answer rules
    if require_citations and len(result.citations) == 0:
        raise ValueError("Non-refusal answer must include citations.")

    if require_inline_markers:
        used = extract_citation_keys(result.answer)
        known = {c.key for c in result.citations}
        if not used:
            raise ValueError("Answer must include inline citation markers like [c1].")
        unknown = used - known
        if unknown:
            raise ValueError(f"Answer uses unknown citation keys: {sorted(unknown)}")
