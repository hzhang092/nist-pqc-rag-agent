"""
Chunks cleaned text from a `pages_clean.jsonl` file into smaller pieces.

This script implements a content-aware chunking strategy suitable for preparing
text for retrieval-augmented generation (RAG) systems. The goal is to create
appropriately sized, overlapping text chunks that respect document structure
(like paragraph breaks) to provide good context for embedding models.

The main steps are:
1. Load cleaned page data from a JSONL file.
2. Group pages by document.
3. For each document, concatenate all its page text into a single string.
4. Apply a greedy chunking algorithm with a sliding window and overlap.
5. The algorithm tries to split at "nice" boundaries (paragraph breaks, newlines,
   or sentence-ending punctuation) rather than cutting words in half.
6. Each chunk is saved as a JSON object with metadata, including the original
   document ID, the page range it covers, and its character length.
7. The final list of chunks is written to an output JSONL file.

The main entry point is `run_chunking`.
"""

# rag/chunk.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


@dataclass
class ChunkConfig:
    """Configuration for the chunking process."""
    target_chars: int = 4000
    overlap_chars: int = 600
    min_chars: int = 400
    breakpoint_lookback: int = 600   # how far back to look for a nice split
    prefer_paragraph_breaks: bool = True


def load_jsonl(path: str | Path) -> List[dict]:
    """Loads a JSONL file into a list of dictionaries."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: List[dict]) -> None:
    """Writes a list of dictionaries to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _best_split_point(text: str, start: int, hard_end: int, cfg: ChunkConfig) -> int:
    """
    Finds the best position to split a text chunk.

    It searches backwards from a `hard_end` position to find a preferred
    boundary, such as a paragraph break, newline, or sentence-ending punctuation.
    This helps avoid splitting in the middle of a sentence or word.

    Args:
        text: The full text of the document.
        start: The start index of the current chunk attempt.
        hard_end: The desired (but not strict) end index for the chunk.
        cfg: The chunking configuration.

    Returns:
        The optimal index at which to split the text.
    """
    if hard_end >= len(text):
        return len(text)

    window_start = max(start, hard_end - cfg.breakpoint_lookback)
    window = text[window_start:hard_end]

    # Prefer paragraph breaks
    if cfg.prefer_paragraph_breaks:
        idx = window.rfind("\n\n")
        if idx != -1:
            return window_start + idx + 2  # split after the break

    # Next: single newline
    idx = window.rfind("\n")
    if idx != -1:
        return window_start + idx + 1

    # Next: sentence-ish punctuation followed by space
    # (Not perfect, but helpful)
    m = list(re.finditer(r"[.!?]\s", window))
    if m:
        return window_start + m[-1].end()

    # Fallback: hard cut
    return hard_end


def _page_span_for_range(page_ranges: List[Tuple[int, int, int]], start: int, end: int) -> Tuple[int, int]:
    """
    Determines the start and end page numbers for a given character range.

    Args:
        page_ranges: A list of tuples, where each tuple contains
                     (page_number, start_char_index, end_char_index) for the document.
        start: The starting character index of the chunk.
        end: The ending character index of the chunk.

    Returns:
        A tuple containing the (start_page, end_page).
    """
    start_page = None
    end_page = None

    for page_no, p_start, p_end in page_ranges:
        # overlap condition
        if p_end > start and p_start < end:
            if start_page is None:
                start_page = page_no
            end_page = page_no

    # Fallback if something weird happens
    if start_page is None:
        start_page = page_ranges[0][0]
        end_page = page_ranges[-1][0]

    return start_page, end_page


def chunk_doc_pages(pages: List[dict], cfg: ChunkConfig,
                    doc_id_key: str, page_key: str, text_key: str) -> List[dict]:
    """
    Chunks the text of a single document, which is provided as a list of pages.

    It first concatenates the text from all pages, then applies a greedy chunking
    algorithm to split it into overlapping chunks of a target size.

    Args:
        pages: A list of page dictionaries, sorted by page number.
        cfg: The chunking configuration.
        doc_id_key: Key for the document ID in the page dictionaries.
        page_key: Key for the page number.
        text_key: Key for the cleaned text.

    Returns:
        A list of chunk dictionaries, each with metadata.
    """
    # Build concatenated doc text while tracking per-page char ranges
    parts: List[str] = []
    page_ranges: List[Tuple[int, int, int]] = []

    cursor = 0
    for p in pages:
        page_no = int(p[page_key])
        txt = (p.get(text_key) or "").strip()
        if not txt:
            continue

        # Add a separator between pages (helps avoid accidental word-joins)
        if parts:
            sep = "\n\n"
            parts.append(sep)
            cursor += len(sep)

        start_idx = cursor
        parts.append(txt)
        cursor += len(txt)
        end_idx = cursor

        page_ranges.append((page_no, start_idx, end_idx))

    doc_text = "".join(parts)
    if not doc_text.strip():
        return []

    # Greedy chunking with overlap
    chunks: List[dict] = []
    start = 0
    chunk_i = 0

    while start < len(doc_text):
        hard_end = min(len(doc_text), start + cfg.target_chars)
        end = _best_split_point(doc_text, start, hard_end, cfg)

        # Ensure progress (avoid infinite loops)
        if end <= start:
            end = min(len(doc_text), start + cfg.target_chars)

        chunk_text = doc_text[start:end].strip()

        # Drop tiny chunks (usually noise at end)
        if len(chunk_text) >= cfg.min_chars:
            sp, ep = _page_span_for_range(page_ranges, start, end)
            chunks.append({
                "chunk_id": f"{pages[0][doc_id_key]}::c{chunk_i:05d}",
                "doc_id": pages[0][doc_id_key],
                "start_page": sp,
                "end_page": ep,
                "text": chunk_text,
                "char_len": len(chunk_text),
                "approx_tokens": max(1, len(chunk_text) // 4),  # rough heuristic
            })
            chunk_i += 1

        if end >= len(doc_text):
            break

        # Overlap: step back a bit for next chunk
        start = max(0, end - cfg.overlap_chars)

        # Optional: avoid starting in the middle of whitespace runs
        while start < len(doc_text) and doc_text[start].isspace():
            start += 1

    return chunks


def run_chunking(
    pages_clean_path: str | Path,
    chunks_out_path: str | Path,
    cfg: ChunkConfig = ChunkConfig(),
    doc_id_key: str = "doc_id",
    page_key: str = "page_number",
    text_key: str = "text_clean",
) -> None:
    """
    Main function to run the chunking process.

    It loads cleaned pages, groups them by document, chunks each document's text,
    and writes the resulting chunks to an output file.

    Args:
        pages_clean_path: Path to the input `pages_clean.jsonl` file.
        chunks_out_path: Path to write the output `chunks.jsonl` file.
        cfg: Chunking configuration object.
        doc_id_key: Key for the document identifier.
        page_key: Key for the page number.
        text_key: Key for the text to be chunked.
    """
    rows = load_jsonl(pages_clean_path)

    # Group by doc_id
    by_doc: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        doc_id = r.get(doc_id_key)
        if not doc_id:
            doc_id = r.get("source_path") or r.get("source") or "unknown_doc"
            r[doc_id_key] = doc_id
        # normalize page key if you used "page"
        if page_key not in r and "page" in r:
            r[page_key] = r["page"]
        by_doc[doc_id].append(r)

    all_chunks: List[dict] = []
    for doc_id, pages in by_doc.items():
        # sort pages
        pages_sorted = sorted(pages, key=lambda x: int(x[page_key]))
        doc_chunks = chunk_doc_pages(pages_sorted, cfg, doc_id_key, page_key, text_key)
        all_chunks.extend(doc_chunks)

    write_jsonl(chunks_out_path, all_chunks)
