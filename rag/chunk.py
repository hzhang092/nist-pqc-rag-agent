"""
Chunking engine to split documents into smaller, context-aware pieces.

This module provides a set of functions to split text from PDF pages into
semantically coherent chunks suitable for retrieval-augmented generation (RAG).
The core idea is to respect the structural integrity of technical content
like tables, algorithms, and mathematical formulas, while also breaking down
prose into manageable sizes.

Key components:
- Heuristics to identify "verbatim-ish" content (tables, code, math).
- `split_into_blocks`: Splits a page's text into blocks based on blank lines,
  then decides whether to preserve line breaks (for verbatim content) or
  join lines (for prose).
- `pack_blocks_into_chunks`: Greedily packs these blocks into larger chunks
  that meet certain size constraints, with optional overlap.
- `run_chunking_per_page`: The main entry point that processes a JSONL file
  of cleaned pages, chunks each page, and writes the results to a new
  JSONL file.
"""
# rag/chunk.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


@dataclass
class ChunkConfig:
    """Configuration for the chunking process."""
    target_chars: int = 1400      # << smaller for precision (250–400 tokens-ish)
    overlap_blocks: int = 1       # overlap by repeating last N blocks into next chunk
    min_chars: int = 250          # drop tiny junk chunks
    max_chars: int = 2200         # allow a bit of spill if a block is big


# --- Heuristics to preserve technical structure ---
_REPEAT_SPACES_RE = re.compile(r"( {2,}|\t{1,})")
_MATH_RE = re.compile(r"(\$|\\\(|\\\)|\\\[|\\\])")

def looks_like_table_line(line: str) -> bool:
    """Heuristically checks if a line appears to be part of a table."""
    s = line.rstrip()
    if not s.strip():
        return False
    if s.strip().startswith("|") or s.count("|") >= 2:
        return True
    if len(_REPEAT_SPACES_RE.findall(s)) >= 2:
        return True
    return False

def looks_like_code_or_algo_line(line: str) -> bool:
    """Heuristically checks if a line appears to be part of code or an algorithm."""
    s = line.rstrip()
    if not s.strip():
        return False
    if s.startswith("    ") or s.startswith("\t"):
        return True
    if any(tok in s for tok in ("::=", ":=", "->", "<-", "{", "}", "[", "]")):
        return True
    if re.match(r"^\s*(\d+\.|\(\d+\)|Step\s+\d+[:.]|Algorithm\s+\d+[:.])", s, re.I):
        return True
    if re.match(r"^\s*(Input|Output|Require|Ensure|Given)[:\s]", s, re.I):
        return True
    return False

def looks_like_math_line(line: str) -> bool:
    """Heuristically checks if a line appears to be part of a mathematical expression."""
    s = line.rstrip()
    if not s.strip():
        return False
    if _MATH_RE.search(s):
        return True
    if sum(ch in s for ch in "=<>±×÷∑∏∈∉≈≡≤≥⊕⊗") >= 1:
        return True
    return False

def is_verbatimish_line(line: str) -> bool:
    """
    Determines if a line should be treated as "verbatim-ish" content.

    This includes tables, code, algorithms, or mathematical notation, where
    preserving the original line breaks and spacing is important.
    """
    return looks_like_table_line(line) or looks_like_code_or_algo_line(line) or looks_like_math_line(line)


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


def split_into_blocks(page_text: str) -> List[str]:
    """
    Splits a page of text into semantic blocks.

    The function first divides the text into groups of lines separated by blank
    lines. It then analyzes each group to determine if it's "verbatim-ish"
    (like a table, code, or math) or prose.

    - For verbatim blocks, line breaks are preserved.
    - For prose blocks, lines are joined to reverse hard wraps from PDF extraction.

    Args:
        page_text: The text content of a single page.

    Returns:
        A list of strings, where each string is a content block.
    """
    lines = page_text.split("\n")
    blocks: List[List[str]] = []
    cur: List[str] = []

    for ln in lines:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln.rstrip())
    if cur:
        blocks.append(cur)

    # Light post-process:
    # If a block is mostly verbatimish lines (tables/algo/math), keep line breaks.
    # Otherwise, it’s prose-ish; we can join lines within the block to avoid PDF hard wraps.
    out_blocks: List[str] = []
    for b in blocks:
        if not b:
            continue
        verb_count = sum(1 for ln in b if is_verbatimish_line(ln))
        if verb_count >= max(1, len(b) // 2):
            out_blocks.append("\n".join(b).strip())
        else:
            out_blocks.append(" ".join(ln.strip() for ln in b).strip())

    return [b for b in out_blocks if b]


def pack_blocks_into_chunks(blocks: List[str], cfg: ChunkConfig) -> List[str]:
    """
    Greedily packs content blocks into larger chunks based on character limits.

    This function iterates through blocks and adds them to the current chunk
    until the `target_chars` limit is approached. It includes logic to handle
    oversized blocks and to overlap chunks by carrying over a specified number
    of blocks from the end of one chunk to the beginning of the next.

    Args:
        blocks: A list of content blocks from `split_into_blocks`.
        cfg: The chunking configuration.

    Returns:
        A list of strings, where each string is a final chunk of text.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        text = "\n\n".join(cur).strip()
        if len(text) >= cfg.min_chars:
            chunks.append(text)
        # overlap: carry last N blocks
        if cfg.overlap_blocks > 0:
            carry = cur[-cfg.overlap_blocks:]
        else:
            carry = []
        cur = carry[:]
        cur_len = sum(len(x) for x in cur) + (2 * max(0, len(cur) - 1))

    for blk in blocks:
        blk_len = len(blk)
        # If a single block is huge (big table), allow it as its own chunk
        if blk_len > cfg.max_chars and not cur:
            chunks.append(blk.strip())
            continue

        # Would adding this block exceed target?
        proposed = cur_len + (2 if cur else 0) + blk_len
        if proposed <= cfg.target_chars or (proposed <= cfg.max_chars and cur_len < cfg.min_chars):
            cur.append(blk)
            cur_len = proposed
        else:
            flush()
            # try to add after flush
            if len(blk) > cfg.max_chars and not cur:
                chunks.append(blk.strip())
            else:
                cur.append(blk)
                cur_len = len(blk)

    flush()
    return chunks


def run_chunking_per_page(
    pages_clean_path: str | Path,
    chunks_out_path: str | Path,
    cfg: ChunkConfig = ChunkConfig(),
    doc_id_key: str = "doc_id",
    page_key: str = "page_number",
    text_key: str = "text_clean",
) -> None:
    """
    Runs the full chunking pipeline on a file of cleaned pages.

    This function reads a JSONL file where each line is a dictionary representing
    a cleaned page from a document. It groups pages by document, then processes
    each page's text to generate chunks. The resulting chunks are written to a
    new JSONL file with detailed metadata.

    Args:
        pages_clean_path: Path to the input JSONL file containing cleaned pages.
        chunks_out_path: Path to write the output JSONL file of chunks.
        cfg: The chunking configuration.
        doc_id_key: The dictionary key for the document identifier.
        page_key: The dictionary key for the page number.
        text_key: The dictionary key for the cleaned page text.
    """
    pages = load_jsonl(pages_clean_path)

    # Normalize page key if needed
    for p in pages:
        if page_key not in p and "page" in p:
            p[page_key] = p["page"]

    # Group by doc
    by_doc: Dict[str, List[dict]] = defaultdict(list)
    for p in pages:
        doc_id = p.get(doc_id_key) or p.get("source_path") or p.get("source") or "unknown_doc"
        p[doc_id_key] = doc_id
        by_doc[doc_id].append(p)

    out_chunks: List[dict] = []

    for doc_id, ps in by_doc.items():
        ps = sorted(ps, key=lambda x: int(x[page_key]))
        for p in ps:
            page_no = int(p[page_key])
            text = (p.get(text_key) or "").strip()
            if not text:
                continue

            blocks = split_into_blocks(text)
            chunk_texts = pack_blocks_into_chunks(blocks, cfg)

            for i, ct in enumerate(chunk_texts):
                out_chunks.append({
                    "chunk_id": f"{doc_id}::p{page_no:04d}::c{i:03d}",
                    "doc_id": doc_id,
                    "page_number": page_no,
                    "start_page": page_no,
                    "end_page": page_no,
                    "text": ct,
                    "char_len": len(ct),
                    "approx_tokens": max(1, len(ct) // 4),
                })

    write_jsonl(chunks_out_path, out_chunks)
