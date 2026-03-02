
from __future__ import annotations
"""
Chunking module for splitting document pages into semantically meaningful chunks.

This module provides functionality to split cleaned document pages into chunks suitable
for RAG (Retrieval-Augmented Generation) systems. It supports two chunking strategies:
- v1: Simple block-based chunking with paragraph detection
- v2: Structure-aware chunking that preserves markdown elements (headings, tables, algorithms, lists)

Classes:
    ChunkConfig: Configuration parameters for the chunking process
    StructuredBlock: Represents a single structural element (paragraph, table, algorithm, etc.)

Functions:
    looks_like_table_line: Detects if a line is part of a table
    looks_like_code_or_algo_line: Detects if a line is code or algorithm pseudocode
    looks_like_math_line: Detects if a line contains mathematical notation
    is_verbatimish_line: Checks if a line should be preserved verbatim
    load_jsonl: Loads data from a JSONL file
    write_jsonl: Writes data to a JSONL file
    split_into_blocks: Splits page text into logical blocks (v1 chunker)
    pack_blocks_into_chunks: Packs blocks into chunks respecting size limits
    split_markdown_into_structured_blocks: Parses markdown into structured blocks (v2 chunker)
    pack_structured_blocks_into_chunks: Packs structured blocks into chunks
    run_chunking_per_page: Main entry point for chunking documents

The chunking process:
1. Loads cleaned pages from JSONL
2. Groups pages by document
3. For each page, applies the selected chunking strategy
4. Outputs chunks with metadata (chunk_id, doc_id, page_number, text, etc.)
5. Optionally updates a manifest file with chunking statistics

Chunk configuration parameters:
- target_chars: Target character count per chunk (default: 1400)
- overlap_blocks: Number of blocks to overlap between chunks (default: 1)
- min_chars: Minimum characters required for a valid chunk (default: 250)
- max_chars: Maximum characters allowed per chunk (default: 2200)
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from rag.versioning import update_manifest


@dataclass
class ChunkConfig:
    target_chars: int = 1400
    overlap_blocks: int = 1
    min_chars: int = 250
    max_chars: int = 2200


@dataclass(frozen=True)
class StructuredBlock:
    text: str
    block_type: str
    section_path: str


_REPEAT_SPACES_RE = re.compile(r"( {2,}|\t{1,})")
_MATH_RE = re.compile(r"(\$|\\\(|\\\)|\\\[|\\\])")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_NUMERIC_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\s+([A-Za-z][^\n]{1,120})\s*$")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]|[0-9]+\.)\s+")
_ALGO_START_RE = re.compile(r"^\s*Algorithm\b", re.I)
_ALGO_LINE_RE = re.compile(
    r"^\s*(Algorithm\b|Input\b|Output\b|Require\b|Ensure\b|Step\s+\d+|[0-9]+\.)",
    re.I,
)


def looks_like_table_line(line: str) -> bool:
    s = line.rstrip()
    if not s.strip():
        return False
    if s.strip().startswith("|") or s.count("|") >= 2:
        return True
    if len(_REPEAT_SPACES_RE.findall(s)) >= 2:
        return True
    return False


def looks_like_code_or_algo_line(line: str) -> bool:
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
    s = line.rstrip()
    if not s.strip():
        return False
    if _MATH_RE.search(s):
        return True
    if sum(ch in s for ch in "=<>±×÷∑∏∈∉≈≡≤≥⊕⊗") >= 1:
        return True
    return False


def is_verbatimish_line(line: str) -> bool:
    return looks_like_table_line(line) or looks_like_code_or_algo_line(line) or looks_like_math_line(line)


def load_jsonl(path: str | Path) -> List[dict]:
    p = Path(path)
    rows: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: List[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def split_into_blocks(page_text: str) -> List[str]:
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
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        text = "\n\n".join(cur).strip()
        if len(text) >= cfg.min_chars:
            chunks.append(text)
        carry = cur[-cfg.overlap_blocks:] if cfg.overlap_blocks > 0 else []
        cur = carry[:]
        cur_len = sum(len(x) for x in cur) + (2 * max(0, len(cur) - 1))

    for blk in blocks:
        blk_len = len(blk)
        if blk_len > cfg.max_chars and not cur:
            chunks.append(blk.strip())
            continue

        proposed = cur_len + (2 if cur else 0) + blk_len
        if proposed <= cfg.target_chars or (proposed <= cfg.max_chars and cur_len < cfg.min_chars):
            cur.append(blk)
            cur_len = proposed
        else:
            flush()
            if len(blk) > cfg.max_chars and not cur:
                chunks.append(blk.strip())
            else:
                cur.append(blk)
                cur_len = len(blk)

    flush()
    return chunks


def _heading_from_line(line: str) -> tuple[int, str] | None:
    m = _MD_HEADING_RE.match(line)
    if m:
        level = len(m.group(1))
        title = m.group(2).strip().strip("#").strip()
        if title:
            return level, title

    m = _NUMERIC_HEADING_RE.match(line)
    if m:
        numeric = m.group(1)
        title = m.group(2).strip()
        if title:
            level = min(6, numeric.count(".") + 1)
            return level, f"{numeric} {title}"
    return None


def _update_heading_stack(stack: list[tuple[int, str]], level: int, title: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _section_path(stack: list[tuple[int, str]]) -> str:
    if not stack:
        return ""
    return " > ".join(title for _, title in stack)


def _is_markdown_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.count("|") >= 2:
        return True
    return False


def _is_algorithm_start_line(line: str) -> bool:
    return bool(_ALGO_START_RE.match(line)) or bool(
        re.match(r"^\s*(Input|Output|Require|Ensure)\b", line, re.I)
    )


def _is_algorithm_continuation_line(line: str) -> bool:
    if not line.strip():
        return False
    if _ALGO_LINE_RE.match(line):
        return True
    return looks_like_code_or_algo_line(line) or looks_like_math_line(line)


def _is_list_line(line: str) -> bool:
    return bool(_LIST_LINE_RE.match(line))


def split_markdown_into_structured_blocks(markdown: str) -> List[StructuredBlock]:
    lines = (markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(lines)
    i = 0
    heading_stack: list[tuple[int, str]] = []
    blocks: list[StructuredBlock] = []

    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        heading = _heading_from_line(line)
        if heading is not None:
            level, title = heading
            _update_heading_stack(heading_stack, level, title)
            blocks.append(
                StructuredBlock(
                    text=title,
                    block_type="text",
                    section_path=_section_path(heading_stack),
                )
            )
            i += 1
            continue

        if _is_algorithm_start_line(line):
            j = i
            block_lines = [line]
            j += 1
            while j < n:
                nxt = lines[j].rstrip()
                if _heading_from_line(nxt) is not None:
                    break
                if not nxt.strip():
                    k = j + 1
                    while k < n and not lines[k].strip():
                        k += 1
                    if k >= n:
                        break
                    probe = lines[k].rstrip()
                    if _heading_from_line(probe) is not None or not _is_algorithm_continuation_line(probe):
                        break
                block_lines.append(nxt)
                j += 1

            text = "\n".join(x for x in block_lines).strip()
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="algorithm",
                        section_path=_section_path(heading_stack),
                    )
                )
            i = j
            continue

        if _is_markdown_table_line(line):
            j = i
            block_lines = [line]
            j += 1
            while j < n:
                nxt = lines[j].rstrip()
                if _is_markdown_table_line(nxt):
                    block_lines.append(nxt)
                    j += 1
                    continue
                if not nxt.strip():
                    k = j + 1
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and _is_markdown_table_line(lines[k].rstrip()):
                        block_lines.append("")
                        j += 1
                        continue
                break
            text = "\n".join(x for x in block_lines).strip()
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="table",
                        section_path=_section_path(heading_stack),
                    )
                )
            i = j
            continue

        if _is_list_line(line):
            j = i
            block_lines = [line]
            j += 1
            while j < n:
                nxt = lines[j].rstrip()
                if _is_list_line(nxt):
                    block_lines.append(nxt)
                    j += 1
                    continue
                if nxt.startswith("  ") and nxt.strip():
                    block_lines.append(nxt)
                    j += 1
                    continue
                break
            text = "\n".join(block_lines).strip()
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="list",
                        section_path=_section_path(heading_stack),
                    )
                )
            i = j
            continue

        j = i
        para_parts = [line.strip()]
        j += 1
        while j < n:
            nxt = lines[j].rstrip()
            if not nxt.strip():
                break
            if _heading_from_line(nxt) is not None:
                break
            if _is_markdown_table_line(nxt):
                break
            if _is_algorithm_start_line(nxt):
                break
            if _is_list_line(nxt):
                break
            para_parts.append(nxt.strip())
            j += 1

        text = " ".join(part for part in para_parts if part).strip()
        if text:
            blocks.append(
                StructuredBlock(
                    text=text,
                    block_type="text",
                    section_path=_section_path(heading_stack),
                )
            )
        i = j

    return [b for b in blocks if b.text.strip()]


def pack_structured_blocks_into_chunks(blocks: List[StructuredBlock], cfg: ChunkConfig) -> List[List[StructuredBlock]]:
    chunks: List[List[StructuredBlock]] = []
    cur: List[StructuredBlock] = []
    cur_len = 0

    def cur_text_len(items: List[StructuredBlock]) -> int:
        return sum(len(x.text) for x in items) + (2 * max(0, len(items) - 1))

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        text = "\n\n".join(x.text for x in cur).strip()
        if len(text) >= cfg.min_chars:
            chunks.append(cur[:])
        carry = cur[-cfg.overlap_blocks:] if cfg.overlap_blocks > 0 else []
        cur = carry[:]
        cur_len = cur_text_len(cur)

    for blk in blocks:
        blk_len = len(blk.text)
        if blk_len > cfg.max_chars and not cur:
            chunks.append([blk])
            continue

        proposed = cur_len + (2 if cur else 0) + blk_len
        if proposed <= cfg.target_chars or (proposed <= cfg.max_chars and cur_len < cfg.min_chars):
            cur.append(blk)
            cur_len = proposed
        else:
            flush()
            if blk_len > cfg.max_chars and not cur:
                chunks.append([blk])
            else:
                cur.append(blk)
                cur_len = cur_text_len(cur)

    flush()
    return chunks


def _dominant_block_type(blocks: List[StructuredBlock]) -> str:
    if not blocks:
        return "text"
    counts = Counter(b.block_type for b in blocks)
    # Prefer high-signal technical block classes when mixed with generic text.
    for preferred in ("algorithm", "table", "code", "math"):
        if preferred in counts and counts[preferred] >= counts.get("text", 0):
            return preferred
    if len(counts) == 1:
        return next(iter(counts))
    most_common = counts.most_common()
    if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
        return "mixed"
    return str(most_common[0][0])


def _dominant_section_path(blocks: List[StructuredBlock]) -> str:
    paths = [b.section_path for b in blocks if b.section_path]
    if not paths:
        return ""
    counts = Counter(paths)
    return counts.most_common(1)[0][0]


def run_chunking_per_page(
    pages_clean_path: str | Path,
    chunks_out_path: str | Path,
    cfg: ChunkConfig = ChunkConfig(),
    doc_id_key: str = "doc_id",
    page_key: str = "page_number",
    text_key: str = "text_clean",
    chunker_version: str = "v1",
    markdown_key: str = "markdown",
    write_manifest: bool = True,
) -> None:
    pages = load_jsonl(pages_clean_path)

    selected_chunker = str(chunker_version).strip().lower()
    if selected_chunker not in {"v1", "v2"}:
        raise ValueError(f"Unsupported chunker_version={chunker_version!r}. Expected 'v1' or 'v2'.")

    for p in pages:
        if page_key not in p and "page" in p:
            p[page_key] = p["page"]

    by_doc: Dict[str, List[dict]] = defaultdict(list)
    for p in pages:
        doc_id = p.get(doc_id_key) or p.get("source_path") or p.get("source") or "unknown_doc"
        p[doc_id_key] = doc_id
        by_doc[doc_id].append(p)

    out_chunks: List[dict] = []

    for doc_id in sorted(by_doc.keys()):
        ps = sorted(by_doc[doc_id], key=lambda x: int(x[page_key]))
        for p in ps:
            page_no = int(p[page_key])
            base_text = str(p.get(text_key, "") or "").strip()

            def append_v1_chunks() -> None:
                text = base_text
                if not text:
                    return
                blocks = split_into_blocks(text)
                chunk_texts = pack_blocks_into_chunks(blocks, cfg)
                for i, ct in enumerate(chunk_texts):
                    out_chunks.append(
                        {
                            "chunk_id": f"{doc_id}::p{page_no:04d}::c{i:03d}",
                            "doc_id": doc_id,
                            "page_number": page_no,
                            "start_page": page_no,
                            "end_page": page_no,
                            "text": ct,
                            "char_len": len(ct),
                            "approx_tokens": max(1, len(ct) // 4),
                            "chunker_version": selected_chunker,
                        }
                    )

            if selected_chunker == "v2":
                markdown = str(p.get(markdown_key, "") or "").strip()
                if markdown:
                    blocks = split_markdown_into_structured_blocks(markdown)
                    chunk_blocks = pack_structured_blocks_into_chunks(blocks, cfg)
                    if not chunk_blocks:
                        append_v1_chunks()
                        continue

                    for i, group in enumerate(chunk_blocks):
                        ct = "\n\n".join(b.text for b in group).strip()
                        if not ct:
                            continue
                        row = {
                            "chunk_id": f"{doc_id}::p{page_no:04d}::c{i:03d}",
                            "doc_id": doc_id,
                            "page_number": page_no,
                            "start_page": page_no,
                            "end_page": page_no,
                            "text": ct,
                            "char_len": len(ct),
                            "approx_tokens": max(1, len(ct) // 4),
                            "block_type": _dominant_block_type(group),
                            "chunker_version": selected_chunker,
                        }
                        section_path = _dominant_section_path(group)
                        if section_path:
                            row["section_path"] = section_path
                        out_chunks.append(row)
                    continue

            append_v1_chunks()

    write_jsonl(chunks_out_path, out_chunks)

    if write_manifest:
        update_manifest(
            stage_name="chunk",
            stage_payload={
                "chunker_version": selected_chunker,
                "config": asdict(cfg),
                "num_chunks": len(out_chunks),
            },
            artifact_paths=[Path(chunks_out_path)],
        )
