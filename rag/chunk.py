
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
    # v2 fallback for oversized blocks (~250-400 token windows with overlap).
    v2_window_tokens: int = 320
    v2_overlap_tokens: int = 64
    v2_max_block_tokens: int = 400


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
_ALGO_START_RE = re.compile(r"^\s*Algorithm\b", re.I) # 
_ALGO_LINE_RE = re.compile(
    r"^\s*(Algorithm\b|Input\b|Output\b|Require\b|Ensure\b|Step\s+\d+|[0-9]+\.)",
    re.I,
)
_ALGO_TERMINAL_EQ_RE = re.compile(r"^\s*[0-9]+\s*:\s*")
_FENCED_CODE_RE = re.compile(r"^\s*```")
_MATH_BLOCK_DELIM_RE = re.compile(r"^\s*(\$\$|\\\[|\\\])\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_NUMERIC_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\b")
_SENTENCE_LIKE_HEADING_RE = re.compile(
    r"\b(is|are|was|were|be|being|been|should|may|can|could|would|will|must)\b",
    re.I,
)
_SHORT_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")
_TABLEISH_ROW_RE = re.compile(r"^\s*[^|]{0,80}(?:\|[^|]{0,80}){2,}\s*$")


def looks_like_table_line(line: str) -> bool:
    s = line.rstrip()
    if not s.strip():
        return False
    stripped = s.strip()
    if "||" in stripped and not stripped.startswith("|"):
        return False
    if "$$" in stripped or "\\dots" in stripped:
        return False
    if _TABLE_SEPARATOR_RE.match(stripped):
        return True
    if stripped.startswith("|") and stripped.count("|") >= 3:
        return True
    if stripped.count("|") >= 3 and _TABLEISH_ROW_RE.match(stripped):
        return True
    if len(_REPEAT_SPACES_RE.findall(s)) >= 3 and len(_SHORT_WORD_RE.findall(s)) <= 24:
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
    symbol_count = sum(ch in s for ch in "=<>±×÷∑∏∈∉≈≡≤≥⊕⊗")
    if symbol_count >= 2:
        return True
    if symbol_count == 1:
        # Long prose often includes one equality/comparison symbol in examples.
        # Keep those as text unless there is additional math-like evidence.
        word_count = len(_SHORT_WORD_RE.findall(s))
        if word_count >= 12:
            return False
        if re.search(r"\b(mod|log|gcd|lcm)\b", s, re.I):
            return True
        if re.search(r"\b[A-Za-z]\s*[=<>]\s*[A-Za-z0-9]\b", s):
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
            if cur and (cur_len + 2 + blk_len) > cfg.max_chars:
                cur = []
                cur_len = 0
            if len(blk) > cfg.max_chars and not cur:
                chunks.append(blk.strip())
            else:
                cur.append(blk)
                cur_len = len(blk) if len(cur) == 1 else cur_len + 2 + len(blk)

    flush()
    return chunks


def _heading_from_line(line: str) -> tuple[int, str] | None:
    def _normalize_title(title: str) -> str:
        t = re.sub(r"\s+", " ", str(title or "").strip())
        # Docling occasionally injects broken numeric prefixes in algorithm headings,
        # e.g., "## -1 Algorithm 10 ..."; keep the meaningful algorithm title only.
        m_artifact = re.match(r"^-?\d+\s+(Algorithm\s+\d+\b.*)$", t, re.I)
        if m_artifact:
            t = m_artifact.group(1).strip()
        return t

    def _numeric_depth(text: str) -> int | None:
        m_depth = _NUMERIC_PREFIX_RE.match(str(text or ""))
        if not m_depth:
            return None
        return min(6, m_depth.group(1).count(".") + 1)

    def _looks_like_numeric_heading(numeric: str, title: str) -> bool:
        t = _normalize_title(title)
        if not t:
            return False
        words = t.split()
        if len(words) > 12:
            return False
        if t[-1] in ".:;!?":
            return False
        if _SENTENCE_LIKE_HEADING_RE.search(t):
            return False
        # Top-level headings in standards are short labels, not full sentences.
        if "." not in numeric and len(words) > 8:
            return False
        return True

    m = _MD_HEADING_RE.match(line)
    if m:
        level = len(m.group(1))
        title = _normalize_title(m.group(2).strip().strip("#").strip())
        if title:
            numeric_depth = _numeric_depth(title)
            if numeric_depth is not None:
                level = numeric_depth
            return level, title

    m = _NUMERIC_HEADING_RE.match(line)
    if m:
        numeric = str(m.group(1))
        title = _normalize_title(m.group(2))
        if title and _looks_like_numeric_heading(numeric, title):
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
    if "||" in s and not s.startswith("|"):
        return False
    if "$$" in s or "\\dots" in s:
        return False
    if _TABLE_SEPARATOR_RE.match(s):
        return True
    if s.startswith("|") and s.count("|") >= 3:
        return True
    # Support no-leading-pipe markdown rows while rejecting prose like "|x|".
    if s.count("|") >= 3 and _TABLEISH_ROW_RE.match(s):
        parts = [part.strip() for part in s.split("|") if part.strip()]
        if len(parts) >= 3 and max(len(part) for part in parts) <= 80:
            # Sentence punctuation strongly suggests prose, not a table row.
            if not re.search(r"[.!?;:]\s", s):
                return True
        return False
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


def _is_code_block_start(line: str) -> bool:
    return bool(_FENCED_CODE_RE.match(line))


def _is_math_block_delimiter(line: str) -> bool:
    return bool(_MATH_BLOCK_DELIM_RE.match(line))


def _is_algorithm_terminal_equation_line(line: str) -> bool:
    return bool(_ALGO_TERMINAL_EQ_RE.match(line))


def _looks_like_fenced_algorithm_block(block_lines: List[str]) -> bool:
    """Return True if a fenced block appears to contain algorithm pseudocode."""
    if not block_lines:
        return False

    content = [ln.rstrip() for ln in block_lines if ln.strip() and not _is_code_block_start(ln)]
    if not content:
        return False

    first = content[0]
    if _is_algorithm_start_line(first) or _is_algorithm_terminal_equation_line(first):
        return True

    signals = 0
    for ln in content[:20]:
        if _is_algorithm_start_line(ln) or _is_algorithm_terminal_equation_line(ln) or _is_algorithm_continuation_line(ln):
            signals += 1
        if signals >= 2:
            return True
    return False


def _approx_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def _clean_inline_markdown(text: str) -> str:
    s = _MD_LINK_RE.sub(r"\1", text)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    return s.strip()


def _is_algorithm_heading_title(title: str, following_lines: List[str] | None = None) -> bool:
    """Return True if a markdown heading title likely denotes an algorithm block.

    We primarily rely on NIST-style headings like "Algorithm 19 ML-KEM.KeyGen".
    Some parsed sources may produce a bare "Algorithm" heading; in that case we
    require a short lookahead for algorithm-ish lines to avoid false positives.
    """
    t = _clean_inline_markdown(str(title or "")).strip()
    if not t:
        return False

    # Most common: "Algorithm <number> ...".
    if re.match(r"^Algorithm\s+\d+\b", t, flags=re.IGNORECASE):
        return True

    # If heading begins with Algorithm and contains any digit, accept.
    if t.lower().startswith("algorithm") and re.search(r"\d", t):
        return True

    # Bare "Algorithm" headings exist in some conversions; require lookahead.
    if t.strip().lower() in {"algorithm", "algorithms"} and following_lines:
        nonempty = 0
        for raw in following_lines:
            s = (raw or "").rstrip()
            if not s.strip():
                continue
            nonempty += 1
            if _is_algorithm_start_line(s) or _is_algorithm_terminal_equation_line(s) or _is_algorithm_continuation_line(s):
                return True
            if nonempty >= 8:
                break
    return False


def _clean_block_text(block_type: str, text: str) -> str:
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []

    if block_type == "code":
        for line in lines:
            if _is_code_block_start(line):
                continue
            out.append(line.rstrip())
        return "\n".join(x for x in out if x.strip()).strip()

    if block_type == "table":
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if _TABLE_SEPARATOR_RE.match(s):
                continue
            if "|" in s:
                cells = [cell.strip() for cell in s.strip("|").split("|")]
                out.append(" | ".join(cell for cell in cells if cell))
            else:
                out.append(_clean_inline_markdown(s))
        return "\n".join(out).strip()

    if block_type == "list":
        for line in lines:
            s = _LIST_LINE_RE.sub("", line.strip())
            if s:
                out.append(_clean_inline_markdown(s))
        return "\n".join(out).strip()

    if block_type in {"algorithm", "math"}:
        for line in lines:
            if _is_math_block_delimiter(line):
                continue
            if block_type == "algorithm" and _is_code_block_start(line):
                continue
            s = _clean_inline_markdown(line.rstrip())
            if s:
                out.append(s)
        return "\n".join(out).strip()

    # Default text cleanup: keep prose compact.
    prose = " ".join(_clean_inline_markdown(line) for line in lines if line.strip())
    return re.sub(r"\s{2,}", " ", prose).strip()


def _split_tokens_with_overlap(text: str, window_tokens: int, overlap_tokens: int) -> List[str]:
    tokens = re.findall(r"\S+", text or "")
    if not tokens:
        return []
    if len(tokens) <= window_tokens:
        return [" ".join(tokens).strip()]

    step = max(1, window_tokens - max(0, overlap_tokens))
    out: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + window_tokens)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            out.append(chunk)
        if end >= len(tokens):
            break
        start += step
    return out


def _split_chars_with_overlap(text: str, window_chars: int, overlap_chars: int) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    if len(s) <= window_chars:
        return [s]

    out: list[str] = []
    step = max(1, window_chars - max(0, overlap_chars))
    start = 0
    while start < len(s):
        end = min(len(s), start + window_chars)
        piece = s[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(s):
            break
        start += step
    return out


def _split_lines_with_overlap(lines: List[str], window_tokens: int, overlap_tokens: int) -> List[str]:
    filtered = [ln.rstrip() for ln in lines if ln.strip()]
    expanded: list[str] = []
    for line in filtered:
        if _approx_token_count(line) > window_tokens:
            expanded.extend(_split_tokens_with_overlap(line, window_tokens, overlap_tokens))
        else:
            expanded.append(line)
    filtered = expanded
    if not filtered:
        return []

    out: list[str] = []
    start = 0
    n = len(filtered)
    while start < n:
        token_budget = 0
        end = start
        while end < n:
            line_tokens = _approx_token_count(filtered[end])
            if end == start:
                token_budget += line_tokens
                end += 1
                continue
            if token_budget + line_tokens > window_tokens:
                break
            token_budget += line_tokens
            end += 1

        chunk = "\n".join(filtered[start:end]).strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break

        if overlap_tokens <= 0:
            start = end
            continue

        back_tokens = 0
        next_start = end
        while next_start > start and back_tokens < overlap_tokens:
            next_start -= 1
            back_tokens += _approx_token_count(filtered[next_start])
        # Ensure progress even with tiny windows.
        start = next_start if next_start > start else end

    return out


def _split_block_for_window_fallback(block: StructuredBlock, cfg: ChunkConfig) -> List[StructuredBlock]:
    if not block.text.strip():
        return []

    block_tokens = _approx_token_count(block.text)
    if len(block.text) <= cfg.max_chars and block_tokens <= cfg.v2_max_block_tokens:
        return [block]

    if block.block_type in {"table", "algorithm", "code", "math", "list"}:
        parts = _split_lines_with_overlap(
            lines=block.text.split("\n"),
            window_tokens=cfg.v2_window_tokens,
            overlap_tokens=cfg.v2_overlap_tokens,
        )
    else:
        parts = _split_tokens_with_overlap(
            text=block.text,
            window_tokens=cfg.v2_window_tokens,
            overlap_tokens=cfg.v2_overlap_tokens,
        )

    if not parts:
        return [block]

    normalized_parts: list[str] = []
    for part in parts:
        if len(part) > cfg.max_chars:
            normalized_parts.extend(
                _split_chars_with_overlap(
                    text=part,
                    window_chars=cfg.max_chars,
                    overlap_chars=max(1, cfg.max_chars // 5),
                )
            )
        else:
            normalized_parts.append(part)
    if not normalized_parts:
        return [block]

    return [
        StructuredBlock(text=part, block_type=block.block_type, section_path=block.section_path)
        for part in normalized_parts
    ]


def split_markdown_into_structured_blocks(
    markdown: str,
    heading_stack: list[tuple[int, str]] | None = None,
) -> List[StructuredBlock]:
    lines = (markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(lines)
    i = 0
    stack = heading_stack if heading_stack is not None else []
    blocks: list[StructuredBlock] = []

    in_algorithm_section = False

    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        md_heading_match = _MD_HEADING_RE.match(line)
        heading = _heading_from_line(line)
        if heading is not None:
            level, title = heading
            lookahead = lines[i + 1 : min(n, i + 40)]
            heading_is_algorithm = _is_algorithm_heading_title(title, following_lines=lookahead)

            # Section headings should not stay nested under a previously seen
            # algorithm heading from an earlier block/page.
            if not heading_is_algorithm:
                while stack and _is_algorithm_heading_title(stack[-1][1]):
                    stack.pop()

            # Docling often emits flat markdown heading levels (for example all "##").
            # For algorithm headings, keep them under the surrounding non-algorithm
            # section to avoid algorithm-under-algorithm path chains.
            if md_heading_match and heading_is_algorithm and stack:
                non_algorithm_parent_level = None
                for stack_level, stack_title in reversed(stack):
                    if not _is_algorithm_heading_title(stack_title):
                        non_algorithm_parent_level = stack_level
                        break
                if non_algorithm_parent_level is not None:
                    desired_level = min(6, non_algorithm_parent_level + 1)
                    if level <= desired_level:
                        level = desired_level

            _update_heading_stack(stack, level, title)
            in_algorithm_section = heading_is_algorithm

            cleaned_title = _clean_block_text("text", title)
            if cleaned_title:
                blocks.append(
                    StructuredBlock(
                        text=cleaned_title,
                        block_type="text",
                        section_path=_section_path(stack),
                    )
                )
            i += 1
            continue

        if _is_code_block_start(line):
            fence_start = line.strip()[:3]
            j = i + 1
            block_lines = [line]
            while j < n:
                nxt = lines[j].rstrip()
                block_lines.append(nxt)
                if nxt.strip().startswith(fence_start):
                    j += 1
                    break
                j += 1

            block_type = "algorithm" if _looks_like_fenced_algorithm_block(block_lines) else "code"
            text = _clean_block_text(block_type, "\n".join(block_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type=block_type,
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        if _is_math_block_delimiter(line):
            delim = line.strip()
            closing_delim = "$$" if delim == "$$" else "\\]"
            j = i + 1
            block_lines = [line]
            while j < n:
                nxt = lines[j].rstrip()
                block_lines.append(nxt)
                if nxt.strip() == closing_delim:
                    j += 1
                    break
                j += 1

            text = _clean_block_text("math", "\n".join(block_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="math",
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        # If we are inside an Algorithm heading section, capture contiguous content as
        # an algorithm block (stops at next heading). This handles Docling-style output
        # where the algorithm starts with prose and step markers like "1:".
        if in_algorithm_section and _heading_from_line(line) is None:
            j = i + 1
            block_lines = [line]
            while j < n:
                nxt = lines[j].rstrip()
                if _heading_from_line(nxt) is not None:
                    break
                block_lines.append(nxt)
                j += 1

            text = _clean_block_text("algorithm", "\n".join(block_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="algorithm",
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        if _is_algorithm_start_line(line):
            j = i + 1
            block_lines = [line]
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

                if _is_algorithm_terminal_equation_line(nxt):
                    k = j
                    while k < n and not lines[k].strip():
                        k += 1
                    if k >= n:
                        break
                    probe = lines[k].rstrip()
                    if _heading_from_line(probe) is not None or not _is_algorithm_continuation_line(probe):
                        break

            text = _clean_block_text("algorithm", "\n".join(block_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="algorithm",
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        if _is_markdown_table_line(line):
            j = i + 1
            block_lines = [line]
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
            text = _clean_block_text("table", "\n".join(block_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="table",
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        if _is_list_line(line):
            j = i + 1
            block_lines = [line]
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
            text = _clean_block_text("list", "\n".join(block_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="list",
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        if line.startswith("    ") or line.startswith("\t"):
            j = i + 1
            block_lines = [line]
            while j < n:
                nxt = lines[j].rstrip()
                if not nxt.strip():
                    break
                if nxt.startswith("    ") or nxt.startswith("\t"):
                    block_lines.append(nxt)
                    j += 1
                    continue
                break
            text = _clean_block_text("code", "\n".join(block_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="code",
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        if looks_like_math_line(line):
            j = i + 1
            math_lines = [line]
            while j < n:
                nxt = lines[j].rstrip()
                if not nxt.strip():
                    break
                if _heading_from_line(nxt) is not None or _is_markdown_table_line(nxt) or _is_list_line(nxt):
                    break
                if not looks_like_math_line(nxt):
                    break
                math_lines.append(nxt)
                j += 1
            text = _clean_block_text("math", "\n".join(math_lines))
            if text:
                blocks.append(
                    StructuredBlock(
                        text=text,
                        block_type="math",
                        section_path=_section_path(stack),
                    )
                )
            i = j
            continue

        j = i + 1
        para_parts = [line.strip()]
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
            if _is_code_block_start(nxt) or _is_math_block_delimiter(nxt):
                break
            para_parts.append(nxt.strip())
            j += 1

        text = _clean_block_text("text", " ".join(part for part in para_parts if part).strip())
        if text:
            blocks.append(
                StructuredBlock(
                    text=text,
                    block_type="text",
                    section_path=_section_path(stack),
                )
            )
        i = j

    return [b for b in blocks if b.text.strip()]


def pack_structured_blocks_into_chunks(blocks: List[StructuredBlock], cfg: ChunkConfig) -> List[List[StructuredBlock]]:
    expanded_blocks: list[StructuredBlock] = []
    for blk in blocks:
        expanded_blocks.extend(_split_block_for_window_fallback(blk, cfg))

    chunks: List[List[StructuredBlock]] = []
    cur: List[StructuredBlock] = []
    cur_len = 0

    def cur_text_len(items: List[StructuredBlock]) -> int:
        return sum(len(x.text) for x in items) + (2 * max(0, len(items) - 1))

    def is_hard_boundary(prev_blk: StructuredBlock, next_blk: StructuredBlock) -> bool:
        if prev_blk.section_path and next_blk.section_path and prev_blk.section_path != next_blk.section_path:
            return True
        strong_types = {"algorithm", "table", "code"}
        if (prev_blk.block_type in strong_types or next_blk.block_type in strong_types) and prev_blk.block_type != next_blk.block_type:
            return True
        return False

    def flush(carry_overlap: bool = True) -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        text = "\n\n".join(x.text for x in cur).strip()
        if len(text) >= cfg.min_chars:
            chunks.append(cur[:])
        carry = cur[-cfg.overlap_blocks:] if carry_overlap and cfg.overlap_blocks > 0 else []
        cur = carry[:]
        cur_len = cur_text_len(cur)

    for blk in expanded_blocks:
        if cur and cur_len >= cfg.min_chars and is_hard_boundary(cur[-1], blk):
            # Prevent overlap bleed across section/strong-structure boundaries.
            flush(carry_overlap=False)

        blk_len = len(blk.text)
        proposed = cur_len + (2 if cur else 0) + blk_len
        if proposed <= cfg.target_chars or (proposed <= cfg.max_chars and cur_len < cfg.min_chars):
            cur.append(blk)
            cur_len = proposed
        else:
            flush()
            if cur and (cur_text_len(cur) + 2 + blk_len) > cfg.max_chars:
                cur = []
                cur_len = 0
            cur.append(blk)
            cur_len = cur_text_len(cur)

    flush()
    return chunks


def _dominant_block_type(blocks: List[StructuredBlock]) -> str:
    if not blocks:
        return "text"

    # Preserve strong structure labels when mixed with generic text.
    if any(b.block_type == "algorithm" for b in blocks):
        return "algorithm"
    if any(b.block_type == "table" for b in blocks):
        return "table"
    counts = Counter(b.block_type for b in blocks)
    allowed = ("algorithm", "table", "code", "math", "list", "text")
    ranked = sorted(
        allowed,
        key=lambda name: (counts.get(name, 0), -allowed.index(name)),
        reverse=True,
    )
    for name in ranked:
        if counts.get(name, 0) > 0:
            return name
    return "text"


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
        doc_heading_stack: list[tuple[int, str]] = []
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
                    row = {
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
                    if selected_chunker == "v2":
                        row["block_type"] = "text"
                        path = _section_path(doc_heading_stack)
                        if path:
                            row["section_path"] = path
                    out_chunks.append(row)

            if selected_chunker == "v2":
                markdown = str(p.get(markdown_key, "") or "").strip()
                if markdown:
                    blocks = split_markdown_into_structured_blocks(markdown, heading_stack=doc_heading_stack)
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
                "config": asdict(cfg, dict_factory=dict),
                "num_chunks": len(out_chunks),
            },
            artifact_paths=[Path(chunks_out_path)],
        )
