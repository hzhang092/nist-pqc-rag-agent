"""
Cleans raw text extracted from PDFs, stored in a `pages.jsonl` file.

This script performs several cleaning and normalization steps:
- Unicode normalization (NFKC) and ligature replacement.
- Whitespace normalization (line endings, extra spaces).
- De-hyphenation of words split across lines.
- Removal of standalone page numbers.
- Detection and removal of repeated boilerplate content (headers/footers).
- **Smart, content-aware line joining:** It joins lines that are part of
  prose paragraphs while preserving the line-by-line structure of verbatim
  content like tables, code/pseudocode, and mathematical equations.

The main entry point is `run_clean`, which reads a JSONL file, adds a new
field with the cleaned text (e.g., "text_clean"), and writes a new JSONL file.
"""

# rag/clean.py
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


# ----------------------------
# Config
# ----------------------------

@dataclass
class CleanConfig:
    """Configuration for the text cleaning process."""
    header_footer_lines: int = 3          # look at first/last N lines per page
    boilerplate_ratio: float = 0.6        # remove lines appearing on >= 60% pages (per doc)
    min_line_len: int = 3                 # ignore tiny lines for boilerplate detection
    max_boilerplate_len: int = 160        # ignore huge lines (likely content)
    join_wrapped_lines: bool = True       # join mid-sentence line wraps into paragraphs


# ----------------------------
# Low-level normalizers
# ----------------------------

_ZWSP_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")   # zero-width chars
#_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$", re.I)
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")  # "algo-\nrithm" -> "algorithm"
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_REPEAT_SPACES_RE = re.compile(r"( {2,}|\t{1,})")          # column-like spacing
_TABLE_PIPE_RE = re.compile(r"\|")                         # markdown-like tables
_MATH_RE = re.compile(r"(\$|\\\(|\\\)|\\\[|\\\])")         # latex-ish math markers

def looks_like_table_line(line: str) -> bool:
    """Heuristically checks if a line looks like it's part of a table."""
    s = line.rstrip("\n")
    if not s.strip():
        return False
    # Markdown / LlamaParse table patterns
    if s.strip().startswith("|") or s.count("|") >= 2:
        return True
    # Column alignment in plaintext tables: repeated multi-spaces separating fields
    if len(_REPEAT_SPACES_RE.findall(s)) >= 2:
        return True
    return False

def looks_like_code_or_algo_line(line: str) -> bool:
    """Heuristically checks if a line looks like code, pseudocode, or an algorithm step."""
    s = line.rstrip("\n")
    if not s.strip():
        return False
    # Indentation often indicates code/pseudocode or nested structure
    if s.startswith("    ") or s.startswith("\t"):
        return True
    # Common pseudocode tokens / syntax-ish symbols
    if any(tok in s for tok in ("::=", ":=", "->", "<-", "{", "}", "[", "]")):
        return True
    # Step numbering / algorithm style
    if re.match(r"^\s*(\d+\.|\(\d+\)|Step\s+\d+[:.]|Algorithm\s+\d+[:.])", s, re.I):
        return True
    # Input/Output/Require/Ensure patterns
    if re.match(r"^\s*(Input|Output|Require|Ensure|Given)[:\s]", s, re.I):
        return True
    return False

def looks_like_math_line(line: str) -> bool:
    """Heuristically checks if a line contains mathematical notation."""
    s = line.rstrip("\n")
    if not s.strip():
        return False
    if _MATH_RE.search(s):
        return True
    # Heuristic: lots of mathy symbols
    if sum(ch in s for ch in "=<>±×÷∑∏∈∉≈≡≤≥⊕⊗") >= 1:
        return True
    return False

def is_verbatim_line(line: str) -> bool:
    """Determines if a line should be treated as verbatim (table, code, or math)."""
    return (
        looks_like_table_line(line)
        or looks_like_code_or_algo_line(line)
        or looks_like_math_line(line)
    )

def should_join_as_wrapped_prose(curr: str, nxt: str) -> bool:
    """
    Decides whether to join the current line with the next one.

    This is the core logic for distinguishing between hard line wraps within a
    prose paragraph and meaningful line breaks (e.g., between paragraphs,
    before a list item, or around verbatim content).

    Returns `True` if `curr` and `nxt` should be joined into a single line.
    """
    c = curr.rstrip()
    n = nxt.lstrip()

    if not c or not n:
        return False

    # Don't join verbatim-ish lines
    if is_verbatim_line(c) or is_verbatim_line(n):
        return False

    # Don't join headings / labels (often end with colon)
    if c.endswith(":"):
        return False

    # If curr ends with sentence punctuation, keep newline
    if c.endswith((".", "!", "?", ";")):
        return False

    # If next line starts like a new section/bullet, keep newline
    if re.match(r"^(\-|\*|•|\d+\.|\(\d+\))\s+", n):
        return False

    # Typical wrapped-prose signature: next line starts lowercase or continuation punctuation
    if re.match(r"^[a-z(]", n):
        return True

    # Also join if curr is long-ish (likely wrapped) and next starts with a word
    if len(c) >= 60 and re.match(r"^[A-Za-z]", n):
        return True

    return False

def smart_join_lines(lines: List[str]) -> List[str]:
    """
    Intelligently joins wrapped prose lines while preserving verbatim blocks.

    It iterates through lines, deciding whether to join a line with the next
    based on `should_join_as_wrapped_prose`. It keeps verbatim lines (tables,
    code, math) and paragraph breaks intact.
    """
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip() == "":
            out.append("")
            i += 1
            continue

        # Preserve verbatim blocks line-by-line
        if is_verbatim_line(line):
            out.append(line)
            i += 1
            continue

        # Build a prose paragraph by joining only where it looks like PDF wrapping
        buf = line
        j = i
        while j + 1 < len(lines):
            nxt = lines[j + 1].rstrip()
            if nxt.strip() == "":
                break
            if should_join_as_wrapped_prose(buf, nxt):
                buf = buf + " " + nxt.lstrip()
                j += 1
            else:
                break

        out.append(_MULTI_SPACE_RE.sub(" ", buf).strip())
        i = j + 1

    # Collapse multiple blank lines to a single blank line
    cleaned: List[str] = []
    empty_run = 0
    for ln in out:
        if ln.strip() == "":
            empty_run += 1
            if empty_run <= 1:
                cleaned.append("")
        else:
            empty_run = 0
            cleaned.append(ln)
    return cleaned


def normalize_unicode(s: str) -> str:
    """
    Performs unicode normalization and replaces common ligatures.
    - Applies NFKC normalization.
    - Removes soft hyphens and zero-width spaces.
    - Replaces 'fi' and 'fl' ligatures.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")           # soft hyphen
    s = _ZWSP_RE.sub("", s)               # zero-width spaces
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return s


def normalize_whitespace(s: str) -> str:
    """
    Normalizes newlines and collapses multiple spaces/tabs.
    - Converts all newlines to `\n`.
    - Strips trailing whitespace from each line.
    - Collapses multiple spaces/tabs into a single space.
    - Strips leading/trailing whitespace from the whole string.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = "\n".join(_MULTI_SPACE_RE.sub(" ", line) for line in s.split("\n"))
    return s.strip()


def dehyphenate(s: str) -> str:
    """Joins words that were hyphenated across a line break."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", s)


def remove_standalone_page_numbers(lines: List[str]) -> List[str]:
    """Removes lines that appear to be just page numbers (e.g., "Page 12", "12")."""
    return [ln for ln in lines if not _PAGE_NUM_RE.match(ln)]


def join_wrapped_paragraph_lines(lines: List[str]) -> List[str]:
    """
    Join lines within paragraphs; keep blank lines as paragraph boundaries.
    """
    paras: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                paras.append(cur)
                cur = []
        else:
            cur.append(ln.strip())
    if cur:
        paras.append(cur)

    joined: List[str] = []
    for p in paras:
        joined.append(" ".join(p))
        joined.append("")  # paragraph break
    if joined and joined[-1] == "":
        joined.pop()
    return joined


# ----------------------------
# Boilerplate detection
# ----------------------------

def canon_line(s: str) -> str:
    """
    Canonicalize for boilerplate detection:
    - lowercase
    - remove digits (page numbers vary)
    - collapse whitespace
    """
    s = s.lower()
    s = re.sub(r"\d+", "", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s.strip()


def detect_boilerplate(
    pages_by_doc: Dict[str, List[dict]],
    cfg: CleanConfig
) -> Dict[str, set]:
    """
    Find repeated header/footer lines per doc, using frequency across pages.

    It works by:
    1. Iterating through each document.
    2. For each page, collecting candidate lines from the top and bottom.
    3. Canonicalizing these lines (lowercase, no digits) to handle variations.
    4. Counting the frequency of each canonical line across all pages in the doc.
    5. Flagging lines that appear on a high percentage of pages as boilerplate.

    Returns:
        A dictionary mapping each `doc_id` to a set of canonical boilerplate
        lines that should be removed from that document.
    """
    boiler_by_doc: Dict[str, set] = {}

    for doc_id, pages in pages_by_doc.items():
        counts = Counter()
        total_pages = 0

        for p in pages:
            text = p.get("text", "") or ""
            # Keep blank lines for later; boilerplate detection uses non-empty
            nonempty_lines = [ln for ln in text.split("\n") if ln.strip() != ""]
            if not nonempty_lines:
                continue

            total_pages += 1
            candidates = (
                nonempty_lines[:cfg.header_footer_lines]
                + nonempty_lines[-cfg.header_footer_lines:]
            )

            # De-duplicate per page so one page doesn't overweight counts
            seen = set()
            for ln in candidates:
                if not (cfg.min_line_len <= len(ln) <= cfg.max_boilerplate_len):
                    continue
                cl = canon_line(ln)
                if cl:
                    seen.add(cl)
            for cl in seen:
                counts[cl] += 1

        if total_pages == 0:
            boiler_by_doc[doc_id] = set()
            continue

        threshold = max(2, int(cfg.boilerplate_ratio * total_pages))
        boiler_by_doc[doc_id] = {cl for cl, c in counts.items() if c >= threshold}

    return boiler_by_doc


# ----------------------------
# Cleaning pipeline
# ----------------------------

def clean_page_text(raw: str, boiler_canon: set, cfg: CleanConfig) -> str:
    """
    Applies the full, structure-aware cleaning pipeline to a single page's text.

    The process is as follows:
    1. Basic normalization (Unicode, whitespace, de-hyphenation).
    2. Remove lines that are just page numbers.
    3. Remove detected boilerplate (headers/footers).
    4. Use `smart_join_lines` to selectively join wrapped prose lines while
       preserving the structure of tables, code, and math.
    5. Collapse any remaining runs of multiple blank lines.

    Args:
        raw: The raw text content of the page.
        boiler_canon: A set of canonical boilerplate lines for this document.
        cfg: The cleaning configuration.

    Returns:
        The cleaned and normalized text as a single string.
    """
    # 1) Unicode + whitespace normalization
    s = normalize_unicode(raw)
    s = normalize_whitespace(s)

    # 2) Fix hyphenation across line breaks: "algo-\nrithm" -> "algorithm"
    s = dehyphenate(s)

    # 3) Split into lines and remove stand-alone page number lines
    lines = s.split("\n")
    lines = remove_standalone_page_numbers(lines)

    # 4) Remove header/footer boilerplate (canonical match)
    kept = []
    for ln in lines:
        if canon_line(ln) in boiler_canon:
            continue
        kept.append(ln.rstrip())

    # 5) Preserve technical structure, only join where it looks like wrapped prose
    if cfg.join_wrapped_lines:
        kept = smart_join_lines(kept)
    else:
        # Still collapse multi-blank runs even if not joining
        squashed = []
        empty_run = 0
        for ln in kept:
            if ln.strip() == "":
                empty_run += 1
                if empty_run <= 1:
                    squashed.append("")
            else:
                empty_run = 0
                squashed.append(ln)
        kept = squashed

    # 6) Final output
    return "\n".join(kept).strip()


def load_pages_jsonl(path: Path) -> List[dict]:
    """Loads a JSONL file where each line is a JSON object representing a page."""
    pages: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def write_pages_jsonl(path: Path, pages: Iterable[dict]) -> None:
    """Writes a list of page objects to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def run_clean(
    pages_path: str | Path,
    out_path: str | Path,
    cfg: CleanConfig = CleanConfig(),
    text_key: str = "text",
    out_text_key: str = "text_clean",
    doc_id_key: str = "doc_id",
) -> None:
    """
    Reads pages from a JSONL file, cleans their text, and writes to a new file.

    This is the main entry point for the cleaning script. It orchestrates the
    process of loading data, detecting boilerplate on a per-document basis,
    cleaning each page's text, and saving the result.

    Args:
        pages_path: Path to the input `pages.jsonl` file.
        out_path: Path to write the output `pages_clean.jsonl` file.
        cfg: Cleaning configuration object.
        text_key: The key in the input JSON objects containing the text to clean.
        out_text_key: The key to use for the cleaned text in the output objects.
        doc_id_key: The key for the document identifier, used to group pages.
    """
    pages_path = Path(pages_path)
    out_path = Path(out_path)

    pages = load_pages_jsonl(pages_path)

    # Group by doc_id (fallback to source_path if doc_id missing)
    pages_by_doc: Dict[str, List[dict]] = defaultdict(list)
    for p in pages:
        doc_id = p.get(doc_id_key) or p.get("source_path") or p.get("source") or "unknown_doc"
        pages_by_doc[doc_id].append(p)

    boiler_by_doc = detect_boilerplate(pages_by_doc, cfg)

    cleaned_pages = []
    for p in pages:
        doc_id = p.get(doc_id_key) or p.get("source_path") or p.get("source") or "unknown_doc"
        raw = p.get(text_key) or ""
        p2 = dict(p)
        p2[out_text_key] = clean_page_text(raw, boiler_by_doc.get(doc_id, set()), cfg)
        cleaned_pages.append(p2)

    write_pages_jsonl(out_path, cleaned_pages)
