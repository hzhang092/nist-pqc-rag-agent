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
    header_footer_lines: int = 3          # look at first/last N lines per page
    boilerplate_ratio: float = 0.6        # remove lines appearing on >= 60% pages (per doc)
    min_line_len: int = 3                 # ignore tiny lines for boilerplate detection
    max_boilerplate_len: int = 160        # ignore huge lines (likely content)
    join_wrapped_lines: bool = True       # join mid-sentence line wraps into paragraphs


# ----------------------------
# Low-level normalizers
# ----------------------------

_ZWSP_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")   # zero-width chars
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$", re.I)
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")  # "algo-\nrithm" -> "algorithm"


def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")           # soft hyphen
    s = _ZWSP_RE.sub("", s)               # zero-width spaces
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return s


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = "\n".join(_MULTI_SPACE_RE.sub(" ", line) for line in s.split("\n"))
    return s.strip()


def dehyphenate(s: str) -> str:
    return _HYPHEN_BREAK_RE.sub(r"\1\2", s)


def remove_standalone_page_numbers(lines: List[str]) -> List[str]:
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
    Returns: {doc_id: set(canonical_lines_to_remove)}
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
    s = normalize_unicode(raw)
    s = normalize_whitespace(s)
    s = dehyphenate(s)

    lines = s.split("\n")
    lines = remove_standalone_page_numbers(lines)

    # Remove boilerplate by canonical match
    kept = []
    for ln in lines:
        if canon_line(ln) in boiler_canon:
            continue
        kept.append(ln.rstrip())

    # Collapse multiple blank lines to a single blank line
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

    if cfg.join_wrapped_lines:
        squashed = join_wrapped_paragraph_lines(squashed)

    return "\n".join(squashed).strip()


def load_pages_jsonl(path: Path) -> List[dict]:
    pages: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def write_pages_jsonl(path: Path, pages: Iterable[dict]) -> None:
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
    Reads pages.jsonl, writes pages_clean.jsonl with an added `out_text_key`.
    You can adapt key names via parameters.
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
