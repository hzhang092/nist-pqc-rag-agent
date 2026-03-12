"""
Helper functions for graph construction and manipulation."""


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


ALGORITHM_PATTERNS = [
    re.compile(r"\bAlgorithm\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(ML-KEM\.(?:KeyGen|Encaps|Decaps))\b"),
    re.compile(r"\b(ML-DSA\.[A-Za-z0-9_]+)\b"),
    re.compile(r"\b(SLH-DSA\.[A-Za-z0-9_]+)\b"),
]
ALGORITHM_HEADER_PATTERN = re.compile(r"^\s*Algorithm\s+\d+\b[^\n]*", re.IGNORECASE | re.MULTILINE)

DEFINITION_SECTION_HINTS = {
    "terms",
    "notation",
    "definitions",
    "acronyms",
    "symbols",
}
FRONT_MATTER_SECTION_TITLES = {
    "contents",
    "table of contents",
    "list of algorithms",
    "list of figures",
    "list of tables",
}


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_term(term: str) -> str:
    return " ".join(term.strip().lower().split())


def make_document_id(doc_id: str) -> str:
    return f"doc::{doc_id}"


def make_section_id(doc_id: str, full_section_path: str) -> str:
    return f"section::{doc_id}::{full_section_path}"


def make_algorithm_id(doc_id: str, algorithm_key: str) -> str:
    return f"alg::{doc_id}::{algorithm_key}"


def make_term_id(normalized_term: str) -> str:
    return f"term::{normalized_term}"


def make_edge_id(edge_type: str, source_id: str, target_id: str) -> str:
    return f"edge::{edge_type}::{source_id}::{target_id}"


def coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def get_chunk_text(chunk: dict) -> str:
    return str(chunk.get("text", "") or "")


def get_doc_id(chunk: dict) -> str:
    return str(chunk.get("doc_id", "") or "").strip()


def get_page_span(chunk: dict) -> tuple[int | None, int | None]:
    start_page = coerce_int(chunk.get("start_page", chunk.get("page_start")))
    end_page = coerce_int(chunk.get("end_page", chunk.get("page_end")))
    return start_page, end_page


def get_section_path(chunk: dict) -> list[str]:
    raw = chunk.get("section_path")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        # conservative fallback if stored as a single string
        if ">" in text:
            return [part.strip() for part in text.split(">") if part.strip()]
        return [text]
    return []


def section_path_to_string(path_parts: list[str]) -> str:
    return " > ".join(path_parts)


def leaf_section_title(path_parts: list[str]) -> str:
    return path_parts[-1] if path_parts else ""


def is_definition_like_section(path_parts: list[str]) -> bool:
    low_parts = [p.lower() for p in path_parts]
    return any(any(hint in p for hint in DEFINITION_SECTION_HINTS) for p in low_parts)


def is_front_matter_section(path_parts: list[str]) -> bool:
    leaf = normalize_term(leaf_section_title(path_parts))
    return leaf in FRONT_MATTER_SECTION_TITLES


def _algorithm_search_scopes(text: str, *, header_only: bool) -> list[str]:
    if not header_only:
        return [text]
    return [match.group(0) for match in ALGORITHM_HEADER_PATTERN.finditer(text)]


def detect_algorithms(text: str, *, header_only: bool = False) -> list[str]:
    found: set[str] = set()
    for scope in _algorithm_search_scopes(text, header_only=header_only):
        for pattern in ALGORITHM_PATTERNS:
            for match in pattern.findall(scope):
                if isinstance(match, tuple):
                    for item in match:
                        if item:
                            found.add(str(item))
                else:
                    found.add(str(match))
    return sorted(found)


def term_occurs_in_text(term: str, text: str) -> bool:
    return normalize_term(term) in normalize_term(text)
