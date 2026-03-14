"""Helper functions for graph construction and lookup."""

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
ALGORITHM_HEADER_PATTERN = re.compile(
    r"^\s*(?P<header>Algorithm\s+(?P<number>\d+)\b(?P<rest>[^\n]*))",
    re.IGNORECASE | re.MULTILINE,
)
IDENTIFIER_PATTERNS = (
    re.compile(r"\b(?P<term>(?:ML-KEM|ML-DSA|SLH-DSA)(?:\.[A-Za-z0-9_]+)?)\b"),
    re.compile(r"\b(?P<term>SHAKE(?:128|256)|KEM|PKE|XOF|PRF|PRG|NTT|CBD|LWE|MLWE|SIS|MSIS|PQC)\b"),
)
SECTION_NUMBER_RE = re.compile(r"^\s*(?P<number>[A-Z]?\d+(?:\.\d+)*)\b[.)]?\s*")
DEFINITION_PIPE_RE = re.compile(
    r"(?m)^\s*(?P<term>[A-Za-z0-9][A-Za-z0-9 ._()/,-]{1,80}?)\s*\|"
)

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
GENERIC_TERM_HINTS = {
    "terms",
    "definitions",
    "notation",
    "symbols",
    "requirements",
    "external functions",
    "interoperability",
    "introduction",
    "overview",
}
TERM_PHRASE_HINTS = (
    "key generation",
    "encapsulation",
    "decapsulation",
    "public key",
    "secret key",
    "ciphertext",
    "shared secret",
    "parameter set",
    "signature generation",
    "signature verification",
)
OPERATION_TERMS = {phrase for phrase in TERM_PHRASE_HINTS}


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
    return " ".join(str(term or "").strip().lower().split())


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


def get_block_type(chunk: dict) -> str:
    return str(chunk.get("block_type", "") or "").strip()


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
        if ">" in text:
            return [part.strip() for part in text.split(">") if part.strip()]
        return [text]
    return []


def section_path_to_string(path_parts: list[str]) -> str:
    return " > ".join(path_parts)


def leaf_section_title(path_parts: list[str]) -> str:
    return path_parts[-1] if path_parts else ""


def strip_section_number(title: str) -> str:
    text = str(title or "").strip()
    return SECTION_NUMBER_RE.sub("", text).strip()


def is_definition_like_section(path_parts: list[str]) -> bool:
    low_parts = [p.lower() for p in path_parts]
    return any(any(hint in p for hint in DEFINITION_SECTION_HINTS) for p in low_parts)


def is_front_matter_section(path_parts: list[str]) -> bool:
    leaf = normalize_term(leaf_section_title(path_parts))
    return leaf in FRONT_MATTER_SECTION_TITLES


def _algorithm_search_scopes(text: str, *, header_only: bool) -> list[str]:
    if not header_only:
        return [text]
    return [match.group("header") for match in ALGORITHM_HEADER_PATTERN.finditer(text)]


def _looks_like_algorithm_name(token: str) -> bool:
    if not token:
        return False
    if any(mark in token for mark in (".", "-", "_")):
        return True
    if token.isupper() and any(ch.isalpha() for ch in token):
        return True
    tail = token[1:] if len(token) > 1 else ""
    return any(ch.isupper() for ch in tail)


def extract_algorithm_header_info(text: str) -> list[dict[str, str | None]]:
    infos: list[dict[str, str | None]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in ALGORITHM_HEADER_PATTERN.finditer(text or ""):
        algorithm_number = str(match.group("number") or "").strip()
        rest = " ".join(str(match.group("rest") or "").split()).strip(" :;-")
        raw_header = " ".join(str(match.group("header") or "").split())
        algorithm_name = None
        if rest:
            token = rest.split()[0].strip("()[]{}:;,.")
            if _looks_like_algorithm_name(token):
                algorithm_name = token
        dedupe_key = (algorithm_number, str(algorithm_name or ""), raw_header)
        if not algorithm_number or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        infos.append(
            {
                "algorithm_number": algorithm_number,
                "algorithm_label": f"Algorithm {algorithm_number}",
                "algorithm_name": algorithm_name,
                "raw_header": raw_header,
            }
        )
    return infos


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


def classify_term_type(term: str) -> str:
    text = str(term or "").strip()
    normalized = normalize_term(text)
    if not normalized:
        return "concept"
    if normalized in OPERATION_TERMS or any(
        normalized.endswith(suffix)
        for suffix in ("generation", "encapsulation", "decapsulation", "verification")
    ):
        return "operation"
    if re.fullmatch(r"[A-Z][A-Z0-9-]{1,10}", text):
        return "acronym"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z0-9_]+)+", text):
        return "identifier"
    if "-" in text and any(ch.isupper() for ch in text):
        return "identifier"
    if any(ch.isupper() for ch in text[1:]):
        return "identifier"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,3}", text):
        return "symbol"
    return "concept"


def _clean_term_candidate(term: str) -> str | None:
    text = strip_section_number(str(term or "").replace("\n", " ").strip())
    text = re.sub(r"\s+", " ", text).strip(" .,:;()[]{}")
    normalized = normalize_term(text)
    if not text or not normalized:
        return None
    if len(text) > 80:
        return None
    if len(normalized.split()) > 6:
        return None
    if not any(ch.isalpha() for ch in text):
        return None
    if normalized.startswith(("algorithm ", "table ", "figure ", "section ")):
        return None
    if normalized in GENERIC_TERM_HINTS:
        return None
    if classify_term_type(text) == "concept" and len(normalized.split()) == 1 and normalized.isalpha():
        return None
    return text


def _append_term_candidate(
    out: list[dict[str, str]],
    seen: set[tuple[str, str]],
    term: str,
    *,
    source: str,
) -> None:
    clean = _clean_term_candidate(term)
    if not clean:
        return
    normalized = normalize_term(clean)
    key = (normalized, source)
    if key in seen:
        return
    seen.add(key)
    out.append(
        {
            "normalized_term": normalized,
            "surface_form": clean,
            "source": source,
            "term_type": classify_term_type(clean),
        }
    )


def _extract_definition_terms(text: str) -> list[str]:
    out: list[str] = []
    for match in DEFINITION_PIPE_RE.finditer(text or ""):
        out.append(str(match.group("term") or ""))
    return out


def _extract_identifier_terms(text: str) -> list[str]:
    out: list[str] = []
    for pattern in IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text or ""):
            out.append(str(match.group("term") or ""))
    return out


def _extract_phrase_terms(text: str) -> list[str]:
    normalized_text = normalize_term(text)
    out: list[str] = []
    for phrase in TERM_PHRASE_HINTS:
        if phrase in normalized_text:
            out.append(phrase)
    return out


def _extract_heading_terms(title: str) -> list[str]:
    heading = strip_section_number(title)
    out = _extract_identifier_terms(heading)
    out.extend(_extract_phrase_terms(heading))
    return out


def extract_term_candidates(text: str, path_parts: list[str], block_type: str) -> list[dict[str, str]]:
    _ = block_type
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    if is_definition_like_section(path_parts):
        for term in _extract_definition_terms(text):
            _append_term_candidate(out, seen, term, source="definition_section")

    for term in _extract_identifier_terms(text):
        _append_term_candidate(out, seen, term, source="identifier_regex")

    for algorithm_info in extract_algorithm_header_info(text):
        if algorithm_info.get("algorithm_name"):
            _append_term_candidate(
                out,
                seen,
                str(algorithm_info.get("algorithm_name") or ""),
                source="algorithm_header",
            )
        for phrase in _extract_phrase_terms(str(algorithm_info.get("raw_header") or "")):
            _append_term_candidate(out, seen, phrase, source="algorithm_header")

    for term in _extract_heading_terms(leaf_section_title(path_parts)):
        _append_term_candidate(out, seen, term, source="section_heading")

    return out
