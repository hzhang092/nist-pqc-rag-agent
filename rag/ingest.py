from __future__ import annotations
"""
Module for ingesting and parsing PDF documents from a raw data directory.

This module handles the workflow of reading PDF files, parsing their content using
configurable parser backends, validating the parsed results, and writing them to
a unified JSONL dataset along with per-document JSON files.

Key responsibilities:
- Load PDF files from the raw data directory
- Parse PDFs using specified backend (e.g., LLM-based, rule-based)
- Validate parsed page numbers against the true page count
- Normalize and structure parsed page data
- Write output to both per-document JSON files and a unified JSONL file
- Track parsing statistics and update project manifest

Environment:
- Requires .env file for configuration via dotenv
- Reads settings from rag.config.SETTINGS
- Uses raw PDFs from: data/raw_pdfs
- Writes processed data to: data/processed

Output:
- pages.jsonl: Unified JSONL file with all parsed pages
- {pdf_stem}_parsed.json: Per-document JSON files with parsed content
- Updated manifest entry with parsing statistics and artifact paths

How to Use:
1. Place PDF files to be ingested in the `data/raw_pdfs` directory.
2. Configure parser backend and settings in the .env file or directly in SETTINGS.
3. Run this script to perform ingestion and parsing, which will output structured data and update the manifest for tracking.
"""

import json
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

from rag.config import SETTINGS, validate_settings
from rag.parsers.base import ParsedPage, ParserBackend
from rag.parsers.factory import get_parser_backend
from rag.versioning import update_manifest


RAW_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_page_record(rec: ParsedPage, pdf_path: Path, parser_backend: str) -> dict:
    page_no = int(rec.get("page_number", 0))
    text = str(rec.get("text", "") or "")
    markdown = str(rec.get("markdown", "") or "")
    out = {
        "doc_id": str(rec.get("doc_id", pdf_path.stem)),
        "source_path": str(rec.get("source_path", pdf_path.as_posix())),
        "page_number": page_no,
        "text": text,
        "parser_backend": str(rec.get("parser_backend", parser_backend)),
    }
    if markdown:
        out["markdown"] = markdown
    return out


def _validate_page_numbers(parsed_pages: list[dict], *, true_pages: int, strict_page_match: bool) -> None:
    page_numbers = [int(row.get("page_number", 0)) for row in parsed_pages]
    invalid = [p for p in page_numbers if p <= 0 or p > true_pages]
    if invalid and strict_page_match:
        raise ValueError(
            "Parsed page numbers out of bounds: "
            f"{sorted(set(invalid))[:10]} (true_pages={true_pages})"
        )
    if invalid:
        print(
            f"[WARN] Parsed page numbers out of bounds: "
            f"{sorted(set(invalid))[:10]} (true_pages={true_pages})"
        )

    expected = set(range(1, int(true_pages) + 1))
    observed = set(page_numbers)
    missing = sorted(expected - observed)

    seen: set[int] = set()
    duplicates_set: set[int] = set()
    for page_no in page_numbers:
        if page_no in seen:
            duplicates_set.add(page_no)
        else:
            seen.add(page_no)
    duplicates = sorted(duplicates_set)

    if strict_page_match and (missing or duplicates):
        raise ValueError(
            "Parsed pages do not match expected 1..N coverage: "
            f"missing={missing[:10]}, duplicates={duplicates[:10]}, true_pages={true_pages}"
        )
    if missing or duplicates:
        print(
            "[WARN] Parsed pages do not match expected 1..N coverage: "
            f"missing={missing[:10]}, duplicates={duplicates[:10]}, true_pages={true_pages}"
        )


def parse_and_validate(
    pdf_path: Path,
    pages_jsonl_f: TextIO,
    *,
    parser_backend: ParserBackend,
    strict_page_match: bool = True,
) -> tuple[list[dict], Path]:
    print(f"🚀 Parsing: {pdf_path.name} with backend={parser_backend.name} ...")

    true_pages = len(PdfReader(str(pdf_path)).pages)
    parsed_raw = parser_backend.parse_pdf(pdf_path, expected_pages=true_pages)
    parsed_pages = [
        _normalize_page_record(row, pdf_path, parser_backend.name)
        for row in sorted(parsed_raw, key=lambda r: int(r.get("page_number", 0)))
    ]

    if len(parsed_pages) != true_pages:
        msg = (
            f"{pdf_path.name} has {true_pages} pages, "
            f"but parser returned {len(parsed_pages)} pages."
        )
        if strict_page_match:
            raise ValueError(msg)
        print(f"[WARN] {msg}")

    _validate_page_numbers(parsed_pages, true_pages=true_pages, strict_page_match=strict_page_match)

    per_doc_path = PROCESSED_DIR / f"{pdf_path.stem}_parsed.json"
    per_doc_path.write_text(
        json.dumps(parsed_pages, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for rec in parsed_pages:
        pages_jsonl_f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"✅ Saved {len(parsed_pages)} pages to {per_doc_path.name}")
    return parsed_pages, per_doc_path


def main() -> None:
    validate_settings()

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR}")

    parser_backend = get_parser_backend(SETTINGS.PARSER_BACKEND)
    pages_jsonl_path = PROCESSED_DIR / "pages.jsonl"

    per_doc_counts: dict[str, int] = {}
    per_doc_debug_paths: list[Path] = []

    with pages_jsonl_path.open("w", encoding="utf-8") as pages_jsonl_f:
        for pdf_path in pdf_paths:
            parsed_pages, debug_path = parse_and_validate(
                pdf_path,
                pages_jsonl_f,
                parser_backend=parser_backend,
                strict_page_match=SETTINGS.PARSER_STRICT_PAGE_MATCH,
            )
            per_doc_counts[pdf_path.stem] = len(parsed_pages)
            per_doc_debug_paths.append(debug_path)

    update_manifest(
        stage_name="ingest",
        stage_payload={
            "parser_backend": parser_backend.name,
            "parser_backend_version": parser_backend.backend_version(),
            "strict_page_match": bool(SETTINGS.PARSER_STRICT_PAGE_MATCH),
            "num_docs": len(pdf_paths),
            "total_pages": int(sum(per_doc_counts.values())),
            "per_doc_pages": dict(sorted(per_doc_counts.items(), key=lambda kv: kv[0])),
        },
        artifact_paths=[pages_jsonl_path, *sorted(per_doc_debug_paths, key=lambda p: p.name)],
    )

    print(f"✅ Wrote unified dataset: {pages_jsonl_path}")


if __name__ == "__main__":
    main()
