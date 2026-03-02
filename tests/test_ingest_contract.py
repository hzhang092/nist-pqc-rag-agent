from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from rag import ingest


class _FakeParser:
    name = "fake"

    def backend_version(self) -> str:
        return "0.0-test"

    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None):
        _ = pdf_path
        _ = expected_pages
        return [
            {"doc_id": "D", "source_path": "x.pdf", "page_number": 2, "text": "p2", "markdown": "## p2"},
            {"doc_id": "D", "source_path": "x.pdf", "page_number": 1, "text": "p1", "markdown": "# p1"},
        ]


class _FakeParserMismatch(_FakeParser):
    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None):
        _ = pdf_path
        _ = expected_pages
        return [{"doc_id": "D", "source_path": "x.pdf", "page_number": 1, "text": "p1"}]


class _FakeParserDuplicate(_FakeParser):
    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None):
        _ = pdf_path
        _ = expected_pages
        return [
            {"doc_id": "D", "source_path": "x.pdf", "page_number": 1, "text": "p1-a"},
            {"doc_id": "D", "source_path": "x.pdf", "page_number": 1, "text": "p1-b"},
        ]


def _make_pdf(path: Path, n_pages: int) -> None:
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)


def test_parse_and_validate_deterministic_order_and_fields(tmp_path, monkeypatch):
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path, 2)
    monkeypatch.setattr(ingest, "PROCESSED_DIR", tmp_path)

    pages_jsonl = tmp_path / "pages.jsonl"
    with pages_jsonl.open("w", encoding="utf-8") as f:
        parsed_pages, debug_path = ingest.parse_and_validate(
            pdf_path,
            f,
            parser_backend=_FakeParser(),
            strict_page_match=True,
        )

    assert debug_path.exists()
    assert [row["page_number"] for row in parsed_pages] == [1, 2]
    assert all("parser_backend" in row for row in parsed_pages)
    assert all("markdown" in row for row in parsed_pages)

    rows = [json.loads(line) for line in pages_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["page_number"] for row in rows] == [1, 2]


def test_parse_and_validate_strict_page_count_raises(tmp_path, monkeypatch):
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path, 2)
    monkeypatch.setattr(ingest, "PROCESSED_DIR", tmp_path)

    with (tmp_path / "pages.jsonl").open("w", encoding="utf-8") as pages_jsonl_open:
        with pytest.raises(ValueError, match="has 2 pages"):
            ingest.parse_and_validate(
                pdf_path,
                pages_jsonl_open,
                parser_backend=_FakeParserMismatch(),
                strict_page_match=True,
            )


def test_parse_and_validate_strict_page_coverage_raises_on_duplicates(tmp_path, monkeypatch):
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path, 2)
    monkeypatch.setattr(ingest, "PROCESSED_DIR", tmp_path)

    with (tmp_path / "pages.jsonl").open("w", encoding="utf-8") as pages_jsonl_open:
        with pytest.raises(ValueError, match="1..N coverage"):
            ingest.parse_and_validate(
                pdf_path,
                pages_jsonl_open,
                parser_backend=_FakeParserDuplicate(),
                strict_page_match=True,
            )
