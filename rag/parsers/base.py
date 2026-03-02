"""
Module for parsing and normalizing document content from PDF files.

This module provides base protocols and utilities for PDF parsing backends and
markdown-to-text conversion for document ingestion.

Classes:
    ParsedPage: TypedDict containing parsed page data with optional fields for
        document ID, source path, page number, plain text, markdown content,
        and parser backend identifier.
    ParserBackend: Protocol defining the interface for PDF parser implementations.

Functions:
    markdown_to_text: Converts markdown-formatted text to plain text by removing
        markdown syntax elements while preserving content for ingestion purposes.

How to Use:
    1. Implement a PDF parser backend by creating a class that adheres to the
       ParserBackend protocol, providing methods for parsing PDFs and returning
       structured page data.
    2. Use the markdown_to_text function to convert markdown content to plain text
       when ingesting documents, ensuring a consistent and clean text representation
       for downstream processing and retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, TypedDict


class ParsedPage(TypedDict, total=False):
    doc_id: str
    source_path: str
    page_number: int
    text: str
    markdown: str
    parser_backend: str


class ParserBackend(Protocol):
    name: str

    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None) -> list[ParsedPage]:
        ...

    def backend_version(self) -> str:
        ...


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_MD_CODE_FENCE_RE = re.compile(r"^\s*```")
_MD_EMPH_RE = re.compile(r"(\*\*|\*|__|_)")
_MD_HTML_TAG_RE = re.compile(r"<[^>]+>")


def markdown_to_text(markdown: str) -> str:
    """Deterministic markdown-to-plain normalization for ingestion fallback text."""
    s = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw in s.split("\n"):
        if _MD_CODE_FENCE_RE.match(raw):
            continue
        ln = _MD_HEADING_RE.sub("", raw)
        ln = _MD_LINK_RE.sub(r"\1", ln)
        ln = ln.replace("`", "")
        ln = _MD_EMPH_RE.sub("", ln)
        ln = _MD_HTML_TAG_RE.sub("", ln)
        lines.append(ln.rstrip())

    out: list[str] = []
    empty_run = 0
    for ln in lines:
        if ln.strip() == "":
            empty_run += 1
            if empty_run <= 1:
                out.append("")
            continue
        empty_run = 0
        out.append(ln)
    return "\n".join(out).strip()
