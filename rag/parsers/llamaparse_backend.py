from __future__ import annotations
"""
LlamaParseBackend module for parsing PDF documents using the LlamaParse API.

This module provides a backend implementation for parsing PDF files, particularly
NIST technical standards, using the LlamaParse service. It preserves tables and
LaTeX mathematical expressions during parsing.

Classes:
    LlamaParseBackend: A parser backend that uses LlamaParse to convert PDF documents
                       to markdown format with structured page information.
"""

from pathlib import Path

from llama_parse import LlamaParse

from .base import ParsedPage, ParserBackend


class LlamaParseBackend(ParserBackend):
    name = "llamaparse"

    def __init__(self) -> None:
        self._parser = None

    def backend_version(self) -> str:
        try:
            from importlib.metadata import version

            return version("llama-parse")
        except Exception:
            return "unknown"

    def _get_parser(self) -> LlamaParse:
        if self._parser is None:
            self._parser = LlamaParse(
                result_type="markdown",
                verbose=False,
                parsing_instruction=(
                    "This is a NIST technical standard. Preserve all tables and LaTeX math."
                ),
            )
        return self._parser

    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None) -> list[ParsedPage]:
        _ = expected_pages

        parser = self._get_parser()
        json_objs = parser.get_json_result(str(pdf_path))
        if not json_objs:
            return []
        parsed_pages = json_objs[0].get("pages", [])

        out: list[ParsedPage] = []
        for page in parsed_pages:
            page_no = int(page.get("page", 0))
            markdown = str(page.get("text", "") or "")
            out.append(
                ParsedPage(
                    doc_id=pdf_path.stem,
                    source_path=pdf_path.as_posix(),
                    page_number=page_no,
                    text=markdown,
                    markdown=markdown,
                    parser_backend=self.name,
                )
            )
        out.sort(key=lambda row: int(row.get("page_number", 0)))
        return out
