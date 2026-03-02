from __future__ import annotations
"""
Factory function that returns a parser backend instance based on the specified name.

Args:
    name: Optional string specifying the parser backend to use. If None, defaults to
          the PARSER_BACKEND setting from SETTINGS. The name is case-insensitive and
          whitespace is stripped.

Returns:
    ParserBackend: An instance of the requested parser backend (LlamaParseBackend
                   or DoclingBackend).

Raises:
    ValueError: If the specified backend name is not recognized or supported.
"""

from rag.config import SETTINGS

from .base import ParserBackend


def get_parser_backend(name: str | None = None) -> ParserBackend:
    backend = (name or SETTINGS.PARSER_BACKEND).strip().lower()
    if backend == "llamaparse":
        from .llamaparse_backend import LlamaParseBackend

        return LlamaParseBackend()
    if backend == "docling":
        from .docling_backend import DoclingBackend

        return DoclingBackend()
    raise ValueError(f"Unknown parser backend: {backend!r}")
