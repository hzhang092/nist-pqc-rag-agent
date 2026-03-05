from __future__ import annotations
'''
Normalize technical identifiers with escaped or space-broken separators.

This module provides utilities to clean up identifier-like spans in text that may
contain escaped separators (\\_, \\., \\-) or extra whitespace around separators,
which can occur from PDF parsing artifacts or LaTeX rendering.

The normalization is conservative and only affects spans that:
- Contain alphanumeric characters separated by ., _, or - 
- Have at least one separator present
- Are bounded by non-identifier characters

Examples:
    >>> normalize_identifier_like_spans("MAC _ Data")
    'MAC_Data'
    >>> normalize_identifier_like_spans("some\\_var\\_name")
    'some_var_name'
    >>> normalize_identifier_like_spans("foo . bar . baz")
    'foo.bar.baz'
    >>> normalize_identifier_like_spans("LaTeX \\command is unchanged")
    'LaTeX \\command is unchanged'
'''

import re

# Identifier-like spans:
# - must contain at least one separator from . _ -
# - separators may be escaped (\_, \., \-)
# - allows whitespace around separators (for parser artifacts like "MAC _ Data")
_IDENTIFIER_LIKE_RE = re.compile(
    r"(?<![A-Za-z0-9\\])"
    r"(?:[A-Za-z0-9]+(?:\s*\\?[._-]\s*[A-Za-z0-9]+)+)"
    r"(?![A-Za-z0-9\\])"
)
_SEP_WITH_OPTIONAL_ESCAPE_RE = re.compile(r"\s*\\?([._-])\s*")


def normalize_identifier_like_spans(text: str) -> str:
    """
    Normalizes escaped/space-broken technical identifiers in-place.

    Scope is intentionally strict:
    - only identifier-like spans are changed,
    - only separators . _ - are unescaped/compacted,
    - non-identifier backslash contexts (LaTeX/macros/commands) are untouched.
    """
    if not text:
        return ""

    def _normalize_match(match: re.Match[str]) -> str:
        token = match.group(0)
        return _SEP_WITH_OPTIONAL_ESCAPE_RE.sub(r"\1", token)

    return _IDENTIFIER_LIKE_RE.sub(_normalize_match, text)
