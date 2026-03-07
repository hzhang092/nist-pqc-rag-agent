from __future__ import annotations

"""
Normalize technical identifiers with escaped or space-broken separators.

This module provides utilities to clean up identifier-like spans in text that may
contain escaped separators (\\_, \\., \\-) or extra whitespace around separators,
which can occur from PDF parsing artifacts or LaTeX rendering.

Normalization is pairwise and conservative:
- escaped separators are unescaped between local identifier tokens,
- spacing around `_` and `-` is compacted between local identifier tokens,
- spacing around `.` is compacted only for dotted identifier patterns that are
  likely to be technical identifiers instead of prose.

Examples:
    >>> normalize_identifier_like_spans("MAC _ Data")
    'MAC_Data'
    >>> normalize_identifier_like_spans("some\\_var\\_name")
    'some_var_name'
    >>> normalize_identifier_like_spans("ML-KEM . Decaps")
    'ML-KEM.Decaps'
    >>> normalize_identifier_like_spans("U.S. Department")
    'U.S. Department'
"""

import re

_LOCAL_TOKEN = r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*"
_PAIR_RE = re.compile(
    rf"(?<![A-Za-z0-9\\])"
    rf"(?P<left>{_LOCAL_TOKEN})"
    rf"(?P<between>(?:\s*\\[._-]\s*|\s+[._-]\s*|\s*[._-]\s+))"
    rf"(?P<right>{_LOCAL_TOKEN})"
    rf"(?![A-Za-z0-9\\])"
)


def _is_all_caps_token(token: str) -> bool:
    letters = [ch for ch in token if ch.isalpha()]
    return bool(letters) and len(letters) >= 2 and all(ch.isupper() for ch in letters)


def _should_compact_unescaped_dot(left: str, right: str) -> bool:
    if left.isdigit() and right.isdigit():
        return True
    if ("-" in left or "_" in left) and (right[0].isupper() or right[0].isdigit()):
        return True
    if _is_all_caps_token(left) and (_is_all_caps_token(right) or right.isdigit()):
        return True
    return False


def normalize_identifier_like_spans(text: str) -> str:
    """
    Normalizes escaped/space-broken technical identifiers in-place.

    Scope is intentionally strict:
    - only local identifier-token pairs are changed,
    - escaped separators are unescaped between identifier tokens,
    - unescaped dots are compacted only when they look identifier-like,
    - non-identifier backslash contexts (LaTeX/macros/commands) are untouched.
    """
    if not text:
        return ""

    normalized = text
    while True:
        changed = False

        def _normalize_match(match: re.Match[str]) -> str:
            nonlocal changed

            left = match.group("left")
            right = match.group("right")
            between = match.group("between")
            escaped = "\\" in between
            sep = next(ch for ch in between if ch in "._-")
            original = match.group(0)

            if sep == "." and not escaped and not _should_compact_unescaped_dot(left, right):
                return original

            if escaped or sep in {"_", "-"} or _should_compact_unescaped_dot(left, right):
                replacement = f"{left}{sep}{right}"
                if replacement != original:
                    changed = True
                return replacement

            return original

        updated = _PAIR_RE.sub(_normalize_match, normalized)
        if not changed:
            return updated
        normalized = updated
