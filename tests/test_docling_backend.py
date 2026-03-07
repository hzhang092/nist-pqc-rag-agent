from __future__ import annotations

from pathlib import Path
import re

import pytest

pytest.importorskip("docling")

from rag.parsers.docling_backend import (
    DoclingBackend,
    _sanitize_docling_markdown,
    _sanitize_formula_latex,
)


def test_docling_parse_pdf_page_failure_falls_back(monkeypatch):
    backend = DoclingBackend()

    class _FakeDoc:
        def __init__(self, mode: str, page_no: int | None = None):
            self._mode = mode
            self._page_no = page_no

        def export_to_markdown(self, page_no=None):
            if self._mode == "bulk":
                if page_no == 2:
                    raise RuntimeError("boom")
                return f"# p{page_no}"
            # Per-page fallback conversion exports without page_no.
            return f"# p{self._page_no}"

    class _FakeResult:
        def __init__(self, mode: str, page_no: int | None = None):
            self.document = _FakeDoc(mode=mode, page_no=page_no)

    calls: list[tuple[Path, tuple[int, int]]] = []

    class _FakeConverter:
        def convert(self, pdf_path: Path, page_range=(1, 1)):
            calls.append((pdf_path, tuple(page_range)))
            lo, hi = tuple(page_range)
            if lo == 1 and hi == 3:
                return _FakeResult(mode="bulk")
            if lo == 2 and hi == 2:
                raise RuntimeError("fallback boom")
            return _FakeResult(mode="single", page_no=lo)

    monkeypatch.setattr(backend, "_converter", _FakeConverter())

    pages = backend.parse_pdf(Path("dummy.pdf"), expected_pages=3)

    assert [int(p["page_number"]) for p in pages] == [1, 2, 3]
    assert pages[1]["markdown"] == ""
    assert pages[1]["text"] == ""
    assert pages[1]["parser_backend"] == "docling"
    assert len(calls) >= 1
    assert calls[0][1] == (1, 3)


def test_docling_sanitizer_removes_loc_and_broken_formula_tags():
    raw = (
        "<!-- image -->\n"
        "<text><loc_465><loc_0><loc_471><loc_0>)</text>\n"
        "$$x + y</formula$$\n"
        "<loc_99>\n"
    )
    cleaned = _sanitize_docling_markdown(raw)

    assert "<!-- image -->" not in cleaned
    assert "<loc_" not in cleaned
    assert "</formula" not in cleaned
    assert "$$x + y$$" in cleaned


def test_docling_sanitizer_removes_quad_spam_and_hat_only_lines():
    quad_run = " ".join(["\\quad"] * 20)
    bare_backslash_run = " ".join(["\\"] * 60)
    hat = "\u0302"
    raw = (
        "keep-before\n"
        f"$$ z \\leftarrow x + y {quad_run} $$\n"
        f"$$ a {bare_backslash_run} b $$\n"
        f"{hat}\n"
        f" {hat} \n"
        "keep-after\n"
    )
    cleaned = _sanitize_docling_markdown(raw)

    assert "keep-before" in cleaned
    assert "keep-after" in cleaned
    assert not re.search(r"(?:\\quad\s*){6,}", cleaned)
    assert not re.search(r"(?:\\(?:\s|$)){20,}", cleaned)
    assert all(line.strip() != hat for line in cleaned.splitlines())


def test_docling_sanitizer_drops_alignment_layout_math_noise():
    noisy = (
        "$$\\quad & \\quad & \\quad & \\quad & \\quad & \\quad & \\quad & \\quad & "
        "\\quad & \\quad & \\quad & \\quad & \\quad & \\quad & \\quad & \\quad & $$\n"
        "Fig. 9. Using a KEM for key establishment with unilateral authentication\n"
    )
    cleaned = _sanitize_docling_markdown(noisy)
    assert "Fig. 9." in cleaned
    assert "\\quad &" not in cleaned

    formula_noisy = (
        "\\quad & \\quad & \\quad & \\quad & \\quad & \\quad & "
        "\\quad & \\quad & \\quad & \\quad & \\quad & \\quad & "
        "\\quad & \\quad & \\quad & \\quad &"
    )
    assert _sanitize_formula_latex(formula_noisy) == ""


def test_docling_parse_pdf_adaptive_batch_shrinks_on_oom(monkeypatch):
    backend = DoclingBackend()

    class _FakeDoc:
        def __init__(self, lo: int):
            self._lo = lo

        def export_to_markdown(self, page_no=None):
            _ = page_no
            return f"# p{self._lo}"

    class _FakeResult:
        def __init__(self, lo: int):
            self.document = _FakeDoc(lo)

    calls: list[tuple[int, int]] = []

    class _FakeConverter:
        def convert(self, _pdf_path: Path, page_range=(1, 1)):
            lo, hi = tuple(page_range)
            calls.append((lo, hi))
            if hi - lo + 1 > 1:
                raise RuntimeError("CUDA out of memory")
            return _FakeResult(lo)

    monkeypatch.setenv("DOCLING_PAGE_BATCH_SIZE", "4")
    monkeypatch.setenv("DOCLING_MIN_PAGE_BATCH_SIZE", "1")
    monkeypatch.setenv("DOCLING_ADAPTIVE_BATCHING", "1")
    monkeypatch.setattr(backend, "_converter", _FakeConverter())

    pages = backend.parse_pdf(Path("dummy.pdf"), expected_pages=4)

    assert [int(p["page_number"]) for p in pages] == [1, 2, 3, 4]
    assert [p["markdown"] for p in pages] == ["# p1", "# p2", "# p3", "# p4"]
    assert calls[:3] == [(1, 4), (1, 2), (1, 1)]
