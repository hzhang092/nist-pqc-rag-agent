from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("docling")

from scripts import docling_preflight


def _make_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # The preflight only needs the path to exist; parser behavior is mocked.
    (raw_dir / "sample.pdf").write_bytes(b"%PDF-1.4\n")
    return raw_dir


def test_docling_preflight_passes_on_non_empty_page(monkeypatch, tmp_path):
    raw_dir = _make_raw_dir(tmp_path)

    class _FakeBackend:
        def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None):
            _ = pdf_path
            assert expected_pages == 1
            return [
                {
                    "page_number": 1,
                    "markdown": "# ok",
                    "text": "ok",
                }
            ]

        def backend_version(self) -> str:
            return "test"

    monkeypatch.setattr(docling_preflight, "DoclingBackend", lambda: _FakeBackend())
    monkeypatch.setattr(docling_preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["docling_preflight.py", "--raw-dir", str(raw_dir)],
    )

    rc = docling_preflight.main()
    assert rc == 0
    assert (tmp_path / "data" / "debug" / "docling_preflight_debug.json").exists()


def test_docling_preflight_fails_on_empty_page_content(monkeypatch, tmp_path):
    raw_dir = _make_raw_dir(tmp_path)

    class _FakeBackend:
        def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None):
            _ = pdf_path
            assert expected_pages == 1
            return [
                {
                    "page_number": 1,
                    "markdown": "",
                    "text": "",
                }
            ]

        def backend_version(self) -> str:
            return "test"

    monkeypatch.setattr(docling_preflight, "DoclingBackend", lambda: _FakeBackend())
    monkeypatch.setattr(docling_preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["docling_preflight.py", "--raw-dir", str(raw_dir)],
    )

    rc = docling_preflight.main()
    assert rc == 1


def test_docling_preflight_selects_requested_pages(monkeypatch, tmp_path):
    raw_dir = _make_raw_dir(tmp_path)

    class _FakeBackend:
        def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None):
            _ = pdf_path
            assert expected_pages == 4
            return [
                {
                    "page_number": 1,
                    "markdown": "# p1",
                    "text": "p1",
                },
                {
                    "page_number": 2,
                    "markdown": "# p2",
                    "text": "p2",
                },
                {
                    "page_number": 3,
                    "markdown": "# p3",
                    "text": "p3",
                },
                {
                    "page_number": 4,
                    "markdown": "# p4",
                    "text": "p4",
                },
            ]

        def backend_version(self) -> str:
            return "test"

    monkeypatch.setattr(docling_preflight, "DoclingBackend", lambda: _FakeBackend())
    monkeypatch.setattr(docling_preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "docling_preflight.py",
            "--raw-dir",
            str(raw_dir),
            "--pages",
            "2,4",
        ],
    )

    rc = docling_preflight.main()
    assert rc == 0

    debug_file = tmp_path / "data" / "debug" / "docling_preflight_debug.json"
    assert debug_file.exists()
    payload = debug_file.read_text(encoding="utf-8")
    assert '"page_number": 2' in payload
    assert '"page_number": 4' in payload
    assert '"page_number": 1' not in payload
    assert '"page_number": 3' not in payload
