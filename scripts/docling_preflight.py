from __future__ import annotations
"""
Preflight validation script for Docling PDF parsing backend.

This script verifies that the Docling backend is properly installed and configured
by attempting to parse the first PDF found in a specified directory. It provides
helpful error messages and hints if the preflight check fails, particularly for
Windows symlink privilege issues.

Module-level functions:
    _hint_for_error: Generates contextual hints based on error messages.
    main: Executes the preflight validation and returns exit code.
    
How to Use:
1. Ensure you have at least one PDF file in the `data/raw_pdfs` directory.
2. Run this script to perform the preflight check for the Docling parser backend.
3. If the preflight fails, review the error message and hints to resolve common issues.


"""

import argparse
import json
from pathlib import Path
import sys

PAGE = 20

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.parsers.docling_backend import DoclingBackend


def _hint_for_error(msg: str) -> str:
    lower = msg.lower()
    if "winerror 1314" in lower or "symlink" in lower:
        return (
            "Detected Windows symlink privilege issue while Docling downloads models. "
            "Enable Developer Mode or run shell with required privilege; "
            "keep PARSER_BACKEND=llamaparse until this preflight passes."
        )
    return "Keep PARSER_BACKEND=llamaparse and inspect Docling runtime logs."


def main() -> int:
    parser = argparse.ArgumentParser(prog="python scripts/docling_preflight.py")
    parser.add_argument("--raw-dir", type=str, default="data/raw_pdfs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[FAIL] No PDFs found in {raw_dir}")
        return 1

    pdf_path = pdfs[0]
    backend = DoclingBackend()

    print(f"[INFO] Running Docling preflight on {pdf_path.name} page {PAGE} ...")
    try:
        pages = backend.parse_pdf(pdf_path, expected_pages=PAGE)
    except Exception as exc:
        msg = str(exc)
        print("[FAIL] Docling preflight failed.")
        print(f"[ERROR] {msg}")
        print(f"[HINT] {_hint_for_error(msg)}")
        return 1

    if len(pages) != PAGE:
        print(f"[FAIL] Docling preflight did not return exactly {PAGE} pages for page-{PAGE} probe.")
        print(f"[ERROR] parsed_pages={len(pages)}")
        return 1

    if not pages:
        print("[FAIL] Docling returned zero parsed pages for page 1.")
        return 1

    page = pages[0]
    md = str(page.get("markdown", "") or "").strip()
    text = str(page.get("text", "") or "").strip()

    if not md or not text:
        print("[FAIL] Docling preflight produced empty content for page 1.")
        print(f"[ERROR] markdown_len={len(md)} text_len={len(text)}")
        print("[HINT] Keep PARSER_BACKEND=llamaparse and inspect Docling runtime logs.")
        return 1
    
    # save the pages for debugging/inspection
    debug_path = REPO_ROOT / "data" / "debug" / "docling_preflight_debug.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[OK] Docling preflight passed.")
    print(f"[INFO] backend_version={backend.backend_version()}")
    print(f"[INFO] markdown_len={len(md)} text_len={len(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
