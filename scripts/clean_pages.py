# scripts/clean_pages.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.clean import run_clean, CleanConfig

PAGES_IN = "data/processed/pages.jsonl"
PAGES_OUT = "data/processed/pages_clean.jsonl"

cfg = CleanConfig(
    header_footer_lines=3,
    boilerplate_ratio=0.6,
    join_wrapped_lines=True,
)

run_clean(
    pages_path=PAGES_IN,
    out_path=PAGES_OUT,
    cfg=cfg,
    text_key="text",          # change if your key is different
    out_text_key="text_clean",
    doc_id_key="doc_id",      # change if your doc id key is different
)

print("Wrote:", PAGES_OUT)
