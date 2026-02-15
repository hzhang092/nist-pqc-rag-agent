# scripts/clean_pages.py
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
