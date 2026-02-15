from rag.chunk import run_chunking, ChunkConfig

PAGES_CLEAN = "data/processed/pages_clean.jsonl"
CHUNKS_OUT = "data/processed/chunks.jsonl"

cfg = ChunkConfig(
    target_chars=4000,
    overlap_chars=600,
    min_chars=400,
)

run_chunking(
    pages_clean_path=PAGES_CLEAN,
    chunks_out_path=CHUNKS_OUT,
    cfg=cfg,
    doc_id_key="doc_id",
    page_key="page_number",   # or "page" if that's what you use
    text_key="text_clean",
)

print("Wrote:", CHUNKS_OUT)
