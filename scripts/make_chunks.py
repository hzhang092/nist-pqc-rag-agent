from rag.chunk import run_chunking_per_page, ChunkConfig

run_chunking_per_page(
    pages_clean_path="data/processed/pages_clean.jsonl",
    chunks_out_path="data/processed/chunks.jsonl",
    cfg=ChunkConfig(target_chars=1400, overlap_blocks=1, min_chars=250, max_chars=2200),
)
print("Wrote: data/processed/chunks.jsonl")
