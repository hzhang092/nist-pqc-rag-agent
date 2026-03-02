from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.chunk import ChunkConfig, run_chunking_per_page
from rag.config import SETTINGS


run_chunking_per_page(
    pages_clean_path="data/processed/pages_clean.jsonl",
    chunks_out_path="data/processed/chunks.jsonl",
    cfg=ChunkConfig(target_chars=1400, overlap_blocks=1, min_chars=250, max_chars=2200),
    chunker_version=SETTINGS.CHUNKER_VERSION,
)
print(f"Wrote: data/processed/chunks.jsonl (chunker_version={SETTINGS.CHUNKER_VERSION})")
