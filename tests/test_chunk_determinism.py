from __future__ import annotations

import json
from pathlib import Path

from rag.chunk import ChunkConfig, run_chunking_per_page


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_chunk_v2_deterministic_output(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_a = tmp_path / "chunks_a.jsonl"
    chunks_b = tmp_path / "chunks_b.jsonl"

    rows = [
        {
            "doc_id": "DOC",
            "page_number": 1,
            "text_clean": "Fallback page one.",
            "markdown": "# Intro\nThis is intro text.\n\nAlgorithm 1: A\nInput: z\nOutput: q\n",
        },
        {
            "doc_id": "DOC",
            "page_number": 2,
            "text_clean": "Fallback page two.",
            "markdown": "## Table Section\n| a | b |\n| --- | --- |\n| 3 | 4 |\n",
        },
    ]
    _write_jsonl(pages_in, rows)

    cfg = ChunkConfig(target_chars=120, overlap_blocks=0, min_chars=1, max_chars=500)
    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_a,
        cfg=cfg,
        chunker_version="v2",
        write_manifest=False,
    )
    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_b,
        cfg=cfg,
        chunker_version="v2",
        write_manifest=False,
    )

    assert chunks_a.read_text(encoding="utf-8") == chunks_b.read_text(encoding="utf-8")

    parsed = _load_jsonl(chunks_a)
    assert parsed
    assert all(int(row["start_page"]) <= int(row["end_page"]) for row in parsed)
    assert all(row.get("block_type") in {"text", "list", "table", "algorithm", "code", "math"} for row in parsed)
