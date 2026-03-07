from __future__ import annotations

import json
from pathlib import Path

from rag.chunk import ChunkConfig, run_chunking_per_page


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def test_chunk_v2_detects_algorithm_and_table_blocks(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    markdown = """# Section A
Algorithm 2: Example Procedure
Input: x
Output: y
1. Do thing

| col1 | col2 |
| --- | --- |
| 1 | 2 |
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=80, overlap_blocks=0, min_chars=1, max_chars=500),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows
    block_types = {row.get("block_type") for row in rows}
    assert "algorithm" in block_types
    assert "table" in block_types
    assert all(row.get("chunker_version") == "v2" for row in rows)
    assert any(row.get("section_path") for row in rows)


def test_chunk_v2_supports_code_math_fallback_and_section_carry(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    long_para = " ".join(f"tok{i}" for i in range(240))
    huge_list_item = "- " + ("X" * 1500)
    markdown_page_1 = "# 1 Intro\nShort opener paragraph."
    markdown_page_2 = f"""{long_para}

{huge_list_item}

```python
def add(x, y):
    return x + y
```

$$
a = b + c
$$

Algorithm 7: Demo
1: init
2: loop
8: finalize
This sentence should not be captured as algorithm.
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback page one",
                "markdown": markdown_page_1,
            },
            {
                "doc_id": "DOC",
                "page_number": 2,
                "text_clean": "fallback page two",
                "markdown": markdown_page_2,
            },
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(
            target_chars=30,
            overlap_blocks=0,
            min_chars=1,
            max_chars=120,
            v2_window_tokens=40,
            v2_overlap_tokens=8,
            v2_max_block_tokens=40,
        ),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows

    block_types = {row.get("block_type") for row in rows}
    assert "code" in block_types
    assert "math" in block_types
    assert "mixed" not in block_types

    page2_rows = [row for row in rows if int(row.get("page_number", 0)) == 2]
    assert page2_rows
    assert all(row.get("section_path") == "1 Intro" for row in page2_rows)

    text_rows_page2 = [row for row in page2_rows if row.get("block_type") == "text"]
    assert len(text_rows_page2) >= 2

    algo_texts = [row.get("text", "") for row in page2_rows if row.get("block_type") == "algorithm"]
    assert algo_texts
    assert all("This sentence should not be captured as algorithm." not in t for t in algo_texts)

    assert max(int(row.get("char_len", 0)) for row in rows) < 1000


def test_chunk_v2_detects_algorithm_under_algorithm_heading(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    # Mirrors Docling output for NIST.FIPS.203 where the algorithm is introduced by
    # a markdown heading and the body begins with prose + step labels like "1:".
    markdown = """## Algorithm 1 ForExample ()

Performs two simple 'for' loops.

1:

for

(

i

←

0

;

i

<

10

;

i

++)

2:

A[i] ← i

3: end for
"""

    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=200, overlap_blocks=0, min_chars=1, max_chars=1000),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows

    assert any(row.get("block_type") == "algorithm" for row in rows)
    assert any("Algorithm 1" in str(row.get("section_path", "")) for row in rows)


def test_chunk_v2_classifies_fenced_algorithm_block_as_algorithm(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    markdown = """## 4.1 Functions

```
Algorithm 2 SHAKE128example(str1, str2)
Input: byte arrays
1: init
2: absorb
3: return
```
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=300, overlap_blocks=0, min_chars=1, max_chars=1000),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows
    assert any(row.get("block_type") == "algorithm" for row in rows)


def test_chunk_v2_does_not_treat_absolute_value_bars_as_table(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    markdown = """## 4.1 Cryptographic Functions

This equivalence holds whether or not | str_i | and b_j are multiples of the SHAKE128 block length.
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=400, overlap_blocks=0, min_chars=1, max_chars=1000),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows
    bar_rows = [row for row in rows if "| str_i |" in str(row.get("text", ""))]
    assert bar_rows
    assert all(row.get("block_type") != "table" for row in bar_rows)


def test_chunk_v2_does_not_treat_concat_operator_bars_as_table(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    markdown = """## 3.7 Use of Symmetric Cryptography

$$\\text {output} \\leftarrow \\text {SHAKE256(str||\\dots||str_{m}, 8b_{1}+dots+8b_{k}).}$$
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=400, overlap_blocks=0, min_chars=1, max_chars=1000),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows
    formula_rows = [row for row in rows if "str||\\dots||str_{m}" in str(row.get("text", ""))]
    assert formula_rows
    assert all(row.get("block_type") != "table" for row in formula_rows)


def test_chunk_v2_does_not_label_long_prose_as_math_from_single_symbol(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    markdown = """## 2. Notes

This explanatory sentence includes x = y as an example and continues with many words so it should remain prose text rather than a standalone math block.
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=400, overlap_blocks=0, min_chars=1, max_chars=1000),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows
    prose_rows = [row for row in rows if "x = y" in str(row.get("text", ""))]
    assert prose_rows
    assert all(row.get("block_type") != "math" for row in prose_rows)


def test_chunk_v2_algorithm_headings_are_siblings_not_chained(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    markdown = """## 4.2 General Algorithms

## Algorithm 3 BitsToBytes (b)
Performs conversion from bits to bytes.
1: init
2: return

## Algorithm 4 BytesToBits (B)
Performs conversion from bytes to bits.
1: init
2: return
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=220, overlap_blocks=0, min_chars=1, max_chars=1000),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows
    algo_rows = [row for row in rows if row.get("block_type") == "algorithm"]
    assert algo_rows

    algo3_paths = [str(row.get("section_path", "")) for row in algo_rows if "Algorithm 3" in str(row.get("section_path", ""))]
    algo4_paths = [str(row.get("section_path", "")) for row in algo_rows if "Algorithm 4" in str(row.get("section_path", ""))]
    assert algo3_paths and algo4_paths
    assert all(path.startswith("4.2 General Algorithms > Algorithm 3") for path in algo3_paths)
    assert all(path.startswith("4.2 General Algorithms > Algorithm 4") for path in algo4_paths)
    assert all("Algorithm 3 BitsToBytes (b) > Algorithm 4" not in path for path in algo4_paths)


def test_chunk_v2_numeric_section_heading_pops_prior_algorithm_heading(tmp_path: Path):
    pages_in = tmp_path / "pages_clean.jsonl"
    chunks_out = tmp_path / "chunks.jsonl"

    markdown = """## -1 Algorithm 10 NTT (f)
Input: x
1: return

## 4.3.1 Multiplication in the NTT Domain

## Algorithm 11 MultiplyNTTs (f, g)
Input: f, g
1: return
"""
    _write_jsonl(
        pages_in,
        [
            {
                "doc_id": "DOC",
                "page_number": 1,
                "text_clean": "fallback text",
                "markdown": markdown,
            }
        ],
    )

    run_chunking_per_page(
        pages_clean_path=pages_in,
        chunks_out_path=chunks_out,
        cfg=ChunkConfig(target_chars=220, overlap_blocks=0, min_chars=1, max_chars=1000),
        chunker_version="v2",
        write_manifest=False,
    )

    rows = _read_jsonl(chunks_out)
    assert rows
    algo11_paths = [
        str(row.get("section_path", ""))
        for row in rows
        if row.get("block_type") == "algorithm" and "Algorithm 11" in str(row.get("section_path", ""))
    ]
    assert algo11_paths
    assert all("Algorithm 10" not in path for path in algo11_paths)
    assert all(path.startswith("4.3.1 Multiplication in the NTT Domain > Algorithm 11") for path in algo11_paths)
