"""Tests for embedding store record construction.

What this tests
---------------
This module verifies that ``rag.embed.build_store_records``:

1. Skips chunks whose text is empty/whitespace-only after trimming.
2. Returns a ``texts`` list containing only normalized, non-empty chunk text.
3. Produces store records aligned 1:1 with ``texts``.
4. Assigns contiguous ``vector_id`` values starting at 0.
5. Preserves correct ``chunk_id`` mapping between kept texts and store records.

How to run
----------
From the project root, run either:

- ``pytest tests/test_embed_store_records.py``
- ``python -m pytest tests/test_embed_store_records.py``
"""

from rag.embed import build_store_records


def test_build_store_records_vector_id_contiguous_and_aligned():
    """Ensure kept chunks are aligned to contiguous vector IDs and chunk IDs."""
    chunks = [
        {"chunk_id": "c0", "doc_id": "D", "start_page": 1, "end_page": 1, "text": "hello"},
        {"chunk_id": "c1", "doc_id": "D", "start_page": 1, "end_page": 1, "text": ""},
        {"chunk_id": "c2", "doc_id": "D", "start_page": 2, "end_page": 2, "text": "  world  "},
        {"chunk_id": "c3", "doc_id": "D", "start_page": 3, "end_page": 3, "text": "\n\t"},
    ]

    texts, store = build_store_records(chunks)
    assert texts == ["hello", "world"]

    assert len(store) == len(texts)
    assert [rec["vector_id"] for rec in store] == [0, 1]
    assert store[0]["chunk_id"] == "c0"
    assert store[1]["chunk_id"] == "c2"
