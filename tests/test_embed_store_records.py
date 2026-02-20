from rag.embed import build_store_records


def test_build_store_records_vector_id_contiguous_and_aligned():
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
