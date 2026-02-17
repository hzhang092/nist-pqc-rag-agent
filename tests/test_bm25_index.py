import json
from pathlib import Path

from rag.index_bm25 import build_bm25_artifact, tokenize


def test_tokenize_preserves_technical_tokens():
    text = "ML-KEM.KeyGen K-PKE.KeyGen ML-KEM-768 SHAKE128 Algorithm 19"
    tokens = tokenize(text)

    assert "ml-kem.keygen" in tokens
    assert "k-pke.keygen" in tokens
    assert "ml-kem-768" in tokens
    assert "shake128" in tokens
    assert "algorithm" in tokens
    assert "19" in tokens

    assert "ml" in tokens
    assert "kem" in tokens
    assert "keygen" in tokens
    assert "pke" in tokens
    assert "768" in tokens


def test_bm25_artifact_stable_for_same_input(tmp_path: Path):
    chunk_store = tmp_path / "chunk_store.jsonl"
    rows = [
        {
            "vector_id": 2,
            "chunk_id": "d1::p1::c1",
            "doc_id": "d1",
            "start_page": 1,
            "end_page": 1,
            "text": "Algorithm 19 ML-KEM.KeyGen",
        },
        {
            "vector_id": 1,
            "chunk_id": "d2::p1::c1",
            "doc_id": "d2",
            "start_page": 1,
            "end_page": 1,
            "text": "SHAKE128 K-PKE.KeyGen",
        },
    ]
    with chunk_store.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row) + "\n")

    a1 = build_bm25_artifact(chunk_store_path=chunk_store)
    a2 = build_bm25_artifact(chunk_store_path=chunk_store)

    assert a1 == a2
    assert a1["docs"][0]["vector_id"] == 1
    assert a1["docs"][1]["vector_id"] == 2
