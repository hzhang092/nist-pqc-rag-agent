"""
Build and persist a BM25 lexical index from chunk_store.jsonl.

This script creates `data/processed/bm25.pkl`, which is consumed by
`rag.retriever.bm25_retriever.BM25Retriever`.

How to use:
    python -m rag.index_bm25
    python -m rag.index_bm25 --chunk-store data/processed/chunk_store.jsonl --out data/processed/bm25.pkl
    python -m rag.index_bm25 --k1 1.5 --b 0.75

Inputs:
    - chunk store JSONL with deterministic row ordering by `vector_id`

Outputs:
    - BM25 artifact pickle with docs, postings, idf, and tokenizer metadata

Used by:
    - `rag/retriever/bm25_retriever.py` (loads the built artifact at query time)
    - `tests/test_bm25_index.py` (validates tokenizer and artifact behavior)
    - indirectly by hybrid retrieval via `rag.retrieve.hybrid_search`

Usage:
    python -m rag.index_bm25
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


OUT_DIR = Path("data/processed")
STORE_PATH = OUT_DIR / "chunk_store.jsonl"
BM25_PATH = OUT_DIR / "bm25.pkl"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-._][a-z0-9]+)+|[a-z0-9]+")
COMPOUND_RE = re.compile(r"^[a-z0-9]+(?:[-._][a-z0-9]+)+$")


def tokenize(text: str) -> List[str]:
    """Tokenizes text while preserving technical compounds like ML-KEM.KeyGen."""
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)

    expanded: List[str] = []
    for token in tokens:
        expanded.append(token)

        if COMPOUND_RE.match(token):
            for part in re.split(r"[-._]", token):
                if part:
                    expanded.append(part)

    return expanded


def _load_chunk_store(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rows.sort(key=lambda rec: int(rec.get("vector_id", 0)))
    return rows


def build_bm25_artifact(
    chunk_store_path: Path = STORE_PATH,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, Any]:
    """Builds an in-memory BM25 artifact from chunk_store rows."""
    rows = _load_chunk_store(chunk_store_path)
    if not rows:
        raise ValueError(f"No rows found in {chunk_store_path}")

    doc_freq: Dict[str, int] = defaultdict(int)
    postings: Dict[str, List[List[float]]] = defaultdict(list)
    doc_lens: List[int] = []
    doc_records: List[Dict[str, Any]] = []

    for doc_idx, rec in enumerate(rows):
        text = str(rec.get("text", ""))
        tokens = tokenize(text)
        tf = Counter(tokens)

        doc_lens.append(sum(tf.values()))
        doc_records.append(
            {
                "chunk_id": rec.get("chunk_id", ""),
                "doc_id": rec.get("doc_id", ""),
                "start_page": int(rec.get("start_page", 0)),
                "end_page": int(rec.get("end_page", 0)),
                "text": text,
                "vector_id": int(rec.get("vector_id", doc_idx)),
            }
        )

        for term in tf:
            doc_freq[term] += 1
        for term, freq in tf.items():
            postings[term].append([doc_idx, float(freq)])

    n_docs = len(doc_records)
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0

    idf: Dict[str, float] = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    return {
        "version": 1,
        "tokenizer": "regex_compound_v1",
        "params": {"k1": k1, "b": b},
        "n_docs": n_docs,
        "avgdl": avgdl,
        "doc_lens": doc_lens,
        "idf": dict(sorted(idf.items())),
        "postings": {term: postings[term] for term in sorted(postings)},
        "docs": doc_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag.index_bm25")
    parser.add_argument("--chunk-store", type=str, default=str(STORE_PATH))
    parser.add_argument("--out", type=str, default=str(BM25_PATH))
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    store_path = Path(args.chunk_store)
    out_path = Path(args.out)
    if not store_path.exists():
        raise FileNotFoundError(f"Missing chunk store: {store_path}")

    artifact = build_bm25_artifact(chunk_store_path=store_path, k1=args.k1, b=args.b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as outfile:
        pickle.dump(artifact, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[OK] saved {out_path} docs={artifact['n_docs']} "
        f"vocab={len(artifact['idf'])} avgdl={artifact['avgdl']:.2f}"
    )


if __name__ == "__main__":
    main()
