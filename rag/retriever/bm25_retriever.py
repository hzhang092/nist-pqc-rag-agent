from __future__ import annotations

"""BM25 retriever implementation over the persisted lexical index.

How to use (Python):
    from rag.retriever.bm25_retriever import BM25Retriever

    retriever = BM25Retriever()  # expects data/processed/bm25.pkl
    hits = retriever.search("ML-KEM.KeyGen", k=5)

Prerequisite:
    Build the artifact first with:
        python -m rag.index_bm25

Used by:
    - `rag.retrieve.hybrid_search` (lexical leg in hybrid retrieval)
    - `rag.retriever.factory.get_retriever` when backend=`bm25`
"""

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rag.index_bm25 import tokenize

from .base import ChunkHit


class BM25Retriever:
    def __init__(self, bm25_path: str = "data/processed/bm25.pkl"):
        self.bm25_path = Path(bm25_path)
        if not self.bm25_path.exists():
            raise FileNotFoundError(
                f"Missing BM25 artifact: {self.bm25_path}. Build it with `python -m rag.index_bm25`."
            )

        with self.bm25_path.open("rb") as infile:
            artifact = pickle.load(infile)

        self.k1 = float(artifact["params"]["k1"])
        self.b = float(artifact["params"]["b"])
        self.avgdl = float(artifact["avgdl"])
        self.doc_lens = artifact["doc_lens"]
        self.idf: Dict[str, float] = artifact["idf"]
        self.postings = artifact["postings"]
        self.docs = artifact["docs"]

    def search(self, query: str, k: int = 5) -> List[ChunkHit]:
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores = defaultdict(float)
        qtf = defaultdict(int)
        for term in q_terms:
            qtf[term] += 1

        for term, q_weight in qtf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue

            for doc_idx, tf in self.postings.get(term, []):
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-9))
                scores[int(doc_idx)] += term_score * float(q_weight)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        hits: List[ChunkHit] = []
        for doc_idx, score in ranked:
            rec = self.docs[doc_idx]
            hits.append(
                ChunkHit(
                    score=float(score),
                    chunk_id=rec.get("chunk_id", ""),
                    doc_id=rec.get("doc_id", ""),
                    start_page=int(rec.get("start_page", 0)),
                    end_page=int(rec.get("end_page", 0)),
                    text=rec.get("text", ""),
                )
            )

        return hits
