# rag/retriever/faiss_retriever.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import ChunkHit

class FaissRetriever:
    def __init__(
        self,
        processed_dir: str = "data/processed",
        ef_candidates: int = 25,
        max_hits_per_page: int = 1,
    ):
        self.out_dir = Path(processed_dir)
        self.meta_path = self.out_dir / "emb_meta.json"
        self.index_path = self.out_dir / "faiss.index"
        self.store_path = self.out_dir / "chunk_store.jsonl"

        self.candidates_k = ef_candidates
        self.max_hits_per_page = max_hits_per_page

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.model_name = meta["model_name"]
        self.dim = int(meta["dim"])

        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(str(self.index_path))
        self.store = self._load_store()

    def _load_store(self) -> Dict[int, dict]:
        store = {}
        with self.store_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                store[int(rec["vector_id"])] = rec
        return store

    @staticmethod
    def _page_key(rec: dict) -> Tuple[str, int, int]:
        return (rec.get("doc_id"), int(rec.get("start_page")), int(rec.get("end_page")))

    def search(self, query: str, k: int = 5) -> List[ChunkHit]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        assert q.shape[1] == self.dim

        scores, ids = self.index.search(q, max(k, self.candidates_k))

        kept: List[ChunkHit] = []
        page_counts = {}

        for vid, score in zip(ids[0], scores[0]):
            if vid < 0:
                continue
            rec = self.store.get(int(vid))
            if not rec:
                continue

            key = self._page_key(rec)
            page_counts[key] = page_counts.get(key, 0) + 1
            if page_counts[key] > self.max_hits_per_page:
                continue

            kept.append(
                ChunkHit(
                    score=float(score),
                    chunk_id=rec["chunk_id"],
                    doc_id=rec["doc_id"],
                    start_page=int(rec["start_page"]),
                    end_page=int(rec["end_page"]),
                    text=rec.get("text", ""),   # make sure your chunk_store includes text
                )
            )
            if len(kept) >= k:
                break

        return kept
