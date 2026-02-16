# rag/retriever/factory.py
from __future__ import annotations

from .base import Retriever
from .faiss_retriever import FaissRetriever

def get_retriever(backend: str = "faiss") -> Retriever:
    backend = backend.lower().strip()
    if backend == "faiss":
        return FaissRetriever()
    raise ValueError(f"Unknown backend: {backend}")
