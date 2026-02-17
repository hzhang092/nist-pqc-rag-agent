# rag/retriever/factory.py
from __future__ import annotations

"""Retriever backend factory.

How to use:
    from rag.retriever.factory import get_retriever

    retriever = get_retriever("faiss")
    hits = retriever.search("ML-DSA", k=5)

Supported backends:
    - "faiss"
    - "bm25"

Used by:
    - `rag.retrieve.base_search` (single backend retrieval path)
    - `rag.retrieve.main` (CLI backend validation)
"""

from .base import Retriever
from .bm25_retriever import BM25Retriever
from .faiss_retriever import FaissRetriever

def get_retriever(backend: str = "faiss") -> Retriever:
    """Returns a configured retriever instance for a backend name."""
    backend = backend.lower().strip()
    if backend == "faiss":
        return FaissRetriever()
    if backend == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unknown backend: {backend}")
