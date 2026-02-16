"""
Command-line interface for performing a vector search over the document chunks.

This script provides a simple way to query the RAG system from the command line.
It initializes a retriever based on the configured backend (e.g., FAISS, LanceDB),
takes a query string from the command-line arguments, and prints the top K
most relevant document chunks.

Usage:
    python -m rag.search "your question here"
"""

# rag/search.py 
# backend-agnostic
import sys
from rag.retriever.factory import get_retriever
from rag.config import SETTINGS, validate_settings

TOP_K = SETTINGS.TOP_K
BACKEND = SETTINGS.VECTOR_BACKEND

def main():
    """
    Parses command-line arguments, runs a search, and prints the results.
    """
    validate_settings()
    qtext = " ".join(sys.argv[1:]).strip()
    if not qtext:
        print('Usage: python -m rag.search "your question"')
        sys.exit(1)

    retriever = get_retriever(BACKEND)
    hits = retriever.search(qtext, k=TOP_K)

    print(f"\nQuery: {qtext}\n")
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] score={h.score:.4f}  {h.doc_id}  p{h.start_page}-p{h.end_page}  ({h.chunk_id})")
        if h.text:
            preview = h.text[:300].replace("\n", " ")
            print(f"    {preview}...")
        print()

if __name__ == "__main__":
    main()
