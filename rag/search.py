"""
Command-line interface for performing retrieval over the document chunks.

This script provides a simple way to query the RAG system from the command line.
It uses the shared retrieval entrypoint (`rag.retrieve.retrieve`) so search and
ask stay aligned on behavior.

How to use:
    python -m rag.search "ML-KEM key generation"
    python -m rag.search "Algorithm 19" --mode hybrid --k 8
    python -m rag.search "ML-DSA verification" --mode base --backend bm25
    python -m rag.search "SLH-DSA parameters" --no-query-fusion

Used by:
    - Direct CLI entrypoint for developers (not imported by other project modules).
    - Shares retrieval logic with `rag.ask` through `rag.retrieve.retrieve`.

Usage:
    python -m rag.search "your question here"
"""

import argparse

from rag.config import SETTINGS, validate_settings
from rag.retrieve import retrieve

def main():
    """
    Parses command-line arguments, runs a search, and prints the results.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.search")
    parser.add_argument("query", nargs="+", help="Question text (wrap in quotes recommended).")
    parser.add_argument("--k", type=int, default=SETTINGS.TOP_K)
    parser.add_argument("--mode", type=str, default=SETTINGS.RETRIEVAL_MODE, choices=["base", "hybrid"])
    parser.add_argument("--backend", type=str, default=SETTINGS.VECTOR_BACKEND)
    parser.add_argument("--no-query-fusion", action="store_true")
    args = parser.parse_args()

    validate_settings()

    qtext = " ".join(args.query).strip()
    if not qtext:
        print('Usage: python -m rag.search "your question"')
        raise SystemExit(1)

    hits = retrieve(
        query=qtext,
        k=args.k,
        mode=args.mode,
        backend=args.backend,
        use_query_fusion=not args.no_query_fusion,
    )

    print(f"\nQuery: {qtext}\n")
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] score={h.score:.4f}  {h.doc_id}  p{h.start_page}-p{h.end_page}  ({h.chunk_id})")
        if h.text:
            preview = h.text[:300].replace("\n", " ")
            print(f"    {preview}...")
        print()

if __name__ == "__main__":
    main()
