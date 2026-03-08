"""
Command-line interface for performing retrieval over the document chunks.

This script provides a simple way to query the RAG system from the command line.
It uses the shared retrieval service (`rag.service.search_query`) to perform
searches with configurable retrieval modes and parameters.

How to use:
    python -m rag.search "ML-KEM key generation"
    python -m rag.search "Algorithm 19" --mode hybrid --k 8
    python -m rag.search "ML-DSA verification" --mode base --backend bm25
    python -m rag.search "SLH-DSA parameters" --no-query-fusion
    python -m rag.search "ML-KEM.Decaps" --candidate-multiplier 6 --k0 55
    python -m rag.search "Algorithm 19" --no-rerank --rerank-pool 50

Flags:
    --k
        Final number of results returned (default from SETTINGS.TOP_K).
    --mode
        Retrieval mode: "base" for single backend, "hybrid" for FAISS+BM25 fusion
        (default from SETTINGS.RETRIEVAL_MODE).
    --backend
        Backend used by base mode, e.g., faiss, bm25
        (default from SETTINGS.VECTOR_BACKEND).
    --k0
        Reciprocal Rank Fusion constant. Lower values emphasize top ranks more
        (default from SETTINGS.RETRIEVAL_RRF_K0).
    --candidate-multiplier
        Expands per-source candidate pool before fusion (k * multiplier)
        (default from SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER).
    --rerank-pool
        Number of fused candidates considered before final rerank truncation
        (default from SETTINGS.RETRIEVAL_RERANK_POOL).
    --no-query-fusion
        Disable deterministic query rewrites before retrieval.
    --no-rerank
        Disable lightweight lexical reranking step.

Used by:
    - Direct CLI entrypoint for developers.
    - Shares retrieval logic with other modules through `rag.service.search_query`.

Output:
    Prints query results with score, doc_id, page range, chunk_id, and preview text.

Usage:
    python -m rag.search "your question here"
"""

import argparse

from rag.config import SETTINGS, validate_settings
from rag.service import search_query

def main():
    """
    Parses command-line arguments, runs a search, and prints the results.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.search")
    parser.add_argument("query", nargs="+", help="Question text (wrap in quotes recommended).")
    parser.add_argument("--k", type=int, default=SETTINGS.TOP_K, help="Final number of results to return.")
    parser.add_argument(
        "--mode",
        type=str,
        default=SETTINGS.RETRIEVAL_MODE,
        choices=["base", "hybrid"],
        help="Retrieval mode: base backend or hybrid (faiss+bm25).",
    )
    parser.add_argument("--backend", type=str, default=SETTINGS.VECTOR_BACKEND, help="Backend for base mode.")
    parser.add_argument("--k0", type=int, default=SETTINGS.RETRIEVAL_RRF_K0, help="RRF constant (1/(k0+rank)).")
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
        help="Per-source candidate expansion before fusion (k * multiplier).",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=SETTINGS.RETRIEVAL_RERANK_POOL,
        help="Number of fused candidates considered before final rerank truncation.",
    )
    parser.add_argument("--no-query-fusion", action="store_true", help="Disable deterministic query rewrites.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable lightweight lexical reranking.")
    args = parser.parse_args()

    validate_settings()

    qtext = " ".join(args.query).strip()
    if not qtext:
        print('Usage: python -m rag.search "your question"')
        raise SystemExit(1)

    payload = search_query(
        query=qtext,
        k=args.k,
        mode=args.mode,
        backend=args.backend,
        use_query_fusion=not args.no_query_fusion,
        candidate_multiplier=args.candidate_multiplier,
        k0=args.k0,
        enable_rerank=not args.no_rerank,
        rerank_pool=args.rerank_pool,
    )

    print(f"\nQuery: {qtext}\n")
    for i, hit in enumerate(payload["hits"], start=1):
        print(
            f"[{i}] score={hit['score']:.4f}  {hit['doc_id']}  "
            f"p{hit['start_page']}-p{hit['end_page']}  ({hit['chunk_id']})"
        )
        preview = hit["preview_text"]
        if preview:
            print(f"    {preview}")
        print()

if __name__ == "__main__":
    main()
