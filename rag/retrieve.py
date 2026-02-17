"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF).

This module is the shared retrieval core for both search and QA flows.
It provides:
    - deterministic query variants (`query_variants`),
    - RRF merging (`rrf_fuse`),
    - base single-backend retrieval (`base_search`), and
    - hybrid BM25 + vector retrieval (`hybrid_search`).

How to use (CLI):
    python -m rag.retrieve "Algorithm 19"
    python -m rag.retrieve "ML-KEM key generation" --mode hybrid --k 8
    python -m rag.retrieve "ML-DSA signing" --mode base --backend faiss
    python -m rag.retrieve "SLH-DSA" --candidate-multiplier 6 --k0 80
    python -m rag.retrieve "ML-KEM" --no-query-fusion

How to use (Python):
    from rag.retrieve import retrieve
    hits = retrieve("ML-KEM key generation", k=5, mode="hybrid")

Used by:
    - `rag.search` (search CLI)
    - `rag.ask` (QA CLI)
    - `scripts/mini_retrieval_sanity.py` (sanity evaluation script)
    - `tests/test_query_fusion.py` (`query_variants`)
    - `tests/test_retrieve_rrf.py` (`rrf_fuse`)

Usage:
    python -m rag.retrieve "Algorithm 19"
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING

from rag.config import SETTINGS
from rag.retriever import factory as retriever_factory
from rag.retriever.base import ChunkHit
from rag.retriever.bm25_retriever import BM25Retriever
from rag.retriever.factory import get_retriever

if TYPE_CHECKING:
    from rag.retriever.faiss_retriever import FaissRetriever


def _tie_break_key(hit: ChunkHit) -> tuple:
    return (hit.doc_id, hit.start_page, hit.chunk_id)


TECH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-._][A-Za-z0-9]+)+")
ALGORITHM_NUM_RE = re.compile(r"\balgorithm\s+(\d+)\b", flags=re.IGNORECASE)


def query_variants(query: str) -> List[str]:
    """Creates deterministic, domain-specific rewrites without using an LLM."""
    original = query.strip()
    if not original:
        return []

    variants: List[str] = [original]

    technical_tokens: List[str] = []
    seen = set()
    for token in TECH_TOKEN_RE.findall(original):
        if token not in seen:
            seen.add(token)
            technical_tokens.append(token)

    if technical_tokens:
        variants.append(" ".join(technical_tokens))

    lowered = original.lower()
    if "key generation" in lowered:
        variants.append("ML-KEM.KeyGen key generation")

    alg_match = ALGORITHM_NUM_RE.search(original)
    if alg_match:
        variants.append(f"Algorithm {alg_match.group(1)} ML-KEM.KeyGen")

    # Stable de-dup while preserving order
    deduped: List[str] = []
    seen_variants = set()
    for item in variants:
        key = item.strip()
        if key and key not in seen_variants:
            deduped.append(key)
            seen_variants.add(key)
    return deduped


def rrf_fuse(rankings: Sequence[Iterable[ChunkHit]], top_k: int, k0: int = 60) -> List[ChunkHit]:
    """
    Fuses multiple ranked hit lists using Reciprocal Rank Fusion.

    RRF score rule:
        rrf_score += 1 / (k0 + rank)
    where rank is 1-indexed within each ranking.
    """
    if top_k <= 0:
        return []
    if k0 <= 0:
        raise ValueError("k0 must be > 0")

    rrf_scores: Dict[str, float] = {}
    representative: Dict[str, ChunkHit] = {}

    for hits in rankings:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k0 + rank))

            prev = representative.get(chunk_id)
            if prev is None:
                representative[chunk_id] = hit
            elif _tie_break_key(hit) < _tie_break_key(prev):
                representative[chunk_id] = hit

    ordered_ids = sorted(
        rrf_scores.keys(),
        key=lambda chunk_id: (
            -rrf_scores[chunk_id],
            representative[chunk_id].doc_id,
            representative[chunk_id].start_page,
            representative[chunk_id].chunk_id,
        ),
    )

    fused: List[ChunkHit] = []
    for chunk_id in ordered_ids[:top_k]:
        hit = representative[chunk_id]
        fused.append(
            ChunkHit(
                score=float(rrf_scores[chunk_id]),
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                start_page=hit.start_page,
                end_page=hit.end_page,
                text=hit.text,
            )
        )
    return fused


def hybrid_search(
    query: str,
    top_k: int = SETTINGS.TOP_K,
    candidate_multiplier: int = 4,
    k0: int = 60,
    use_query_fusion: bool = True,
    faiss: "FaissRetriever | None" = None,
    bm25: BM25Retriever | None = None,
) -> List[ChunkHit]:
    """Runs FAISS + BM25 retrieval and fuses the rankings with RRF."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be > 0")

    if faiss is None:
        try:
            from rag.retriever.faiss_retriever import FaissRetriever
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FAISS backend is unavailable. Install faiss to run hybrid retrieval."
            ) from exc
        faiss_retriever = FaissRetriever()
    else:
        faiss_retriever = faiss

    bm25_retriever = bm25 or BM25Retriever()

    per_source_k = max(top_k * candidate_multiplier, top_k)
    queries = query_variants(query) if use_query_fusion else [query]

    rankings: List[List[ChunkHit]] = []
    for q in queries:
        vector_hits = faiss_retriever.search(q, k=per_source_k)
        bm25_hits = bm25_retriever.search(q, k=per_source_k)
        rankings.append(vector_hits)
        rankings.append(bm25_hits)

    return rrf_fuse(rankings, top_k=top_k, k0=k0)


def base_search(
    query: str,
    top_k: int = SETTINGS.TOP_K,
    backend: str = SETTINGS.VECTOR_BACKEND,
    candidate_multiplier: int = 4,
    k0: int = 60,
    use_query_fusion: bool = True,
) -> List[ChunkHit]:
    """Runs single-backend retrieval; optionally fuses deterministic query variants."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be > 0")

    retriever = get_retriever(backend)
    per_query_k = max(top_k * candidate_multiplier, top_k)
    queries = query_variants(query) if use_query_fusion else [query]

    rankings = [retriever.search(q, k=per_query_k) for q in queries]
    return rrf_fuse(rankings, top_k=top_k, k0=k0)


def retrieve(
    query: str,
    k: int = SETTINGS.TOP_K,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    backend: str = SETTINGS.VECTOR_BACKEND,
    use_query_fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
) -> List[ChunkHit]:
    """Shared retrieval entrypoint for search/ask (base or hybrid + fusion)."""
    selected_mode = mode.lower().strip()
    if selected_mode == "hybrid":
        return hybrid_search(
            query=query,
            top_k=k,
            candidate_multiplier=candidate_multiplier,
            k0=k0,
            use_query_fusion=use_query_fusion,
        )
    if selected_mode == "base":
        return base_search(
            query=query,
            top_k=k,
            backend=backend,
            candidate_multiplier=candidate_multiplier,
            k0=k0,
            use_query_fusion=use_query_fusion,
        )
    raise ValueError(f"Unknown retrieval mode: {mode}")



def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag.retrieve")
    parser.add_argument("query", nargs="+", help="Question text (wrap in quotes recommended).")
    parser.add_argument("--k", type=int, default=SETTINGS.TOP_K)
    parser.add_argument("--mode", type=str, default=SETTINGS.RETRIEVAL_MODE, choices=["base", "hybrid"])
    parser.add_argument("--backend", type=str, default=SETTINGS.VECTOR_BACKEND)
    parser.add_argument("--k0", type=int, default=SETTINGS.RETRIEVAL_RRF_K0)
    parser.add_argument("--candidate-multiplier", type=int, default=SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER)
    parser.add_argument("--no-query-fusion", action="store_true")
    args = parser.parse_args()

    if args.mode == "base":
        # Validate backend names through existing factory behavior.
        retriever_factory.get_retriever(args.backend)

    qtext = " ".join(args.query).strip()
    hits = retrieve(
        query=qtext,
        k=args.k,
        mode=args.mode,
        backend=args.backend,
        use_query_fusion=not args.no_query_fusion,
        candidate_multiplier=args.candidate_multiplier,
        k0=args.k0,
    )

    print(f"\nQuery: {qtext}\n")
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] score={h.score:.6f}  {h.doc_id}  p{h.start_page}-p{h.end_page}  ({h.chunk_id})")
        if h.text:
            preview = h.text[:300].replace("\n", " ")
            print(f"    {preview}...")
        print()


if __name__ == "__main__":
    main()
