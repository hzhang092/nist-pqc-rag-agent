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
    python -m rag.retrieve "ML-KEM.Decaps" --rerank-pool 40
    python -m rag.retrieve "Algorithm 19" --no-rerank
"""

from __future__ import annotations

import argparse
import re
from time import perf_counter
from typing import Dict, Iterable, List, Literal, Sequence, TYPE_CHECKING, TypedDict

from rag.config import SETTINGS
from rag.text_normalize import normalize_identifier_like_spans
from rag.retriever import factory as retriever_factory
from rag.retriever.base import ChunkHit
from rag.retriever.bm25_retriever import BM25Retriever
from rag.retriever.factory import get_retriever

if TYPE_CHECKING:
    from rag.retriever.faiss_retriever import FaissRetriever


ModeHint = Literal["definition", "algorithm", "compare", "general"]
MODE_HINT_VALUES = ("definition", "algorithm", "compare", "general")

MODE_WEIGHTS: Dict[ModeHint, tuple[float, float, float]] = {
    "definition": (0.35, 0.45, 0.20),  # prior, bm25, anchors
    "algorithm": (0.45, 0.20, 0.35),
    "compare": (0.60, 0.30, 0.10),
    "general": (0.70, 0.20, 0.10),
}

KNOWN_ACRONYM_ANCHORS = {
    "mlwe": "MLWE",
    "lwe": "LWE",
    "sis": "SIS",
    "msis": "MSIS",
    "kdf": "KDF",
}

TECH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-._][A-Za-z0-9]+)+")
ALGORITHM_NUM_RE = re.compile(r"\balgorithm\s+(\d+)\b", flags=re.IGNORECASE)
ACRONYM_RE = re.compile(r"\b[A-Z]{2,8}\b")
COMPARE_TOPICS_RE = re.compile(
    r"(?:compare|comparison\s+of|difference(?:s)?\s+between)\s+"
    r"(?P<a>.+?)\s+(?:and|vs|versus)\s+(?P<b>.+?)(?:\?|$)",
    flags=re.IGNORECASE,
)
DEFINITION_TERM_RE = re.compile(
    r"(?:definition\s+of|define|what\s+is|what(?:'s|\s+is)|what\s+does)\s+"
    r"(?P<term>[A-Za-z0-9][A-Za-z0-9._-]*)",
    flags=re.IGNORECASE,
)
WHAT_DOES_MEAN_RE = re.compile(
    r"what\s+does\s+(?P<term>[A-Za-z0-9][A-Za-z0-9._-]*)\s+mean",
    flags=re.IGNORECASE,
)


class RetrievalStageOutputs(TypedDict):
    final_hits: List[ChunkHit]
    pre_rerank_fused_hits: List[ChunkHit]
    post_rerank_hits: List[ChunkHit]
    rerank_pool: int


class RetrievalTimingSummary(TypedDict):
    retrieve_ms: float
    rerank_ms: float


def _tie_break_key(hit: ChunkHit) -> tuple:
    return (hit.doc_id, hit.start_page, hit.chunk_id)


def _dedupe_preserve_order(items: Iterable[str], *, limit: int | None = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
        if limit is not None and len(out) >= limit:
            break
    return out


def _resolve_mode_hint(mode_hint: str | None, query: str) -> ModeHint:
    if mode_hint in MODE_HINT_VALUES:
        return mode_hint  # type: ignore[return-value]

    normalized_query = normalize_identifier_like_spans(query or "")
    ql = normalized_query.lower()
    if any(x in ql for x in ("compare", "difference between", "differences between", " vs ", " versus ")):
        return "compare"
    if "algorithm" in ql or "shake" in ql or "table" in ql:
        return "algorithm"
    if ql.startswith(("define", "what is", "what's", "what does")) or "stands for" in ql:
        return "definition"
    return "general"


def _extract_compare_topics(query: str) -> tuple[str | None, str | None]:
    match = COMPARE_TOPICS_RE.search(query)
    if not match:
        return None, None
    topic_a = str(match.group("a") or "").strip(" .;,:")
    topic_b = str(match.group("b") or "").strip(" .;,:")
    if not topic_a or not topic_b:
        return None, None
    return topic_a, topic_b


def _extract_definition_term(query: str) -> str | None:
    match = WHAT_DOES_MEAN_RE.search(query)
    if match:
        term = str(match.group("term") or "").strip(" .;,:")
        if term:
            return term
    match = DEFINITION_TERM_RE.search(query)
    if match:
        term = str(match.group("term") or "").strip(" .;,:")
        if term:
            return term
    return None


def query_variants(
    query: str,
    *,
    mode_hint: str | None = None,
    max_variants: int = 4,
    enable_mode_templates: bool = True,
) -> List[str]:
    """Creates deterministic, domain-specific rewrites without using an LLM."""
    original = normalize_identifier_like_spans(query.strip())
    if not original:
        return []
    if max_variants <= 0:
        raise ValueError("max_variants must be > 0")

    resolved_mode = _resolve_mode_hint(mode_hint, original)
    lowered = original.lower()
    variants: List[str] = [original]

    if enable_mode_templates and resolved_mode == "definition":
        term = _extract_definition_term(original)
        if term:
            variants.extend(
                [
                    f"definition of {term}",
                    f"{term} stands for",
                    f"{term} notation",
                ]
            )

    if enable_mode_templates and resolved_mode == "compare":
        topic_a, topic_b = _extract_compare_topics(original)
        if topic_a and topic_b:
            variants.extend(
                [
                    f"{topic_a} intended use-cases",
                    f"{topic_b} intended use-cases",
                    f"{topic_a} vs {topic_b}",
                ]
            )

    if enable_mode_templates and "pqc" in lowered and "standards" in lowered:
        variants.append("FIPS 203 FIPS 204 FIPS 205 ML-KEM ML-DSA SLH-DSA")

    technical_tokens = _dedupe_preserve_order(TECH_TOKEN_RE.findall(original))
    if technical_tokens:
        variants.append(" ".join(technical_tokens))

    if "ml-kem" in lowered and "key generation" in lowered:
        variants.append("ML-KEM.KeyGen key generation")
    if "ml-dsa" in lowered and "signing" in lowered:
        variants.append("ML-DSA.Sign")
    if "ml-dsa" in lowered and "verify" in lowered:
        variants.append("ML-DSA.Verify")
    if "slh-dsa" in lowered and "keygen" in lowered:
        variants.append("SLH-DSA.KeyGen")
    if "ml-kem" in lowered and "decapsulation" in lowered:
        variants.append("ML-KEM.Decaps")

    alg_match = ALGORITHM_NUM_RE.search(original)
    if alg_match:
        variants.append(f"Algorithm {alg_match.group(1)}")
        variants.append(f"Algorithm {alg_match.group(1)} ML-KEM.KeyGen")

    return _dedupe_preserve_order(variants, limit=max_variants)


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


def _query_anchor_tokens(query: str, mode_hint: str | None = None) -> List[str]:
    resolved_mode = _resolve_mode_hint(mode_hint, query)
    normalized_query = normalize_identifier_like_spans(query)
    tokens = [tok.lower() for tok in TECH_TOKEN_RE.findall(normalized_query)]

    if resolved_mode == "definition":
        for acronym in ACRONYM_RE.findall(normalized_query):
            tokens.append(acronym.lower())
        lower_query = normalized_query.lower()
        for k, v in KNOWN_ACRONYM_ANCHORS.items():
            if k in lower_query:
                tokens.append(v.lower())
        term = _extract_definition_term(normalized_query)
        if term:
            tokens.append(term.lower())

    return _dedupe_preserve_order(tokens)


def _anchor_overlap_count(anchor_tokens: List[str], text: str) -> int:
    if not anchor_tokens or not text:
        return 0
    haystack = normalize_identifier_like_spans(text).lower()
    overlap = 0
    for token in anchor_tokens:
        if not token:
            continue
        if re.search(r"[._-]", token):
            if token in haystack:
                overlap += 1
            continue
        if re.search(rf"\b{re.escape(token)}\b", haystack):
            overlap += 1
    return overlap


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _rank_prior_norm(n: int) -> List[float]:
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    return [1.0 - (idx / float(n - 1)) for idx in range(n)]


def _passes_promotion_gate(mode: ModeHint, *, anchor_overlap: int, bm25_norm: float) -> bool:
    if mode == "definition":
        return anchor_overlap >= 1 or bm25_norm >= 0.55
    if mode == "algorithm":
        return anchor_overlap >= 2 or (anchor_overlap >= 1 and bm25_norm >= 0.60)
    if mode == "compare":
        return anchor_overlap >= 1 and bm25_norm >= 0.65
    return anchor_overlap >= 2 or bm25_norm >= 0.70


def rerank_fused_hits(
    query: str,
    hits: List[ChunkHit],
    top_k: int,
    bm25: BM25Retriever,
    mode_hint: str | None = None,
) -> List[ChunkHit]:
    """
    Deterministic do-no-harm rerank:
    - preserve fused order by default;
    - only promote candidates that clear strict mode-aware gates.
    """
    if top_k <= 0:
        return []
    if not hits:
        return []

    resolved_mode = _resolve_mode_hint(mode_hint, query)
    anchor_tokens = _query_anchor_tokens(query, mode_hint=resolved_mode)
    prior_w, bm25_w, anchors_w = MODE_WEIGHTS[resolved_mode]

    prior_scores = [float(hit.score) for hit in hits]
    prior_norm = _minmax_norm(prior_scores)
    if all(v == 0.0 for v in prior_norm):
        prior_norm = _rank_prior_norm(len(hits))

    bm25_scores = [float(bm25.score_text(query, hit.text or "")) for hit in hits]
    bm25_norm = _minmax_norm(bm25_scores)
    anchor_overlap = [_anchor_overlap_count(anchor_tokens, hit.text or "") for hit in hits]
    max_overlap = max(anchor_overlap) if anchor_overlap else 0
    anchor_norm = [
        (float(v) / float(max_overlap)) if max_overlap > 0 else 0.0
        for v in anchor_overlap
    ]

    rows = []
    for idx, hit in enumerate(hits):
        promotion_score = (
            prior_w * prior_norm[idx]
            + bm25_w * bm25_norm[idx]
            + anchors_w * anchor_norm[idx]
        )
        rows.append(
            {
                "hit": hit,
                "idx": idx,
                "anchor_overlap": anchor_overlap[idx],
                "bm25_norm": bm25_norm[idx],
                "promotion_score": float(promotion_score),
                "promote": _passes_promotion_gate(
                    resolved_mode,
                    anchor_overlap=anchor_overlap[idx],
                    bm25_norm=bm25_norm[idx],
                ),
            }
        )

    promoted = [row for row in rows if row["promote"]]
    if not promoted:
        return hits[:top_k]

    promoted_sorted = sorted(
        promoted,
        key=lambda row: (
            -row["promotion_score"],
            row["idx"],
            row["hit"].doc_id,
            row["hit"].start_page,
            row["hit"].chunk_id,
        ),
    )
    max_promoted = min(8, top_k // 2)
    if max_promoted <= 0:
        return hits[:top_k]
    promoted_hits = [row["hit"] for row in promoted_sorted[:max_promoted]]
    promoted_ids = {h.chunk_id for h in promoted_hits}
    remaining_hits = [h for h in hits if h.chunk_id not in promoted_ids]
    reranked = promoted_hits + remaining_hits
    return reranked[:top_k]


def _build_hybrid_stage_outputs(
    query: str,
    top_k: int = SETTINGS.TOP_K,
    candidate_multiplier: int = 4,
    k0: int = 60,
    use_query_fusion: bool = True,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    diagnostic_pre_rerank_depth: int = 60,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
    faiss: "FaissRetriever | None" = None,
    bm25: BM25Retriever | None = None,
    timing_ms: RetrievalTimingSummary | None = None,
) -> RetrievalStageOutputs:
    """Runs hybrid retrieval and returns stage outputs for diagnostics."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be > 0")
    if rerank_pool <= 0:
        raise ValueError("rerank_pool must be > 0")
    if diagnostic_pre_rerank_depth <= 0:
        raise ValueError("diagnostic_pre_rerank_depth must be > 0")

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

    retrieve_started = perf_counter()
    normalized_query = normalize_identifier_like_spans(query)
    resolved_mode = _resolve_mode_hint(mode_hint, normalized_query)
    per_source_k = max(top_k * candidate_multiplier, top_k)
    fused_pool = max(top_k, rerank_pool)
    queries = (
        query_variants(
            normalized_query,
            mode_hint=resolved_mode,
            max_variants=4,
            enable_mode_templates=enable_mode_variants,
        )
        if use_query_fusion
        else [normalized_query]
    )

    rankings: List[List[ChunkHit]] = []
    for q in queries:
        vector_hits = faiss_retriever.search(q, k=per_source_k)
        bm25_hits = bm25_retriever.search(q, k=per_source_k)
        rankings.append(vector_hits)
        rankings.append(bm25_hits)

    pre_rerank_depth = max(fused_pool, diagnostic_pre_rerank_depth)
    pre_rerank_fused = rrf_fuse(rankings, top_k=pre_rerank_depth, k0=k0)
    rerank_input = pre_rerank_fused[:fused_pool]
    retrieve_elapsed_ms = (perf_counter() - retrieve_started) * 1000.0

    rerank_elapsed_ms = 0.0
    if enable_rerank:
        rerank_started = perf_counter()
        post_rerank_hits = rerank_fused_hits(
            query=normalized_query,
            hits=rerank_input,
            top_k=fused_pool,
            bm25=bm25_retriever,
            mode_hint=resolved_mode,
        )
        rerank_elapsed_ms = (perf_counter() - rerank_started) * 1000.0
    else:
        post_rerank_hits = rerank_input

    if timing_ms is not None:
        timing_ms["retrieve_ms"] = retrieve_elapsed_ms
        timing_ms["rerank_ms"] = rerank_elapsed_ms

    return {
        "final_hits": post_rerank_hits[:top_k],
        "pre_rerank_fused_hits": pre_rerank_fused,
        "post_rerank_hits": post_rerank_hits,
        "rerank_pool": fused_pool,
    }


def hybrid_search(
    query: str,
    top_k: int = SETTINGS.TOP_K,
    candidate_multiplier: int = 4,
    k0: int = 60,
    use_query_fusion: bool = True,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
    faiss: "FaissRetriever | None" = None,
    bm25: BM25Retriever | None = None,
) -> List[ChunkHit]:
    """Runs FAISS + BM25 retrieval and fuses the rankings with RRF."""
    outputs = _build_hybrid_stage_outputs(
        query=query,
        top_k=top_k,
        candidate_multiplier=candidate_multiplier,
        k0=k0,
        use_query_fusion=use_query_fusion,
        enable_rerank=enable_rerank,
        rerank_pool=rerank_pool,
        diagnostic_pre_rerank_depth=max(60, rerank_pool, top_k),
        mode_hint=mode_hint,
        enable_mode_variants=enable_mode_variants,
        faiss=faiss,
        bm25=bm25,
    )
    return outputs["final_hits"]


def _build_base_stage_outputs(
    query: str,
    top_k: int = SETTINGS.TOP_K,
    backend: str = SETTINGS.VECTOR_BACKEND,
    candidate_multiplier: int = 4,
    k0: int = 60,
    use_query_fusion: bool = True,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    diagnostic_pre_rerank_depth: int = 60,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
    timing_ms: RetrievalTimingSummary | None = None,
) -> RetrievalStageOutputs:
    """Runs base retrieval and returns stage outputs for diagnostics."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be > 0")
    if rerank_pool <= 0:
        raise ValueError("rerank_pool must be > 0")
    if diagnostic_pre_rerank_depth <= 0:
        raise ValueError("diagnostic_pre_rerank_depth must be > 0")

    retrieve_started = perf_counter()
    normalized_query = normalize_identifier_like_spans(query)
    resolved_mode = _resolve_mode_hint(mode_hint, normalized_query)
    retriever = get_retriever(backend)
    per_query_k = max(top_k * candidate_multiplier, top_k)
    fused_pool = max(top_k, rerank_pool)
    queries = (
        query_variants(
            normalized_query,
            mode_hint=resolved_mode,
            max_variants=4,
            enable_mode_templates=enable_mode_variants,
        )
        if use_query_fusion
        else [normalized_query]
    )

    rankings = [retriever.search(q, k=per_query_k) for q in queries]
    pre_rerank_depth = max(fused_pool, diagnostic_pre_rerank_depth)
    pre_rerank_fused = rrf_fuse(rankings, top_k=pre_rerank_depth, k0=k0)
    rerank_input = pre_rerank_fused[:fused_pool]
    retrieve_elapsed_ms = (perf_counter() - retrieve_started) * 1000.0

    rerank_elapsed_ms = 0.0
    if enable_rerank:
        bm25_retriever = BM25Retriever()
        rerank_started = perf_counter()
        post_rerank_hits = rerank_fused_hits(
            query=normalized_query,
            hits=rerank_input,
            top_k=fused_pool,
            bm25=bm25_retriever,
            mode_hint=resolved_mode,
        )
        rerank_elapsed_ms = (perf_counter() - rerank_started) * 1000.0
    else:
        post_rerank_hits = rerank_input

    if timing_ms is not None:
        timing_ms["retrieve_ms"] = retrieve_elapsed_ms
        timing_ms["rerank_ms"] = rerank_elapsed_ms

    return {
        "final_hits": post_rerank_hits[:top_k],
        "pre_rerank_fused_hits": pre_rerank_fused,
        "post_rerank_hits": post_rerank_hits,
        "rerank_pool": fused_pool,
    }


def base_search(
    query: str,
    top_k: int = SETTINGS.TOP_K,
    backend: str = SETTINGS.VECTOR_BACKEND,
    candidate_multiplier: int = 4,
    k0: int = 60,
    use_query_fusion: bool = True,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
) -> List[ChunkHit]:
    """Runs single-backend retrieval; optionally fuses deterministic query variants."""
    outputs = _build_base_stage_outputs(
        query=query,
        top_k=top_k,
        backend=backend,
        candidate_multiplier=candidate_multiplier,
        k0=k0,
        use_query_fusion=use_query_fusion,
        enable_rerank=enable_rerank,
        rerank_pool=rerank_pool,
        diagnostic_pre_rerank_depth=max(60, rerank_pool, top_k),
        mode_hint=mode_hint,
        enable_mode_variants=enable_mode_variants,
    )
    return outputs["final_hits"]


def retrieve(
    query: str,
    k: int = SETTINGS.TOP_K,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    backend: str = SETTINGS.VECTOR_BACKEND,
    use_query_fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
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
            enable_rerank=enable_rerank,
            rerank_pool=rerank_pool,
            mode_hint=mode_hint,
            enable_mode_variants=enable_mode_variants,
        )
    if selected_mode == "base":
        return base_search(
            query=query,
            top_k=k,
            backend=backend,
            candidate_multiplier=candidate_multiplier,
            k0=k0,
            use_query_fusion=use_query_fusion,
            enable_rerank=enable_rerank,
            rerank_pool=rerank_pool,
            mode_hint=mode_hint,
            enable_mode_variants=enable_mode_variants,
        )
    raise ValueError(f"Unknown retrieval mode: {mode}")


def retrieve_with_stages(
    query: str,
    k: int = SETTINGS.TOP_K,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    backend: str = SETTINGS.VECTOR_BACKEND,
    use_query_fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    diagnostic_pre_rerank_depth: int = 60,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
) -> RetrievalStageOutputs:
    """Retrieval entrypoint that also returns stage outputs for evaluation diagnostics."""
    selected_mode = mode.lower().strip()
    if selected_mode == "hybrid":
        return _build_hybrid_stage_outputs(
            query=query,
            top_k=k,
            candidate_multiplier=candidate_multiplier,
            k0=k0,
            use_query_fusion=use_query_fusion,
            enable_rerank=enable_rerank,
            rerank_pool=rerank_pool,
            diagnostic_pre_rerank_depth=diagnostic_pre_rerank_depth,
            mode_hint=mode_hint,
            enable_mode_variants=enable_mode_variants,
        )
    if selected_mode == "base":
        return _build_base_stage_outputs(
            query=query,
            top_k=k,
            backend=backend,
            candidate_multiplier=candidate_multiplier,
            k0=k0,
            use_query_fusion=use_query_fusion,
            enable_rerank=enable_rerank,
            rerank_pool=rerank_pool,
            diagnostic_pre_rerank_depth=diagnostic_pre_rerank_depth,
            mode_hint=mode_hint,
            enable_mode_variants=enable_mode_variants,
        )
    raise ValueError(f"Unknown retrieval mode: {mode}")


def retrieve_with_stages_and_timing(
    query: str,
    k: int = SETTINGS.TOP_K,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    backend: str = SETTINGS.VECTOR_BACKEND,
    use_query_fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    diagnostic_pre_rerank_depth: int = 60,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
) -> tuple[RetrievalStageOutputs, RetrievalTimingSummary]:
    timing_ms: RetrievalTimingSummary = {
        "retrieve_ms": 0.0,
        "rerank_ms": 0.0,
    }

    selected_mode = mode.lower().strip()
    if selected_mode == "hybrid":
        outputs = _build_hybrid_stage_outputs(
            query=query,
            top_k=k,
            candidate_multiplier=candidate_multiplier,
            k0=k0,
            use_query_fusion=use_query_fusion,
            enable_rerank=enable_rerank,
            rerank_pool=rerank_pool,
            diagnostic_pre_rerank_depth=diagnostic_pre_rerank_depth,
            mode_hint=mode_hint,
            enable_mode_variants=enable_mode_variants,
            timing_ms=timing_ms,
        )
        return outputs, timing_ms
    if selected_mode == "base":
        outputs = _build_base_stage_outputs(
            query=query,
            top_k=k,
            backend=backend,
            candidate_multiplier=candidate_multiplier,
            k0=k0,
            use_query_fusion=use_query_fusion,
            enable_rerank=enable_rerank,
            rerank_pool=rerank_pool,
            diagnostic_pre_rerank_depth=diagnostic_pre_rerank_depth,
            mode_hint=mode_hint,
            enable_mode_variants=enable_mode_variants,
            timing_ms=timing_ms,
        )
        return outputs, timing_ms
    raise ValueError(f"Unknown retrieval mode: {mode}")


def _hits_to_eval_rows(hits: List[ChunkHit], *, mode: str) -> List[dict]:
    rows: List[dict] = []
    for rank, hit in enumerate(hits, start=1):
        rows.append(
            {
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "start_page": hit.start_page,
                "end_page": hit.end_page,
                "score": float(hit.score),
                "text": hit.text,
                "rank": rank,
                "mode": mode,
            }
        )
    return rows


def retrieve_for_eval_with_stages(
    query: str,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    k: int = SETTINGS.TOP_K,
    backend: str = SETTINGS.VECTOR_BACKEND,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    evidence_window: bool = False,
    cheap_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    debug: bool = False,
    diagnostic_pre_rerank_depth: int = 60,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
) -> dict:
    """Eval adapter that returns final hits plus stage-level diagnostic hit lists."""
    _ = evidence_window
    _ = debug

    stages = retrieve_with_stages(
        query=query,
        k=k,
        mode=mode,
        backend=backend,
        use_query_fusion=fusion,
        candidate_multiplier=candidate_multiplier,
        k0=k0,
        enable_rerank=cheap_rerank,
        rerank_pool=rerank_pool,
        diagnostic_pre_rerank_depth=diagnostic_pre_rerank_depth,
        mode_hint=mode_hint,
        enable_mode_variants=enable_mode_variants,
    )

    return {
        "hits": _hits_to_eval_rows(stages["final_hits"], mode=mode),
        "pre_rerank_fused_hits": _hits_to_eval_rows(stages["pre_rerank_fused_hits"], mode=mode),
        "post_rerank_hits": _hits_to_eval_rows(stages["post_rerank_hits"], mode=mode),
        "rerank_pool": int(stages["rerank_pool"]),
    }


def retrieve_for_eval(
    query: str,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    k: int = SETTINGS.TOP_K,
    backend: str = SETTINGS.VECTOR_BACKEND,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    evidence_window: bool = False,
    cheap_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    debug: bool = False,
    mode_hint: str | None = None,
    enable_mode_variants: bool = True,
) -> List[dict]:
    """
    Eval-friendly retrieval adapter with a stable row schema.

    Returns a deterministic list of dictionaries so evaluation code can stay
    decoupled from `ChunkHit` internals.
    """
    _ = evidence_window
    _ = debug

    stage_rows = retrieve_for_eval_with_stages(
        query=query,
        mode=mode,
        k=k,
        backend=backend,
        k0=k0,
        candidate_multiplier=candidate_multiplier,
        fusion=fusion,
        evidence_window=evidence_window,
        cheap_rerank=cheap_rerank,
        rerank_pool=rerank_pool,
        debug=debug,
        diagnostic_pre_rerank_depth=max(60, rerank_pool, k),
        mode_hint=mode_hint,
        enable_mode_variants=enable_mode_variants,
    )
    return stage_rows["hits"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag.retrieve")
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
    parser.add_argument(
        "--mode-hint",
        type=str,
        default=None,
        choices=list(MODE_HINT_VALUES),
        help="Optional intent hint used for rerank weighting and query variants.",
    )
    parser.add_argument(
        "--no-mode-variants",
        action="store_true",
        help="Disable mode-aware variant templates and keep legacy-style variants only.",
    )
    parser.add_argument("--no-query-fusion", action="store_true", help="Disable deterministic query rewrites.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable lightweight lexical reranking.")
    args = parser.parse_args()

    if args.mode == "base":
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
        enable_rerank=not args.no_rerank,
        rerank_pool=args.rerank_pool,
        mode_hint=args.mode_hint,
        enable_mode_variants=not args.no_mode_variants,
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
