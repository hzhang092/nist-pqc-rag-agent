"""
LangChain tool layer for bounded, citation-grounded retrieval in the RAG agent.

Tools
-----
- retrieve
    - Purpose: Return evidence chunks for a query via repo-configured retrieval/fusion.
    - Key args:
        - query (str)
        - k (int, default=8)
        - doc_id (Optional[str]) and/or doc_ids (Optional[List[str]]) for doc filtering
        - mode_hint (Optional[str]); inferred automatically if omitted
        - use_query_fusion (bool, default=True)
        - enable_mode_variants (bool, default=True)
    - Returns: JSON-serializable dict with `evidence` and per-item citation fields:
        `doc_id`, `start_page`, `end_page`, `chunk_id`, `text`, `score`.

- resolve_definition
    - Purpose: Force definition/notation-oriented retrieval for a term/symbol.
    - Key args:
        - term (str)
        - k (int, default=8)
        - doc_id/doc_ids filters
        - optional query override
        - use_query_fusion, enable_mode_variants
    - Returns: Same evidence schema as `retrieve`.

- compare
    - Purpose: Retrieve evidence for two topics, then merge and deduplicate by `chunk_id`.
    - Key args:
        - topic_a (str), topic_b (str)
        - k (int, default=6) per topic
        - doc_ids filter
        - use_query_fusion, enable_mode_variants
    - Returns: Merged evidence plus counts (`n_a`, `n_b`, `n_merged`).

- summarize
    - Purpose: Deterministically collect chunks overlapping a page range for grounded summary generation.
    - Key args:
        - doc_id (str), start_page (int), end_page (int), k (int, default=30)
    - Behavior:
        - Reads `data/processed/chunks.jsonl`
        - Selects overlapping page spans
        - Stable sort by `(start_page, chunk_id)`
    - Returns: Evidence items with required citation fields.

Helper behavior
---------------
- `_load_chunks_meta`: Lazy-loads/caches `chunks.jsonl` metadata.
- `_chunks_for_doc_pages`: Page-overlap selection with deterministic ordering.
- `_find_retrieve_entrypoint`: Locates retrieval function from known module/function candidates.
- `_call_with_flexible_signature`: Passes only supported kwargs to retrieval entrypoint.
- `_normalize_hit`: Converts dict/object hits into `EvidenceItem`.
- `_merge_doc_ids`: Merges `doc_id` + `doc_ids`, removes empties/duplicates, preserves order.
- `_run_retrieve`: Adapter for retrieval call + normalization.
- `_dedupe_evidence`: Deduplicates evidence by `chunk_id`.
- `_mode_hint_from_query`: Lightweight heuristic mode inference.

Notes
-----
- This module is adapter-oriented so retrieval backends remain swappable.
- Citation integrity is preserved by carrying page-level fields on every evidence item.
"""

# rag/lc/tools.py
from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from .state import EvidenceItem


DATA_DIR = Path("data/processed")
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"


# ---------------------------
# Helpers: load chunk metadata (for summarize tool)
# ---------------------------

_CHUNKS_META_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_chunks_meta() -> List[Dict[str, Any]]:
    global _CHUNKS_META_CACHE
    if _CHUNKS_META_CACHE is not None:
        return _CHUNKS_META_CACHE

    if not CHUNKS_JSONL.exists():
        _CHUNKS_META_CACHE = []
        return _CHUNKS_META_CACHE

    rows: List[Dict[str, Any]] = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    _CHUNKS_META_CACHE = rows
    return rows


def _chunks_for_doc_pages(doc_id: str, start_page: int, end_page: int) -> List[Dict[str, Any]]:
    rows = _load_chunks_meta()
    out = []
    for r in rows:
        if r.get("doc_id") != doc_id:
            continue
        sp = int(r.get("start_page", -1))
        ep = int(r.get("end_page", -1))
        if sp <= end_page and ep >= start_page:  # overlap
            out.append(r)
    # stable sort = determinism
    out.sort(key=lambda x: (int(x.get("start_page", 10**9)), str(x.get("chunk_id", ""))))
    return out


# ---------------------------
# Helpers: call your retrieval entrypoint (adapter)
# ---------------------------

def _call_with_flexible_signature(fn, **kwargs):
    """Call fn but only pass parameters it actually accepts."""
    sig = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return fn(**{k: v for k, v in kwargs.items() if v is not None})
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return fn(**allowed)


def _find_retrieve_entrypoint():
    """
    Try a few likely retrieval entrypoints in your repo.
    Edit the candidates list if your function has a different name.
    """
    candidates: List[Tuple[str, str]] = [
        ("rag.retrieve", "retrieve"),
        ("rag.retrieve", "hybrid_retrieve"),
        ("rag.fusion", "fused_retrieve"),
        ("rag.fusion", "retrieve_fusion"),
    ]
    last_err = None
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            return fn
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not find retrieval entrypoint. "
        "Update candidates in rag/lc/tools.py::_find_retrieve_entrypoint(). "
        f"Last error: {last_err}"
    )


def _find_planner_retrieve_entrypoint():
    candidates: List[Tuple[str, str]] = [
        ("rag.retrieve", "execute_query_plan"),
    ]
    last_err = None
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            return fn
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not find planner retrieval entrypoint. "
        "Update candidates in rag/lc/tools.py::_find_planner_retrieve_entrypoint(). "
        f"Last error: {last_err}"
    )


def _normalize_hit(hit: Any) -> EvidenceItem:
    """
    Convert your ChunkHit-like object/dict to EvidenceItem.
    Supports:
      - dataclass/object with attributes
      - dict with keys
    """
    if isinstance(hit, dict):
        return EvidenceItem(
            score=float(hit["score"]),
            chunk_id=str(hit["chunk_id"]),
            doc_id=str(hit["doc_id"]),
            start_page=int(hit["start_page"]),
            end_page=int(hit["end_page"]),
            text=str(hit["text"]),
        )
    # attribute style
    return EvidenceItem(
        score=float(getattr(hit, "score")),
        chunk_id=str(getattr(hit, "chunk_id")),
        doc_id=str(getattr(hit, "doc_id")),
        start_page=int(getattr(hit, "start_page")),
        end_page=int(getattr(hit, "end_page")),
        text=str(getattr(hit, "text")),
    )


def _merge_doc_ids(doc_ids: Optional[List[str]], doc_id: Optional[str]) -> Optional[List[str]]:
    merged = []
    if doc_id:
        merged.append(str(doc_id))
    merged.extend([str(item) for item in (doc_ids or []) if str(item or "").strip()])
    if not merged:
        return None
    seen = set()
    out: List[str] = []
    for item in merged:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _run_retrieve(
    query: str,
    k: int = 8,
    mode_hint: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    use_query_fusion: bool = True,
    enable_mode_variants: bool = True,
    canonical_query: Optional[str] = None,
    sparse_query: Optional[str] = None,
    dense_query: Optional[str] = None,
    subqueries: Optional[List[str]] = None,
    protected_spans: Optional[List[str]] = None,
) -> List[EvidenceItem]:
    planner_requested = any(
        [
            str(sparse_query or "").strip(),
            str(dense_query or "").strip(),
            list(subqueries or []),
            list(protected_spans or []),
        ]
    )
    filters = {"doc_ids": list(doc_ids or [])} if doc_ids else None
    if planner_requested:
        try:
            fn = _find_planner_retrieve_entrypoint()
            hits = _call_with_flexible_signature(
                fn,
                query=query,
                canonical_query=canonical_query or query,
                sparse_query=sparse_query or query,
                dense_query=dense_query or query,
                subqueries=subqueries,
                protected_spans=protected_spans,
                k=k,
                mode_hint=mode_hint,
                doc_ids=doc_ids,
            )
        except RuntimeError:
            fn = _find_retrieve_entrypoint()
            hits = _call_with_flexible_signature(
                fn,
                query=canonical_query or query,
                k=k,
                mode_hint=mode_hint,
                doc_ids=doc_ids,
                filters=filters,
                use_query_fusion=use_query_fusion,
                enable_mode_variants=enable_mode_variants,
            )
    else:
        fn = _find_retrieve_entrypoint()
        hits = _call_with_flexible_signature(
            fn,
            query=query,
            k=k,
            mode_hint=mode_hint,
            doc_ids=doc_ids,
            filters=filters,
            use_query_fusion=use_query_fusion,
            enable_mode_variants=enable_mode_variants,
        )
    # Expect list of hits
    return [_normalize_hit(h) for h in hits]


def _dedupe_evidence(items: List[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    out = []
    for it in items:
        if it.chunk_id in seen:
            continue
        seen.add(it.chunk_id)
        out.append(it)
    return out


def _mode_hint_from_query(q: str) -> Optional[str]:
    ql = q.lower()
    if "algorithm" in ql or "shake" in ql:
        return "algorithm"
    if any(tok in q for tok in ["§", "(", ")", "η", "k", "d"]) or "fips" in ql:
        return "symbolic"
    if ql.startswith(("define", "what is", "what's")):
        return "definition"
    return "general"


# ---------------------------
# Tools
# ---------------------------

@tool
def retrieve(
    query: str,
    k: int = 8,
    doc_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    mode_hint: Optional[str] = None,
    canonical_query: Optional[str] = None,
    sparse_query: Optional[str] = None,
    dense_query: Optional[str] = None,
    subqueries: Optional[List[str]] = None,
    protected_spans: Optional[List[str]] = None,
    use_query_fusion: bool = True,
    enable_mode_variants: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve evidence chunks for a query (hybrid + fusion + optional rerank if enabled in your pipeline).
    Returns JSON with evidence items including page spans for citations.
    """
    resolved_mode_hint = mode_hint or _mode_hint_from_query(query)
    resolved_doc_ids = _merge_doc_ids(doc_ids, doc_id)
    evidence = _run_retrieve(
        query=query,
        k=k,
        mode_hint=resolved_mode_hint,
        doc_ids=resolved_doc_ids,
        canonical_query=canonical_query or query,
        sparse_query=sparse_query,
        dense_query=dense_query,
        subqueries=subqueries,
        protected_spans=protected_spans,
        use_query_fusion=use_query_fusion,
        enable_mode_variants=enable_mode_variants,
    )

    return {
        "tool": "retrieve",
        "query": query,
        "k": k,
        "mode_hint": resolved_mode_hint,
        "filters": {"doc_ids": resolved_doc_ids or []},
        "planner": {
            "canonical_query": canonical_query or query,
            "sparse_query": sparse_query or query,
            "dense_query": dense_query or query,
            "subqueries": list(subqueries or []),
            "protected_spans": list(protected_spans or []),
        },
        "evidence": [asdict(e) for e in evidence],
        "stats": {
            "n": len(evidence),
        },
    }


@tool
def resolve_definition(
    term: str,
    k: int = 8,
    doc_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    query: Optional[str] = None,
    use_query_fusion: bool = True,
    enable_mode_variants: bool = True,
) -> Dict[str, Any]:
    """
    Force a definitions/notation-oriented retrieval pass for a term/symbol.
    """
    resolved_query = str(query or f"definition of {term}; notation; definitions")
    resolved_doc_ids = _merge_doc_ids(doc_ids, doc_id)
    evidence = _run_retrieve(
        query=resolved_query,
        k=k,
        mode_hint="definition",
        doc_ids=resolved_doc_ids,
        use_query_fusion=use_query_fusion,
        enable_mode_variants=enable_mode_variants,
    )

    return {
        "tool": "resolve_definition",
        "term": term,
        "query": resolved_query,
        "k": k,
        "mode_hint": "definition",
        "filters": {"doc_ids": resolved_doc_ids or []},
        "evidence": [asdict(e) for e in evidence],
        "stats": {"n": len(evidence)},
    }


@tool
def compare(
    topic_a: str,
    topic_b: str,
    k: int = 6,
    doc_ids: Optional[List[str]] = None,
    use_query_fusion: bool = True,
    enable_mode_variants: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve evidence for two topics and merge/dedupe.
    (The graph will do the actual comparison answer generation with citations.)
    """
    qa = f"{topic_a} intended use-cases; definition; key properties"
    qb = f"{topic_b} intended use-cases; definition; key properties"

    ea = _run_retrieve(
        query=qa,
        k=k,
        mode_hint="compare",
        doc_ids=doc_ids,
        use_query_fusion=use_query_fusion,
        enable_mode_variants=enable_mode_variants,
    )
    eb = _run_retrieve(
        query=qb,
        k=k,
        mode_hint="compare",
        doc_ids=doc_ids,
        use_query_fusion=use_query_fusion,
        enable_mode_variants=enable_mode_variants,
    )

    merged = _dedupe_evidence(ea + eb)

    return {
        "tool": "compare",
        "topic_a": topic_a,
        "topic_b": topic_b,
        "k": k,
        "mode_hint": "compare",
        "filters": {"doc_ids": list(doc_ids or [])},
        "evidence": [asdict(e) for e in merged],
        "stats": {"n_a": len(ea), "n_b": len(eb), "n_merged": len(merged)},
    }


@tool
def summarize(doc_id: str, start_page: int, end_page: int, k: int = 30) -> Dict[str, Any]:
    """
    Fetch chunks overlapping a doc page range and return them as evidence for a grounded summary.
    (The graph/answer node will generate the summary text with strict citations.)
    """
    # Pull chunks by metadata first (deterministic), then keep at most k.
    chunks = _chunks_for_doc_pages(doc_id, start_page, end_page)[:k]

    # Convert metadata rows into EvidenceItem-like dicts.
    # Assumes chunks.jsonl rows already contain "text". If not, wire chunk_store lookup here.
    evidence: List[EvidenceItem] = []
    for r in chunks:
        if "text" not in r:
            continue
        evidence.append(
            EvidenceItem(
                score=0.0,
                chunk_id=str(r["chunk_id"]),
                doc_id=str(r["doc_id"]),
                start_page=int(r["start_page"]),
                end_page=int(r["end_page"]),
                text=str(r["text"]),
            )
        )

    return {
        "tool": "summarize",
        "doc_id": doc_id,
        "start_page": start_page,
        "end_page": end_page,
        "k": k,
        "evidence": [asdict(e) for e in evidence],
        "stats": {"n": len(evidence)},
    }
