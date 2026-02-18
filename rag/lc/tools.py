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


def _run_retrieve(query: str, k: int = 8, mode_hint: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[EvidenceItem]:
    fn = _find_retrieve_entrypoint()
    hits = _call_with_flexible_signature(fn, query=query, k=k, mode_hint=mode_hint, filters=filters)
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
def retrieve(query: str, k: int = 8, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve evidence chunks for a query (hybrid + fusion + optional rerank if enabled in your pipeline).
    Returns JSON with evidence items including page spans for citations.
    """
    mode_hint = _mode_hint_from_query(query)
    filters = {"doc_id": doc_id} if doc_id else None
    evidence = _run_retrieve(query=query, k=k, mode_hint=mode_hint, filters=filters)

    return {
        "tool": "retrieve",
        "query": query,
        "k": k,
        "mode_hint": mode_hint,
        "filters": filters or {},
        "evidence": [asdict(e) for e in evidence],
        "stats": {
            "n": len(evidence),
        },
    }


@tool
def resolve_definition(term: str, k: int = 8, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Force a definitions/notation-oriented retrieval pass for a term/symbol.
    """
    query = f"definition of {term}; notation; definitions"
    filters = {"doc_id": doc_id} if doc_id else None
    evidence = _run_retrieve(query=query, k=k, mode_hint="definition", filters=filters)

    return {
        "tool": "resolve_definition",
        "term": term,
        "query": query,
        "k": k,
        "filters": filters or {},
        "evidence": [asdict(e) for e in evidence],
        "stats": {"n": len(evidence)},
    }


@tool
def compare(topic_a: str, topic_b: str, k: int = 6) -> Dict[str, Any]:
    """
    Retrieve evidence for two topics and merge/dedupe.
    (The graph will do the actual comparison answer generation with citations.)
    """
    qa = f"{topic_a} intended use-cases; definition; key properties"
    qb = f"{topic_b} intended use-cases; definition; key properties"

    ea = _run_retrieve(query=qa, k=k, mode_hint=_mode_hint_from_query(qa), filters=None)
    eb = _run_retrieve(query=qb, k=k, mode_hint=_mode_hint_from_query(qb), filters=None)

    merged = _dedupe_evidence(ea + eb)

    return {
        "tool": "compare",
        "topic_a": topic_a,
        "topic_b": topic_b,
        "k": k,
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
