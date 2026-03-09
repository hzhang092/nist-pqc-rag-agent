'''RAG (Retrieval-Augmented Generation) service implementation.
This module provides the core logic for handling search and question-answering requests, including retrieval of relevant document chunks, generation of answers using a language model, and construction of cited answers with evidence. It also includes health check functionality to ensure that the necessary components are available and functioning properly.

The main functions in this module are:
- `health_status`: Checks the health of the LLM backend and retrieval components.
- `search_query`: Handles search requests by retrieving relevant document chunks based on a query.
- `ask_question`: Handles question-answering requests by retrieving relevant document chunks, generating an answer using the LLM, and constructing a cited answer with evidence.'''


from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from rag.config import SETTINGS, validate_settings
from rag.lc.graph import run_agent
from rag.llm.factory import get_backend
from rag.rag_answer import build_cited_answer, select_evidence
from rag.retrieve import retrieve_with_stages_and_timing
from rag.types import AnswerResult, REFUSAL_TEXT


_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
_BM25_PATH = Path("data/processed/bm25.pkl")
_FAISS_PATH = Path("data/processed/faiss.index")


@lru_cache(maxsize=1)
def _chunk_metadata_map() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    if not _CHUNKS_PATH.exists():
        return metadata

    with _CHUNKS_PATH.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id", "") or "")
            if not chunk_id:
                continue
            metadata[chunk_id] = {
                "section_path": str(row.get("section_path", "") or ""),
                "block_type": str(row.get("block_type", "") or ""),
            }
    return metadata


def _preview_text(text: str, *, limit: int = 280) -> str:
    preview = (text or "").strip().replace("\n", " ")
    if len(preview) > limit:
        return preview[:limit] + "..."
    return preview


def _metadata_for_chunk(chunk_id: str) -> dict[str, str]:
    return _chunk_metadata_map().get(chunk_id, {})


def _hit_to_payload(hit: Any, *, include_text: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": float(hit.score),
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "start_page": int(hit.start_page),
        "end_page": int(hit.end_page),
        "preview_text": _preview_text(hit.text or ""),
    }
    if include_text:
        payload["text"] = hit.text
    metadata = _metadata_for_chunk(hit.chunk_id)
    if metadata.get("section_path"):
        payload["section_path"] = metadata["section_path"]
    if metadata.get("block_type"):
        payload["block_type"] = metadata["block_type"]
    return payload


def _retrieval_ready(*, mode: str, backend: str) -> bool:
    selected_mode = (mode or SETTINGS.RETRIEVAL_MODE).strip().lower()
    selected_backend = (backend or SETTINGS.VECTOR_BACKEND).strip().lower()

    if selected_mode == "hybrid":
        return _CHUNKS_PATH.exists() and _BM25_PATH.exists() and _FAISS_PATH.exists()
    if selected_backend == "faiss":
        return _CHUNKS_PATH.exists() and _FAISS_PATH.exists()
    if selected_backend == "bm25":
        return _CHUNKS_PATH.exists() and _BM25_PATH.exists()
    return _CHUNKS_PATH.exists()


def health_status(
    *,
    llm_backend_name: str | None = None,
    retrieval_mode: str | None = None,
    vector_backend: str | None = None,
) -> dict[str, Any]:
    validate_settings()
    llm_backend = get_backend(llm_backend_name)
    mode = retrieval_mode or SETTINGS.RETRIEVAL_MODE
    backend = vector_backend or SETTINGS.VECTOR_BACKEND
    backend_reachable = llm_backend.ping()
    retrieval_ready = _retrieval_ready(mode=mode, backend=backend)
    status = "ok" if backend_reachable and retrieval_ready else "degraded"
    return {
        "status": status,
        "llm_backend": llm_backend.backend_name,
        "llm_model": llm_backend.model_name,
        "backend_reachable": backend_reachable,
        "retrieval_ready": retrieval_ready,
    }


def search_query(
    query: str,
    *,
    k: int | None = None,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    backend: str = SETTINGS.VECTOR_BACKEND,
    use_query_fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
) -> dict[str, Any]:
    validate_settings()
    final_k = k or SETTINGS.TOP_K
    stages, timing_ms = retrieve_with_stages_and_timing(
        query=query,
        k=final_k,
        mode=mode,
        backend=backend,
        use_query_fusion=use_query_fusion,
        candidate_multiplier=candidate_multiplier,
        k0=k0,
        enable_rerank=enable_rerank,
        rerank_pool=rerank_pool,
    )

    hits = stages["final_hits"]
    return {
        "query": query,
        "hits": [_hit_to_payload(hit, include_text=False) for hit in hits],
        "retrieval": {
            "mode": mode,
            "backend": backend,
            "k": final_k,
            "query_fusion": use_query_fusion,
            "candidate_multiplier": candidate_multiplier,
            "k0": k0,
            "rerank": enable_rerank,
            "rerank_pool": rerank_pool,
        },
        "timing_ms": {
            "retrieve": round(timing_ms["retrieve_ms"], 3),
            "rerank": round(timing_ms["rerank_ms"], 3),
        },
    }


def _derive_refusal_reason(
    *,
    evidence_count: int,
    min_evidence_hits: int,
    raw_answer: str | None,
    backend_error: str | None,
    result: AnswerResult,
) -> str | None:
    if not result.is_refusal():
        return None
    if backend_error:
        return "backend_error"
    if evidence_count < min_evidence_hits:
        return "insufficient_evidence"
    if raw_answer is None:
        return "missing_citations_after_generation"
    if not raw_answer.strip():
        return "empty_answer"
    return "missing_citations_after_generation"


def _agent_analysis_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_query": str(state.get("original_query") or state.get("question") or ""),
        "canonical_query": str(state.get("canonical_query") or state.get("question") or ""),
        "mode_hint": str(state.get("mode_hint") or ""),
        "rewrite_needed": bool(state.get("rewrite_needed", False)),
        "protected_spans": list(state.get("protected_spans") or []),
        "required_anchors": list(state.get("required_anchors") or []),
        "sparse_query": str(state.get("sparse_query") or state.get("canonical_query") or state.get("question") or ""),
        "dense_query": str(state.get("dense_query") or state.get("canonical_query") or state.get("question") or ""),
        "subqueries": list(state.get("subqueries") or []),
        "confidence": float(state.get("confidence", 0.0) or 0.0),
        "compare_topics": state.get("compare_topics"),
        "doc_ids": list(state.get("doc_ids") or []),
        "doc_family": str(state.get("doc_family") or ""),
        "analysis_notes": str(state.get("analysis_notes") or ""),
        "answer_prompt_question": str(state.get("answer_prompt_question") or state.get("question") or ""),
    }


def _agent_trace_summary(state: dict[str, Any]) -> dict[str, Any]:
    analysis = _agent_analysis_payload(state)
    evidence = state.get("evidence") or []
    trace = state.get("trace") or []
    plan = state.get("plan") or {}
    first_step = next(
        (event.get("node") for event in trace if event.get("type") == "step" and event.get("node")),
        "",
    )
    return {
        "entry_node": first_step,
        "plan_action": plan.get("action"),
        "steps": int(state.get("steps", 0)),
        "retrieval_rounds": int(state.get("retrieval_round", 0)),
        "tool_calls": int(state.get("tool_calls", 0)),
        "stop_reason": str(state.get("stop_reason") or ""),
        "refusal_reason": str(state.get("refusal_reason") or ""),
        "trace_events": len(trace),
        "top_chunk_ids": [item.get("chunk_id") for item in evidence[:5] if isinstance(item, dict)],
        "original_query": analysis["original_query"],
        "canonical_query": analysis["canonical_query"],
        "mode_hint": analysis["mode_hint"],
        "compare_topics": analysis["compare_topics"],
        "doc_ids": analysis["doc_ids"],
        "answer_prompt_question": analysis["answer_prompt_question"],
    }


def ask_question(
    question: str,
    *,
    k: int | None = None,
    mode: str = SETTINGS.RETRIEVAL_MODE,
    vector_backend: str = SETTINGS.VECTOR_BACKEND,
    use_query_fusion: bool = SETTINGS.RETRIEVAL_QUERY_FUSION,
    candidate_multiplier: int = SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
    k0: int = SETTINGS.RETRIEVAL_RRF_K0,
    enable_rerank: bool = SETTINGS.RETRIEVAL_ENABLE_RERANK,
    rerank_pool: int = SETTINGS.RETRIEVAL_RERANK_POOL,
    llm_backend_name: str | None = None,
) -> dict[str, Any]:
    validate_settings()

    final_k = k or SETTINGS.TOP_K
    llm_backend = get_backend(llm_backend_name)
    total_started = perf_counter()
    stages, retrieval_timing = retrieve_with_stages_and_timing(
        query=question,
        k=final_k,
        mode=mode,
        backend=vector_backend,
        use_query_fusion=use_query_fusion,
        candidate_multiplier=candidate_multiplier,
        k0=k0,
        enable_rerank=enable_rerank,
        rerank_pool=rerank_pool,
    )
    hits = stages["final_hits"]
    evidence = select_evidence(hits)

    raw_answer_box: dict[str, str | None] = {"raw_answer": None}
    backend_error_box: dict[str, str | None] = {"backend_error": None}
    generation_timing_ms = {"value": 0.0}

    def _generate(prompt: str) -> str:
        started = perf_counter()
        try:
            raw_answer = llm_backend.generate(prompt, temperature=SETTINGS.LLM_TEMPERATURE)
            raw_answer_box["raw_answer"] = raw_answer
            return raw_answer
        except Exception as exc:
            backend_error_box["backend_error"] = str(exc)
            raise
        finally:
            generation_timing_ms["value"] = (perf_counter() - started) * 1000.0

    try:
        result = build_cited_answer(question=question, hits=hits, generate_fn=_generate)
    except Exception:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])

    refusal_reason = _derive_refusal_reason(
        evidence_count=len(evidence),
        min_evidence_hits=SETTINGS.ASK_MIN_EVIDENCE_HITS,
        raw_answer=raw_answer_box["raw_answer"],
        backend_error=backend_error_box["backend_error"],
        result=result,
    )

    total_ms = (perf_counter() - total_started) * 1000.0
    trace_summary = {
        "llm_backend": llm_backend.backend_name,
        "llm_model": llm_backend.model_name,
        "retrieval_mode": mode,
        "vector_backend": vector_backend,
        "query_fusion": use_query_fusion,
        "rerank_enabled": enable_rerank,
        "retrieved_hit_count": len(hits),
        "evidence_chunk_count": len(evidence),
        "top_chunk_ids": [hit.chunk_id for hit in hits[:5]],
    }

    return {
        "question": question,
        "answer_result": result,
        "answer": result.answer,
        "citations": [citation.__dict__ for citation in result.citations],
        "refusal_reason": refusal_reason,
        "trace_summary": trace_summary,
        "timing_ms": {
            "retrieve": round(retrieval_timing["retrieve_ms"], 3),
            "rerank": round(retrieval_timing["rerank_ms"], 3),
            "generate": round(generation_timing_ms["value"], 3),
            "total": round(total_ms, 3),
        },
        "llm_backend": llm_backend.backend_name,
        "llm_model": llm_backend.model_name,
        "retrieval": {
            "mode": mode,
            "backend": vector_backend,
            "k": final_k,
            "query_fusion": use_query_fusion,
            "candidate_multiplier": candidate_multiplier,
            "k0": k0,
            "rerank": enable_rerank,
            "rerank_pool": rerank_pool,
        },
        "evidence": [_hit_to_payload(hit, include_text=True) for hit in hits],
    }


def ask_agent_question(
    question: str,
    *,
    k: int | None = None,
) -> dict[str, Any]:
    validate_settings()

    total_started = perf_counter()
    state = run_agent(question, k=k)
    total_ms = (perf_counter() - total_started) * 1000.0
    timing_ms = dict(state.get("timing_ms") or {})
    timing_ms["total"] = round(float(timing_ms.get("total") or total_ms), 3)
    timing_ms["analyze"] = round(float(timing_ms.get("analyze", 0.0)), 3)
    timing_ms["retrieve"] = round(float(timing_ms.get("retrieve", 0.0)), 3)
    timing_ms["generate"] = round(float(timing_ms.get("generate", 0.0)), 3)

    answer = str(state.get("final_answer") or state.get("draft_answer") or "")
    citations = list(state.get("citations") or [])
    refusal_reason = str(state.get("refusal_reason") or "") or None
    analysis = _agent_analysis_payload(state)

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "refusal_reason": refusal_reason,
        "trace_summary": _agent_trace_summary(state),
        "timing_ms": timing_ms,
        "analysis": analysis,
    }
