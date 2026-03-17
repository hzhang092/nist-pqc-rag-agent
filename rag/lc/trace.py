"""
Trace utilities for LangGraph agent runs.

What this module is for:
- Builds human-readable trace summaries from the canonical flat `state["trace"]` event stream.
- Writes a readable trace summary artifact and a sibling raw state dump for debugging.
- Reuses the same summary shape for saved traces and API responses.

How it is used:
- Called by `rag.agent.ask` to save trace artifacts after agent execution.
- Imported by `rag.service` to derive `trace_summary` and `analysis` for `/ask-agent`.
- Works with both current trace events and older `verify`/`plan`/`analysis` style traces.

CLI flags:
- None. This is a library module (non-CLI) and is not executed via command-line flags.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_TIMELINE_TEXT_LIMIT = 240
_EVIDENCE_PREVIEW_LIMIT = 280
_LIST_PREVIEW_LIMIT = 5


def _slugify(text: str, max_len: int = 80) -> str:
    s = text.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:max_len] if s else "question"


def _safe(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _json_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=_safe))


def _preview_text(text: str, *, limit: int) -> str:
    preview = str(text or "").strip().replace("\n", " ")
    if len(preview) > limit:
        return preview[:limit] + "..."
    return preview


def _preview_list(values: Any, *, limit: int = _LIST_PREVIEW_LIMIT) -> List[Any]:
    if isinstance(values, str):
        text = _preview_text(values, limit=_TIMELINE_TEXT_LIMIT)
        return [text] if text else []
    out: List[Any] = []
    for value in list(values or [])[:limit]:
        if isinstance(value, str):
            out.append(_preview_text(value, limit=_TIMELINE_TEXT_LIMIT))
        elif isinstance(value, dict):
            out.append(_preview_mapping(value))
        else:
            out.append(value)
    return out


def _preview_mapping(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            out[key] = _preview_text(value, limit=_TIMELINE_TEXT_LIMIT)
        elif isinstance(value, list):
            out[key] = _preview_list(value)
        elif isinstance(value, dict):
            out[key] = _preview_mapping(value)
        else:
            out[key] = value
    return out


def _preview_topic_gaps(value: Any) -> Any:
    if isinstance(value, dict):
        return _preview_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return _preview_list(list(value))
    if isinstance(value, str):
        text = _preview_text(value, limit=_TIMELINE_TEXT_LIMIT)
        return [text] if text else []
    return []


def _plan_args_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_ids": _preview_list(args.get("doc_ids") or []),
        "protected_spans": _preview_list(args.get("protected_spans") or []),
        "subqueries": _preview_list(args.get("subqueries") or []),
        "sparse_query": _preview_text(str(args.get("sparse_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
        "dense_query": _preview_text(str(args.get("dense_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
    }


def _canonical_event_type(event_type: str) -> str:
    mapping = {
        "analysis": "analysis_applied",
        "plan": "plan_applied",
        "evidence": "evidence_updated",
        "answer": "answer_drafted",
        "final_answer": "final_answer_set",
        "verify": "verification_decision",
    }
    return mapping.get(event_type, event_type)


def _compact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    event_type = str(event.get("type") or "")
    node = str(event.get("node") or "")

    def _base() -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": event_type}
        if node:
            payload["node"] = node
        return payload

    if event_type == "step":
        payload = _base()
        payload["steps"] = int(event.get("steps", 0))
        payload["tool_calls"] = int(event.get("tool_calls", 0))
        payload["retrieval_round"] = int(event.get("retrieval_round", 0))
        return payload

    if event_type == "analysis_applied":
        payload = _base()
        payload.update(
            {
                "mode_hint": str(event.get("mode_hint") or ""),
                "rewrite_needed": bool(event.get("rewrite_needed", False)),
                "protected_spans": _preview_list(event.get("protected_spans") or []),
                "doc_ids": _preview_list(event.get("doc_ids") or []),
                "sparse_query": _preview_text(str(event.get("sparse_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "dense_query": _preview_text(str(event.get("dense_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "subqueries_count": int(event.get("subqueries_count", 0)),
                "analysis_notes": _preview_text(str(event.get("analysis_notes") or ""), limit=_TIMELINE_TEXT_LIMIT),
            }
        )
        return payload

    if event_type == "plan_applied":
        payload = _base()
        payload.update(
            {
                "action": str(event.get("action") or ""),
                "reason": _preview_text(str(event.get("reason") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "query": _preview_text(str(event.get("query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "mode_hint": str(event.get("mode_hint") or ""),
                "args_summary": _preview_mapping(dict(event.get("args_summary") or {})),
            }
        )
        return payload

    if event_type == "graph_lookup_applied":
        payload = _base()
        payload.update(
            {
                "lookup_type": str(event.get("lookup_type") or ""),
                "matched": bool(event.get("matched", False)),
                "match_reason": str(event.get("match_reason") or ""),
                "lookup_value": _preview_text(str(event.get("lookup_value") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "candidate_doc_ids": _preview_list(event.get("candidate_doc_ids") or []),
                "candidate_section_ids": _preview_list(event.get("candidate_section_ids") or []),
                "required_anchors": _preview_list(event.get("required_anchors") or []),
                "ambiguous": bool(event.get("ambiguous", False)),
                "fallback_used": bool(event.get("fallback_used", False)),
                "applied_doc_ids": _preview_list(event.get("applied_doc_ids") or []),
            }
        )
        return payload

    if event_type == "evidence_updated":
        payload = _base()
        payload.update(
            {
                "round": int(event.get("round", 0)),
                "total_hits": int(event.get("total_hits", 0)),
                "top_chunk_ids": _preview_list(event.get("top_chunk_ids") or []),
                "top_doc_ids": _preview_list(event.get("top_doc_ids") or []),
            }
        )
        return payload

    if event_type == "retrieval_round_started":
        payload = _base()
        payload.update(
            {
                "round": int(event.get("round", 0)),
                "action": str(event.get("action") or ""),
                "tool_calls": int(event.get("tool_calls", 0)),
            }
        )
        return payload

    if event_type == "retrieval_round_result":
        payload = _base()
        payload.update(
            {
                "round": int(event.get("round", 0)),
                "action": str(event.get("action") or ""),
                "new_hits": int(event.get("new_hits", 0)),
                "total_hits": int(event.get("total_hits", 0)),
                "mode_hint": str(event.get("mode_hint") or ""),
                "doc_ids": _preview_list(event.get("doc_ids") or []),
                "sparse_query": _preview_text(str(event.get("sparse_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "dense_query": _preview_text(str(event.get("dense_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "subqueries": _preview_list(event.get("subqueries") or []),
                "protected_spans": _preview_list(event.get("protected_spans") or []),
                "candidate_section_ids_count": int(event.get("candidate_section_ids_count", 0) or 0),
                "section_prior_applied": bool(event.get("section_prior_applied", False)),
                "timing_ms_retrieve": round(float(event.get("timing_ms_retrieve", 0.0) or 0.0), 3),
            }
        )
        return payload

    if event_type == "assessment_decision":
        payload = _base()
        payload.update(
            {
                "sufficient": bool(event.get("sufficient", False)),
                "reasons": _preview_list(event.get("reasons") or []),
                "budget_reason": str(event.get("budget_reason") or ""),
                "evidence_hits": int(event.get("evidence_hits", 0)),
                "doc_diversity": int(event.get("doc_diversity", 0)),
                "missing_protected_spans": _preview_list(event.get("missing_protected_spans") or []),
                "missing_compare_topics": _preview_topic_gaps(event.get("missing_compare_topics")),
            }
        )
        return payload

    if event_type == "query_refined":
        payload = _base()
        payload.update(
            {
                "strategy": str(event.get("strategy") or ""),
                "previous_query": _preview_text(str(event.get("previous_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "refined_query": _preview_text(str(event.get("refined_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "refined_sparse_query": _preview_text(str(event.get("refined_sparse_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "refined_dense_query": _preview_text(str(event.get("refined_dense_query") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "refined_subqueries": _preview_list(event.get("refined_subqueries") or []),
            }
        )
        return payload

    if event_type == "answer_drafted":
        payload = _base()
        payload.update(
            {
                "answer_prompt_question": _preview_text(str(event.get("answer_prompt_question") or ""), limit=_TIMELINE_TEXT_LIMIT),
                "draft_len": int(event.get("draft_len", 0)),
                "citations_count": int(event.get("citations_count", 0)),
                "citation_keys": _preview_list(event.get("citation_keys") or []),
                "timing_ms_generate": round(float(event.get("timing_ms_generate", 0.0) or 0.0), 3),
            }
        )
        return payload

    if event_type == "final_answer_set":
        payload = _base()
        payload.update(
            {
                "result": str(event.get("result") or ""),
                "final_len": int(event.get("final_len", 0)),
                "used_refusal_template": bool(event.get("used_refusal_template", False)),
                "stop_reason": str(event.get("stop_reason") or ""),
                "refusal_reason": str(event.get("refusal_reason") or ""),
            }
        )
        return payload

    if event_type == "verification_decision":
        payload = _base()
        payload.update(
            {
                "result": str(event.get("result") or ""),
                "stop_reason": str(event.get("stop_reason") or ""),
                "refusal_reason": str(event.get("refusal_reason") or ""),
                "citations_count": int(event.get("citations_count", 0)),
                "used_refusal_template": bool(event.get("used_refusal_template", False)),
            }
        )
        return payload

    if event_type in {"answer_skip", "loop_stop", "tool_skip"}:
        payload = _base()
        payload["reason"] = str(event.get("reason") or "")
        return payload

    return _preview_mapping(_base() | dict(event))


def _normalize_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []
    active_node = ""
    for raw_event in trace:
        event = dict(raw_event)
        event["type"] = _canonical_event_type(str(event.get("type") or ""))
        node = str(event.get("node") or "")
        if event["type"] == "step" and node:
            active_node = node
        elif not node and active_node:
            event["node"] = active_node

        if event["type"] == "analysis_applied" and "subqueries_count" not in event:
            event["subqueries_count"] = len(list(event.get("subqueries") or []))
        if event["type"] == "plan_applied" and "args_summary" not in event:
            event["args_summary"] = _plan_args_summary(dict(event.get("args") or {}))
        if event["type"] == "evidence_updated":
            if "total_hits" not in event:
                event["total_hits"] = int(event.get("n", 0))
            if "round" not in event:
                event["round"] = int(event.get("retrieval_round", 0) or 0)
        if event["type"] == "answer_drafted":
            if "citations_count" not in event:
                event["citations_count"] = int(event.get("citations", 0) or 0)
        if event["type"] == "verification_decision":
            if "citations_count" not in event:
                event["citations_count"] = int(event.get("citations", 0) or 0)
            if "used_refusal_template" not in event:
                event["used_refusal_template"] = str(event.get("result") or "") == "refuse"

        timeline.append(_compact_event(event))
    return timeline


def _analysis_payload(state: Dict[str, Any]) -> Dict[str, Any]:
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
        "graph_lookup": _json_copy(dict(state.get("graph_lookup") or {})),
    }


def _graph_lookup_summary(state: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    graph_lookup = dict(state.get("graph_lookup") or {})
    if not graph_lookup:
        return {}
    section_prior_applied = any(
        bool(event.get("section_prior_applied", False))
        for event in timeline
        if event.get("type") == "retrieval_round_result"
    )
    return {
        "lookup_type": str(graph_lookup.get("lookup_type") or ""),
        "matched": bool(graph_lookup.get("matched", False)),
        "match_reason": str(graph_lookup.get("match_reason") or ""),
        "ambiguous": bool(graph_lookup.get("ambiguous", False)),
        "applied_doc_ids": list(graph_lookup.get("applied_doc_ids") or []),
        "candidate_doc_ids": list(graph_lookup.get("candidate_doc_ids") or []),
        "candidate_section_ids_count": len(list(graph_lookup.get("candidate_section_ids") or [])),
        "section_prior_applied": section_prior_applied,
    }


def _evidence_preview(state: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for item in list(state.get("evidence") or [])[:_LIST_PREVIEW_LIMIT]:
        if not isinstance(item, dict):
            continue
        preview.append(
            {
                "score": round(float(item.get("score", 0.0) or 0.0), 6),
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": str(item.get("doc_id") or ""),
                "start_page": int(item.get("start_page", 0) or 0),
                "end_page": int(item.get("end_page", 0) or 0),
                "preview_text": _preview_text(str(item.get("text") or ""), limit=limit),
            }
        )
    return preview


def _answer_summary(state: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    draft_event = next((event for event in reversed(timeline) if event.get("type") == "answer_drafted"), {})
    final_event = next((event for event in reversed(timeline) if event.get("type") == "final_answer_set"), {})
    verify_event = next((event for event in reversed(timeline) if event.get("type") == "verification_decision"), {})
    citations = [item for item in list(state.get("citations") or []) if isinstance(item, dict)]
    citation_keys = [str(item.get("key")) for item in citations if item.get("key")][:_LIST_PREVIEW_LIMIT]
    draft_answer = str(state.get("draft_answer") or "")
    final_answer = str(state.get("final_answer") or "")
    result = str(
        verify_event.get("result")
        or final_event.get("result")
        or ("refuse" if state.get("refusal_reason") else ("ok" if final_answer else ""))
    )
    used_refusal_template = bool(
        verify_event.get("used_refusal_template", final_event.get("used_refusal_template", result == "refuse"))
    )
    return {
        "answer_prompt_question": str(state.get("answer_prompt_question") or state.get("question") or ""),
        "draft_len": int(draft_event.get("draft_len", len(draft_answer.strip()))),
        "final_len": int(final_event.get("final_len", len(final_answer.strip()))),
        "citations_count": len(citations),
        "citation_keys": citation_keys,
        "result": result,
        "used_refusal_template": used_refusal_template,
        "stop_reason": str(verify_event.get("stop_reason") or state.get("stop_reason") or ""),
        "refusal_reason": str(verify_event.get("refusal_reason") or state.get("refusal_reason") or ""),
        "timing_ms_generate": round(float(draft_event.get("timing_ms_generate", 0.0) or 0.0), 3),
        "draft_preview": _preview_text(draft_answer, limit=_TIMELINE_TEXT_LIMIT),
        "final_preview": _preview_text(final_answer, limit=_TIMELINE_TEXT_LIMIT),
    }


def _trace_by_node(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visits: List[Dict[str, Any]] = []
    visit_counts: Dict[str, int] = {}
    current: Optional[Dict[str, Any]] = None

    for event in timeline:
        if event.get("type") == "step":
            node = str(event.get("node") or "")
            visit_counts[node] = int(visit_counts.get(node, 0)) + 1
            current = {
                "node": node,
                "visit": visit_counts[node],
                "step": int(event.get("steps", 0)),
                "tool_calls": int(event.get("tool_calls", 0)),
                "retrieval_round": int(event.get("retrieval_round", 0)),
                "event_types": [],
                "events": [],
            }
            visits.append(current)
            continue

        if current is None:
            current = {
                "node": str(event.get("node") or "unknown"),
                "visit": 1,
                "step": 0,
                "tool_calls": 0,
                "retrieval_round": 0,
                "event_types": [],
                "events": [],
            }
            visits.append(current)
        current["event_types"].append(str(event.get("type") or ""))
        current["events"].append(event)

    return visits


def _retrieval_summary(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rounds: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    current_round = 0

    def _round(round_no: int) -> Dict[str, Any]:
        if round_no not in rounds:
            rounds[round_no] = {
                "round": round_no,
                "action": "",
                "tool_calls": 0,
                "doc_ids": [],
                "protected_spans": [],
                "sparse_query": "",
                "dense_query": "",
                "subqueries": [],
                "new_hits": 0,
                "total_hits": 0,
                "top_chunk_ids": [],
                "top_doc_ids": [],
                "candidate_section_ids_count": 0,
                "section_prior_applied": False,
                "sufficient": None,
                "assessment_reasons": [],
                "budget_reason": "",
                "refinement": None,
                "timing_ms_retrieve": 0.0,
            }
            order.append(round_no)
        return rounds[round_no]

    for event in timeline:
        event_type = str(event.get("type") or "")
        if event_type == "retrieval_round_started":
            current_round = int(event.get("round", 0) or current_round or 0)
            round_summary = _round(current_round)
            round_summary["action"] = str(event.get("action") or "")
            round_summary["tool_calls"] = int(event.get("tool_calls", 0))
            continue

        if event_type == "evidence_updated":
            round_no = int(event.get("round", 0) or current_round or 0)
            if round_no <= 0:
                continue
            round_summary = _round(round_no)
            round_summary["total_hits"] = int(event.get("total_hits", 0))
            round_summary["top_chunk_ids"] = _preview_list(event.get("top_chunk_ids") or [])
            round_summary["top_doc_ids"] = _preview_list(event.get("top_doc_ids") or [])
            continue

        if event_type == "retrieval_round_result":
            round_no = int(event.get("round", 0) or current_round or 0)
            if round_no <= 0:
                continue
            current_round = round_no
            round_summary = _round(round_no)
            round_summary["action"] = str(event.get("action") or round_summary["action"])
            round_summary["new_hits"] = int(event.get("new_hits", 0))
            round_summary["total_hits"] = int(event.get("total_hits", round_summary["total_hits"]))
            round_summary["doc_ids"] = _preview_list(event.get("doc_ids") or [])
            round_summary["protected_spans"] = _preview_list(event.get("protected_spans") or [])
            round_summary["sparse_query"] = str(event.get("sparse_query") or "")
            round_summary["dense_query"] = str(event.get("dense_query") or "")
            round_summary["subqueries"] = _preview_list(event.get("subqueries") or [])
            round_summary["candidate_section_ids_count"] = int(event.get("candidate_section_ids_count", 0) or 0)
            round_summary["section_prior_applied"] = bool(event.get("section_prior_applied", False))
            round_summary["timing_ms_retrieve"] = round(float(event.get("timing_ms_retrieve", 0.0) or 0.0), 3)
            continue

        if event_type == "assessment_decision" and current_round > 0:
            round_summary = _round(current_round)
            round_summary["sufficient"] = bool(event.get("sufficient", False))
            round_summary["assessment_reasons"] = _preview_list(event.get("reasons") or [])
            round_summary["budget_reason"] = str(event.get("budget_reason") or "")
            continue

        if event_type == "query_refined" and current_round > 0:
            round_summary = _round(current_round)
            round_summary["refinement"] = {
                "strategy": str(event.get("strategy") or ""),
                "previous_query": str(event.get("previous_query") or ""),
                "refined_query": str(event.get("refined_query") or ""),
                "refined_sparse_query": str(event.get("refined_sparse_query") or ""),
                "refined_dense_query": str(event.get("refined_dense_query") or ""),
                "refined_subqueries": _preview_list(event.get("refined_subqueries") or []),
            }

    return [rounds[round_no] for round_no in order]


def summarize_trace(
    state: Dict[str, Any],
    *,
    evidence_preview_chars: int = _EVIDENCE_PREVIEW_LIMIT,
) -> Dict[str, Any]:
    payload = _json_copy(state)
    payload.pop("_trace_active_node", None)

    analysis = _analysis_payload(payload)
    timeline = _normalize_trace(list(payload.get("trace") or []))
    evidence_preview = _evidence_preview(
        payload,
        limit=min(max(int(evidence_preview_chars), 1), _EVIDENCE_PREVIEW_LIMIT),
    )
    answer_summary = _answer_summary(payload, timeline)
    first_step = next(
        (event.get("node") for event in timeline if event.get("type") == "step" and event.get("node")),
        "",
    )
    result = answer_summary["result"]
    graph_lookup_summary = _graph_lookup_summary(payload, timeline)
    timing_ms = {
        "analyze": round(float((payload.get("timing_ms") or {}).get("analyze", 0.0) or 0.0), 3),
        "retrieve": round(float((payload.get("timing_ms") or {}).get("retrieve", 0.0) or 0.0), 3),
        "generate": round(float((payload.get("timing_ms") or {}).get("generate", 0.0) or 0.0), 3),
        "total": round(float((payload.get("timing_ms") or {}).get("total", 0.0) or 0.0), 3),
    }
    run = {
        "question": str(payload.get("question") or ""),
        "entry_node": str(first_step or ""),
        "mode_hint": analysis["mode_hint"],
        "rewrite_needed": analysis["rewrite_needed"],
        "protected_span_count": len(analysis["protected_spans"]),
        "subquery_count": len(analysis["subqueries"]),
        "doc_ids": analysis["doc_ids"],
        "result": result,
        "stop_reason": str(payload.get("stop_reason") or answer_summary["stop_reason"] or ""),
        "refusal_reason": str(payload.get("refusal_reason") or answer_summary["refusal_reason"] or ""),
        "steps": int(payload.get("steps", 0)),
        "tool_calls": int(payload.get("tool_calls", 0)),
        "retrieval_rounds": int(payload.get("retrieval_round", 0)),
        "evidence_hits": len(list(payload.get("evidence") or [])),
        "citation_count": len(list(payload.get("citations") or [])),
        "top_chunk_ids": [item.get("chunk_id") for item in evidence_preview if item.get("chunk_id")],
        "timing_ms": timing_ms,
        "graph_lookup": graph_lookup_summary,
    }

    return {
        "run": run,
        "analysis": analysis,
        "retrieval_summary": _retrieval_summary(timeline),
        "answer_summary": answer_summary,
        "trace_by_node": _trace_by_node(timeline),
        "timeline": timeline,
        "evidence_preview": evidence_preview,
    }


def write_trace(
    state: Dict[str, Any],
    out_dir: str = "runs/agent",
    filename_prefix: Optional[str] = None,
    truncate_evidence_chars: int = 800,
) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    question = state.get("question", "")
    slug = _slugify(question)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix = filename_prefix or "agent"
    summary_file = out_path / f"{prefix}_{ts}_{slug}.json"
    raw_file = out_path / f"{prefix}_{ts}_{slug}.raw.json"

    raw_payload = _json_copy(state)
    raw_payload.pop("_trace_active_node", None)
    summary_payload = summarize_trace(
        state,
        evidence_preview_chars=min(max(int(truncate_evidence_chars), 1), _EVIDENCE_PREVIEW_LIMIT),
    )

    summary_file.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_file.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(summary_file)
