from __future__ import annotations
"""
Utilities for term-to-graph lookup over precomputed lightweight graph artifacts.

This module loads node/edge JSONL files, builds an in-memory cached index, and
resolves user-provided terms to matching Term entities and related context
(document and section candidates).

Graph sources:
- ``data/processed/graph_lite_nodes.jsonl``
- ``data/processed/graph_lite_edges.jsonl``

High-level behavior:
1. Normalize and deduplicate optional document filters.
2. Lazily build and cache graph indexes (nodes, outgoing edges, and term lookup maps).
3. Match a query term by priority:
    - exact normalized term
    - surface form
    - display name
4. Collect:
    - matched term entities (with lightweight metadata),
    - candidate document IDs,
    - candidate section IDs (definitions and mentions),
    - required anchors (display names),
    while honoring optional ``doc_ids`` filtering for section candidates.

Notes:
- If graph artifacts are missing, lookup returns empty candidates with
  ``match_reason="graph_artifacts_missing"``.
- Section IDs preserve discovery order and are deduplicated.
- Document IDs are returned sorted.
- Index construction is memoized with ``lru_cache(maxsize=1)`` for efficiency.

Public API:
- ``lookup_term(term: str, *, doc_ids: list[str] | None = None) -> dict[str, Any]``

Return schema for ``lookup_term``:
- ``matched_entities``: list of matched Term node summaries.
- ``candidate_doc_ids``: sorted list of related document IDs.
- ``candidate_section_ids``: ordered, deduplicated related section IDs.
- ``required_anchors``: display-name anchors for downstream retrieval.
- ``match_reason``: reason label indicating match path or failure mode.
"""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag.graph.helpers import normalize_term, read_jsonl


NODES_PATH = Path("data/processed/graph_lite_nodes.jsonl")
EDGES_PATH = Path("data/processed/graph_lite_edges.jsonl")


def _normalize_doc_ids(doc_ids: list[str] | None) -> list[str]:
    out: list[str] = []
    seen = set()
    for doc_id in doc_ids or []:
        text = str(doc_id or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@lru_cache(maxsize=1)
def _graph_index() -> dict[str, Any]:
    if not NODES_PATH.exists() or not EDGES_PATH.exists():
        return {
            "nodes": {},
            "out_edges": defaultdict(list),
            "term_by_norm": defaultdict(list),
            "term_by_surface": defaultdict(list),
            "term_by_display": defaultdict(list),
        }

    nodes = {row["node_id"]: row for row in read_jsonl(NODES_PATH)}
    out_edges: dict[str, list[dict]] = defaultdict(list)
    term_by_norm: dict[str, list[dict]] = defaultdict(list)
    term_by_surface: dict[str, list[dict]] = defaultdict(list)
    term_by_display: dict[str, list[dict]] = defaultdict(list)

    for edge in read_jsonl(EDGES_PATH):
        out_edges[str(edge.get("source_id") or "")].append(edge)

    for node in nodes.values():
        if str(node.get("label") or "") != "Term":
            continue
        properties = dict(node.get("properties") or {})
        normalized = normalize_term(
            str(properties.get("normalized_term") or node.get("display_name") or "")
        )
        if normalized:
            term_by_norm[normalized].append(node)
        for surface_form in list(properties.get("surface_forms") or []):
            normalized_surface = normalize_term(surface_form)
            if normalized_surface:
                term_by_surface[normalized_surface].append(node)
        display_key = normalize_term(str(node.get("display_name") or ""))
        if display_key:
            term_by_display[display_key].append(node)

    return {
        "nodes": nodes,
        "out_edges": out_edges,
        "term_by_norm": term_by_norm,
        "term_by_surface": term_by_surface,
        "term_by_display": term_by_display,
    }


def _node_sort_key(node: dict) -> tuple[str, int, str]:
    return (
        str(node.get("doc_id") or ""),
        int(node.get("start_page") or 10**9),
        str(node.get("node_id") or ""),
    )


def _select_matches(term: str, index: dict[str, Any]) -> tuple[list[dict], str]:
    normalized = normalize_term(term)
    if not normalized:
        return [], "empty_term"
    if normalized in index["term_by_norm"]:
        return sorted(index["term_by_norm"][normalized], key=_node_sort_key), "exact_normalized_term"
    if normalized in index["term_by_surface"]:
        return sorted(index["term_by_surface"][normalized], key=_node_sort_key), "surface_form"
    if normalized in index["term_by_display"]:
        return sorted(index["term_by_display"][normalized], key=_node_sort_key), "display_name"
    return [], "no_match"


def lookup_term(term: str, *, doc_ids: list[str] | None = None) -> dict[str, Any]:
    index = _graph_index()
    if not index["nodes"]:
        return {
            "matched_entities": [],
            "candidate_doc_ids": [],
            "candidate_section_ids": [],
            "required_anchors": [],
            "match_reason": "graph_artifacts_missing",
        }

    matches, match_reason = _select_matches(term, index)
    if not matches:
        return {
            "matched_entities": [],
            "candidate_doc_ids": [],
            "candidate_section_ids": [],
            "required_anchors": [],
            "match_reason": match_reason,
        }

    allowed_doc_ids = set(_normalize_doc_ids(doc_ids))
    candidate_doc_ids: set[str] = set()
    candidate_section_ids: list[str] = []
    required_anchors: list[str] = []
    seen_sections = set()
    matched_entities: list[dict[str, Any]] = []

    for node in matches:
        node_id = str(node.get("node_id") or "")
        node_properties = dict(node.get("properties") or {})
        matched_entities.append(
            {
                "node_id": node_id,
                "display_name": str(node.get("display_name") or ""),
                "normalized_term": str(node_properties.get("normalized_term") or ""),
                "term_type": str(node_properties.get("term_type") or ""),
                "definition_strength": str(node_properties.get("definition_strength") or ""),
            }
        )
        anchor = str(node.get("display_name") or "")
        if anchor and anchor not in required_anchors:
            required_anchors.append(anchor)

        definition_sections: list[str] = []
        mention_sections: list[str] = []
        for edge in index["out_edges"].get(node_id, []):
            edge_type = str(edge.get("type") or "")
            target_id = str(edge.get("target_id") or "")
            if edge_type == "IN_DOCUMENT":
                doc_id = str(edge.get("doc_id") or "")
                if doc_id:
                    candidate_doc_ids.add(doc_id)
            elif edge_type == "DEFINED_IN":
                definition_sections.append(target_id)
            elif edge_type == "APPEARS_IN":
                mention_sections.append(target_id)

        for section_id in definition_sections + mention_sections:
            section_node = index["nodes"].get(section_id, {})
            section_doc_id = str(section_node.get("doc_id") or "")
            if allowed_doc_ids and section_doc_id not in allowed_doc_ids:
                continue
            if section_id in seen_sections:
                continue
            seen_sections.add(section_id)
            candidate_section_ids.append(section_id)

    if allowed_doc_ids:
        filtered_doc_ids = [doc_id for doc_id in sorted(candidate_doc_ids) if doc_id in allowed_doc_ids]
        if filtered_doc_ids:
            candidate_doc_ids = set(filtered_doc_ids)

    return {
        "matched_entities": matched_entities,
        "candidate_doc_ids": sorted(candidate_doc_ids),
        "candidate_section_ids": candidate_section_ids,
        "required_anchors": required_anchors,
        "match_reason": match_reason,
    }
