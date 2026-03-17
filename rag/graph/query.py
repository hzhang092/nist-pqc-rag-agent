from __future__ import annotations
"""
Utilities for term and algorithm lookup over precomputed graph-lite artifacts.
"""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from rag.graph.helpers import normalize_term, read_jsonl
from rag.text_normalize import normalize_identifier_like_spans


NODES_PATH = Path("data/processed/graph_lite_nodes.jsonl")
EDGES_PATH = Path("data/processed/graph_lite_edges.jsonl")
_ALGORITHM_LABEL_RE = re.compile(r"\bAlgorithm\s+(?P<number>\d+)\b", flags=re.IGNORECASE)
_ALGORITHM_NAME_RE = re.compile(
    r"\b(?P<name>(?:ML-KEM|ML-DSA|SLH-DSA)(?:\.[A-Za-z0-9_]+)+)\b"
)


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
            "algorithm_by_number": defaultdict(list),
            "algorithm_by_name": defaultdict(list),
            "algorithm_by_label": defaultdict(list),
        }

    nodes = {row["node_id"]: row for row in read_jsonl(NODES_PATH)}
    out_edges: dict[str, list[dict]] = defaultdict(list)
    term_by_norm: dict[str, list[dict]] = defaultdict(list)
    term_by_surface: dict[str, list[dict]] = defaultdict(list)
    term_by_display: dict[str, list[dict]] = defaultdict(list)
    algorithm_by_number: dict[str, list[dict]] = defaultdict(list)
    algorithm_by_name: dict[str, list[dict]] = defaultdict(list)
    algorithm_by_label: dict[str, list[dict]] = defaultdict(list)

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
        continue

    for node in nodes.values():
        if str(node.get("label") or "") != "Algorithm":
            continue
        properties = dict(node.get("properties") or {})
        algorithm_number = str(properties.get("algorithm_number") or "").strip()
        if algorithm_number:
            algorithm_by_number[algorithm_number].append(node)

        algorithm_name = _normalize_algorithm_name(str(properties.get("algorithm_name") or ""))
        if algorithm_name:
            algorithm_by_name[algorithm_name].append(node)

        algorithm_label = _normalize_algorithm_label(
            str(properties.get("algorithm_label") or node.get("display_name") or "")
        )
        if algorithm_label:
            algorithm_by_label[algorithm_label].append(node)

    return {
        "nodes": nodes,
        "out_edges": out_edges,
        "term_by_norm": term_by_norm,
        "term_by_surface": term_by_surface,
        "term_by_display": term_by_display,
        "algorithm_by_number": algorithm_by_number,
        "algorithm_by_name": algorithm_by_name,
        "algorithm_by_label": algorithm_by_label,
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


def _normalize_algorithm_name(value: str) -> str:
    normalized = normalize_identifier_like_spans(str(value or "").strip())
    return normalize_term(normalized)


def _normalize_algorithm_label(value: str) -> str:
    match = _ALGORITHM_LABEL_RE.search(str(value or ""))
    if not match:
        return ""
    return f"algorithm {match.group('number')}"


def _extract_algorithm_queries(query: str) -> tuple[list[str], list[str]]:
    normalized_query = normalize_identifier_like_spans(str(query or "").strip())
    labels = []
    names = []
    for match in _ALGORITHM_LABEL_RE.finditer(normalized_query):
        labels.append(f"Algorithm {match.group('number')}")
    for match in _ALGORITHM_NAME_RE.finditer(normalized_query):
        names.append(str(match.group("name") or ""))
    return _dedupe_strings(labels), _dedupe_strings(names)


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _node_doc_id(node: dict) -> str:
    return str(node.get("doc_id") or "")


def _filter_nodes_by_doc_ids(nodes: list[dict], doc_ids: list[str] | None) -> list[dict]:
    allowed = set(_normalize_doc_ids(doc_ids))
    if not allowed:
        return list(nodes)
    return [node for node in nodes if _node_doc_id(node) in allowed]


def _algorithm_entity(node: dict) -> dict[str, Any]:
    properties = dict(node.get("properties") or {})
    return {
        "node_id": str(node.get("node_id") or ""),
        "display_name": str(node.get("display_name") or ""),
        "doc_id": str(node.get("doc_id") or ""),
        "algorithm_label": str(properties.get("algorithm_label") or ""),
        "algorithm_name": str(properties.get("algorithm_name") or ""),
        "algorithm_number": str(properties.get("algorithm_number") or ""),
        "section_id": str(properties.get("section_id") or ""),
    }


def _lookup_empty(
    *,
    lookup_type: str,
    lookup_value: str,
    match_reason: str,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "lookup_type": lookup_type,
        "lookup_value": lookup_value,
        "matched_entities": [],
        "candidate_doc_ids": [],
        "candidate_section_ids": [],
        "required_anchors": [],
        "match_reason": match_reason,
        "ambiguous": False,
        "fallback_used": fallback_used,
    }


def _collect_term_result(
    matches: list[dict],
    *,
    match_reason: str,
    lookup_value: str,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
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
        for edge in _graph_index()["out_edges"].get(node_id, []):
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
            section_node = _graph_index()["nodes"].get(section_id, {})
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
        "lookup_type": "term",
        "lookup_value": lookup_value,
        "matched_entities": matched_entities,
        "candidate_doc_ids": sorted(candidate_doc_ids),
        "candidate_section_ids": candidate_section_ids,
        "required_anchors": required_anchors,
        "match_reason": match_reason,
        "ambiguous": False,
        "fallback_used": False,
    }


def _collect_algorithm_result(
    matches: list[dict],
    *,
    lookup_type: str,
    lookup_value: str,
    match_reason: str,
    query_labels: list[str],
    query_names: list[str],
) -> dict[str, Any]:
    index = _graph_index()
    candidate_doc_ids: set[str] = set()
    candidate_section_ids: list[str] = []
    matched_entities: list[dict[str, Any]] = []
    required_anchors: list[str] = []
    seen_sections = set()

    if lookup_type == "algorithm_number":
        required_anchors.extend(query_labels)
    elif lookup_type == "algorithm_name":
        required_anchors.extend(query_names)
        if query_labels:
            required_anchors.extend(query_labels)

    for node in matches:
        node_id = str(node.get("node_id") or "")
        properties = dict(node.get("properties") or {})
        matched_entities.append(_algorithm_entity(node))

        doc_id = str(node.get("doc_id") or "")
        if doc_id:
            candidate_doc_ids.add(doc_id)

        section_id = str(properties.get("section_id") or "")
        if section_id and section_id not in seen_sections:
            seen_sections.add(section_id)
            candidate_section_ids.append(section_id)

        for edge in index["out_edges"].get(node_id, []):
            edge_type = str(edge.get("type") or "")
            target_id = str(edge.get("target_id") or "")
            if edge_type == "IN_DOCUMENT":
                doc_id = str(edge.get("doc_id") or "")
                if doc_id:
                    candidate_doc_ids.add(doc_id)
            elif edge_type == "APPEARS_IN" and target_id and target_id not in seen_sections:
                seen_sections.add(target_id)
                candidate_section_ids.append(target_id)

    distinct_doc_ids = {entity["doc_id"] for entity in matched_entities if entity.get("doc_id")}
    return {
        "lookup_type": lookup_type,
        "lookup_value": lookup_value,
        "matched_entities": matched_entities,
        "candidate_doc_ids": sorted(candidate_doc_ids),
        "candidate_section_ids": candidate_section_ids,
        "required_anchors": _dedupe_strings(required_anchors) or [lookup_value],
        "match_reason": match_reason,
        "ambiguous": len(distinct_doc_ids) > 1,
        "fallback_used": False,
    }


def lookup_term(term: str, *, doc_ids: list[str] | None = None) -> dict[str, Any]:
    index = _graph_index()
    if not index["nodes"]:
        return _lookup_empty(
            lookup_type="term",
            lookup_value=str(term or ""),
            match_reason="graph_artifacts_missing",
        )

    matches, match_reason = _select_matches(term, index)
    if not matches:
        return _lookup_empty(
            lookup_type="term",
            lookup_value=str(term or ""),
            match_reason=match_reason,
        )

    return _collect_term_result(
        matches,
        match_reason=match_reason,
        lookup_value=str(term or ""),
        doc_ids=doc_ids,
    )


def lookup_algorithm(query: str, *, doc_ids: list[str] | None = None) -> dict[str, Any]:
    index = _graph_index()
    if not index["nodes"]:
        return _lookup_empty(
            lookup_type="algorithm",
            lookup_value=str(query or ""),
            match_reason="graph_artifacts_missing",
        )

    allowed_doc_ids = _normalize_doc_ids(doc_ids)
    query_labels, query_names = _extract_algorithm_queries(query)
    lookup_value = query_names[0] if query_names else (query_labels[0] if query_labels else str(query or ""))

    for algorithm_name in query_names:
        normalized_name = _normalize_algorithm_name(algorithm_name)
        matches = sorted(
            _filter_nodes_by_doc_ids(index["algorithm_by_name"].get(normalized_name, []), allowed_doc_ids),
            key=_node_sort_key,
        )
        if matches:
            return _collect_algorithm_result(
                matches,
                lookup_type="algorithm_name",
                lookup_value=algorithm_name,
                match_reason="exact_algorithm_name",
                query_labels=query_labels,
                query_names=[algorithm_name],
            )

    term_hint = None
    hinted_doc_ids: list[str] = []
    if query_names:
        term_hint = lookup_term(query_names[0], doc_ids=allowed_doc_ids or None)
        hinted_doc_ids = list(term_hint.get("candidate_doc_ids") or [])

    for algorithm_label in query_labels:
        match = _ALGORITHM_LABEL_RE.search(algorithm_label)
        if match is None:
            continue
        number = str(match.group("number") or "").strip()
        if not number:
            continue
        matches = sorted(index["algorithm_by_number"].get(number, []), key=_node_sort_key)
        filtered_matches = _filter_nodes_by_doc_ids(matches, allowed_doc_ids)
        match_reason = "exact_algorithm_number"
        if not filtered_matches and allowed_doc_ids:
            continue
        if not filtered_matches:
            filtered_matches = matches
        if hinted_doc_ids:
            hinted_matches = _filter_nodes_by_doc_ids(filtered_matches, hinted_doc_ids)
            if hinted_matches:
                filtered_matches = hinted_matches
                match_reason = "exact_algorithm_number_name_hint"
        if filtered_matches:
            return _collect_algorithm_result(
                filtered_matches,
                lookup_type="algorithm_number",
                lookup_value=algorithm_label,
                match_reason=match_reason,
                query_labels=[algorithm_label],
                query_names=query_names,
            )

    if query_names and term_hint is not None and term_hint.get("matched_entities"):
        fallback = dict(term_hint)
        fallback["lookup_type"] = "algorithm_name"
        fallback["lookup_value"] = query_names[0]
        fallback["match_reason"] = f"term_fallback:{term_hint.get('match_reason')}"
        fallback["ambiguous"] = bool(len(set(fallback.get("candidate_doc_ids") or [])) > 1)
        fallback["fallback_used"] = True
        fallback["required_anchors"] = _dedupe_strings(
            list(fallback.get("required_anchors") or []) + [query_names[0]]
        )
        return fallback

    lookup_type = "algorithm_name" if query_names else "algorithm_number"
    return _lookup_empty(
        lookup_type=lookup_type,
        lookup_value=lookup_value,
        match_reason="no_match",
        fallback_used=False,
    )
