from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from rag.graph.helpers import (
    extract_algorithm_header_info,
    extract_term_candidates,
    get_block_type,
    get_chunk_text,
    get_doc_id,
    get_page_span,
    get_section_path,
    is_definition_like_section,
    is_front_matter_section,
    leaf_section_title,
    make_algorithm_id,
    make_document_id,
    make_edge_id,
    make_section_id,
    make_term_id,
    read_jsonl,
    section_path_to_string,
    write_jsonl,
)
from rag.graph.types import GraphEdge, GraphNode


INPUT_CHUNKS = Path("data/processed/chunks.jsonl")
OUTPUT_NODES = Path("data/processed/graph_lite_nodes.jsonl")
OUTPUT_EDGES = Path("data/processed/graph_lite_edges.jsonl")

_TERM_TYPE_PRIORITY = {
    "identifier": 0,
    "acronym": 1,
    "operation": 2,
    "concept": 3,
    "symbol": 4,
}


def _merge_page_span(
    current: tuple[int | None, int | None],
    new: tuple[int | None, int | None],
) -> tuple[int | None, int | None]:
    cur_start, cur_end = current
    new_start, new_end = new

    start_candidates = [x for x in [cur_start, new_start] if x is not None]
    end_candidates = [x for x in [cur_end, new_end] if x is not None]

    start = min(start_candidates) if start_candidates else None
    end = max(end_candidates) if end_candidates else None
    return start, end


def _algorithm_anchor_key(meta: dict) -> tuple[int, int, int, str]:
    leaf_title = str(meta.get("leaf_title", "") or "")
    start_page = meta.get("start_page")
    return (
        0 if leaf_title.lower().startswith("algorithm ") else 1,
        start_page if isinstance(start_page, int) else 10**9,
        -int(meta.get("depth", 0) or 0),
        str(meta.get("section_id", "") or ""),
    )


def _should_replace_algorithm_anchor(current: dict | None, candidate: dict) -> bool:
    if current is None:
        return True
    return _algorithm_anchor_key(candidate) < _algorithm_anchor_key(current)


def _new_term_stats() -> dict:
    return {
        "surface_forms": Counter(),
        "doc_ids": set(),
        "section_ids": set(),
        "definition_section_ids": set(),
        "anchor_section_ids": set(),
        "chunk_ids": set(),
        "sources": set(),
        "term_type_counts": Counter(),
        "identifier_regex": False,
    }


def _preferred_surface_form(surface_forms: Counter) -> str:
    ranked = sorted(
        surface_forms.items(),
        key=lambda item: (
            -int(item[1]),
            0 if any(ch.isupper() for ch in item[0]) else 1,
            0 if any(mark in item[0] for mark in (".", "-", "_")) else 1,
            item[0].lower(),
        ),
    )
    return ranked[0][0]


def _preferred_term_type(term_type_counts: Counter) -> str:
    ranked = sorted(
        term_type_counts.items(),
        key=lambda item: (-int(item[1]), _TERM_TYPE_PRIORITY.get(item[0], 99), item[0]),
    )
    return ranked[0][0] if ranked else "concept"


def _definition_strength(term_stats: dict) -> str:
    if term_stats["definition_section_ids"]:
        return "heuristic_definition_section"
    return "seed"


def _term_is_valid(term_stats: dict) -> bool:
    if term_stats["definition_section_ids"]:
        return True
    if term_stats["identifier_regex"]:
        return True
    if len(term_stats["chunk_ids"]) >= 2:
        return True
    if term_stats["anchor_section_ids"]:
        return True
    return False


def main() -> None:
    chunks = read_jsonl(INPUT_CHUNKS)

    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}

    document_spans: dict[str, tuple[int | None, int | None]] = {}
    section_spans: dict[str, tuple[int | None, int | None]] = {}
    section_meta: dict[str, dict] = {}
    section_parents: set[tuple[str, str]] = set()
    algorithm_meta: dict[str, dict] = {}

    # pass 1: collect docs, sections, section hierarchy, and algorithm anchors
    for chunk in chunks:
        doc_id = get_doc_id(chunk)
        if not doc_id:
            continue

        start_page, end_page = get_page_span(chunk)
        path_parts = get_section_path(chunk)

        document_spans[doc_id] = _merge_page_span(
            document_spans.get(doc_id, (None, None)),
            (start_page, end_page),
        )

        previous_section_id: str | None = None
        for i in range(1, len(path_parts) + 1):
            prefix = path_parts[:i]
            full_path = section_path_to_string(prefix)
            section_id = make_section_id(doc_id, full_path)
            section_spans[section_id] = _merge_page_span(
                section_spans.get(section_id, (None, None)),
                (start_page, end_page),
            )
            section_meta[section_id] = {
                "doc_id": doc_id,
                "full_section_path": full_path,
                "leaf_title": leaf_section_title(prefix),
                "depth": len(prefix),
            }
            if previous_section_id is not None:
                section_parents.add((section_id, previous_section_id))
            previous_section_id = section_id

        if not path_parts or is_front_matter_section(path_parts):
            continue

        section_id = make_section_id(doc_id, section_path_to_string(path_parts))
        text = get_chunk_text(chunk)
        for header in extract_algorithm_header_info(text):
            algorithm_number = str(header.get("algorithm_number") or "")
            if not algorithm_number:
                continue
            alg_id = make_algorithm_id(doc_id, algorithm_number)
            candidate = {
                "doc_id": doc_id,
                "section_id": section_id,
                "start_page": start_page,
                "end_page": end_page,
                "leaf_title": leaf_section_title(path_parts),
                "depth": len(path_parts),
                "algorithm_number": algorithm_number,
                "algorithm_label": header.get("algorithm_label"),
                "algorithm_name": header.get("algorithm_name"),
                "raw_header": header.get("raw_header"),
            }
            if _should_replace_algorithm_anchor(algorithm_meta.get(alg_id), candidate):
                algorithm_meta[alg_id] = candidate

    algorithm_ids_by_section: dict[str, set[str]] = defaultdict(set)
    for alg_id, meta in algorithm_meta.items():
        algorithm_ids_by_section[str(meta.get("section_id") or "")].add(alg_id)

    term_stats_by_norm: dict[str, dict] = defaultdict(_new_term_stats)

    # pass 2: collect deterministic term candidates across chunks
    for chunk in chunks:
        doc_id = get_doc_id(chunk)
        if not doc_id:
            continue

        path_parts = get_section_path(chunk)
        if not path_parts or is_front_matter_section(path_parts):
            continue

        section_id = make_section_id(doc_id, section_path_to_string(path_parts))
        text = get_chunk_text(chunk)
        chunk_id = str(chunk.get("chunk_id", "") or "")
        block_type = get_block_type(chunk)
        has_algorithm_anchor = bool(algorithm_ids_by_section.get(section_id))
        definition_like = is_definition_like_section(path_parts)

        for candidate in extract_term_candidates(text, path_parts, block_type):
            term_norm = str(candidate.get("normalized_term") or "")
            if not term_norm:
                continue

            term_stats = term_stats_by_norm[term_norm]
            term_stats["surface_forms"][str(candidate.get("surface_form") or term_norm)] += 1
            term_stats["doc_ids"].add(doc_id)
            term_stats["section_ids"].add(section_id)
            term_stats["sources"].add(str(candidate.get("source") or ""))
            term_stats["term_type_counts"][str(candidate.get("term_type") or "concept")] += 1
            if chunk_id:
                term_stats["chunk_ids"].add(chunk_id)
            if str(candidate.get("source") or "") == "definition_section" and definition_like:
                term_stats["definition_section_ids"].add(section_id)
            if str(candidate.get("source") or "") in {"algorithm_header", "section_heading"}:
                term_stats["anchor_section_ids"].add(section_id)
            if str(candidate.get("source") or "") == "identifier_regex":
                term_stats["identifier_regex"] = True
            if has_algorithm_anchor:
                term_stats["anchor_section_ids"].add(section_id)

    # pass 3: materialize nodes and edges
    for doc_id, (start_page, end_page) in document_spans.items():
        node = GraphNode(
            node_id=make_document_id(doc_id),
            label="Document",
            doc_id=doc_id,
            start_page=start_page,
            end_page=end_page,
            display_name=doc_id,
            properties={"doc_id": doc_id},
        )
        nodes[node.node_id] = node

    for section_id, meta in section_meta.items():
        start_page, end_page = section_spans.get(section_id, (None, None))
        node = GraphNode(
            node_id=section_id,
            label="Section",
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            display_name=meta["leaf_title"] or meta["full_section_path"],
            properties={
                "full_section_path": meta["full_section_path"],
                "leaf_title": meta["leaf_title"],
                "depth": meta["depth"],
            },
        )
        nodes[node.node_id] = node

        doc_node_id = make_document_id(meta["doc_id"])
        edges[make_edge_id("IN_DOCUMENT", section_id, doc_node_id)] = GraphEdge(
            edge_id=make_edge_id("IN_DOCUMENT", section_id, doc_node_id),
            type="IN_DOCUMENT",
            source_id=section_id,
            target_id=doc_node_id,
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            properties={},
        )

    for child_section_id, parent_section_id in sorted(section_parents):
        child_node = nodes.get(child_section_id)
        if child_node is None:
            continue
        edge_id = make_edge_id("CHILD_OF", child_section_id, parent_section_id)
        edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            type="CHILD_OF",
            source_id=child_section_id,
            target_id=parent_section_id,
            doc_id=child_node.doc_id,
            start_page=child_node.start_page,
            end_page=child_node.end_page,
            properties={},
        )

    for alg_id, meta in algorithm_meta.items():
        start_page = meta.get("start_page")
        end_page = meta.get("end_page")
        node = GraphNode(
            node_id=alg_id,
            label="Algorithm",
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            display_name=str(meta.get("algorithm_label") or alg_id),
            properties={
                "algorithm_number": meta.get("algorithm_number"),
                "algorithm_label": meta.get("algorithm_label"),
                "algorithm_name": meta.get("algorithm_name"),
                "raw_header": meta.get("raw_header"),
                "section_id": meta.get("section_id"),
            },
        )
        nodes[node.node_id] = node

        section_id = str(meta.get("section_id") or "")
        edge1_id = make_edge_id("APPEARS_IN", alg_id, section_id)
        edges[edge1_id] = GraphEdge(
            edge_id=edge1_id,
            type="APPEARS_IN",
            source_id=alg_id,
            target_id=section_id,
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            properties={},
        )

        doc_node_id = make_document_id(meta["doc_id"])
        edge2_id = make_edge_id("IN_DOCUMENT", alg_id, doc_node_id)
        edges[edge2_id] = GraphEdge(
            edge_id=edge2_id,
            type="IN_DOCUMENT",
            source_id=alg_id,
            target_id=doc_node_id,
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            properties={},
        )

    for term_norm, term_stats in term_stats_by_norm.items():
        if not _term_is_valid(term_stats):
            continue

        display_name = _preferred_surface_form(term_stats["surface_forms"])
        term_node_id = make_term_id(term_norm)
        node = GraphNode(
            node_id=term_node_id,
            label="Term",
            doc_id=None,
            start_page=None,
            end_page=None,
            display_name=display_name,
            properties={
                "normalized_term": term_norm,
                "surface_forms": sorted(term_stats["surface_forms"]),
                "term_type": _preferred_term_type(term_stats["term_type_counts"]),
                "definition_strength": _definition_strength(term_stats),
            },
        )
        nodes[term_node_id] = node

        for section_id in sorted(term_stats["section_ids"]):
            section_node = nodes.get(section_id)
            edge_type = (
                "DEFINED_IN"
                if section_id in term_stats["definition_section_ids"]
                else "APPEARS_IN"
            )
            edge_id = make_edge_id(edge_type, term_node_id, section_id)
            edges[edge_id] = GraphEdge(
                edge_id=edge_id,
                type=edge_type,
                source_id=term_node_id,
                target_id=section_id,
                doc_id=section_node.doc_id if section_node else None,
                start_page=section_node.start_page if section_node else None,
                end_page=section_node.end_page if section_node else None,
                properties={},
            )

        for doc_id in sorted(term_stats["doc_ids"]):
            doc_node_id = make_document_id(doc_id)
            edge_id = make_edge_id("IN_DOCUMENT", term_node_id, doc_node_id)
            edges[edge_id] = GraphEdge(
                edge_id=edge_id,
                type="IN_DOCUMENT",
                source_id=term_node_id,
                target_id=doc_node_id,
                doc_id=doc_id,
                start_page=None,
                end_page=None,
                properties={},
            )

    node_rows = [
        n.to_dict()
        for n in sorted(nodes.values(), key=lambda x: (x.label, x.doc_id or "", x.node_id))
    ]
    edge_rows = [
        e.to_dict()
        for e in sorted(edges.values(), key=lambda x: (x.type, x.source_id, x.target_id, x.doc_id or ""))
    ]

    write_jsonl(OUTPUT_NODES, node_rows)
    write_jsonl(OUTPUT_EDGES, edge_rows)
    print(f"Wrote {len(node_rows)} nodes to {OUTPUT_NODES}")
    print(f"Wrote {len(edge_rows)} edges to {OUTPUT_EDGES}")


if __name__ == "__main__":
    main()
