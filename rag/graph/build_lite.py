from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rag.graph.helpers import (
    detect_algorithms,
    get_chunk_text,
    get_doc_id,
    get_page_span,
    get_section_path,
    is_front_matter_section,
    is_definition_like_section,
    leaf_section_title,
    make_algorithm_id,
    make_document_id,
    make_edge_id,
    make_section_id,
    make_term_id,
    normalize_term,
    section_path_to_string,
    term_occurs_in_text,
    write_jsonl,
    read_jsonl,
)
from rag.graph.types import GraphEdge, GraphNode


INPUT_CHUNKS = Path("data/processed/chunks.jsonl")
OUTPUT_NODES = Path("data/processed/graph_lite_nodes.jsonl")
OUTPUT_EDGES = Path("data/processed/graph_lite_edges.jsonl")

TERM_SEEDS = [
    "ML-KEM",
    "ML-DSA",
    "SLH-DSA",
    "encapsulation",
    "decapsulation",
    "key generation",
    "public key",
    "secret key",
    "ciphertext",
    "parameter set",
]


def _merge_page_span(current: tuple[int | None, int | None], new: tuple[int | None, int | None]) -> tuple[int | None, int | None]:
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


def main() -> None:
    chunks = read_jsonl(INPUT_CHUNKS)

    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}

    document_spans: dict[str, tuple[int | None, int | None]] = {}
    section_spans: dict[str, tuple[int | None, int | None]] = {}
    section_meta: dict[str, dict] = {}
    algorithm_meta: dict[str, dict] = {}

    term_definitions: dict[str, set[str]] = defaultdict(set)
    term_section_mentions: dict[str, set[str]] = defaultdict(set)
    term_doc_mentions: dict[str, set[str]] = defaultdict(set)

    # pass 1: collect docs, sections, algorithms, term mentions
    for chunk in chunks:
        doc_id = get_doc_id(chunk)
        if not doc_id:
            continue

        text = get_chunk_text(chunk)
        start_page, end_page = get_page_span(chunk)
        path_parts = get_section_path(chunk)

        # document span
        document_spans[doc_id] = _merge_page_span(document_spans.get(doc_id, (None, None)), (start_page, end_page))

        # section nodes for cumulative prefixes
        for i in range(1, len(path_parts) + 1):
            prefix = path_parts[:i]
            full_path = section_path_to_string(prefix)
            section_id = make_section_id(doc_id, full_path)

            section_spans[section_id] = _merge_page_span(section_spans.get(section_id, (None, None)), (start_page, end_page))
            section_meta[section_id] = {
                "doc_id": doc_id,
                "full_section_path": full_path,
                "leaf_title": leaf_section_title(prefix),
                "depth": len(prefix),
            }

        # algorithms: anchor nodes to algorithm headers, not later prose mentions or front matter.
        if path_parts and not is_front_matter_section(path_parts):
            section_id = make_section_id(doc_id, section_path_to_string(path_parts))
            algorithm_hits = detect_algorithms(text, header_only=True)
            for algo in algorithm_hits:
                algo_key = normalize_term(algo)
                alg_id = make_algorithm_id(doc_id, algo_key)
                candidate = {
                    "doc_id": doc_id,
                    "name": algo,
                    "section_id": section_id,
                    "start_page": start_page,
                    "end_page": end_page,
                    "leaf_title": leaf_section_title(path_parts),
                    "depth": len(path_parts),
                }
                if _should_replace_algorithm_anchor(algorithm_meta.get(alg_id), candidate):
                    algorithm_meta[alg_id] = candidate

        # seeded term extraction
        for term in TERM_SEEDS:
            if term_occurs_in_text(term, text):
                term_norm = normalize_term(term)
                term_doc_mentions[term_norm].add(doc_id)
                if path_parts:
                    section_id = make_section_id(doc_id, section_path_to_string(path_parts))
                    term_section_mentions[term_norm].add(section_id)
                    if is_definition_like_section(path_parts):
                        term_definitions[term_norm].add(section_id)

    # build document nodes
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

    # build section nodes + section->document edges
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
        edge = GraphEdge(
            edge_id=make_edge_id("IN_DOCUMENT", section_id, doc_node_id),
            type="IN_DOCUMENT",
            source_id=section_id,
            target_id=doc_node_id,
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            properties={},
        )
        edges[edge.edge_id] = edge

    # build algorithm nodes + algorithm->section/document edges
    for alg_id, meta in algorithm_meta.items():
        start_page = meta.get("start_page")
        end_page = meta.get("end_page")
        node = GraphNode(
            node_id=alg_id,
            label="Algorithm",
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            display_name=meta["name"],
            properties={"name": meta["name"]},
        )
        nodes[node.node_id] = node

        section_id = meta["section_id"]
        edge1 = GraphEdge(
            edge_id=make_edge_id("APPEARS_IN", alg_id, section_id),
            type="APPEARS_IN",
            source_id=alg_id,
            target_id=section_id,
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            properties={},
        )
        edges[edge1.edge_id] = edge1

        doc_node_id = make_document_id(meta["doc_id"])
        edge2 = GraphEdge(
            edge_id=make_edge_id("IN_DOCUMENT", alg_id, doc_node_id),
            type="IN_DOCUMENT",
            source_id=alg_id,
            target_id=doc_node_id,
            doc_id=meta["doc_id"],
            start_page=start_page,
            end_page=end_page,
            properties={},
        )
        edges[edge2.edge_id] = edge2

    # build term nodes + term edges
    for term_norm, doc_ids in term_doc_mentions.items():
        display_name = next((t for t in TERM_SEEDS if normalize_term(t) == term_norm), term_norm)
        node = GraphNode(
            node_id=make_term_id(term_norm),
            label="Term",
            doc_id=None,
            start_page=None,
            end_page=None,
            display_name=display_name,
            properties={"normalized_term": term_norm},
        )
        nodes[node.node_id] = node

        term_node_id = node.node_id

        for section_id in sorted(term_section_mentions.get(term_norm, set())):
            section_node = nodes.get(section_id)
            edge_type = "DEFINED_IN" if section_id in term_definitions.get(term_norm, set()) else "APPEARS_IN"
            edge = GraphEdge(
                edge_id=make_edge_id(edge_type, term_node_id, section_id),
                type=edge_type,
                source_id=term_node_id,
                target_id=section_id,
                doc_id=section_node.doc_id if section_node else None,
                start_page=section_node.start_page if section_node else None,
                end_page=section_node.end_page if section_node else None,
                properties={},
            )
            edges[edge.edge_id] = edge

        for doc_id in sorted(doc_ids):
            doc_node_id = make_document_id(doc_id)
            edge = GraphEdge(
                edge_id=make_edge_id("IN_DOCUMENT", term_node_id, doc_node_id),
                type="IN_DOCUMENT",
                source_id=term_node_id,
                target_id=doc_node_id,
                doc_id=doc_id,
                start_page=None,
                end_page=None,
                properties={},
            )
            edges[edge.edge_id] = edge

    node_rows = [n.to_dict() for n in sorted(nodes.values(), key=lambda x: (x.label, x.doc_id or "", x.node_id))]
    edge_rows = [e.to_dict() for e in sorted(edges.values(), key=lambda x: (x.type, x.source_id, x.target_id, x.doc_id or ""))]

    write_jsonl(OUTPUT_NODES, node_rows)
    write_jsonl(OUTPUT_EDGES, edge_rows)
    print(f"Wrote {len(node_rows)} nodes to {OUTPUT_NODES}")
    print(f"Wrote {len(edge_rows)} edges to {OUTPUT_EDGES}")


if __name__ == "__main__":
    main()
