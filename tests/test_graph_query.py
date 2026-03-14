from __future__ import annotations

import json
from pathlib import Path

from rag.graph import query


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_lookup_term_returns_doc_ids_sections_and_anchors(tmp_path: Path, monkeypatch) -> None:
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    _write_jsonl(
        nodes_path,
        [
            {
                "node_id": "doc::NIST.FIPS.203",
                "label": "Document",
                "doc_id": "NIST.FIPS.203",
                "start_page": 1,
                "end_page": 50,
                "display_name": "NIST.FIPS.203",
                "properties": {"doc_id": "NIST.FIPS.203"},
            },
            {
                "node_id": "section::NIST.FIPS.203::2. Terms and Definitions",
                "label": "Section",
                "doc_id": "NIST.FIPS.203",
                "start_page": 11,
                "end_page": 12,
                "display_name": "2. Terms and Definitions",
                "properties": {"full_section_path": "2. Terms and Definitions", "leaf_title": "2. Terms and Definitions", "depth": 1},
            },
            {
                "node_id": "term::encapsulation",
                "label": "Term",
                "doc_id": None,
                "start_page": None,
                "end_page": None,
                "display_name": "encapsulation",
                "properties": {
                    "normalized_term": "encapsulation",
                    "surface_forms": ["encapsulation"],
                    "term_type": "operation",
                    "definition_strength": "heuristic_definition_section",
                },
            },
        ],
    )
    _write_jsonl(
        edges_path,
        [
            {
                "edge_id": "edge::DEFINED_IN::term::encapsulation::section::NIST.FIPS.203::2. Terms and Definitions",
                "type": "DEFINED_IN",
                "source_id": "term::encapsulation",
                "target_id": "section::NIST.FIPS.203::2. Terms and Definitions",
                "doc_id": "NIST.FIPS.203",
                "start_page": 11,
                "end_page": 12,
                "properties": {},
            },
            {
                "edge_id": "edge::IN_DOCUMENT::term::encapsulation::doc::NIST.FIPS.203",
                "type": "IN_DOCUMENT",
                "source_id": "term::encapsulation",
                "target_id": "doc::NIST.FIPS.203",
                "doc_id": "NIST.FIPS.203",
                "start_page": None,
                "end_page": None,
                "properties": {},
            },
        ],
    )

    monkeypatch.setattr(query, "NODES_PATH", nodes_path)
    monkeypatch.setattr(query, "EDGES_PATH", edges_path)
    query._graph_index.cache_clear()

    result = query.lookup_term("encapsulation")

    assert result["match_reason"] == "exact_normalized_term"
    assert result["candidate_doc_ids"] == ["NIST.FIPS.203"]
    assert result["candidate_section_ids"] == ["section::NIST.FIPS.203::2. Terms and Definitions"]
    assert result["required_anchors"] == ["encapsulation"]
