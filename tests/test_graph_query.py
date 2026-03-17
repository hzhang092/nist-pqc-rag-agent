from __future__ import annotations

import json
from pathlib import Path

from rag.graph import query


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _patch_graph(tmp_path: Path, monkeypatch, *, nodes: list[dict], edges: list[dict]) -> None:
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    _write_jsonl(nodes_path, nodes)
    _write_jsonl(edges_path, edges)
    monkeypatch.setattr(query, "NODES_PATH", nodes_path)
    monkeypatch.setattr(query, "EDGES_PATH", edges_path)
    query._graph_index.cache_clear()


def test_lookup_term_returns_doc_ids_sections_and_anchors(tmp_path: Path, monkeypatch) -> None:
    _patch_graph(
        tmp_path,
        monkeypatch,
        nodes=[
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
        edges=[
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

    result = query.lookup_term("encapsulation")

    assert result["lookup_type"] == "term"
    assert result["lookup_value"] == "encapsulation"
    assert result["match_reason"] == "exact_normalized_term"
    assert result["candidate_doc_ids"] == ["NIST.FIPS.203"]
    assert result["candidate_section_ids"] == ["section::NIST.FIPS.203::2. Terms and Definitions"]
    assert result["required_anchors"] == ["encapsulation"]


def test_lookup_algorithm_exact_name_returns_scoped_section(tmp_path: Path, monkeypatch) -> None:
    _patch_graph(
        tmp_path,
        monkeypatch,
        nodes=[
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
                "node_id": "section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                "label": "Section",
                "doc_id": "NIST.FIPS.203",
                "start_page": 44,
                "end_page": 45,
                "display_name": "7.1 ML-KEM Key Generation",
                "properties": {"full_section_path": "7.1 ML-KEM Key Generation", "leaf_title": "7.1 ML-KEM Key Generation", "depth": 2},
            },
            {
                "node_id": "alg::NIST.FIPS.203::19",
                "label": "Algorithm",
                "doc_id": "NIST.FIPS.203",
                "start_page": 44,
                "end_page": 45,
                "display_name": "Algorithm 19",
                "properties": {
                    "algorithm_label": "Algorithm 19",
                    "algorithm_name": "ML-KEM.KeyGen",
                    "algorithm_number": "19",
                    "raw_header": "Algorithm 19 ML-KEM.KeyGen ()",
                    "section_id": "section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                },
            },
        ],
        edges=[
            {
                "edge_id": "edge::APPEARS_IN::alg::NIST.FIPS.203::19::section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                "type": "APPEARS_IN",
                "source_id": "alg::NIST.FIPS.203::19",
                "target_id": "section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                "doc_id": "NIST.FIPS.203",
                "start_page": 44,
                "end_page": 45,
                "properties": {},
            },
            {
                "edge_id": "edge::IN_DOCUMENT::alg::NIST.FIPS.203::19::doc::NIST.FIPS.203",
                "type": "IN_DOCUMENT",
                "source_id": "alg::NIST.FIPS.203::19",
                "target_id": "doc::NIST.FIPS.203",
                "doc_id": "NIST.FIPS.203",
                "start_page": 44,
                "end_page": 45,
                "properties": {},
            },
        ],
    )

    result = query.lookup_algorithm("What are the steps in ML-KEM.KeyGen?")

    assert result["lookup_type"] == "algorithm_name"
    assert result["lookup_value"] == "ML-KEM.KeyGen"
    assert result["match_reason"] == "exact_algorithm_name"
    assert result["ambiguous"] is False
    assert result["fallback_used"] is False
    assert result["candidate_doc_ids"] == ["NIST.FIPS.203"]
    assert result["candidate_section_ids"] == ["section::NIST.FIPS.203::7.1 ML-KEM Key Generation"]
    assert result["required_anchors"] == ["ML-KEM.KeyGen"]


def test_lookup_algorithm_bare_number_keeps_multi_doc_candidates(tmp_path: Path, monkeypatch) -> None:
    _patch_graph(
        tmp_path,
        monkeypatch,
        nodes=[
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
                "node_id": "doc::NIST.FIPS.205",
                "label": "Document",
                "doc_id": "NIST.FIPS.205",
                "start_page": 1,
                "end_page": 60,
                "display_name": "NIST.FIPS.205",
                "properties": {"doc_id": "NIST.FIPS.205"},
            },
            {
                "node_id": "section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                "label": "Section",
                "doc_id": "NIST.FIPS.203",
                "start_page": 44,
                "end_page": 45,
                "display_name": "7.1 ML-KEM Key Generation",
                "properties": {"full_section_path": "7.1 ML-KEM Key Generation", "leaf_title": "7.1 ML-KEM Key Generation", "depth": 2},
            },
            {
                "node_id": "section::NIST.FIPS.205::9.2 SLH-DSA Signature Generation",
                "label": "Section",
                "doc_id": "NIST.FIPS.205",
                "start_page": 45,
                "end_page": 46,
                "display_name": "9.2 SLH-DSA Signature Generation",
                "properties": {"full_section_path": "9.2 SLH-DSA Signature Generation", "leaf_title": "9.2 SLH-DSA Signature Generation", "depth": 2},
            },
            {
                "node_id": "alg::NIST.FIPS.203::19",
                "label": "Algorithm",
                "doc_id": "NIST.FIPS.203",
                "start_page": 44,
                "end_page": 45,
                "display_name": "Algorithm 19",
                "properties": {
                    "algorithm_label": "Algorithm 19",
                    "algorithm_name": "ML-KEM.KeyGen",
                    "algorithm_number": "19",
                    "raw_header": "Algorithm 19 ML-KEM.KeyGen ()",
                    "section_id": "section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                },
            },
            {
                "node_id": "alg::NIST.FIPS.205::19",
                "label": "Algorithm",
                "doc_id": "NIST.FIPS.205",
                "start_page": 45,
                "end_page": 46,
                "display_name": "Algorithm 19",
                "properties": {
                    "algorithm_label": "Algorithm 19",
                    "algorithm_name": "slh_sign_internal",
                    "algorithm_number": "19",
                    "raw_header": "Algorithm 19 slh_sign_internal (M, SK, addrnd)",
                    "section_id": "section::NIST.FIPS.205::9.2 SLH-DSA Signature Generation",
                },
            },
            {
                "node_id": "alg::NIST.FIPS.204::18",
                "label": "Algorithm",
                "doc_id": "NIST.FIPS.204",
                "start_page": 41,
                "end_page": 42,
                "display_name": "Algorithm 18",
                "properties": {
                    "algorithm_label": "Algorithm 18",
                    "algorithm_name": "SimpleBitUnpack",
                    "algorithm_number": "18",
                    "raw_header": "Algorithm 18 SimpleBitUnpack ... Algorithm 19 BitUnpack (v, a, b)",
                    "section_id": "section::NIST.FIPS.204::7.1 Conversion Between Data Types",
                },
            },
        ],
        edges=[
            {
                "edge_id": "edge::APPEARS_IN::alg::NIST.FIPS.203::19::section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                "type": "APPEARS_IN",
                "source_id": "alg::NIST.FIPS.203::19",
                "target_id": "section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
                "doc_id": "NIST.FIPS.203",
                "start_page": 44,
                "end_page": 45,
                "properties": {},
            },
            {
                "edge_id": "edge::APPEARS_IN::alg::NIST.FIPS.205::19::section::NIST.FIPS.205::9.2 SLH-DSA Signature Generation",
                "type": "APPEARS_IN",
                "source_id": "alg::NIST.FIPS.205::19",
                "target_id": "section::NIST.FIPS.205::9.2 SLH-DSA Signature Generation",
                "doc_id": "NIST.FIPS.205",
                "start_page": 45,
                "end_page": 46,
                "properties": {},
            },
        ],
    )

    result = query.lookup_algorithm("Algorithm 19")

    assert result["lookup_type"] == "algorithm_number"
    assert result["lookup_value"] == "Algorithm 19"
    assert result["match_reason"] == "exact_algorithm_number"
    assert result["ambiguous"] is True
    assert result["candidate_doc_ids"] == ["NIST.FIPS.203", "NIST.FIPS.205"]
    assert result["candidate_section_ids"] == [
        "section::NIST.FIPS.203::7.1 ML-KEM Key Generation",
        "section::NIST.FIPS.205::9.2 SLH-DSA Signature Generation",
    ]
    assert result["required_anchors"] == ["Algorithm 19"]
    assert all(entity["algorithm_number"] == "19" for entity in result["matched_entities"])


def test_lookup_algorithm_falls_back_to_term_for_identifier_scope(tmp_path: Path, monkeypatch) -> None:
    _patch_graph(
        tmp_path,
        monkeypatch,
        nodes=[
            {
                "node_id": "doc::NIST.FIPS.204",
                "label": "Document",
                "doc_id": "NIST.FIPS.204",
                "start_page": 1,
                "end_page": 55,
                "display_name": "NIST.FIPS.204",
                "properties": {"doc_id": "NIST.FIPS.204"},
            },
            {
                "node_id": "section::NIST.FIPS.204::5.1 ML-DSA Key Generation",
                "label": "Section",
                "doc_id": "NIST.FIPS.204",
                "start_page": 27,
                "end_page": 27,
                "display_name": "5.1 ML-DSA Key Generation",
                "properties": {"full_section_path": "5.1 ML-DSA Key Generation", "leaf_title": "5.1 ML-DSA Key Generation", "depth": 2},
            },
            {
                "node_id": "alg::NIST.FIPS.204::6",
                "label": "Algorithm",
                "doc_id": "NIST.FIPS.204",
                "start_page": 32,
                "end_page": 32,
                "display_name": "Algorithm 6",
                "properties": {
                    "algorithm_label": "Algorithm 6",
                    "algorithm_name": "ML-DSA.KeyGen_internal",
                    "algorithm_number": "6",
                    "raw_header": "Algorithm 6 ML-DSA.KeyGen_internal (xi)",
                    "section_id": "section::NIST.FIPS.204::6.1 ML-DSA Key Generation (Internal)",
                },
            },
            {
                "node_id": "term::ml-dsa.keygen",
                "label": "Term",
                "doc_id": None,
                "start_page": None,
                "end_page": None,
                "display_name": "ML-DSA.KeyGen",
                "properties": {
                    "normalized_term": "ml-dsa.keygen",
                    "surface_forms": ["ML-DSA.KeyGen"],
                    "term_type": "identifier",
                    "definition_strength": "seed",
                },
            },
        ],
        edges=[
            {
                "edge_id": "edge::APPEARS_IN::term::ml-dsa.keygen::section::NIST.FIPS.204::5.1 ML-DSA Key Generation",
                "type": "APPEARS_IN",
                "source_id": "term::ml-dsa.keygen",
                "target_id": "section::NIST.FIPS.204::5.1 ML-DSA Key Generation",
                "doc_id": "NIST.FIPS.204",
                "start_page": 27,
                "end_page": 27,
                "properties": {},
            },
            {
                "edge_id": "edge::IN_DOCUMENT::term::ml-dsa.keygen::doc::NIST.FIPS.204",
                "type": "IN_DOCUMENT",
                "source_id": "term::ml-dsa.keygen",
                "target_id": "doc::NIST.FIPS.204",
                "doc_id": "NIST.FIPS.204",
                "start_page": None,
                "end_page": None,
                "properties": {},
            },
        ],
    )

    result = query.lookup_algorithm("ML-DSA.KeyGen")

    assert result["lookup_type"] == "algorithm_name"
    assert result["lookup_value"] == "ML-DSA.KeyGen"
    assert result["fallback_used"] is True
    assert result["match_reason"] == "term_fallback:exact_normalized_term"
    assert result["candidate_doc_ids"] == ["NIST.FIPS.204"]
    assert result["candidate_section_ids"] == ["section::NIST.FIPS.204::5.1 ML-DSA Key Generation"]
    assert result["required_anchors"] == ["ML-DSA.KeyGen"]
