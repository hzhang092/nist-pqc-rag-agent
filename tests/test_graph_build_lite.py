from __future__ import annotations

import json
from pathlib import Path

from rag.graph import build_lite


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_fixture(tmp_path: Path, monkeypatch, rows: list[dict]) -> tuple[Path, Path]:
    chunks_path = tmp_path / "chunks.jsonl"
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    _write_jsonl(chunks_path, rows)
    monkeypatch.setattr(build_lite, "INPUT_CHUNKS", chunks_path)
    monkeypatch.setattr(build_lite, "OUTPUT_NODES", nodes_path)
    monkeypatch.setattr(build_lite, "OUTPUT_EDGES", edges_path)
    build_lite.main()
    return nodes_path, edges_path


def test_build_lite_canonicalizes_algorithms_and_adds_child_edges(tmp_path: Path, monkeypatch) -> None:
    nodes_path, edges_path = _build_fixture(
        tmp_path,
        monkeypatch,
        [
            {
                "chunk_id": "DOC::p0001::c000",
                "doc_id": "DOC",
                "start_page": 1,
                "end_page": 1,
                "section_path": "List of Tables",
                "block_type": "table",
                "text": "Algorithm 1 | ML-DSA.KeyGen | 7",
            },
            {
                "chunk_id": "DOC::p0007::c000",
                "doc_id": "DOC",
                "start_page": 7,
                "end_page": 7,
                "section_path": "5. External Functions > 5.1 ML-DSA Key Generation",
                "block_type": "algorithm",
                "text": "Algorithm 1 ML-DSA.KeyGen ()\nGenerates a public-private key pair.",
            },
            {
                "chunk_id": "DOC::p0008::c000",
                "doc_id": "DOC",
                "start_page": 8,
                "end_page": 8,
                "section_path": "Appendix A",
                "block_type": "text",
                "text": "Algorithm 1 is described above.",
            },
        ],
    )

    nodes = {row["node_id"]: row for row in _load_jsonl(nodes_path)}
    edges = _load_jsonl(edges_path)

    algorithm = nodes["alg::DOC::1"]
    leaf_section_id = "section::DOC::5. External Functions > 5.1 ML-DSA Key Generation"
    parent_section_id = "section::DOC::5. External Functions"

    assert "alg::DOC::ml-dsa.keygen" not in nodes
    assert algorithm["display_name"] == "Algorithm 1"
    assert algorithm["properties"] == {
        "algorithm_label": "Algorithm 1",
        "algorithm_name": "ML-DSA.KeyGen",
        "algorithm_number": "1",
        "raw_header": "Algorithm 1 ML-DSA.KeyGen ()",
        "section_id": leaf_section_id,
    }

    child_edge = next(
        edge
        for edge in edges
        if edge["type"] == "CHILD_OF" and edge["source_id"] == leaf_section_id
    )
    assert child_edge["target_id"] == parent_section_id


def test_build_lite_extracts_deterministic_terms_and_rejects_generic_noise(tmp_path: Path, monkeypatch) -> None:
    nodes_path, edges_path = _build_fixture(
        tmp_path,
        monkeypatch,
        [
            {
                "chunk_id": "DOC::p0011::c000",
                "doc_id": "DOC",
                "start_page": 11,
                "end_page": 11,
                "section_path": "2. Terms and Definitions",
                "block_type": "text",
                "text": "encapsulation | The process of applying Encaps.\npublic key | Public component.",
            },
            {
                "chunk_id": "DOC::p0012::c000",
                "doc_id": "DOC",
                "start_page": 12,
                "end_page": 12,
                "section_path": "5. External Functions > 5.1 ML-KEM Encapsulation",
                "block_type": "algorithm",
                "text": "Algorithm 2 ML-KEM.Encaps ()\nThe ML-KEM.Encaps routine uses SHAKE128 for encapsulation.",
            },
            {
                "chunk_id": "DOC::p0013::c000",
                "doc_id": "DOC",
                "start_page": 13,
                "end_page": 13,
                "section_path": "5. External Functions > 5.2 ML-KEM Encapsulation Notes",
                "block_type": "text",
                "text": "ML-KEM.Encaps returns a ciphertext and a shared secret.",
            },
            {
                "chunk_id": "DOC::p0014::c000",
                "doc_id": "DOC",
                "start_page": 14,
                "end_page": 14,
                "section_path": "3. Additional Requirements",
                "block_type": "text",
                "text": "These requirements are critical for conformance.",
            },
        ],
    )

    nodes = {row["node_id"]: row for row in _load_jsonl(nodes_path)}
    edges = _load_jsonl(edges_path)

    encapsulation = nodes["term::encapsulation"]
    ml_kem_encaps = nodes["term::ml-kem.encaps"]
    shake128 = nodes["term::shake128"]

    assert "term::requirements" not in nodes
    assert encapsulation["properties"]["definition_strength"] == "heuristic_definition_section"
    assert encapsulation["properties"]["term_type"] == "operation"
    assert ml_kem_encaps["properties"]["term_type"] == "identifier"
    assert shake128["properties"]["term_type"] == "acronym"
    assert ml_kem_encaps["properties"]["surface_forms"] == ["ML-KEM.Encaps"]

    definition_edge = next(
        edge
        for edge in edges
        if edge["type"] == "DEFINED_IN" and edge["source_id"] == encapsulation["node_id"]
    )
    assert definition_edge["target_id"] == "section::DOC::2. Terms and Definitions"


def test_build_lite_is_byte_deterministic(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {
            "chunk_id": "DOC::p0011::c000",
            "doc_id": "DOC",
            "start_page": 11,
            "end_page": 11,
            "section_path": "2. Terms and Definitions",
            "block_type": "text",
            "text": "encapsulation | The process of applying Encaps.",
        },
        {
            "chunk_id": "DOC::p0012::c000",
            "doc_id": "DOC",
            "start_page": 12,
            "end_page": 12,
            "section_path": "5. External Functions > 5.1 ML-KEM Encapsulation",
            "block_type": "algorithm",
            "text": "Algorithm 2 ML-KEM.Encaps ()\nThe ML-KEM.Encaps routine uses SHAKE128.",
        },
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    _write_jsonl(chunks_path, rows)

    monkeypatch.setattr(build_lite, "INPUT_CHUNKS", chunks_path)
    monkeypatch.setattr(build_lite, "OUTPUT_NODES", nodes_path)
    monkeypatch.setattr(build_lite, "OUTPUT_EDGES", edges_path)

    build_lite.main()
    first_nodes = nodes_path.read_bytes()
    first_edges = edges_path.read_bytes()

    build_lite.main()
    assert nodes_path.read_bytes() == first_nodes
    assert edges_path.read_bytes() == first_edges
