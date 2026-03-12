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


def test_build_lite_anchors_algorithms_to_definition_chunks(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"

    _write_jsonl(
        chunks_path,
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
                "chunk_id": "DOC::p0002::c000",
                "doc_id": "DOC",
                "start_page": 2,
                "end_page": 2,
                "section_path": "2. Notes",
                "block_type": "text",
                "text": "The computation involves running the Compress algorithm 256 times.",
            },
            {
                "chunk_id": "DOC::p0003::c000",
                "doc_id": "DOC",
                "start_page": 3,
                "end_page": 3,
                "section_path": "3. Additional Requirements",
                "block_type": "text",
                "text": "This section describes several requirements when implementing ML-DSA.These are critical.",
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

    monkeypatch.setattr(build_lite, "INPUT_CHUNKS", chunks_path)
    monkeypatch.setattr(build_lite, "OUTPUT_NODES", nodes_path)
    monkeypatch.setattr(build_lite, "OUTPUT_EDGES", edges_path)

    build_lite.main()

    nodes = {row["node_id"]: row for row in _load_jsonl(nodes_path)}
    edges = _load_jsonl(edges_path)

    numeric = nodes["alg::DOC::1"]
    named = nodes["alg::DOC::ml-dsa.keygen"]
    section = nodes["section::DOC::5. External Functions > 5.1 ML-DSA Key Generation"]

    assert numeric["start_page"] == 7
    assert numeric["end_page"] == 7
    assert named["start_page"] == 7
    assert named["end_page"] == 7
    assert "alg::DOC::256" not in nodes
    assert "alg::DOC::ml-dsa.these" not in nodes

    edge = next(e for e in edges if e["source_id"] == numeric["node_id"] and e["type"] == "APPEARS_IN")
    assert edge["target_id"] == section["node_id"]
    assert edge["start_page"] == section["start_page"] == numeric["start_page"]
