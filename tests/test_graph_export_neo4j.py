from __future__ import annotations

import csv
import json
from pathlib import Path

from rag.graph import export_neo4j


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_export_neo4j_serializes_json_properties(tmp_path: Path, monkeypatch) -> None:
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    out_dir = tmp_path / "neo4j"

    _write_jsonl(
        nodes_path,
        [
            {
                "node_id": "term::ml-kem",
                "label": "Term",
                "doc_id": None,
                "start_page": None,
                "end_page": None,
                "display_name": "ML-KEM",
                "properties": {
                    "normalized_term": "ml-kem",
                    "surface_forms": ["ML-KEM"],
                    "term_type": "identifier",
                },
            }
        ],
    )
    _write_jsonl(
        edges_path,
        [
            {
                "edge_id": "edge::IN_DOCUMENT::term::ml-kem::doc::NIST.FIPS.203",
                "type": "IN_DOCUMENT",
                "source_id": "term::ml-kem",
                "target_id": "doc::NIST.FIPS.203",
                "doc_id": "NIST.FIPS.203",
                "start_page": None,
                "end_page": None,
                "properties": {"reason": "test"},
            }
        ],
    )

    monkeypatch.setattr(export_neo4j, "NODES_PATH", nodes_path)
    monkeypatch.setattr(export_neo4j, "EDGES_PATH", edges_path)
    monkeypatch.setattr(export_neo4j, "OUT_DIR", out_dir)

    export_neo4j.main()

    with (out_dir / "nodes.csv").open("r", encoding="utf-8", newline="") as f:
        node_rows = list(csv.DictReader(f))
    with (out_dir / "edges.csv").open("r", encoding="utf-8", newline="") as f:
        edge_rows = list(csv.DictReader(f))

    assert json.loads(node_rows[0]["json_properties"]) == {
        "normalized_term": "ml-kem",
        "surface_forms": ["ML-KEM"],
        "term_type": "identifier",
    }
    assert json.loads(edge_rows[0]["json_properties"]) == {"reason": "test"}
