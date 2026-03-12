from __future__ import annotations

import csv
from pathlib import Path

from rag.graph.helpers import read_jsonl


NODES_PATH = Path("data/processed/graph_lite_nodes.jsonl")
EDGES_PATH = Path("data/processed/graph_lite_edges.jsonl")
OUT_DIR = Path("data/processed/neo4j_import")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    nodes = read_jsonl(NODES_PATH)
    edges = read_jsonl(EDGES_PATH)

    node_rows = []
    for n in nodes:
        props = n.get("properties", {})
        node_rows.append(
            {
                "node_id:ID": n["node_id"],
                "label": n["label"],
                "doc_id": n.get("doc_id"),
                "start_page:int": n.get("start_page"),
                "end_page:int": n.get("end_page"),
                "display_name": n.get("display_name"),
                "json_properties": str(props),
            }
        )

    edge_rows = []
    for e in edges:
        props = e.get("properties", {})
        edge_rows.append(
            {
                "edge_id": e["edge_id"],
                ":START_ID": e["source_id"],
                ":END_ID": e["target_id"],
                ":TYPE": e["type"],
                "doc_id": e.get("doc_id"),
                "start_page:int": e.get("start_page"),
                "end_page:int": e.get("end_page"),
                "json_properties": str(props),
            }
        )

    write_csv(
        OUT_DIR / "nodes.csv",
        ["node_id:ID", "label", "doc_id", "start_page:int", "end_page:int", "display_name", "json_properties"],
        node_rows,
    )
    write_csv(
        OUT_DIR / "edges.csv",
        ["edge_id", ":START_ID", ":END_ID", ":TYPE", "doc_id", "start_page:int", "end_page:int", "json_properties"],
        edge_rows,
    )

    constraints = """CREATE CONSTRAINT node_id_unique IF NOT EXISTS
FOR (n:Entity) REQUIRE n.node_id IS UNIQUE;
"""
    load = """LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {node_id: row.`node_id:ID`})
SET n.label = row.label,
    n.doc_id = row.doc_id,
    n.start_page = CASE WHEN row.`start_page:int` = '' THEN null ELSE toInteger(row.`start_page:int`) END,
    n.end_page = CASE WHEN row.`end_page:int` = '' THEN null ELSE toInteger(row.`end_page:int`) END,
    n.display_name = row.display_name,
    n.json_properties = row.json_properties;

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Entity {node_id: row.`:START_ID`})
MATCH (t:Entity {node_id: row.`:END_ID`})
CALL apoc.create.relationship(
  s,
  row.`:TYPE`,
  {
    edge_id: row.edge_id,
    doc_id: row.doc_id,
    start_page: CASE WHEN row.`start_page:int` = '' THEN null ELSE toInteger(row.`start_page:int`) END,
    end_page: CASE WHEN row.`end_page:int` = '' THEN null ELSE toInteger(row.`end_page:int`) END,
    json_properties: row.json_properties
  },
  t
) YIELD rel
RETURN count(rel);
"""
    # fallback: if APOC isn't available, use separate relationship loads later
    (OUT_DIR / "constraints.cypher").write_text(constraints, encoding="utf-8")
    (OUT_DIR / "load.cypher").write_text(load, encoding="utf-8")

    print(f"Wrote Neo4j import files to {OUT_DIR}")


if __name__ == "__main__":
    main()