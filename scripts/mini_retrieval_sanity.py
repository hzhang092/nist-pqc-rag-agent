"""
Mini local retrieval sanity check (base vs hybrid+query-fusion).

Usage:
    python scripts/mini_retrieval_sanity.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rag.retrieve import retrieve
from rag.retriever.base import ChunkHit


QUERIES = [
    "ML-KEM.KeyGen",
    "Algorithm 19 ML-KEM.KeyGen",
    "K-PKE Key Generation",
    "ML-KEM-768 parameter set",
    "decapsulation algorithm ML-KEM.Decaps",
]

REPORT_JSON = Path("reports") / "mini_retrieval_sanity.json"
REPORT_MD = Path("reports") / "mini_retrieval_sanity.md"


def _is_spec_like(hit: ChunkHit) -> bool:
    text = (hit.text or "").lower()
    return hit.doc_id == "NIST.FIPS.203" and (
        hit.start_page >= 20
        or "algorithm" in text
        or "keygen" in text
        or "decaps" in text
        or "parameter set" in text
    )


def _summarize(hits: List[ChunkHit]) -> Dict[str, object]:
    top = []
    for h in hits[:5]:
        top.append(
            {
                "score": round(h.score, 6),
                "doc_id": h.doc_id,
                "start_page": h.start_page,
                "end_page": h.end_page,
                "chunk_id": h.chunk_id,
                "spec_like": _is_spec_like(h),
            }
        )
    spec_count = sum(1 for item in top if item["spec_like"])
    return {"top5": top, "spec_like_top5": spec_count}


def main() -> None:
    rows = []
    for query in QUERIES:
        base_hits = retrieve(query, k=5, mode="base", backend="faiss", use_query_fusion=False)
        hybrid_hits = retrieve(query, k=5, mode="hybrid", backend="faiss", use_query_fusion=True)

        base_summary = _summarize(base_hits)
        hybrid_summary = _summarize(hybrid_hits)

        rows.append(
            {
                "query": query,
                "base": base_summary,
                "hybrid": hybrid_summary,
                "delta_spec_like_top5": hybrid_summary["spec_like_top5"] - base_summary["spec_like_top5"],
            }
        )

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Mini Retrieval Sanity (Base vs Hybrid+Fusion)", ""]
    for row in rows:
        lines.append(f"## {row['query']}")
        lines.append(
            f"- base spec-like@5: {row['base']['spec_like_top5']} | "
            f"hybrid spec-like@5: {row['hybrid']['spec_like_top5']} | "
            f"delta: {row['delta_spec_like_top5']}"
        )
        lines.append("- base top-5:")
        for hit in row["base"]["top5"]:
            lines.append(
                f"  - {hit['doc_id']} p{hit['start_page']}-p{hit['end_page']} ({hit['chunk_id']}) "
                f"spec_like={hit['spec_like']}"
            )
        lines.append("- hybrid top-5:")
        for hit in row["hybrid"]["top5"]:
            lines.append(
                f"  - {hit['doc_id']} p{hit['start_page']}-p{hit['end_page']} ({hit['chunk_id']}) "
                f"spec_like={hit['spec_like']}"
            )
        lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {REPORT_JSON}")
    print(f"[OK] wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
