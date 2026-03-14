from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.lc import graph as agent_graph
from rag.lc.state_utils import init_state


DEFAULT_DATASET = Path("eval/graph_definition_sanity.jsonl")
DEFAULT_OUT = Path("reports/eval/graph_definition_sanity.md")


def _load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_analyze_query(question: str, *, use_graph_lookup: bool) -> dict[str, Any]:
    original = agent_graph._call_llm_query_analysis
    agent_graph._call_llm_query_analysis = lambda _question, deterministic: deterministic
    try:
        state = init_state(question, use_graph_lookup=use_graph_lookup)
        out = agent_graph.node_analyze_query(state)
    finally:
        agent_graph._call_llm_query_analysis = original
    return {
        "doc_ids": list(out.get("doc_ids") or []),
        "required_anchors": list(out.get("required_anchors") or []),
        "protected_spans": list(out.get("protected_spans") or []),
        "graph_lookup": dict(out.get("graph_lookup") or {}),
    }


def _bool_mark(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny definition-query sanity check for graph-assisted analyze_query.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    rows = _load_questions(dataset_path)

    report_rows: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question") or "")
        expected_doc_ids = list(row.get("expected_doc_ids") or [])
        expected_anchor = str(row.get("expected_anchor") or "")
        before = _run_analyze_query(question, use_graph_lookup=False)
        after = _run_analyze_query(question, use_graph_lookup=True)

        before_anchor_ok = expected_anchor in before["required_anchors"]
        after_anchor_ok = expected_anchor in after["required_anchors"]
        before_doc_ok = before["doc_ids"] == expected_doc_ids
        after_doc_ok = after["doc_ids"] == expected_doc_ids
        constrained_more = (
            before["doc_ids"] != after["doc_ids"]
            or set(before["required_anchors"]) != set(after["required_anchors"])
        )

        report_rows.append(
            {
                "qid": str(row.get("qid") or ""),
                "question": question,
                "expected_doc_ids": expected_doc_ids,
                "expected_anchor": expected_anchor,
                "before": before,
                "after": after,
                "before_doc_ok": before_doc_ok,
                "after_doc_ok": after_doc_ok,
                "before_anchor_ok": before_anchor_ok,
                "after_anchor_ok": after_anchor_ok,
                "constrained_more": constrained_more,
            }
        )

    lines = [
        "# Graph Definition Sanity",
        "",
        "This report isolates the `analyze_query` stage of `/ask-agent` with the LLM planner pinned to the deterministic fallback.",
        "It compares the same definition queries with graph lookup disabled vs enabled, so the measured change is doc narrowing and anchor enrichment only.",
        "",
        "| qid | before doc ok | after doc ok | before anchor ok | after anchor ok | more constrained |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            "| {qid} | {before_doc_ok} | {after_doc_ok} | {before_anchor_ok} | {after_anchor_ok} | {constrained_more} |".format(
                qid=row["qid"],
                before_doc_ok=_bool_mark(bool(row["before_doc_ok"])),
                after_doc_ok=_bool_mark(bool(row["after_doc_ok"])),
                before_anchor_ok=_bool_mark(bool(row["before_anchor_ok"])),
                after_anchor_ok=_bool_mark(bool(row["after_anchor_ok"])),
                constrained_more=_bool_mark(bool(row["constrained_more"])),
            )
        )

    lines.extend(
        [
            "",
            "## Per-query details",
            "",
        ]
    )

    for row in report_rows:
        lines.extend(
            [
                f"### {row['qid']}: {row['question']}",
                "",
                f"- expected_doc_ids: {row['expected_doc_ids']}",
                f"- expected_anchor: {row['expected_anchor']}",
                f"- before_doc_ids: {row['before']['doc_ids']}",
                f"- after_doc_ids: {row['after']['doc_ids']}",
                f"- before_required_anchors: {row['before']['required_anchors']}",
                f"- after_required_anchors: {row['after']['required_anchors']}",
                f"- graph_match_reason: {row['after']['graph_lookup'].get('match_reason', '')}",
                f"- candidate_doc_ids: {row['after']['graph_lookup'].get('candidate_doc_ids', [])}",
                f"- candidate_section_ids: {row['after']['graph_lookup'].get('candidate_section_ids', [])}",
                "",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
