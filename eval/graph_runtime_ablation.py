from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from eval.metrics import compute_retrieval_metrics_by_ks, hit_matches_gold, safe_mean
from rag.config import SETTINGS
from rag.lc import graph as agent_graph
from rag.retrieve import retrieve_with_stages_and_timing
from rag.service import ask_question
from rag.types import REFUSAL_TEXT


DEFAULT_DATASET = Path("eval/graph_runtime_ablation.jsonl")


@dataclass(frozen=True)
class EvalConfig:
    key: str
    label: str
    mode: str
    use_graph_lookup: bool = False
    use_section_priors: bool = False


CONFIGS = [
    EvalConfig(key="baseline_ask", label="1. /ask baseline", mode="ask"),
    EvalConfig(
        key="agent_graph_lookup_only",
        label="2. /ask-agent + graph lookup",
        mode="agent",
        use_graph_lookup=True,
        use_section_priors=False,
    ),
    EvalConfig(
        key="agent_graph_lookup_plus_section_priors",
        label="3. /ask-agent + graph lookup + section priors",
        mode="agent",
        use_graph_lookup=True,
        use_section_priors=True,
    ),
]

HIGHLIGHT_QIDS = [
    "graph-runtime-001",
    "graph-runtime-002",
    "graph-runtime-007",
    "graph-runtime-009",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rank_hits(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        ranked.append(
            {
                "rank": rank,
                "score": float(item.get("score", 0.0) or 0.0),
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": str(item.get("doc_id") or ""),
                "start_page": int(item.get("start_page", 0) or 0),
                "end_page": int(item.get("end_page", 0) or 0),
                "text": str(item.get("text") or ""),
            }
        )
    return ranked


def _top_hit_summary(hits: list[dict[str, Any]], limit: int = 3) -> str:
    if not hits:
        return "none"
    short = hits[: max(0, limit)]
    return "; ".join(
        f"r{int(hit['rank'])} {hit['doc_id']} p{int(hit['start_page'])}-p{int(hit['end_page'])}"
        for hit in short
    )


def _first_gold_rank(hits: list[dict[str, Any]], gold: list[dict[str, Any]]) -> int | None:
    for hit in hits:
        if any(hit_matches_gold(hit, span) for span in gold):
            return int(hit["rank"])
    return None


def _is_refusal(payload: dict[str, Any]) -> bool:
    return bool(payload.get("refusal_reason")) or str(payload.get("answer") or "").strip() == REFUSAL_TEXT


def _citation_compliance(payload: dict[str, Any], *, answerable: bool) -> float | None:
    if not answerable:
        return None
    if _is_refusal(payload):
        return 0.0
    citations = payload.get("citations") or []
    return 1.0 if isinstance(citations, list) and len(citations) > 0 else 0.0


def _deterministic_answer_payload(question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    if not hits:
        return {
            "answer": REFUSAL_TEXT,
            "citations": [],
            "refusal_reason": "insufficient_evidence",
        }
    top_hits = hits[:2]
    citations = [
        {
            "key": f"c{i}",
            "doc_id": hit["doc_id"],
            "start_page": hit["start_page"],
            "end_page": hit["end_page"],
            "chunk_id": hit["chunk_id"],
        }
        for i, hit in enumerate(top_hits, start=1)
    ]
    preview = str(top_hits[0].get("text") or "").strip().replace("\n", " ")
    preview = preview[:220].strip() or question.strip()
    return {
        "answer": f"{preview} [c1]",
        "citations": citations,
        "refusal_reason": None,
    }


@contextmanager
def _deterministic_agent_planner():
    original = agent_graph._call_llm_query_analysis
    agent_graph._call_llm_query_analysis = lambda _question, deterministic: deterministic
    try:
        yield
    finally:
        agent_graph._call_llm_query_analysis = original


@contextmanager
def _deterministic_agent_answer():
    original = agent_graph._call_rag_answer

    def _fake_answer(question: str, evidence: list[Any]) -> dict[str, Any]:
        hits = _rank_hits(
            [
                {
                    "score": float(item.score),
                    "chunk_id": item.chunk_id,
                    "doc_id": item.doc_id,
                    "start_page": item.start_page,
                    "end_page": item.end_page,
                    "text": item.text,
                }
                for item in evidence
            ]
        )
        return _deterministic_answer_payload(question, hits)

    agent_graph._call_rag_answer = _fake_answer
    try:
        yield
    finally:
        agent_graph._call_rag_answer = original


def _run_direct_question(
    question: str,
    *,
    k: int,
    answer_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if answer_mode == "auto":
        try:
            payload = ask_question(question, k=k)
            hits = _rank_hits(list(payload.get("evidence") or []))
            return payload, hits, "live"
        except Exception:
            pass

    stages, _timing = retrieve_with_stages_and_timing(query=question, k=k)
    hits = _rank_hits(
        [
            {
                "score": float(hit.score),
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "start_page": hit.start_page,
                "end_page": hit.end_page,
                "text": hit.text,
            }
            for hit in stages["final_hits"]
        ]
    )
    return _deterministic_answer_payload(question, hits), hits, "deterministic"


def _run_agent_question(
    question: str,
    *,
    k: int,
    use_graph_lookup: bool,
    use_section_priors: bool,
    answer_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any], bool]:
    with _deterministic_agent_planner():
        if answer_mode == "auto":
            try:
                state = agent_graph.run_agent(
                    question,
                    k=k,
                    use_graph_lookup=use_graph_lookup,
                    use_section_priors=use_section_priors,
                )
                answer_mode = "live"
            except Exception:
                with _deterministic_agent_answer():
                    state = agent_graph.run_agent(
                        question,
                        k=k,
                        use_graph_lookup=use_graph_lookup,
                        use_section_priors=use_section_priors,
                )
                answer_mode = "deterministic_fallback"
        else:
            with _deterministic_agent_answer():
                state = agent_graph.run_agent(
                    question,
                    k=k,
                    use_graph_lookup=use_graph_lookup,
                    use_section_priors=use_section_priors,
                )
            answer_mode = "deterministic"

    payload = {
        "answer": str(state.get("final_answer") or state.get("draft_answer") or ""),
        "citations": list(state.get("citations") or []),
        "refusal_reason": str(state.get("refusal_reason") or "") or None,
        "graph_lookup": dict(state.get("graph_lookup") or {}),
    }
    hits = _rank_hits(list(state.get("evidence") or []))
    section_prior_applied = bool((state.get("last_retrieval_stats") or {}).get("section_prior_applied", False))
    return payload, hits, answer_mode, payload["graph_lookup"], section_prior_applied


def _summarize_config(rows: list[dict[str, Any]], *, ks: list[int]) -> dict[str, Any]:
    answerable_rows = [row for row in rows if row["answerable"] and row["gold_spans"]]
    retrieval_by_k: dict[str, dict[str, float | None]] = {}
    for k in ks:
        retrieval_by_k[f"k{k}"] = {
            metric: safe_mean(
                float(row["retrieval_metrics"][f"k{k}"][metric]) for row in answerable_rows
            )
            for metric in ("recall_at_k", "mrr_at_k", "ndcg_at_k")
        }

    citation_values = [
        row["citation_compliance"]
        for row in rows
        if row["citation_compliance"] is not None
    ]
    refusal_values = [1.0 if row["is_refusal"] else 0.0 for row in rows]
    return {
        "retrieval_by_k": retrieval_by_k,
        "citation_compliance": safe_mean(citation_values),
        "refusal_rate": safe_mean(refusal_values),
        "answer_mode_counts": dict(Counter(row["answer_mode"] for row in rows)),
    }


def _choose_default(summary_by_config: dict[str, dict[str, Any]], *, primary_k: int) -> str:
    def _score(config_key: str) -> tuple[float, float, float]:
        metrics = summary_by_config[config_key]["retrieval_by_k"][f"k{primary_k}"]
        recall = float(metrics.get("recall_at_k") or 0.0)
        mrr = float(metrics.get("mrr_at_k") or 0.0)
        citation = float(summary_by_config[config_key].get("citation_compliance") or 0.0)
        return (recall, mrr, citation)

    return max(summary_by_config.keys(), key=_score)


def _build_report(
    *,
    dataset_path: Path,
    generated_at: datetime,
    k: int,
    ks: list[int],
    per_config_rows: dict[str, list[dict[str, Any]]],
    summary_by_config: dict[str, dict[str, Any]],
) -> str:
    generated_at_iso = generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    best_key = _choose_default(summary_by_config, primary_k=k)
    config_labels = {config.key: config.label for config in CONFIGS}

    lines = [
        "# Graph Runtime Ablation",
        "",
        f"- generated_at: {generated_at_iso}",
        f"- dataset: `{dataset_path}`",
        f"- primary_k: {k}",
        f"- ks: `{','.join(str(v) for v in ks)}`",
        "- agent_planner_mode: deterministic fallback pinned during eval",
        "- note: `/ask` remains the control path; graph section priors only affect `/ask-agent` rerank behavior",
        "",
        "## Summary",
        "",
        "| config | Recall@k | MRR@k | nDCG@k | citation compliance | refusal rate | answer modes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for config in CONFIGS:
        summary = summary_by_config[config.key]
        metrics = summary["retrieval_by_k"][f"k{k}"]
        answer_modes = ", ".join(
            f"{mode}={count}" for mode, count in sorted(summary["answer_mode_counts"].items())
        )
        lines.append(
            "| {label} | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {citation:.4f} | {refusal:.4f} | {answer_modes} |".format(
                label=config.label,
                recall=float(metrics.get("recall_at_k") or 0.0),
                mrr=float(metrics.get("mrr_at_k") or 0.0),
                ndcg=float(metrics.get("ndcg_at_k") or 0.0),
                citation=float(summary.get("citation_compliance") or 0.0),
                refusal=float(summary.get("refusal_rate") or 0.0),
                answer_modes=answer_modes or "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## Default",
            "",
            f"- selected_default: {config_labels[best_key]}",
            "- rationale: best retrieval score at the primary k, with citation compliance used as a tie-breaker.",
            "",
            "## Examples",
            "",
        ]
    )

    for qid in HIGHLIGHT_QIDS:
        sample_row = next((row for row in per_config_rows[CONFIGS[0].key] if row["qid"] == qid), None)
        if sample_row is None:
            continue
        lines.append(f"### {qid}: {sample_row['question']}")
        lines.append("")
        for config in CONFIGS:
            row = next(item for item in per_config_rows[config.key] if item["qid"] == qid)
            graph_lookup = row.get("graph_lookup") or {}
            graph_note = ""
            if config.mode == "agent":
                graph_note = (
                    f"; graph={graph_lookup.get('lookup_type', '')}"
                    f"/{graph_lookup.get('match_reason', '')}"
                    f"; section_priors={row.get('section_prior_applied', False)}"
                )
            lines.append(
                "- {label}: first_gold_rank={rank}; refusal={refusal}; citations={citations}; top_hits={hits}{graph_note}".format(
                    label=config.label,
                    rank=row["first_gold_rank"] if row["first_gold_rank"] is not None else "miss",
                    refusal=row["is_refusal"],
                    citations=row["citation_count"],
                    hits=_top_hit_summary(row["hits"]),
                    graph_note=graph_note,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Tradeoffs",
            "",
            "- This ablation isolates graph runtime effects by pinning the agent planner to the deterministic fallback.",
            "- If any config used `deterministic_fallback` answer mode, answer-side metrics should be treated as lower confidence than retrieval metrics.",
            "- Neo4j remains a dev/debug/export path only; this runtime ablation uses the checked-in graph-lite JSONL artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a focused graph runtime ablation for /ask-agent.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=SETTINGS.TOP_K)
    parser.add_argument("--ks", default="1,3,5,8")
    parser.add_argument("--answer-mode", choices=["deterministic", "auto"], default="deterministic")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    ks = sorted({int(token) for token in str(args.ks).split(",") if str(token).strip()})
    questions = _load_jsonl(dataset_path)

    per_config_rows: dict[str, list[dict[str, Any]]] = {}
    for config in CONFIGS:
        config_rows: list[dict[str, Any]] = []
        for row in questions:
            question = str(row.get("question") or "")
            gold_spans = list(row.get("gold_spans") or [])
            answerable = bool(row.get("answerable", False))

            if config.mode == "ask":
                payload, hits, answer_mode = _run_direct_question(
                    question,
                    k=args.k,
                    answer_mode=args.answer_mode,
                )
                graph_lookup = {}
                section_prior_applied = False
            else:
                payload, hits, answer_mode, graph_lookup, section_prior_applied = _run_agent_question(
                    question,
                    k=args.k,
                    use_graph_lookup=config.use_graph_lookup,
                    use_section_priors=config.use_section_priors,
                    answer_mode=args.answer_mode,
                )

            retrieval_metrics = compute_retrieval_metrics_by_ks(hits=hits, gold=gold_spans, ks=ks) if gold_spans else {}
            config_rows.append(
                {
                    "qid": str(row.get("qid") or ""),
                    "question": question,
                    "answerable": answerable,
                    "gold_spans": gold_spans,
                    "hits": hits,
                    "retrieval_metrics": retrieval_metrics,
                    "first_gold_rank": _first_gold_rank(hits, gold_spans) if gold_spans else None,
                    "answer_mode": answer_mode,
                    "is_refusal": _is_refusal(payload),
                    "refusal_reason": payload.get("refusal_reason"),
                    "citation_count": len(list(payload.get("citations") or [])),
                    "citation_compliance": _citation_compliance(payload, answerable=answerable),
                    "graph_lookup": graph_lookup,
                    "section_prior_applied": section_prior_applied,
                }
            )
        per_config_rows[config.key] = config_rows

    summary_by_config = {
        config.key: _summarize_config(per_config_rows[config.key], ks=ks)
        for config in CONFIGS
    }

    report = _build_report(
        dataset_path=dataset_path,
        generated_at=datetime.now(timezone.utc),
        k=args.k,
        ks=ks,
        per_config_rows=per_config_rows,
        summary_by_config=summary_by_config,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
