"""
Run retrieval (and optional answer) evaluation on a labeled question set.

Usage:
    python -m eval.run
    python -m eval.run --dataset eval/day4/questions.jsonl --allow-unlabeled
    python -m eval.run --dataset eval/day4/questions.jsonl --allow-unlabeled --with-answers
    python -m eval.run --outdir reports/eval

Outputs are timestamped and written into a single output directory.

Flags:
    --dataset: path to JSONL dataset of questions with labels and gold spans
    --outdir: directory to write output artifacts (per-question JSONL, summary JSON, summary MD)
    --mode: retrieval mode to pass to rag.ask (default: base, hybrid)
    --backend: vector database backend to pass to rag.ask (default: SETTINGS.VECTOR_BACKEND)
    --k: primary retrieval depth and primary @k for metrics (default: SETTINGS.TOP_K)
    --ks: comma-separated k values for retrieval metrics (default: "1,3,5,8")
    --k0: k0 parameter for RRF retrieval (default: SETTINGS.RETRIEVAL_RRF_K0)
    --candidate-multiplier: candidate_multiplier for retrieval (default: SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER)
    --fusion / --no-fusion: enable or disable query fusion in retrieval (default: enabled)
    --rerank / --no-rerank: enable or disable rerank in retrieval (default: enabled)
    --rerank-pool: k value for retrieval rerank pool (default: SETTINGS.RETRIEVAL_RERANK_POOL)
    --near-page-tolerance: page slack for relaxed diagnostics (doc match + overlap within ±N pages, default: 1)
    --allow-unlabeled: allow answerable=true questions with empty gold spans (these are skipped in retrieval metrics, default: false)
    --with-answers: also run rag.ask and score citation/refusal metrics (default: false)
    --debug: pass debug flag through eval retrieval adapter (default: false)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from eval.dataset import load_questions, qid_sort_key
from eval.metrics import (
    compute_retrieval_metrics_by_ks,
    evaluate_answer_payload,
    hit_matches_gold,
    hit_matches_gold_doc_only,
    hit_matches_gold_with_tolerance,
    safe_mean,
)
from rag.config import SETTINGS
from rag.retrieve import retrieve_for_eval


def _run_ask_json(
    question: str,
    *,
    mode: str,
    backend: str,
    k: int,
    k0: int,
    candidate_multiplier: int,
    fusion: bool,
    rerank: bool,
    rerank_pool: int,
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "rag.ask",
        question,
        "--json",
        "--no-evidence",
        "--mode",
        mode,
        "--backend",
        backend,
        "--k",
        str(k),
        "--k0",
        str(k0),
        "--candidate-multiplier",
        str(candidate_multiplier),
        "--rerank-pool",
        str(rerank_pool),
    ]
    if not fusion:
        cmd.append("--no-query-fusion")
    if not rerank:
        cmd.append("--no-rerank")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr if stderr else stdout
        raise RuntimeError(f"rag.ask failed: {details}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("rag.ask returned non-JSON output in --json mode") from exc


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _parse_ks(raw: str) -> List[int]:
    ks: List[int] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid k value in --ks: {token!r}") from exc
        if value <= 0:
            raise ValueError(f"All --ks values must be > 0, got {value}")
        ks.append(value)
    ordered = sorted(set(ks))
    if not ordered:
        raise ValueError("--ks produced no valid k values")
    return ordered


def _gold_hit_ranks(
    hits: List[dict],
    gold: List[dict],
    k: int,
    matcher: Callable[[dict, dict], bool] = hit_matches_gold,
) -> List[int]:
    ranks: List[int] = []
    for hit in hits[:k]:
        if any(matcher(hit, g) for g in gold):
            ranks.append(int(hit.get("rank", 0)))
    return ranks


def _top_hit_ids(hits: List[dict], limit: int = 10) -> List[dict]:
    out: List[dict] = []
    for hit in hits[: max(0, limit)]:
        out.append(
            {
                "rank": int(hit.get("rank", 0)),
                "doc_id": hit.get("doc_id", ""),
                "pages": f"p{int(hit.get('start_page', 0))}-p{int(hit.get('end_page', 0))}",
                "chunk_id": hit.get("chunk_id", ""),
            }
        )
    return out


def _hit_rate_at_k(rank_lists: List[List[int]], k: int) -> float | None:
    if not rank_lists:
        return None
    hits = sum(1 for ranks in rank_lists if any(rank <= k for rank in ranks))
    return hits / float(len(rank_lists))


def _has_hit_in_top_k(ranks: List[int], k: int) -> bool:
    return any(rank <= k for rank in ranks)


def _slugify_for_filename(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("._-")
    return slug if slug else "eval"


def _timestamp_slug(dt_utc: datetime) -> str:
    return dt_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_run_id(dataset_path: str | Path, generated_at_utc: datetime) -> str:
    dataset_stem = _slugify_for_filename(Path(dataset_path).stem)
    return f"{dataset_stem}_{_timestamp_slug(generated_at_utc)}"


def _build_artifact_paths(outdir: Path, run_id: str) -> Dict[str, Path]:
    return {
        "per_question": outdir / f"{run_id}_per_question.jsonl",
        "summary_json": outdir / f"{run_id}_summary.json",
        "summary_md": outdir / f"{run_id}_summary.md",
    }


def _format_gold_spans(gold: List[dict]) -> str:
    if not gold:
        return "[]"
    return ", ".join(
        f"{g.get('doc_id', '')}:p{int(g.get('start_page', 0))}-p{int(g.get('end_page', 0))}" for g in gold
    )


def _format_top_hits_for_md(top_hit_ids: List[dict], limit: int = 3) -> str:
    short = top_hit_ids[: max(0, limit)]
    if not short:
        return "none"
    return "; ".join(
        f"r{int(h.get('rank', 0))} {h.get('doc_id', '')} {h.get('pages', '')}" for h in short
    )


def _build_summary_markdown(summary: Dict[str, Any]) -> str:
    retrieval = summary["retrieval"]
    answer = summary.get("answer", {})
    missed = retrieval.get("questions_without_gold_in_primary_k", [])
    sec = retrieval.get("secondary_diagnostics", {})
    near_page_tolerance = sec.get("near_page_tolerance")
    lines = [
        "# Evaluation Summary",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- run_id: {summary.get('run_id')}",
        f"- dataset: {summary['dataset_path']}",
        f"- total_questions: {summary['counts']['total_questions']}",
        f"- answerable_questions: {summary['counts']['answerable_questions']}",
        f"- unanswerable_questions: {summary['counts']['unanswerable_questions']}",
        f"- labeled_answerable_questions: {summary['counts']['labeled_answerable_questions']}",
        f"- unlabeled_answerable_questions: {summary['counts']['unlabeled_answerable_questions']}",
        "",
        "## Retrieval",
        f"- scoring_scope: {retrieval.get('scoring_scope')}",
        f"- primary_k: {retrieval.get('primary_k')}",
        f"- Recall@k: {_fmt(retrieval.get('recall_at_k'))}",
        f"- MRR@k: {_fmt(retrieval.get('mrr_at_k'))}",
        f"- nDCG@k: {_fmt(retrieval.get('ndcg_at_k'))}",
        "",
        "### Retrieval By K",
    ]

    for key in sorted(
        retrieval.get("at_k", {}).keys(),
        key=lambda item: int(item[1:]) if item.startswith("k") and item[1:].isdigit() else item,
    ):
        metric_row = retrieval["at_k"][key]
        lines.append(
            f"- {key}: recall={_fmt(metric_row.get('recall'))}, "
            f"mrr={_fmt(metric_row.get('mrr'))}, ndcg={_fmt(metric_row.get('ndcg'))}"
        )

    sec_at_k = sec.get("hit_rate_at_k", {})
    if sec_at_k:
        lines.extend(
            [
                "",
                "### Secondary Diagnostics",
                f"- near_page_tolerance: {sec.get('near_page_tolerance')}",
            ]
        )
        for key in sorted(
            sec_at_k.keys(),
            key=lambda item: int(item[1:]) if item.startswith("k") and item[1:].isdigit() else item,
        ):
            diag = sec_at_k[key]
            lines.append(
                f"- {key}: strict={_fmt(diag.get('strict_page_overlap'))}, "
                f"doc_only={_fmt(diag.get('doc_only'))}, "
                f"near_page={_fmt(diag.get('near_page_tolerance'))}"
            )
    lines.extend(
        [
            "",
            "### Metric Definitions",
            "- Retrieval By K:",
            "  Recall@k = average fraction of gold spans recovered per question; "
            "MRR@k = average reciprocal first strict-hit rank; "
            "nDCG@k = rank-aware gain over unique gold spans.",
            "- Secondary Diagnostics:",
            "  hit-rate style metrics (per-question success rate): at least one matching hit appears in top-k.",
            "- strict:",
            "  doc_id match + page overlap (same relevance rule as primary retrieval metrics).",
            "- doc_only:",
            "  doc_id must match; page overlap is ignored.",
            "- near_page:",
            "  doc_id match + page overlap with +-near_page_tolerance slack.",
            "- Why numbers differ:",
            "  Retrieval By K is span-coverage/rank quality; Secondary Diagnostics is question-level any-hit rate, "
            "so strict can be higher when questions have multiple gold spans.",
        ]
    )
    if near_page_tolerance == 0:
        lines.append("- Note: near_page_tolerance=0 makes near_page equivalent to strict.")

    lines.extend(
        [
            "",
            "### Questions Missing Gold In Top-k",
            f"- count: {retrieval.get('n_questions_without_gold_in_primary_k')}",
        ]
    )
    if not missed:
        lines.append("- none")
    else:
        for row in missed:
            lines.append(
                f"- {row.get('qid')}: {row.get('question')} | "
                f"gold={_format_gold_spans(row.get('gold', []))} | "
                f"top_hits={_format_top_hits_for_md(row.get('top_hit_ids', []))}"
            )

    lines.extend(
        [
            "",
            "## Answer",
            f"- enabled: {answer.get('enabled')}",
            f"- model_dependent: {answer.get('model_dependent')}",
            f"- note: {answer.get('note')}",
            f"- answer_evaluated: {summary['counts']['answer_evaluated_questions']}",
            f"- citation_presence_rate: {_fmt(answer.get('citation_presence_rate'))}",
            f"- inline_citation_sentence_rate: {_fmt(answer.get('inline_citation_sentence_rate'))}",
            f"- refusal_accuracy: {_fmt(answer.get('refusal_accuracy'))}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run")
    parser.add_argument("--dataset", type=str, default="eval/day4/questions.jsonl")
    parser.add_argument("--outdir", type=str, default="reports/eval")
    parser.add_argument("--mode", type=str, choices=["base", "hybrid"], default=SETTINGS.RETRIEVAL_MODE)
    parser.add_argument("--backend", type=str, default=SETTINGS.VECTOR_BACKEND)
    parser.add_argument("--k", type=int, default=SETTINGS.TOP_K, help="Primary retrieval depth and primary @k.")
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5,8",
        help="Comma-separated k values for retrieval metrics (for example: 1,3,5,8).",
    )
    parser.add_argument("--k0", type=int, default=SETTINGS.RETRIEVAL_RRF_K0)
    parser.add_argument("--candidate-multiplier", type=int, default=SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER)
    parser.add_argument("--rerank-pool", type=int, default=SETTINGS.RETRIEVAL_RERANK_POOL)
    parser.add_argument(
        "--near-page-tolerance",
        type=int,
        default=1,
        help="Page slack for relaxed diagnostics (doc match + overlap within ±N pages).",
    )
    parser.add_argument("--no-fusion", action="store_true", help="Disable query fusion in retrieval.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable rerank in retrieval.")
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Allow answerable=true questions with empty gold spans (these are skipped in retrieval metrics).",
    )
    parser.add_argument("--with-answers", action="store_true", help="Also run rag.ask and score citation/refusal metrics.")
    parser.add_argument("--debug", action="store_true", help="Pass debug flag through eval retrieval adapter.")
    args = parser.parse_args()

    try:
        metric_ks = _parse_ks(args.ks)
    except ValueError as exc:
        raise SystemExit(f"Invalid --ks: {exc}") from exc
    if args.near_page_tolerance < 0:
        raise SystemExit("--near-page-tolerance must be >= 0")

    retrieval_depth = max(args.k, max(metric_ks))
    primary_k = args.k if args.k in metric_ks else max(metric_ks)

    try:
        questions = load_questions(args.dataset, require_labeled=not args.allow_unlabeled)
    except ValueError as exc:
        raise SystemExit(
            f"Dataset validation failed: {exc}\n"
            f"Fix labels in {args.dataset} or rerun with --allow-unlabeled."
        ) from exc
    fusion = not args.no_fusion
    rerank = not args.no_rerank

    per_question: List[Dict[str, Any]] = []
    retrieval_rows_by_k: Dict[int, List[Dict[str, float]]] = {k: [] for k in metric_ks}
    retrieval_eval_qids: List[str] = []
    retrieval_skipped_unanswerable_qids: List[str] = []
    retrieval_skipped_unlabeled_qids: List[str] = []
    strict_rank_lists: List[List[int]] = []
    doc_only_rank_lists: List[List[int]] = []
    near_page_rank_lists: List[List[int]] = []
    questions_with_gold_in_primary_k_qids: List[str] = []
    questions_without_gold_in_primary_k: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    answer_errors = 0

    for row in questions:
        hits = retrieve_for_eval(
            query=row["question"],
            mode=args.mode,
            k=retrieval_depth,
            backend=args.backend,
            k0=args.k0,
            candidate_multiplier=args.candidate_multiplier,
            fusion=fusion,
            evidence_window=False,
            cheap_rerank=rerank,
            rerank_pool=args.rerank_pool,
            debug=args.debug,
        )

        retrieval_metrics = None
        gold_hit_ranks: List[int] = []
        doc_hit_ranks: List[int] = []
        near_page_hit_ranks: List[int] = []
        has_gold_in_primary_k: bool | None = None
        top_hit_ids = _top_hit_ids(hits=hits, limit=min(10, retrieval_depth))

        if row["answerable"] and row["gold"]:
            metrics_by_k = compute_retrieval_metrics_by_ks(
                hits=hits,
                gold=row["gold"],
                ks=metric_ks,
            )
            retrieval_metrics = {
                "primary_k": primary_k,
                "primary": metrics_by_k[f"k{primary_k}"],
                "at_k": metrics_by_k,
            }
            for k in metric_ks:
                retrieval_rows_by_k[k].append(metrics_by_k[f"k{k}"])
            retrieval_eval_qids.append(row["qid"])
            gold_hit_ranks = _gold_hit_ranks(
                hits=hits,
                gold=row["gold"],
                k=retrieval_depth,
                matcher=hit_matches_gold,
            )
            doc_hit_ranks = _gold_hit_ranks(
                hits=hits,
                gold=row["gold"],
                k=retrieval_depth,
                matcher=hit_matches_gold_doc_only,
            )
            near_page_hit_ranks = _gold_hit_ranks(
                hits=hits,
                gold=row["gold"],
                k=retrieval_depth,
                matcher=lambda h, g: hit_matches_gold_with_tolerance(
                    h, g, page_tolerance=args.near_page_tolerance
                ),
            )
            strict_rank_lists.append(gold_hit_ranks)
            doc_only_rank_lists.append(doc_hit_ranks)
            near_page_rank_lists.append(near_page_hit_ranks)

            has_gold_in_primary_k = _has_hit_in_top_k(gold_hit_ranks, primary_k)
            if has_gold_in_primary_k:
                questions_with_gold_in_primary_k_qids.append(row["qid"])
            else:
                questions_without_gold_in_primary_k.append(
                    {
                        "qid": row["qid"],
                        "question": row["question"],
                        "gold": row["gold"],
                        "gold_hit_ranks": gold_hit_ranks,
                        "top_hit_ids": top_hit_ids,
                    }
                )
        elif not row["answerable"]:
            retrieval_skipped_unanswerable_qids.append(row["qid"])
        else:
            retrieval_skipped_unlabeled_qids.append(row["qid"])

        question_result: Dict[str, Any] = {
            "qid": row["qid"],
            "question": row["question"],
            "answerable": row["answerable"],
            "gold": row["gold"],
            "retrieval": {
                "metrics": retrieval_metrics,
                "gold_hit_ranks": gold_hit_ranks,
                "doc_hit_ranks": doc_hit_ranks,
                "near_page_hit_ranks": near_page_hit_ranks,
                "has_gold_in_primary_k": has_gold_in_primary_k,
                "top_hit_ids": top_hit_ids,
                "hits": hits,
            },
        }

        if args.with_answers:
            try:
                payload = _run_ask_json(
                    row["question"],
                    mode=args.mode,
                    backend=args.backend,
                    k=args.k,
                    k0=args.k0,
                    candidate_multiplier=args.candidate_multiplier,
                    fusion=fusion,
                    rerank=rerank,
                    rerank_pool=args.rerank_pool,
                )
                answer_metrics = evaluate_answer_payload(payload=payload, answerable=row["answerable"])
                question_result["answer"] = {
                    "metrics": answer_metrics,
                    "payload": payload,
                }
                answer_rows.append(answer_metrics)
            except Exception as exc:
                answer_errors += 1
                question_result["answer"] = {"error": str(exc)}

        per_question.append(question_result)

    if len(questions_with_gold_in_primary_k_qids) + len(questions_without_gold_in_primary_k) != len(retrieval_eval_qids):
        raise RuntimeError(
            "Internal mismatch while building top-k miss summary: "
            "evaluated != hits + misses."
        )

    retrieval_at_k: Dict[str, Dict[str, float | None]] = {}
    for k in metric_ks:
        rows_k = retrieval_rows_by_k[k]
        retrieval_at_k[f"k{k}"] = {
            "recall": safe_mean(m["recall_at_k"] for m in rows_k),
            "mrr": safe_mean(m["mrr_at_k"] for m in rows_k),
            "ndcg": safe_mean(m["ndcg_at_k"] for m in rows_k),
        }

    primary_metrics = retrieval_at_k.get(f"k{primary_k}", {"recall": None, "mrr": None, "ndcg": None})
    secondary_hit_rate_at_k: Dict[str, Dict[str, float | None]] = {}
    for k in metric_ks:
        secondary_hit_rate_at_k[f"k{k}"] = {
            "strict_page_overlap": _hit_rate_at_k(strict_rank_lists, k),
            "doc_only": _hit_rate_at_k(doc_only_rank_lists, k),
            "near_page_tolerance": _hit_rate_at_k(near_page_rank_lists, k),
        }

    questions_without_gold_in_primary_k = sorted(
        questions_without_gold_in_primary_k,
        key=lambda r: qid_sort_key(str(r.get("qid", ""))),
    )
    questions_with_gold_in_primary_k_qids = sorted(
        questions_with_gold_in_primary_k_qids,
        key=qid_sort_key,
    )

    primary_secondary = secondary_hit_rate_at_k.get(
        f"k{primary_k}",
        {
            "strict_page_overlap": None,
            "doc_only": None,
            "near_page_tolerance": None,
        },
    )
    retrieval_summary = {
        "scoring_scope": "answerable_with_non_empty_gold_only",
        "metric_ks": metric_ks,
        "primary_k": primary_k,
        "n_questions": len(retrieval_eval_qids),
        "skipped_unanswerable_qids": retrieval_skipped_unanswerable_qids,
        "skipped_unlabeled_answerable_qids": retrieval_skipped_unlabeled_qids,
        "at_k": retrieval_at_k,
        "recall_at_k": primary_metrics.get("recall"),
        "mrr_at_k": primary_metrics.get("mrr"),
        "ndcg_at_k": primary_metrics.get("ndcg"),
        "n_questions_with_gold_in_primary_k": len(questions_with_gold_in_primary_k_qids),
        "n_questions_without_gold_in_primary_k": len(questions_without_gold_in_primary_k),
        "questions_with_gold_in_primary_k_qids": questions_with_gold_in_primary_k_qids,
        "questions_without_gold_in_primary_k": questions_without_gold_in_primary_k,
        "secondary_diagnostics": {
            "near_page_tolerance": args.near_page_tolerance,
            "primary_k_hit_rate": primary_secondary,
            "hit_rate_at_k": secondary_hit_rate_at_k,
        },
    }

    answer_summary = {
        "enabled": args.with_answers,
        "model_dependent": True,
        "note": (
            "Answer metrics are model-dependent and less stable than retrieval metrics; "
            "use retrieval metrics as primary regression signals."
        ),
        "citation_presence_rate": safe_mean(
            (1.0 if m["citation_presence_ok"] else 0.0) for m in answer_rows
        ),
        "inline_citation_sentence_rate": safe_mean(
            m["inline_citation_sentence_rate"]
            for m in answer_rows
            if m["inline_citation_sentence_rate"] is not None
        ),
        "refusal_accuracy": safe_mean(m["refusal_accuracy"] for m in answer_rows),
    }

    labeled_answerable = sum(1 for q in questions if q["answerable"] and q["gold"])
    unlabeled_answerable = sum(1 for q in questions if q["answerable"] and not q["gold"])
    generated_at_utc = datetime.now(timezone.utc)
    run_id = _build_run_id(dataset_path=args.dataset, generated_at_utc=generated_at_utc)

    summary = {
        "run_id": run_id,
        "generated_at_utc": generated_at_utc.isoformat(),
        "dataset_path": str(Path(args.dataset)),
        "run_config": {
            "mode": args.mode,
            "backend": args.backend,
            "k": args.k,
            "ks": metric_ks,
            "retrieval_depth": retrieval_depth,
            "near_page_tolerance": args.near_page_tolerance,
            "k0": args.k0,
            "candidate_multiplier": args.candidate_multiplier,
            "fusion": fusion,
            "rerank": rerank,
            "rerank_pool": args.rerank_pool,
            "with_answers": args.with_answers,
            "allow_unlabeled": args.allow_unlabeled,
        },
        "counts": {
            "total_questions": len(questions),
            "answerable_questions": sum(1 for q in questions if q["answerable"]),
            "unanswerable_questions": sum(1 for q in questions if not q["answerable"]),
            "labeled_answerable_questions": labeled_answerable,
            "unlabeled_answerable_questions": unlabeled_answerable,
            "retrieval_evaluated_questions": len(retrieval_eval_qids),
            "retrieval_skipped_unanswerable": len(retrieval_skipped_unanswerable_qids),
            "retrieval_skipped_unlabeled_answerable": len(retrieval_skipped_unlabeled_qids),
            "answer_evaluated_questions": len(answer_rows),
            "answer_errors": answer_errors,
        },
        "retrieval": retrieval_summary,
        "answer": answer_summary,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _build_artifact_paths(outdir=outdir, run_id=run_id)
    per_question_path = artifact_paths["per_question"]
    summary_json_path = artifact_paths["summary_json"]
    summary_md_path = artifact_paths["summary_md"]

    per_question = sorted(per_question, key=lambda r: qid_sort_key(str(r.get("qid", ""))))
    summary["artifact_paths"] = {name: str(path) for name, path in artifact_paths.items()}

    _write_jsonl(per_question_path, per_question)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(_build_summary_markdown(summary), encoding="utf-8")

    print(f"[OK] run_id={run_id}")
    print(f"[OK] wrote {per_question_path}")
    print(f"[OK] wrote {summary_json_path}")
    print(f"[OK] wrote {summary_md_path}")


if __name__ == "__main__":
    main()
