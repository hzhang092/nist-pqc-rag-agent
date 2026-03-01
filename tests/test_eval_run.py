from datetime import datetime, timezone
from pathlib import Path

from eval.run import _build_artifact_paths, _build_run_id, _build_summary_markdown, _has_hit_in_top_k


def test_build_run_id_uses_dataset_stem_and_utc_timestamp():
    dt = datetime(2026, 3, 1, 12, 34, 56, tzinfo=timezone.utc)
    run_id = _build_run_id("eval/day4/questions.jsonl", dt)
    assert run_id == "questions_20260301T123456Z"


def test_build_artifact_paths_are_timestamped_and_in_single_outdir(tmp_path):
    outdir = tmp_path / "reports" / "eval"
    run_id = "questions_20260301T123456Z"
    paths = _build_artifact_paths(outdir=outdir, run_id=run_id)

    assert set(paths.keys()) == {"per_question", "summary_json", "summary_md"}
    assert paths["per_question"] == Path(outdir, f"{run_id}_per_question.jsonl")
    assert paths["summary_json"] == Path(outdir, f"{run_id}_summary.json")
    assert paths["summary_md"] == Path(outdir, f"{run_id}_summary.md")


def test_has_hit_in_top_k():
    assert _has_hit_in_top_k([2, 5], 3) is True
    assert _has_hit_in_top_k([4, 5], 3) is False
    assert _has_hit_in_top_k([], 3) is False


def test_summary_markdown_includes_missing_gold_section():
    summary = {
        "generated_at_utc": "2026-03-01T12:34:56+00:00",
        "run_id": "questions_20260301T123456Z",
        "dataset_path": "eval/day4/questions.jsonl",
        "counts": {
            "total_questions": 1,
            "answerable_questions": 1,
            "unanswerable_questions": 0,
            "labeled_answerable_questions": 1,
            "unlabeled_answerable_questions": 0,
            "answer_evaluated_questions": 0,
        },
        "retrieval": {
            "scoring_scope": "answerable_with_non_empty_gold_only",
            "primary_k": 3,
            "recall_at_k": 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "at_k": {
                "k3": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
            },
            "secondary_diagnostics": {
                "near_page_tolerance": 1,
                "hit_rate_at_k": {
                    "k3": {
                        "strict_page_overlap": 0.0,
                        "doc_only": 1.0,
                        "near_page_tolerance": 1.0,
                    }
                },
            },
            "n_questions_without_gold_in_primary_k": 1,
            "questions_without_gold_in_primary_k": [
                {
                    "qid": "q001",
                    "question": "What is ML-KEM?",
                    "gold": [{"doc_id": "NIST.FIPS.203", "start_page": 3, "end_page": 3}],
                    "top_hit_ids": [
                        {"rank": 1, "doc_id": "NIST.FIPS.204", "pages": "p3-p3", "chunk_id": "c1"}
                    ],
                }
            ],
        },
        "answer": {
            "enabled": False,
            "model_dependent": True,
            "note": "n/a",
            "citation_presence_rate": None,
            "inline_citation_sentence_rate": None,
            "refusal_accuracy": None,
        },
    }

    md = _build_summary_markdown(summary)
    assert "### Questions Missing Gold In Top-k" in md
    assert "q001: What is ML-KEM?" in md
