"""
Dataset utilities for evaluation JSONL files.

Expected question schema:
{
  "qid": "q001",
  "question": "What is ML-KEM?",
  "answerable": true,
  "gold": [
    {"doc_id": "NIST.FIPS.203", "start_page": 8, "end_page": 9}
  ]
}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


GoldSpan = Dict[str, Any]
QuestionRow = Dict[str, Any]
_QID_RE = re.compile(r"^(?P<prefix>[A-Za-z_-]*?)(?P<num>\d+)$")


def qid_sort_key(qid: str) -> tuple:
    """
    Deterministic qid ordering with numeric awareness.

    Examples:
      q2  < q10
      s001 < s010
    """
    s = (qid or "").strip()
    m = _QID_RE.match(s)
    if m:
        return (0, m.group("prefix"), int(m.group("num")), s)
    return (1, s, 0, s)


def _normalize_gold(gold: Iterable[dict]) -> List[GoldSpan]:
    normalized: List[GoldSpan] = []
    for item in gold:
        if not isinstance(item, dict):
            raise ValueError(f"gold item must be an object, got {type(item)!r}")
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            raise ValueError("gold.doc_id must be a non-empty string")

        try:
            start_page = int(item.get("start_page"))
            end_page = int(item.get("end_page"))
        except (TypeError, ValueError) as exc:
            raise ValueError("gold start_page/end_page must be integers") from exc

        if start_page <= 0 or end_page <= 0:
            raise ValueError("gold page spans must be positive")
        if start_page > end_page:
            raise ValueError("gold start_page must be <= end_page")

        normalized.append(
            {
                "doc_id": doc_id,
                "start_page": start_page,
                "end_page": end_page,
            }
        )

    # Stable ordering for deterministic artifacts.
    return sorted(normalized, key=lambda g: (g["doc_id"], g["start_page"], g["end_page"]))


def validate_questions(rows: List[QuestionRow], require_labeled: bool = True) -> None:
    """Validate question rows and enforce deterministic data contracts."""
    seen_qids: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"row {idx}: expected object, got {type(row)!r}")

        qid = str(row.get("qid", "")).strip()
        if not qid:
            raise ValueError(f"row {idx}: qid must be non-empty")
        if qid in seen_qids:
            raise ValueError(f"row {idx}: duplicate qid {qid!r}")
        seen_qids.add(qid)

        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"row {idx}: question must be non-empty")

        answerable = row.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"row {idx}: answerable must be a boolean")

        gold = row.get("gold", [])
        if not isinstance(gold, list):
            raise ValueError(f"row {idx}: gold must be a list")

        normalized_gold = _normalize_gold(gold)
        row["gold"] = normalized_gold

        if answerable and require_labeled and len(normalized_gold) == 0:
            raise ValueError(
                f"row {idx} ({qid}): answerable=true requires at least one gold span"
            )
        if (not answerable) and normalized_gold:
            raise ValueError(
                f"row {idx} ({qid}): answerable=false must not include gold spans"
            )


def load_questions(path: str | Path, require_labeled: bool = True) -> List[QuestionRow]:
    """Load, normalize, and validate questions from JSONL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Question dataset not found: {p}")

    rows: List[QuestionRow] = []
    with p.open("r", encoding="utf-8") as infile:
        for lineno, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON") from exc

            rows.append(
                {
                    "qid": str(rec.get("qid", "")).strip(),
                    "question": str(rec.get("question", "")).strip(),
                    "answerable": rec.get("answerable"),
                    "gold": rec.get("gold", []),
                }
            )

    validate_questions(rows, require_labeled=require_labeled)
    return sorted(rows, key=lambda r: qid_sort_key(str(r.get("qid", ""))))


def write_questions(path: str | Path, rows: List[QuestionRow]) -> None:
    """Write question rows to JSONL with deterministic formatting."""
    validate_questions(rows, require_labeled=False)
    ordered_rows = sorted(rows, key=lambda r: qid_sort_key(str(r.get("qid", ""))))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as outfile:
        for row in ordered_rows:
            payload = {
                "qid": row["qid"],
                "question": row["question"],
                "answerable": bool(row["answerable"]),
                "gold": row["gold"],
            }
            outfile.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
