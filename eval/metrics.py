"""Metrics for retrieval quality and citation/refusal compliance."""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional

from rag.types import REFUSAL_TEXT


_INLINE_CITATION_RE = re.compile(r"\[c\d+\]")


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Returns True when two inclusive page ranges overlap."""
    return not (end_a < start_b or end_b < start_a)


def hit_matches_gold(hit: dict, gold: dict) -> bool:
    """
    Binary relevance contract for retrieval eval.

    A hit is relevant iff:
    1) it has the same doc_id as the gold span, and
    2) the hit page range overlaps the gold page range.
    """
    if hit.get("doc_id") != gold.get("doc_id"):
        return False
    return spans_overlap(
        int(hit.get("start_page", 0)),
        int(hit.get("end_page", 0)),
        int(gold.get("start_page", 0)),
        int(gold.get("end_page", 0)),
    )


def hit_matches_gold_doc_only(hit: dict, gold: dict) -> bool:
    """Relaxed diagnostic: relevant iff document IDs match."""
    return hit.get("doc_id") == gold.get("doc_id")


def hit_matches_gold_with_tolerance(hit: dict, gold: dict, page_tolerance: int = 1) -> bool:
    """
    Relaxed diagnostic: doc_id match plus overlap with ±page_tolerance slack.
    """
    if page_tolerance < 0:
        raise ValueError("page_tolerance must be >= 0")
    if hit.get("doc_id") != gold.get("doc_id"):
        return False

    gold_start = int(gold.get("start_page", 0)) - page_tolerance
    gold_end = int(gold.get("end_page", 0)) + page_tolerance
    return spans_overlap(
        int(hit.get("start_page", 0)),
        int(hit.get("end_page", 0)),
        gold_start,
        gold_end,
    )


def _unique_gold_gain_vector(hits: List[dict], gold: List[dict], k: int) -> List[int]:
    """
    Binary gain vector where each gold span can contribute at most once.

    This keeps nDCG bounded in [0, 1] even if multiple hits overlap the same
    gold span.
    """
    used_gold = [False] * len(gold)
    gains: List[int] = []
    for hit in hits[:k]:
        gain = 0
        for i, g in enumerate(gold):
            if used_gold[i]:
                continue
            if hit_matches_gold(hit, g):
                used_gold[i] = True
                gain = 1
                break
        gains.append(gain)
    return gains


def recall_at_k(hits: List[dict], gold: List[dict], k: int) -> float:
    if not gold:
        return 0.0
    matched = 0
    used_gold = [False] * len(gold)
    for hit in hits[:k]:
        for i, g in enumerate(gold):
            if used_gold[i]:
                continue
            if hit_matches_gold(hit, g):
                used_gold[i] = True
                matched += 1
                break
    return matched / float(len(gold))


def mrr_at_k(hits: List[dict], gold: List[dict], k: int) -> float:
    if not gold:
        return 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if any(hit_matches_gold(hit, g) for g in gold):
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(hits: List[dict], gold: List[dict], k: int) -> float:
    if not gold:
        return 0.0

    gains = _unique_gold_gain_vector(hits, gold, k)
    if not gains:
        return 0.0

    dcg = 0.0
    for i, gain in enumerate(gains, start=1):
        if gain:
            dcg += 1.0 / math.log2(i + 1.0)

    ideal_rel_count = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_rel_count + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(hits: List[dict], gold: List[dict], k: int) -> Dict[str, float]:
    return {
        "recall_at_k": recall_at_k(hits, gold, k),
        "mrr_at_k": mrr_at_k(hits, gold, k),
        "ndcg_at_k": ndcg_at_k(hits, gold, k),
    }


def compute_retrieval_metrics_by_ks(
    hits: List[dict],
    gold: List[dict],
    ks: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """Compute retrieval metrics for multiple k values in one call."""
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted({int(v) for v in ks if int(v) > 0}):
        out[f"k{k}"] = compute_retrieval_metrics(hits=hits, gold=gold, k=k)
    return out


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p for p in parts if p.strip()]


def inline_citation_sentence_rate(answer_text: str) -> Optional[float]:
    sentences = _sentences(answer_text)
    if not sentences:
        return None
    cited = sum(1 for sentence in sentences if _INLINE_CITATION_RE.search(sentence))
    return cited / float(len(sentences))


def evaluate_answer_payload(payload: dict, answerable: bool) -> Dict[str, float | bool | int | None]:
    answer = str(payload.get("answer", "")).strip()
    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    is_refusal = answer.lower() == REFUSAL_TEXT
    citation_presence_ok = (len(citations) == 0) if is_refusal else (len(citations) > 0)
    refusal_accuracy = 1.0 if is_refusal == (not answerable) else 0.0
    inline_rate = None if is_refusal else inline_citation_sentence_rate(answer)

    return {
        "is_refusal": is_refusal,
        "citation_count": len(citations),
        "citation_presence_ok": citation_presence_ok,
        "inline_citation_sentence_rate": inline_rate,
        "refusal_accuracy": refusal_accuracy,
    }


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / float(len(vals))
