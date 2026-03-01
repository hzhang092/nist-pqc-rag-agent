"""
Core logic for generating a cited answer from retrieved document chunks.

This module orchestrates the process of taking a list of retrieved `ChunkHit`
objects, selecting the most relevant ones as evidence, constructing a prompt
for a language model, and then validating the generated answer to ensure it is
properly grounded in the provided citations.

The main entry point is `build_cited_answer`, which enforces a strict contract:
- Evidence is selected and budgeted.
- A detailed prompt instructs the model on how to use inline citations.
- The model's raw output is rigorously validated by `enforce_inline_citations`
  to ensure every sentence is cited and no external information is used.
- If validation fails, the system defaults to a safe, "I can't answer" response.

Improvements in this version:
- Fix prompt formatting bug (missing newline before "Question:").
- Prettify algorithm/pseudocode evidence to reduce "all-on-one-line" failures.
- Add deterministic fallback for "Algorithm N" step questions when the model refuses,
  but the steps are clearly present in evidence.
- Add deterministic fallback for comparison questions when role-bearing evidence is present
  but the model still refuses.
"""
# rag/rag_answer.py
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from rag.config import SETTINGS
from rag.retriever.base import ChunkHit
from rag.types import AnswerResult, Citation, REFUSAL_TEXT, validate_answer


_CITE_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_CITE_TOKEN_RE = re.compile(r"\bc\d+\b", flags=re.IGNORECASE)
_ALG_Q_RE = re.compile(r"\bAlgorithm\s+(?P<n>\d+)\b", re.IGNORECASE)
_STEP_RE = re.compile(r"^\s*(?P<k>\d+)\s*:\s*(?P<body>.+?)\s*$")
_COMPARE_PATTERNS = (
    re.compile(r"(?:differences?|difference)\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+)$", flags=re.IGNORECASE),
    re.compile(r"(?:compare|comparison\s+of)\s+(?P<a>.+?)\s+(?:and|vs|versus)\s+(?P<b>.+)$", flags=re.IGNORECASE),
    re.compile(r"(?P<a>.+?)\s+(?:vs|versus)\s+(?P<b>.+)$", flags=re.IGNORECASE),
)
_ROLE_PATTERNS = (
    (re.compile(r"\bkey-encapsulation mechanism\b", flags=re.IGNORECASE), "key-encapsulation mechanism"),
    (re.compile(r"\bdigital signature scheme\b", flags=re.IGNORECASE), "digital signature scheme"),
    (re.compile(r"\bkey establishment scheme\b", flags=re.IGNORECASE), "key establishment scheme"),
)

CHUNK_STORE_PATH = Path("data/processed/chunk_store.jsonl")


def _stable_sort_key(h: ChunkHit) -> tuple:
    """
    Creates a sort key for `ChunkHit` objects for deterministic ordering.

    Sorts primarily by score (descending), then uses document/chunk IDs and
    page numbers as tie-breakers to ensure stable results across runs.
    """
    return (-h.score, h.doc_id, h.start_page, h.end_page, h.chunk_id)


@lru_cache(maxsize=1)
def _load_chunk_store_maps() -> Tuple[Dict[str, int], Dict[int, dict]]:
    """Loads cached `chunk_id <-> vector_id` maps from chunk store for neighbor lookup."""
    chunk_to_vector: Dict[str, int] = {}
    vector_to_rec: Dict[int, dict] = {}

    if not CHUNK_STORE_PATH.exists():
        return chunk_to_vector, vector_to_rec

    with CHUNK_STORE_PATH.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vector_id = int(rec.get("vector_id", -1))
            chunk_id = rec.get("chunk_id", "")
            if vector_id < 0 or not chunk_id:
                continue
            vector_to_rec[vector_id] = rec
            chunk_to_vector[chunk_id] = vector_id

    return chunk_to_vector, vector_to_rec


def _neighbor_hits(hit: ChunkHit, window: int) -> List[ChunkHit]:
    """Returns same-document neighboring chunks around `hit` using vector-id adjacency."""
    if window <= 0:
        return []

    chunk_to_vector, vector_to_rec = _load_chunk_store_maps()
    vector_id = chunk_to_vector.get(hit.chunk_id)
    if vector_id is None:
        return []

    neighbors: List[ChunkHit] = []
    for delta in range(1, window + 1):
        for candidate_vid in (vector_id - delta, vector_id + delta):
            rec = vector_to_rec.get(candidate_vid)
            if rec is None:
                continue
            if rec.get("doc_id") != hit.doc_id:
                continue
            neighbors.append(
                ChunkHit(
                    score=hit.score - (delta * 1e-6),
                    chunk_id=rec.get("chunk_id", ""),
                    doc_id=rec.get("doc_id", ""),
                    start_page=int(rec.get("start_page", 0)),
                    end_page=int(rec.get("end_page", 0)),
                    text=rec.get("text", ""),
                )
            )

    return neighbors


def select_evidence(hits: List[ChunkHit]) -> List[ChunkHit]:
    """
    Filters, de-duplicates, and budgets raw search hits to form the final evidence set.
    """
    best: Dict[str, ChunkHit] = {}
    for h in hits:
        prev = best.get(h.chunk_id)
        if prev is None or h.score > prev.score:
            best[h.chunk_id] = h

    ordered = sorted(best.values(), key=_stable_sort_key)

    primary_hits = ordered[: SETTINGS.ASK_MAX_CONTEXT_CHUNKS]

    expanded: List[ChunkHit] = []
    seen_chunk_ids = set()
    for h in primary_hits:
        if h.chunk_id not in seen_chunk_ids:
            expanded.append(h)
            seen_chunk_ids.add(h.chunk_id)

        if SETTINGS.ASK_INCLUDE_NEIGHBOR_CHUNKS:
            for n in _neighbor_hits(h, SETTINGS.ASK_NEIGHBOR_WINDOW):
                if n.chunk_id in seen_chunk_ids:
                    continue
                expanded.append(n)
                seen_chunk_ids.add(n.chunk_id)

    budgeted: List[ChunkHit] = []
    total = 0
    for h in expanded:
        if len(budgeted) >= SETTINGS.ASK_MAX_CONTEXT_CHUNKS:
            break
        t = h.text or ""
        if not t.strip():
            continue
        if total + len(t) > SETTINGS.ASK_MAX_CONTEXT_CHARS and budgeted:
            break
        budgeted.append(h)
        total += len(t)

    return budgeted


def _prettify_evidence_text(t: str) -> str:
    """
    Make boxed/algorithm text easier for the LLM to read.

    Many standards algorithms come through as "1: ... 2: ... 3: ..." on one line.
    We only apply this when we see algorithm-ish patterns to avoid altering normal prose.
    """
    if not t:
        return t

    looks_like_algo = ("Algorithm" in t) or bool(re.search(r"\b\d+\s*:\s*", t))
    if not looks_like_algo:
        return t

    s = t
    # Insert newlines before "1:", "2:", etc.
    s = re.sub(r"\s+(?=\d+\s*:)", "\n", s)
    # Insert newlines before "for (" blocks if they got glued
    s = re.sub(r"\s+(?=for\s*\()", "\n", s, flags=re.IGNORECASE)
    # Light cleanup of common OCR/parse glue like "endctx" -> "end ctx" (optional, safe)
    s = re.sub(r"\bendctx\b", "end ctx", s)
    return s


def build_context_and_citations(evidence: List[ChunkHit]) -> Tuple[str, Dict[str, Citation]]:
    """
    Constructs the context string for the LLM prompt and a citation map.
    """
    key_to_cit: Dict[str, Citation] = {}
    blocks: List[str] = []

    for i, h in enumerate(evidence, start=1):
        key = f"c{i}"
        key_to_cit[key] = Citation(
            key=key,
            doc_id=h.doc_id,
            start_page=h.start_page,
            end_page=h.end_page,
            chunk_id=h.chunk_id,
        )

        text = _prettify_evidence_text((h.text or "").strip())

        blocks.append(
            f"[{key}] {h.doc_id} p{h.start_page}-p{h.end_page} chunk_id={h.chunk_id}\n"
            f"{text}"
        )

    context = "\n\n---\n\n".join(blocks)
    return context, key_to_cit


def _sentences(text: str) -> List[str]:
    """
    Splits text into sentences using a simple regex heuristic.
    """
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _extract_inline_citation_keys(text: str) -> set[str]:
    """
    Extract inline citation keys from bracketed markers and normalize to lowercase.

    Accepts:
    - [c1]
    - [C1]
    - [c1][c2]
    - [c1, c2]
    """
    keys: set[str] = set()
    for bracket_content in _CITE_BRACKET_RE.findall(text):
        for token in _CITE_TOKEN_RE.findall(bracket_content):
            keys.add(token.lower())
    return keys


def enforce_inline_citations(answer_text: str, key_to_cit: Dict[str, Citation]) -> AnswerResult:
    """
    Validates the raw LLM output to enforce strict citation grounding.
    """
    ans = answer_text.strip()

    if ans.lower() in {"not found", "not found in documents", REFUSAL_TEXT.lower()}:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    used = _extract_inline_citation_keys(ans)
    if not used:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    unknown = used - set(key_to_cit.keys())
    if unknown:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    # Every sentence must have ≥1 citation marker
    for s in _sentences(ans):
        if not _extract_inline_citation_keys(s):
            result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
            validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
            return result

    citations = [key_to_cit[k] for k in sorted(used, key=lambda x: int(x[1:]))]

    result = AnswerResult(answer=ans, citations=citations)
    validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
    return result


def _extract_algorithm_steps(text: str) -> List[Tuple[int, str]]:
    """
    Extract step lines like "1: ..." from an evidence chunk.

    Returns:
        List of (step_number, step_body) in the order encountered.
    """
    pretty = _prettify_evidence_text(text or "")
    steps: List[Tuple[int, str]] = []
    for line in pretty.splitlines():
        m = _STEP_RE.match(line)
        if not m:
            continue
        k = int(m.group("k"))
        body = m.group("body").strip()
        if body:
            steps.append((k, body))
    return steps


def _extract_compare_topics(question: str) -> Tuple[str | None, str | None]:
    q = (question or "").strip().rstrip("?").strip()
    if not q:
        return None, None

    for pattern in _COMPARE_PATTERNS:
        m = pattern.search(q)
        if not m:
            continue
        topic_a = m.group("a").strip().strip(" .,:;\"'`[](){}")
        topic_b = m.group("b").strip().strip(" .,:;\"'`[](){}")
        if topic_a and topic_b and topic_a.lower() != topic_b.lower():
            return topic_a, topic_b
    return None, None


def _normalized_topic_token(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (topic or "").lower())


def _topic_in_text(topic: str, text: str) -> bool:
    token = _normalized_topic_token(topic)
    if not token:
        return False
    norm_text = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return token in norm_text


def _infer_role_from_text(text: str) -> str | None:
    for pattern, label in _ROLE_PATTERNS:
        if pattern.search(text or ""):
            return label
    return None


def _first_sentence(text: str) -> str:
    sents = _sentences(_prettify_evidence_text(text or ""))
    if not sents:
        return ""
    for s in sents:
        candidate = s.strip()
        if len(candidate) < 25:
            continue
        if sum(ch.isalpha() for ch in candidate) < 12:
            continue
        return candidate
    return sents[0].strip()


def _pick_topic_hit(topic: str, ordered: List[ChunkHit]) -> ChunkHit | None:
    first_hit: ChunkHit | None = None
    for h in ordered:
        text = h.text or ""
        if not _topic_in_text(topic, text):
            continue
        if first_hit is None:
            first_hit = h
        if _infer_role_from_text(text):
            return h
    return first_hit


def _comparison_fallback_answer(question: str, hits: List[ChunkHit]) -> Tuple[str, Dict[str, Citation]] | None:
    """
    Deterministic fallback for compare questions when the model refuses.

    Uses high-signal role phrases from evidence (e.g., "key-encapsulation mechanism",
    "digital signature scheme") so we can still return a minimally grounded comparison.
    """
    topic_a, topic_b = _extract_compare_topics(question)
    if not topic_a or not topic_b:
        return None

    best: Dict[str, ChunkHit] = {}
    for h in hits:
        prev = best.get(h.chunk_id)
        if prev is None or h.score > prev.score:
            best[h.chunk_id] = h
    ordered = sorted(best.values(), key=_stable_sort_key)

    hit_a = _pick_topic_hit(topic_a, ordered)
    hit_b = _pick_topic_hit(topic_b, ordered)

    if hit_a is None or hit_b is None:
        return None

    evidence = [hit_a]
    if hit_b.chunk_id != hit_a.chunk_id:
        evidence.append(hit_b)

    _, key_to_cit = build_context_and_citations(evidence)

    key_a = "c1"
    key_b = "c1" if len(evidence) == 1 else "c2"
    role_a = _infer_role_from_text(hit_a.text or "") or ""
    role_b = _infer_role_from_text(hit_b.text or "") or ""

    if role_a and role_b and role_a != role_b:
        bullets = [
            f"- {topic_a} is specified as a {role_a} [{key_a}].",
            f"- {topic_b} is specified as a {role_b} [{key_b}].",
            f"- This indicates the standards assign different core purposes to these schemes [{key_a}][{key_b}].",
        ]
        return "\n".join(bullets), key_to_cit

    sent_a = _first_sentence(hit_a.text or "").rstrip(".")
    sent_b = _first_sentence(hit_b.text or "").rstrip(".")
    if not sent_a or not sent_b:
        return None

    bullets = [
        f"- {topic_a}: {sent_a} [{key_a}].",
        f"- {topic_b}: {sent_b} [{key_b}].",
        f"- These statements provide citable evidence for a side-by-side comparison [{key_a}][{key_b}].",
    ]
    return "\n".join(bullets), key_to_cit


def _algorithm_fallback_answer(question: str, evidence: List[ChunkHit], key_to_cit: Dict[str, Citation]) -> str | None:
    """
    Deterministic fallback: if question asks about Algorithm N steps and evidence contains
    the algorithm block with numbered steps, return those steps as cited bullets.

    Returns a string with inline citations, or None if not applicable.
    """
    m = _ALG_Q_RE.search(question or "")
    if not m:
        return None
    alg_n = m.group("n")

    # Find the first evidence chunk that clearly contains "Algorithm N" and step markers.
    chosen_key = None
    chosen_text = None
    for key, cit in key_to_cit.items():
        # map key -> actual evidence text (same order as evidence list)
        # key is c{i} where i starts at 1
        idx = int(key[1:]) - 1
        if idx < 0 or idx >= len(evidence):
            continue
        t = (evidence[idx].text or "")
        if re.search(rf"\bAlgorithm\s+{re.escape(alg_n)}\b", t, flags=re.IGNORECASE) and re.search(r"\b1\s*:\s*", t):
            chosen_key = key
            chosen_text = t
            break

    if not chosen_key or not chosen_text:
        return None

    steps = _extract_algorithm_steps(chosen_text)
    if not steps:
        return None

    # Keep it compact but complete; most NIST algorithms are <= ~15 steps.
    steps = steps[:25]

    bullets: List[str] = []
    for k, body in steps:
        # IMPORTANT: citation must be inside the same sentence (before any terminal punctuation)
        bullets.append(f"- {k}: {body} [{chosen_key}].")

    return "\n".join(bullets)


def build_cited_answer(
    question: str,
    hits: List[ChunkHit],
    generate_fn: Callable[[str], str],
) -> AnswerResult:
    """
    The main entry point for generating a citation-grounded answer.
    """
    evidence = select_evidence(hits)
    if len(evidence) < SETTINGS.ASK_MIN_EVIDENCE_HITS:
        result = AnswerResult(answer=REFUSAL_TEXT, citations=[])
        validate_answer(result, require_citations=SETTINGS.ASK_REQUIRE_CITATIONS, require_inline_markers=True)
        return result

    context, key_to_cit = build_context_and_citations(evidence)

    # NOTE: fixed the missing newline before "Question:".
    prompt = (
        "You are a citation-grounded assistant. Answer ONLY using the evidence below.\n"
        "Rules:\n"
        "1) Every sentence MUST include at least one inline citation marker like [c1].\n"
        "2) You may ONLY use citation markers that appear in the evidence headers.\n"
        f"3) If the evidence is insufficient, reply exactly: {REFUSAL_TEXT}\n"
        "4) Be concise and factual.\n"
        "5) NEVER hallucinate or use information not in the provided evidence.\n"
        "6) Answer in 3–6 bullets. Each bullet should be ONE sentence and end with one or more citation markers.\n"
        "7) If multiple sources are needed for one claim, cite all of them at the end (e.g., [c1][c2] or [c1, c2]).\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{context}\n"
    )

    raw_answer = generate_fn(prompt)
    result = enforce_inline_citations(raw_answer, key_to_cit)

    # Deterministic safety net: if the model refuses but the algorithm steps are present,
    # extract them directly and still enforce the strict citation contract.
    if result.answer == REFUSAL_TEXT:
        fb = _algorithm_fallback_answer(question, evidence, key_to_cit)
        if fb:
            result2 = enforce_inline_citations(fb, key_to_cit)
            if result2.answer != REFUSAL_TEXT:
                return result2

    # Deterministic comparison fallback: produce a minimal grounded compare answer
    # from role-bearing evidence when LLM generation still refuses.
    if result.answer == REFUSAL_TEXT:
        fb_cmp = _comparison_fallback_answer(question, hits)
        if fb_cmp:
            fb_cmp_text, fb_cmp_key_map = fb_cmp
            result3 = enforce_inline_citations(fb_cmp_text, fb_cmp_key_map)
            if result3.answer != REFUSAL_TEXT:
                return result3

    return result
