# Technique Report: Comparison Routing + Citation Refusal Failure

Date: February 28, 2026  
Repo: `nist-pqc-rag-agent`  
Author: Codex (implementation + analysis)

## 1. Executive Summary

A high-impact failure occurred after comparison routing was improved in the LangGraph controller: comparison questions (for example, "What are the differences between ML-KEM and ML-DSA?") often ended in refusal even when evidence retrieval succeeded.

The failure mode was not a single bug. It was a chain:
1. Compare routing started returning better multi-document evidence.
2. Answer-generation prompt still constrained output to exactly one citation per sentence.
3. Citation parsing/validation was brittle for common citation variants.
4. When Gemini still refused (`not found in provided docs`), there was no deterministic compare fallback.

The fix introduced:
- comparison-safe citation formatting rules,
- robust citation key extraction,
- a deterministic comparison fallback path (analogous to existing algorithm-step fallback), and
- targeted regression tests.

Outcome: the exact failing query now returns a cited answer and cites both FIPS 203 and FIPS 204 in end-to-end agent execution.

---

## 2. Problem Context

### 2.1 User-reported symptom

From the attached notes file (`D:\henry\Downloads\not done - Routing Fix Citation Issue.md`):
- compare routing was fixed for phrases like "differences between"
- but after that fix, Gemini could not produce a cited answer for compare queries
- before routing fix, the query sometimes succeeded via default retrieve path
- after routing fix, compare path frequently ended with no citations and refusal

### 2.2 Why this mattered

This issue directly violated the Week-1 objective of citation-grounded answering for core PQC comparison questions. It also masked useful evidence behind a strict answer contract and caused false negatives in agent behavior.

---

## 3. Reproduction and Evidence

### 3.1 Before this patch

A representative failed run (post-routing fix, pre-answer-fix):
- `runs/agent/agent_20260228_214548_what_are_the_differences_between_mlkem_and_mldsa.json`

Key facts from trace:
- plan action: `compare`
- evidence retrieved: sufficient, diverse docs
- assessment: `sufficient = true`
- draft answer: `not found in provided docs`
- citations: `[]`
- final refusal reason: `missing_citations`

### 3.2 Before/after behavior comparison

- Earlier run via default retrieve path (older behavior): produced answer with citations but weaker semantics.
- Compare-routed run (new behavior before patch): better retrieval intent, but answer stage refused.
- After this patch: compare-routed run returns cited, non-refusal output.

Representative successful run after fix:
- `runs/agent/agent_20260228_214854_what_are_the_differences_between_mlkem_and_mldsa.json`

---

## 4. Root Cause Analysis

This was a multi-cause failure chain.

### 4.1 Prompt contract conflict with comparison claims

`rag/rag_answer.py` instructed the model to:
- produce one sentence per bullet,
- and end each with exactly one `[c#]` marker.

Comparison claims frequently require evidence from two sources in one sentence. The contract discouraged/blocked that pattern.

### 4.2 Citation parser/validator strictness

Citation extraction expected only exact `[cN]` forms. Common variants like `[C1]` and grouped markers were not robustly handled. This increased false refusal risk from formatting differences.

### 4.3 Missing deterministic compare fallback

There was a deterministic fallback for algorithm step questions, but no equivalent for comparisons. If the model refused despite sufficient evidence, the pipeline had no recovery path.

### 4.4 Subtle evidence-selection interaction

Initial fallback attempt used `select_evidence(...)` output only. In real runs, top selected chunks could overrepresent one topic, causing fallback topic matching to fail. Fallback had to reason over the full hit list (deduped + stably sorted), not only the reduced context window.

---

## 5. Fix Design Goals

1. Preserve strict citation policy (`no claim without citation`).
2. Keep behavior deterministic.
3. Make minimal, testable diffs.
4. Avoid changing ingestion/chunking/index data contracts.
5. Improve compare reliability without widening scope or adding heavy deps.

---

## 6. Implementation Details

### 6.1 `rag/rag_answer.py`

#### A) Robust inline citation extraction

Replaced strict single-pattern extraction with normalized bracket-token parsing:
- accepts `[c1]`
- accepts `[C1]`
- accepts `[c1][c2]`
- accepts `[c1, c2]`

Key additions:
- `_CITE_BRACKET_RE`
- `_CITE_TOKEN_RE`
- `_extract_inline_citation_keys(...)`

Applied in:
- overall used-key extraction
- per-sentence citation enforcement
- refusal recognition normalization (`REFUSAL_TEXT.lower()`).

#### B) Prompt contract made comparison-safe

Updated guidance from:
- "end with exactly one [c#]"

to:
- "end with one or more citation markers"
- explicit multi-source syntax examples: `[c1][c2]` or `[c1, c2]`.

This keeps strictness (citation required per sentence) while allowing valid comparison composition.

#### C) Deterministic comparison fallback

Added `_comparison_fallback_answer(question, hits)` that triggers when model output is refusal.

Algorithm:
1. Parse comparison topics from question.
2. Dedupe hits by `chunk_id`; stable-sort by existing deterministic key.
3. Select one hit per topic, preferring role-bearing text (e.g., "key-encapsulation mechanism", "digital signature scheme").
4. Build temporary two-hit evidence map and deterministic citation keys (`c1`, `c2`).
5. Generate a minimal 3-bullet comparison answer with strict inline citations.
6. Re-run `enforce_inline_citations(...)` before returning.

Why deterministic:
- stable sort key reused
- no LLM involved in fallback construction
- deterministic key assignment via local evidence mapping

#### D) Fallback quality refinement

Added `_pick_topic_hit(...)` and improved `_first_sentence(...)` filtering to avoid low-information snippets (very short/noisy first sentences) when composing fallback bullets.

---

### 6.2 `rag/types.py`

Updated `extract_citation_keys(...)` to match answer-layer citation parsing robustness (same bracket+token approach). This keeps validation behavior consistent across modules.

---

### 6.3 `tests/test_rag_answer.py`

Added/extended regression coverage:

1. `test_accepts_multiple_citations_in_single_sentence`
- Validates `[c1][c2]` in one sentence is accepted.

2. `test_accepts_uppercase_and_comma_grouped_citations`
- Validates `[C1, c2]` parsing and normalization.

3. `test_prompt_allows_multiple_citations_per_sentence`
- Locks prompt contract to avoid regressions back to "exactly one".

4. `test_comparison_fallback_kicks_in_when_model_refuses`
- Forces refusal output from generator and ensures compare fallback returns cited answer with both FIPS doc IDs.

5. `test_comparison_fallback_prefers_role_bearing_hits`
- Ensures fallback does not blindly pick highest-score topic hit if it lacks role-defining semantics.

---

## 7. Verification

### 7.1 Unit/graph tests

Command:

```powershell
D:\Softwares\anaconda\envs\eleven\python.exe -m pytest tests/test_rag_answer.py tests/test_lc_graph.py
```

Result:
- `19 passed`

### 7.2 Retrieval sanity (existing repo entrypoint)

Command:

```powershell
D:\Softwares\anaconda\envs\eleven\python.exe -m rag.search_faiss "ML-KEM key generation"
```

Result:
- returns top FIPS 203 hits with page-level metadata (sanity intact).

### 7.3 End-to-end agent reproduction (target issue)

Command:

```powershell
D:\Softwares\anaconda\envs\eleven\python.exe -m rag.agent.ask "What are the differences between ML-KEM and ML-DSA?"
```

After fix, output is cited (non-refusal), including both topics/documents:
- ML-KEM cited from FIPS 203
- ML-DSA cited from FIPS 204
- combined comparison bullet cites both

Artifact:
- `runs/agent/agent_20260228_214854_what_are_the_differences_between_mlkem_and_mldsa.json`

---

## 8. Data Contracts and Scope Impact

### 8.1 Contracts preserved

No changes to:
- chunk schema (`doc_id`, `start_page`, `end_page`, etc.)
- ingestion/chunk IDs/order determinism
- retrieval interface contracts

### 8.2 Scope preserved

No document scope widening, no new heavy dependencies, no backend changes.

---

## 9. Risks and Residual Gaps

1. Deterministic compare fallback is intentionally minimal.
- It prioritizes groundedness over exhaustive nuance.

2. If retrieval misses one topic entirely, fallback correctly refuses.
- This is expected behavior under strict citation policy.

3. Current fallback relies on role-phrase heuristics.
- It is robust for ML-KEM/ML-DSA framing, but broader semantic coverage would benefit from eval-driven iteration.

---

## 10. Recommended Next Steps

1. Run full eval harness and compare citation-compliance deltas on compare questions:

```powershell
D:\Softwares\anaconda\envs\eleven\python.exe -m eval.run
```

2. Add targeted compare-question set in eval dataset with expected doc diversity and citation requirements.

3. Optionally add a bounded citation-repair pass (second generation prompt) before fallback for improved answer fluency while preserving strict contract.

---

## 11. Files Changed

- `rag/rag_answer.py`
- `rag/types.py`
- `tests/test_rag_answer.py`

No other production contracts or pipeline stages were modified.
