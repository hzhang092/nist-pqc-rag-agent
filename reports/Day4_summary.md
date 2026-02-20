# Day 4 Summary — Evaluation Baseline and Diagnostics

Date: 2026-02-19
Project: `nist-pqc-rag-agent`

## 1) What was done on Day 4

Day 4 focused on implementing a reproducible evaluation harness aligned with the Week-1 milestone in `reports/project_overview.md`.

Implemented:

- A formal eval dataset contract and loader:
  - `eval/dataset.py`
  - deterministic `qid` ordering (`qid_sort_key`)
  - schema validation (`qid`, `question`, `answerable`, `gold`)
  - strict guardrails for labeled vs unlabeled runs

- A retrieval eval adapter:
  - `rag/retrieve.py::retrieve_for_eval(...)`
  - stable output schema for eval rows
  - deterministic tie-breaking for ranking:
    `(-score, doc_id, start_page, end_page, chunk_id)`

- Metric implementations:
  - `eval/metrics.py`
  - `Recall@k`, `MRR@k`, `nDCG@k`
  - binary relevance contract: doc match + page overlap
  - de-duplicated nDCG gain logic to avoid inflation when multiple hits match the same gold span

- End-to-end runner:
  - `eval/run.py`
  - outputs: `per_question.jsonl`, `summary.json`, `summary.md`
  - deterministic ordering in output artifacts
  - multiple-k scoring via `--ks` （k is the number of top retrieval results to consider for metrics)
  - explicit retrieval scoring scope and skipped-question reporting

- Secondary diagnostics (to separate retrieval failure from strict label mismatch):
  - doc-only hit-rate
  - near-page hit-rate (doc match + overlap within ±N pages via `--near-page-tolerance`)
  - per-question debug fields:
    - `gold_hit_ranks` (strict)
    - `doc_hit_ranks` (doc-only)
    - `near_page_hit_ranks` (±N page)
    - `top_hit_ids`

- Tests:
  - `tests/test_retrieve_eval_api.py`
  - `tests/test_eval_dataset.py`
  - `tests/test_eval_metrics.py`

Result: eval harness is now deterministic, inspectable, and regression-friendly.

## 2) Improvements before Day 4 vs improvements implemented on Day 4

### Before Day 4 (Days 1–3 foundation)

- Deterministic ingestion, cleaning, chunking, and index artifacts existed.
- Hybrid retrieval existed (`FAISS + BM25 + RRF + rerank`) with deterministic query rewrites.
- Citation-first answer generation and bounded LangGraph controller were already in place.
- Sanity scripts and baseline run logging existed, but there was no full Week-1 eval harness with standardized metrics and contracts.

### Day 4 improvements

- Converted eval from ad-hoc/sanity outputs into a formal harness with stable data contracts.
- Added a reusable retrieval-eval interface (`retrieve_for_eval`) that decouples eval code from retriever internals.
- Introduced explicit and test-locked metric semantics.
- Added multi-k evaluation in one run (`k1/k3/k5/k8`) to avoid repeated runs for simple curve inspection.
- Added strict-vs-relaxed diagnostics in summary output so interpretation is less ambiguous.
- Added deterministic sorting and output guarantees to avoid run-to-run drift.

## 3) Evaluation criteria: mechanisms and decisions

Primary retrieval criterion (strict, used for headline regression metrics):

- Relevance definition:
  - same `doc_id`
  - overlapping page range (`hit.start_page..hit.end_page` overlaps `gold.start_page..gold.end_page`)
- Metrics:
  - `Recall@k`: fraction of gold spans recovered in top-k
  - `MRR@k`: reciprocal rank of first relevant hit
  - `nDCG@k`: discounted ranking quality with relevance contribution capped per gold span

Decision:

- Primary scoring scope is explicitly:
  - `answerable_with_non_empty_gold_only`
- Unanswerable rows are excluded from retrieval scoring and reported separately.

Secondary diagnostics (analysis-only, not primary regression gate):

- `doc_only`: hit if any top-k result has matching `doc_id`
- `near_page_tolerance`: hit if doc matches and page overlap holds after ±N page expansion
- Purpose:
  - identify whether misses are due to true retrieval failure vs strict page labels
  - support label QA and retriever tuning without weakening primary metric definitions

Answer-side metrics:

- kept optional (`--with-answers`)
- explicitly labeled as model-dependent in summary outputs
- not treated as primary regression signal

## 4) Current Day 4 baseline (from `reports/eval/day4_baseline/summary.json`)

Dataset and scope:

- total questions: 13
- answerable: 12
- unanswerable: 1
- retrieval-evaluated: 12

Primary retrieval (strict):

- `Recall@8 = 0.4167`
- `MRR@8 = 0.2569`
- `nDCG@8 = 0.3011`

By-k trend:

- k1: recall 0.1667
- k3: recall 0.3333
- k5: recall 0.3750
- k8: recall 0.4167

Secondary diagnostics at k8:

- strict hit-rate: 0.4167
- near-page (±1) hit-rate: 0.5833
- doc-only hit-rate: 0.8333

Interpretation:

- Retrieval often reaches the right document family, but strict page-level targeting is still weak.
- This indicates both retriever ranking gaps and potential gold-label strictness on concept-heavy questions.

## 5) What to do next

1. Improve retrieval for concept questions without breaking deterministic behavior.
- Expand deterministic query variants for generic conceptual prompts.
- Add stronger lexical boosts for “standards list / why / compare” style intents.

2. Run small ablation sweeps and compare deltas with this harness.
- candidate knobs: `candidate_multiplier`, `k0`, `rerank_pool`, `--no-rerank`, `--no-fusion`
- keep same dataset and output schema for fair before/after deltas.

3. Tighten label quality where needed.
- Review misses with high doc-only but low strict overlap.
- Resolve whether gold spans are too narrow for broad conceptual questions.

4. Add one eval-shape regression test for `eval.run`.
- Assert required fields in `summary.json` and `per_question.jsonl`:
  - strict metrics
  - secondary diagnostics
  - per-question debug rank lists

5. Optional (after retrieval tuning stabilizes):
- enable `--with-answers` for citation-format trend tracking
- keep answer metrics secondary due to model dependence.

## 6) Day 4 deliverable status

Milestone status: complete for Week-1 Day 4 baseline.

- Eval harness exists and runs end-to-end.
- Deterministic outputs and contracts are in place.
- Retrieval metrics are measurable and debuggable.
- Baseline snapshot is ready for Day 5 improvement work and before/after deltas.

