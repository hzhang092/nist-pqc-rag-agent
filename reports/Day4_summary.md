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
  
  Detailed mechanism implemented:
  - `load_questions(path, require_labeled=...)` reads JSONL line-by-line, normalizes rows to a fixed shape, and rejects malformed JSON with line-aware errors.
  - `validate_questions(...)` enforces:
    - unique non-empty `qid`,
    - non-empty `question`,
    - boolean `answerable`,
    - list-typed `gold`.
  - `_normalize_gold(...)` enforces per-span contract (`doc_id` non-empty, positive pages, `start_page <= end_page`) and stable-sorts gold spans by `(doc_id, start_page, end_page)`.
  - Labeled/unlabeled guardrails are explicit:
    - `answerable=true` requires non-empty gold when `require_labeled=True`,
    - `answerable=false` must have empty gold.
  - Final row order is deterministic via `qid_sort_key(...)` (numeric-aware ordering like `q2 < q10`).

- A retrieval eval adapter:
  - `rag/retrieve.py::retrieve_for_eval(...)`
  - stable output schema for eval rows
  - deterministic tie-breaking for ranking:
    `(-score, doc_id, start_page, end_page, chunk_id)`

  Detailed mechanism implemented:
  - `retrieve_for_eval(...)` forwards eval knobs into the shared retrieval stack (`mode`, `backend`, `k0`, `candidate_multiplier`, `fusion`, `cheap_rerank`, `rerank_pool`).
  - Returned `ChunkHit` objects are re-sorted with `_eval_rank_key` to guarantee deterministic ranking for metrics.
  - Adapter emits a stable row schema per hit:
    - `chunk_id`, `doc_id`, `start_page`, `end_page`, `score`, `text`, `rank`, `mode`.
  - `rank` is regenerated from sorted order (1-indexed), so downstream eval is decoupled from backend-specific ordering.
  - `evidence_window` and `debug` are accepted for API stability (currently no-op in adapter).

- Metric implementations:
  - `eval/metrics.py`
  - `Recall@k`, `MRR@k`, `nDCG@k`
  - binary relevance contract: doc match + page overlap
  - de-duplicated nDCG gain logic to avoid inflation when multiple hits match the same gold span

  Detailed mechanism implemented:
  - Primary relevance function is `hit_matches_gold(hit, gold)`:
    - exact `doc_id` match required,
    - inclusive page-range overlap required via `spans_overlap(...)`.
  - `recall_at_k(...)` matches each gold span at most once (tracks `used_gold`) to avoid double counting.
  - `mrr_at_k(...)` uses rank of first relevant hit among top-k.
  - `ndcg_at_k(...)` uses `_unique_gold_gain_vector(...)` so each gold span contributes gain once; this keeps nDCG bounded and prevents duplicate-hit inflation.
  - Multi-k metrics are computed in one pass contract via `compute_retrieval_metrics_by_ks(...)` with deduped/sorted k values.
  - Relaxed diagnostic matchers are explicit:
    - `hit_matches_gold_doc_only(...)`,
    - `hit_matches_gold_with_tolerance(..., page_tolerance=N)`.

- End-to-end runner:
  - `eval/run.py`
  - outputs: `per_question.jsonl`, `summary.json`, `summary.md`
  - deterministic ordering in output artifacts
  - multiple-k scoring via `--ks` (k is the number of top retrieval results to consider for metrics)
  - explicit retrieval scoring scope and skipped-question reporting

  Detailed mechanism implemented:
  - CLI parsing in `eval/run.py` supports retrieval knobs + eval knobs:
    - `--mode`, `--backend`, `--k`, `--ks`, `--k0`, `--candidate-multiplier`, `--rerank-pool`, `--no-fusion`, `--no-rerank`, `--near-page-tolerance`, `--allow-unlabeled`, `--with-answers`.
  - `_parse_ks(...)` validates/normalizes k-list input; `retrieval_depth = max(primary_k, max(ks))` ensures enough hits are retrieved for all requested metrics.
  - Per question:
    - retrieves via `retrieve_for_eval(...)`,
    - computes retrieval metrics only for `answerable=true` with non-empty gold,
    - tracks skipped qids for unanswerable and unlabeled-answerable cases.
  - Optional answer-side scoring (`--with-answers`) runs `rag.ask --json --no-evidence` via subprocess and evaluates citation/refusal behavior.
  - Aggregation uses `safe_mean(...)`; summary includes run config, counts, retrieval metrics, secondary diagnostics, and optional answer metrics.
  - Output writing is deterministic:
    - `per_question.jsonl` sorted by `qid_sort_key`,
    - JSON serialized with stable keys (`sort_keys=True`),
    - markdown summary generated from summary JSON.

- Secondary diagnostics (to separate retrieval failure from strict label mismatch):
  - doc-only hit-rate
  - near-page hit-rate (doc match + overlap within ±N pages via `--near-page-tolerance`)
  - per-question debug fields:
    - `gold_hit_ranks` (strict)
    - `doc_hit_ranks` (doc-only)
    - `near_page_hit_ranks` (±N page)
    - `top_hit_ids`

  Detailed mechanism implemented:
  - `_gold_hit_ranks(...)` computes rank lists with pluggable matcher, reused for strict/doc-only/near-page diagnostics.
  - `_hit_rate_at_k(...)` converts per-question rank lists into hit-rate@k aggregates.
  - `top_hit_ids` keeps concise traceable identifiers for top results:
    - `rank`, `doc_id`, page span string, `chunk_id`.
  - Summary stores both:
    - `primary_k_hit_rate` (focused diagnostic view),
    - `hit_rate_at_k` (full curve across all ks).

- Tests:
  - `tests/test_retrieve_eval_api.py`
  - `tests/test_eval_dataset.py`
  - `tests/test_eval_metrics.py`

  Detailed mechanism implemented:
  - `tests/test_retrieve_eval_api.py` validates adapter knob mapping and deterministic tie-break/rank assignment.
  - `tests/test_eval_dataset.py` validates dataset contract rules, duplicate qid rejection, unlabeled allowance mode, and deterministic qid sorting.
  - `tests/test_eval_metrics.py` validates strict metric math, duplicate-hit nDCG protection, multi-k output shape, and relaxed diagnostic matchers.

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
