# Week 2 Day 2 Report - Full Timeline (From Initial Diagnosis to Recovery)

## 1. Scope and Context
- Date: `2026-03-03`
- Dataset: `eval/day4/questions.jsonl` (21 labeled answerable questions)
- Goal: explain why `docling + v2` was still below `llamaparse + v1`, implement fixes, and measure recovery with strict eval deltas.

Baseline reference (`llamaparse + v1`):
- Recall@8: `0.5476`
- nDCG@8: `0.4286`
- MRR@8: `0.3968`

## 2. Initial Problem Framing (Start of Conversation)

At the beginning, the key observation was:
- Docling+v2 improved over Docling+v1, but still trailed the baseline.
- Retrieval often found the right document family but missed exact gold pages in top-8.

Critical examples:
- Baseline recovered gold in top-8 for `q018/q019/q023`.
- Docling+v2 missed `q019/q023` (and had an anomaly suspicion around q018 report consistency).

Initial hypotheses raised in feedback:
1. Parser/text fidelity issue (token shape drift hurts lexical retrieval).
2. v2 chunking benefit not always active on pages with weak markdown.
3. Empty-page parser outputs can cap recall.
4. Breadcrumb/section noise can dilute embedding signal.
5. Scorer/report mismatch risk (q018 anomaly) needed verification.

## 3. P0 Plan and Why It Was Necessary

Before tuning reranking, we needed hard attribution for where gold gets lost:
- pre-rerank fused list?
- inside/outside rerank pool?
- lost post-rerank?

Without this, rerank bugs and candidate-generation bugs are easy to conflate.

P0 therefore focused on **instrumentation + contract correctness + token fidelity**.

## 4. P0 Implementation (What Was Added)

### 4.1 Stage diagnostics in retrieval/eval
Updated retrieval/eval plumbing so each query exposes:
- `pre_rerank_fused_hits`
- `post_rerank_hits`
- `final_hits`
- `rerank_pool`

Eval artifact fields added:
- `gold_first_rank_pre_rerank_fused`
- `gold_first_rank_pre_rerank_pool`
- `gold_first_rank_post_rerank`
- `gold_in_pre_rerank_fused`
- `gold_in_rerank_pool`
- `gold_dropped_outside_rerank_pool`
- `gold_suppressed_by_rerank`
- `rerank_miss_cause` (initial coarse form)

### 4.2 Eval ordering contract fix
Ensured eval rank reflects retrieval list order exactly (no post-hoc score reorder in eval path).

### 4.3 Identifier-scoped normalization (index + query symmetry)
Added strict identifier-like normalization and applied across doc/query paths:
- unescape only `\\_`, `\\.`, `\\-` in identifier spans,
- normalize separator spacing inside identifiers,
- avoid touching non-identifier backslash contexts.

Applied to:
- parser text path,
- chunk text path,
- BM25 index/query tokenization,
- retrieval query/anchor path.

### 4.4 Fusion determinism sanity
Added synthetic stability tests for RRF merge and tie behavior.

### 4.5 Detailed P0 diagnostic mechanism (end-to-end)
P0 added a stage-aware evaluation path so every query could be traced through ranking transitions.

Runtime path per question:
1. Build fused ranking at diagnostic depth (`max(60, rerank_pool, k)`).
2. Slice rerank pool (`rerank_pool=40`).
3. Produce post-rerank ordering.
4. Truncate final top-k (`k=8`) for scoring.
5. Compute gold rank at each stage from the exact lists used by scorer.

Formally, for each query `q` with gold set `G`:
- `R_fused`: pre-rerank fused ranking list
- `R_pool = R_fused[:rerank_pool]`
- `R_post`: reranker output over `R_pool`
- `R_final = R_post[:k]`

Recorded diagnostics:
- `gold_first_rank_pre_rerank_fused = min rank of hit in R_fused matching any g in G`
- `gold_first_rank_pre_rerank_pool = min rank of hit in R_pool matching any g in G`
- `gold_first_rank_post_rerank = min rank of hit in R_post matching any g in G`
- boolean presence fields derived from rank-null checks.

This design removed attribution ambiguity:
- `gold missing in R_fused`: upstream retrieval/candidate problem.
- `gold in R_fused but missing in R_pool`: pool depth/candidate pressure problem.
- `gold in R_pool but missing in final top-k`: rerank ordering problem.

### 4.6 Detailed P0 miss-cause logic (initial version)
Initial P0 miss classification used the following rule chain:
1. if `gold_in_pre_rerank_fused == false` -> `upstream_missing`
2. else if `gold_in_rerank_pool == false` -> `outside_pool`
3. else if `gold_first_rank_post_rerank is null or > primary_k` -> `rerank_suppression`
4. else -> `retained_or_recovered`

This was sufficient to prove rerank was dominant, but too coarse to separate:
- demotion vs
- insufficient promotion.

That limitation was fixed in Day-2 with subtype diagnostics.

### 4.7 Detailed P0 identifier normalization mechanism
P0 normalization was intentionally scoped to identifier-like spans only, to prevent collateral text damage.

Mechanism:
1. Detect identifier-like spans containing alnum plus separators (`._-`) and escaped separators.
2. Inside those spans only:
   - unescape `\\_`, `\\.`, `\\-`
   - normalize separator spacing (`MAC _ Data -> MAC_Data`)
3. Leave all non-identifier backslash contexts unchanged (LaTeX/macros/commands).

Why this matters:
- BM25 and lexical anchors depend on exact token shape.
- If index text is normalized but query text is not (or vice versa), lexical matching drifts.
- P0 enforced query/index symmetry by applying normalization in both indexing and query-time paths.

## 5. P0 Results (Control Run E)

Run E artifact:
- `reports/eval/questions_20260303T165754Z_summary.json`

Metrics:
- Recall@8: `0.5000`
- nDCG@8: `0.3955`
- MRR@8: `0.3829`

Interpretation:
- Better than older Docling+v2 pre-P0, but still below baseline parity.

Attribution from P0 diagnostics:
- miss causes: `rerank_suppression=7`, `upstream_missing=2`
- key queries:
  - `q019`: `pre_pool=5`, `post=12` (demoted below top-8)
  - `q023`: `pre_pool=17`, `post=9` (improved, but not enough to top-8)
  - `q003/q012`: absent even pre-fused (upstream missing)

This was the decisive finding: main gap was post-fusion ranking behavior, not only parser/chunking.

## 6. Day 2 Recovery Work (P1, Feedback-Adjusted)

### 6.1 First refinement: attribution granularity upgrade
Coarse `rerank_suppression` was insufficient for diagnosis.

`eval/run.py` upgraded to record:
- `gold_rank_delta_pre_pool_to_post`
- `gold_demoted_by_rerank`
- `gold_not_promoted_enough`

Miss subtype labels became:
- `outside_pool`
- `rerank_demotion`
- `rerank_insufficient_promotion`
- `upstream_missing`
- `retained_or_recovered`

### 6.2 Core fix: new deterministic do-no-harm reranker
Old rerank failure:
- sorted whole pool by lexical signals,
- no intent-awareness,
- no strict promotion gates,
- could demote already-good fused candidates.

New rerank design:
1. Preserve fused order as default.
2. Promote only candidates passing strict mode gates.
3. Keep non-promoted candidates in original fused order.
4. Apply deterministic promotion sort and bounded promotion budget.

### 6.3 Intent-aware rerank modes
Modes:
- `definition`
- `algorithm`
- `compare`
- `general`

If `mode_hint` is absent, infer deterministically from query.

Per-mode scoring weights `(prior, bm25, anchors)`:
- `definition`: `(0.35, 0.45, 0.20)`
- `algorithm`: `(0.45, 0.20, 0.35)`
- `compare`: `(0.60, 0.30, 0.10)`
- `general`: `(0.70, 0.20, 0.10)`

### 6.4 New rerank scoring mechanism
For each candidate in rerank pool:
- `prior_norm`: normalized fused prior score (fallback rank prior when tied)
- `bm25_norm`: min-max normalized BM25
- `anchor_norm`: normalized anchor overlap

Composite score:
- `promotion_score = w_prior*prior_norm + w_bm25*bm25_norm + w_anchor*anchor_norm`

Mode-specific promotion gates:
- `definition`: `anchor_overlap>=1` OR `bm25_norm>=0.55`
- `algorithm`: `anchor_overlap>=2` OR (`anchor_overlap>=1` AND `bm25_norm>=0.60`)
- `compare`: `anchor_overlap>=1` AND `bm25_norm>=0.65`
- `general`: `anchor_overlap>=2` OR `bm25_norm>=0.70`

Promotion ordering:
1. `promotion_score` desc
2. original fused index asc
3. `(doc_id, start_page, chunk_id)` asc

Promotion budget:
- `max_promoted = min(8, rerank_pool // 2)`

### 6.5 Anchor extraction improvement for definitions
Added acronym anchors to reduce separator-free misses:
- `MLWE`, `LWE`, `SIS`, `MSIS`, `KDF`

### 6.6 Mode-aware query variants (capped and deterministic)
Added mode-aware template variants with:
- `max_variants=4`
- stable dedupe/order

Templates:
- definition: `definition of <term>`, `<term> stands for`, `<term> notation`
- compare: `<A> intended use-cases`, `<B> intended use-cases`, `<A> vs <B>`
- general standards: inject `FIPS 203/204/205`, `ML-KEM`, `ML-DSA`, `SLH-DSA` on PQC-standards queries.

### 6.7 Detailed rerank algorithm mechanism (new implementation)
The new reranker is a constrained promotion system, not a full reorder system.

Input:
- query `q`
- fused pool `R_pool` (ordered list)
- mode hint `m` (provided or inferred)

Step 1 - Feature extraction per candidate `i`:
- `prior_i`: fused prior score from RRF
- `bm25_i`: BM25 score of `(q, chunk_i.text)`
- `anchor_i`: count of query anchor tokens overlapping chunk text

Step 2 - Normalization:
- `prior_norm_i = minmax(prior_i)`; if degenerate, fallback to rank-prior curve.
- `bm25_norm_i = minmax(bm25_i)`
- `anchor_norm_i = anchor_i / max(anchor)` (or 0 when no anchor overlap in pool)

Step 3 - Mode-aware weighted score:
- `promotion_score_i = w_prior(m)*prior_norm_i + w_bm25(m)*bm25_norm_i + w_anchor(m)*anchor_norm_i`

Step 4 - Strict gate check:
- candidate is eligible for promotion only if it passes mode gate thresholds.

Step 5 - Deterministic promoted ordering:
- sort promoted candidates by:
  1. `promotion_score` descending
  2. original fused index ascending
  3. `(doc_id, start_page, chunk_id)` ascending

Step 6 - Rebuild final order:
- `max_promoted = min(8, rerank_pool // 2)`
- output is:
  - top promoted subset (bounded), then
  - all remaining candidates in original fused order.

Effect:
- if no candidate clears gates, output == fused order (strict do-no-harm fallback).
- only strongly supported candidates are allowed to move upward.

### 6.8 Why the new rerank fixes observed failures
For q019-like failures:
- old rerank demoted gold despite good pre-pool rank.
- new mechanism keeps fused order unless strong promotion evidence exists.
- result: stable retention (`pre_pool=5 -> post=5` in Run F).

For q023-like failures:
- gold needed meaningful upward movement from outside top-8.
- new mechanism allows controlled promotion when lexical/anchor signals are strong enough.
- result: recovery (`pre_pool=17 -> post=8` in Run F).

For broad/noisy queries:
- gate + promotion budget prevent complete lexical override of fused ranking.
- this reduces catastrophic reshuffling risk compared with old full-pool lexical resorting.

## 7. Experiments (Day 2)

### Run F - intent-aware rerank, mode variants OFF
Command:
- `python -m eval.run --dataset eval/day4/questions.jsonl --outdir reports/eval --mode hybrid --backend faiss --k 8 --ks 1,3,5,8 --rerank-pool 40 --no-mode-variants`

Artifact:
- `reports/eval/questions_20260303T173852Z_summary.json`

Metrics:
- Recall@8: `0.6548`
- nDCG@8: `0.4334`
- MRR@8: `0.3755`

Gate status:
- Recovery Recall gate (`>=0.5476`): pass
- Recovery nDCG gate (`>=0.4286`): pass
- Do-no-harm gate (no new misses with `pre_pool<=8`): pass

Target queries:
- `q018`: `pre_pool=1 -> post=1` (hit)
- `q019`: `pre_pool=5 -> post=5` (hit)
- `q023`: `pre_pool=17 -> post=8` (hit)

### Run G - intent-aware rerank, mode variants ON
Command:
- `python -m eval.run --dataset eval/day4/questions.jsonl --outdir reports/eval --mode hybrid --backend faiss --k 8 --ks 1,3,5,8 --rerank-pool 40`

Artifact:
- `reports/eval/questions_20260303T173944Z_summary.json`

Metrics:
- Recall@8: `0.6429`
- nDCG@8: `0.4113`
- MRR@8: `0.3473`

Gate status:
- Recovery Recall gate: pass
- Recovery nDCG gate: fail
- Do-no-harm gate: fail

Regression example:
- `q009`: `pre_pool=7 -> post=12` (`rerank_demotion`)

## 8. Why Run G Failed (Important)

The mode-aware variant expansion increased candidate competition under fixed pool size (`40`):
- more candidates pushed some relevant items out of stable positions,
- introduced `outside_pool` misses and new demotions (`q009`),
- did not solve persistent upstream-missing queries (`q003`, `q012`).

Conclusion:
- rerank fix was beneficial;
- variant expansion policy was too aggressive for this pool budget/config.

## 9. Final Decision After Day 2

Recommended default:
- keep intent-aware do-no-harm rerank enabled,
- keep mode-aware variant templates disabled by default (`--no-mode-variants` behavior).

Reason:
- best measured recovery,
- parity gates passed,
- no harmful demotions under do-no-harm gate.

## 10. Files and Contracts Updated

Core implementation:
- `rag/retrieve.py`
- `eval/run.py`

Tests updated:
- `tests/test_retrieve_rrf.py`
- `tests/test_query_fusion.py`
- `tests/test_retrieve_eval_api.py`
- `tests/test_eval_run.py`
- `tests/test_retrieve_determinism.py`

Validation result:
- `37 passed, 1 skipped` on targeted test suite.

## 11. Remaining Open Issues
- `q003`: upstream missing (`What are the PQC standards`)
- `q012`: upstream missing (`How do ML-DSA and SLH-DSA differ in intended use-cases?`)

Next work should target upstream recall for these two classes with tighter, lower-risk candidate expansion rules and pool-aware safeguards.

## 12. Key Artifacts
- P0 control: `reports/eval/questions_20260303T165754Z_summary.json`
- Day2 run F (best): `reports/eval/questions_20260303T173852Z_summary.json`
- Day2 run G (regressed): `reports/eval/questions_20260303T173944Z_summary.json`
- Consolidated ablation: `reports/ablation_ingest_chunking_202603xx.md`
