# Week 2 Day 2 Report - Retrieval Regression Attribution and Recovery

## 1. Overview
- Date: 2026-03-03
- Scope: Docling+v2 retrieval regression analysis and recovery
- Dataset: `eval/day4/questions.jsonl`
- Baseline reference: LlamaParse+v1 (`Recall@8=0.5476`, `nDCG@8=0.4286`, `MRR@8=0.3968`)

## 2. Problem Statement
After P0 ingestion/chunking and identifier normalization work, Docling+v2 improved, but still trailed baseline.

P0 control run (`questions_20260303T165754Z`) showed:
- `Recall@8=0.5000`
- `nDCG@8=0.3955`
- `MRR@8=0.3829`

The key diagnostic finding was that most misses were not upstream retrieval absence:
- miss causes: `rerank_suppression=7`, `upstream_missing=2`
- q019 and q023 had gold inside pre-rerank fused list and inside rerank pool, but fell below top-8 after rerank.

So the core bug was ranking behavior after fusion, not only candidate generation.

## 3. Discovery and Debug Process

### Step A - Confirm where gold is lost
We used stage diagnostics (pre-fused, pre-pool, post-rerank) to trace loss point per query.

For P0 control:
- q019: `pre_pool=5`, `post=12` -> clearly demoted by rerank.
- q023: `pre_pool=17`, `post=9` -> not enough promotion to enter top-8.
- q003/q012: `pre_fused=None` -> true upstream missing.

This separated three different failure classes:
1. Upstream missing (never enters fused depth)
2. Rerank demotion (gold rank worsens)
3. Insufficient promotion (gold stays outside top-8)

### Step B - Improve attribution granularity before changing ranking logic
In `eval/run.py`, we extended diagnostics to avoid coarse "rerank_suppression" labels.

Added per-question fields:
- `gold_rank_delta_pre_pool_to_post`
- `gold_demoted_by_rerank`
- `gold_not_promoted_enough`

Replaced miss subtypes with:
- `outside_pool`
- `rerank_demotion`
- `rerank_insufficient_promotion`
- `upstream_missing`
- `retained_or_recovered`

This made q023-style outcomes explicit: sometimes rerank improves rank but still not enough.

### Step C - Replace old reranker with deterministic do-no-harm policy
Old behavior:
- lexical rerank sorted all candidates by `(has_exact_token, bm25, tie-break)`
- no intent awareness
- no promotion gates
- no default preserve-fused-order rule

Failure mode:
- broad BM25-heavy chunks could outrank better fused candidates
- gold could be demoted even when already in good pre-pool position (q019-like)

New behavior in `rag/retrieve.py`:
- fused order is baseline and preserved by default
- only candidates passing strict promotion gates can move up
- non-promoted candidates keep original fused order

## 4. New Fixed Rerank Mechanism

### 4.1 Intent-aware modes
Mode set: `definition`, `algorithm`, `compare`, `general`.

If `mode_hint` is not provided, it is inferred deterministically from query text.

### 4.2 Features
For each candidate in rerank input:
- `prior_norm`: normalized fused prior score (fallback to rank prior if tied)
- `bm25_norm`: min-max normalized BM25 score
- `anchor_norm`: normalized anchor overlap count

Anchor extraction includes:
- technical identifier-like tokens (existing)
- acronym anchors for definition intent (`MLWE`, `LWE`, `SIS`, `MSIS`, `KDF`)

### 4.3 Promotion score
Promotion score (deterministic weighted sum):

`promotion_score = w_prior * prior_norm + w_bm25 * bm25_norm + w_anchor * anchor_norm`

Mode weights:
- `definition`: prior `0.35`, bm25 `0.45`, anchor `0.20`
- `algorithm`: prior `0.45`, bm25 `0.20`, anchor `0.35`
- `compare`: prior `0.60`, bm25 `0.30`, anchor `0.10`
- `general`: prior `0.70`, bm25 `0.20`, anchor `0.10`

### 4.4 Promotion gates (strict)
- `definition`: `anchor_overlap >= 1` OR `bm25_norm >= 0.55`
- `algorithm`: `anchor_overlap >= 2` OR (`anchor_overlap >= 1` AND `bm25_norm >= 0.60`)
- `compare`: `anchor_overlap >= 1` AND `bm25_norm >= 0.65`
- `general`: `anchor_overlap >= 2` OR `bm25_norm >= 0.70`

### 4.5 Deterministic order and budget
Promoted candidates are sorted by:
1. `promotion_score` (desc)
2. original fused index (asc)
3. `(doc_id, start_page, chunk_id)`

Promotion budget:
- `max_promoted = min(8, rerank_pool // 2)`

With this mechanism:
- no qualifying evidence -> rerank output equals fused order
- strong qualifying evidence -> controlled promotion

## 5. Query Variant Changes

We added mode-aware deterministic variants with strict cap and dedupe:
- `max_variants=4`
- definition templates: `definition of <term>`, `<term> stands for`, `<term> notation`
- compare templates: `<A> intended use-cases`, `<B> intended use-cases`, `<A> vs <B>`
- general standards template for PQC-standards queries includes `FIPS 203/204/205`, `ML-KEM`, `ML-DSA`, `SLH-DSA`

Variants remain non-LLM and deterministic.

## 6. Experiments and Results

### Run E (P0 control)
Artifacts:
- `reports/eval/questions_20260303T165754Z_summary.json`
- `reports/eval/questions_20260303T165754Z_per_question.jsonl`

Metrics:
- Recall@8: `0.5000`
- nDCG@8: `0.3955`
- MRR@8: `0.3829`

### Run F (new rerank, mode variants OFF)
Command:
- `python -m eval.run --dataset eval/day4/questions.jsonl --outdir reports/eval --mode hybrid --backend faiss --k 8 --ks 1,3,5,8 --rerank-pool 40 --no-mode-variants`

Artifacts:
- `reports/eval/questions_20260303T173852Z_summary.json`
- `reports/eval/questions_20260303T173852Z_per_question.jsonl`

Metrics:
- Recall@8: `0.6548`
- nDCG@8: `0.4334`
- MRR@8: `0.3755`

Gate status:
- Recovery Recall@8 gate (`>=0.5476`): pass
- Recovery nDCG@8 gate (`>=0.4286`): pass
- Do-no-harm gate: pass (`0` misses with `pre_pool_rank <= 8`)

Target queries:
- q018: `pre_pool=1`, `post=1`, hit top-8
- q019: `pre_pool=5`, `post=5`, hit top-8
- q023: `pre_pool=17`, `post=8`, hit top-8

### Run G (new rerank, mode variants ON)
Command:
- `python -m eval.run --dataset eval/day4/questions.jsonl --outdir reports/eval --mode hybrid --backend faiss --k 8 --ks 1,3,5,8 --rerank-pool 40`

Artifacts:
- `reports/eval/questions_20260303T173944Z_summary.json`
- `reports/eval/questions_20260303T173944Z_per_question.jsonl`

Metrics:
- Recall@8: `0.6429`
- nDCG@8: `0.4113`
- MRR@8: `0.3473`

Gate status:
- Recovery Recall@8 gate: pass
- Recovery nDCG@8 gate: fail
- Do-no-harm gate: fail (new demotion case)

Observed regression:
- q009: `pre_pool=7` -> `post=12` (`rerank_demotion`)

## 7. Why one path failed
The mode-aware variant patch (Run G) increased query expansion breadth and changed fused candidate composition.

Symptoms:
- `outside_pool` misses increased (`q002`, `q010` became pre-fused but outside pool)
- one previously safe pre-pool hit (`q009`) was demoted below top-8

Interpretation:
- for this dataset/config, variant expansion was too aggressive relative to fixed pool depth (`40`), introducing candidate competition and pushing relevant chunks out of stable positions.
- rerank improvements alone were positive; variant expansion was not net-positive yet.

## 8. Code and Contract Updates

### Retrieval API and behavior
Updated `rag/retrieve.py`:
- `mode_hint` support through `retrieve`, `retrieve_with_stages`, `retrieve_for_eval_with_stages`, `retrieve_for_eval`
- deterministic mode inference when `mode_hint=None`
- do-no-harm rerank and mode-aware scoring
- capped mode-aware query variants
- CLI flags: `--mode-hint`, `--no-mode-variants`

### Eval contract
Updated `eval/run.py`:
- new per-question diagnostic fields (delta/demotion/insufficient promotion)
- refined miss subtypes
- markdown summary includes rank delta and subtype in miss rows
- eval flag `--no-mode-variants`

## 9. Validation

Executed tests:
- `pytest tests/test_lc_tools.py tests/test_lc_graph.py tests/test_retrieve_eval_api.py tests/test_query_fusion.py tests/test_retrieve_rrf.py tests/test_eval_run.py tests/test_retrieve_determinism.py`
- Result: `37 passed, 1 skipped`

Added/updated test coverage for:
- do-no-harm behavior
- deterministic promotion behavior
- acronym anchor coverage (`MLWE`)
- mode-aware variant cap/dedupe
- eval miss subtype classification
- eval adapter pass-through for `mode_hint`

## 10. Final Decision for Day 2
Default recommendation after Day 2:
- Keep intent-aware do-no-harm rerank enabled.
- Keep mode-aware variant templates disabled by default for now (`--no-mode-variants` behavior).

Reason:
- This configuration is the best measured recovery and passes parity gates without introducing new harmful demotions.

## 11. Remaining Issues and Next Work
Unresolved upstream-missing queries remain:
- q003 (`What are the PQC standards`)
- q012 (`How do ML-DSA and SLH-DSA differ in intended use-cases?`)

Next iteration should target upstream retrieval for these two classes specifically, with tighter, lower-risk variant policies and pool-aware safeguards.
