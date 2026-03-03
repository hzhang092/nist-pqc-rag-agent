# Ingestion + Chunking Ablation Report

## Run Metadata
- date_utc: `2026-03-03T17:39:44Z`
- dataset: `eval/day4/questions.jsonl`
- baseline_summary: `reports/eval/baseline_summary.md`
- baseline_recall_at_8: `0.5476`
- baseline_mrr_at_8: `0.3968`
- baseline_ndcg_at_8: `0.4286`

## Gates
- recovery_required_recall_at_8: `>= 0.5476` (baseline parity)
- recovery_required_ndcg_at_8: `>= 0.4286` (baseline parity)
- borderline_tolerance: `0.005` (if within 0.005 below gate, rerun required)
- stretch_required_recall_at_8: `>= 0.5776`
- stretch_required_ndcg_at_8: `>= 0.4386`
- do_no_harm_gate: no misses where `gold_first_rank_pre_rerank_pool <= 8`

## Config Matrix
- A: `llamaparse + v1` : baseline
- B: `docling + v1` : `reports/eval/docling+v1_summary.md`
- C: `llamaparse + v2` : pending
- D: `docling + v2` (pre-P0): `reports/eval/docling+v2_summary.md`
- E: `docling + v2` (P0 diagnostics + identifier-normalization + eval-order fix): `reports/eval/questions_20260303T165754Z_summary.md`
- F: `docling + v2` + intent-aware do-no-harm rerank + `--no-mode-variants`: `reports/eval/questions_20260303T173852Z_summary.md`
- G: `docling + v2` + intent-aware do-no-harm rerank + mode-aware variants: `reports/eval/questions_20260303T173944Z_summary.md`

## Results
| config | recall@8 | mrr@8 | ndcg@8 | recovery_gate | stretch_gate | do_no_harm | notes |
|---|---:|---:|---:|---|---|---|---|
| A | 0.5476 | 0.3968 | 0.4286 | pass | fail | n/a | baseline reference |
| B | 0.4286 | 0.2278 | 0.2708 | fail | fail | n/a | docling parser drop vs baseline |
| C |  |  |  |  |  |  | pending run |
| D | 0.4286 | 0.3175 | 0.3295 | fail | fail | fail | docling+v2 improves over B, still below baseline |
| E | 0.5000 | 0.3829 | 0.3955 | fail | fail | fail | P0 improvement, still below parity |
| F | 0.6548 | 0.3755 | 0.4334 | pass | ndcg fail | pass | best recovery run; q019/q023 recovered |
| G | 0.6429 | 0.3473 | 0.4113 | ndcg fail | fail | fail | variants introduced new demotion (q009) |

## Stage Diagnostics

### E (P0 Baseline Control)
- summary: `reports/eval/questions_20260303T165754Z_summary.md`
- per-question: `reports/eval/questions_20260303T165754Z_per_question.jsonl`
- miss_cause_counts: `{"rerank_suppression": 7, "upstream_missing": 2}`

### F (Intent-Aware Rerank, No Mode Variants)
- summary: `reports/eval/questions_20260303T173852Z_summary.md`
- per-question: `reports/eval/questions_20260303T173852Z_per_question.jsonl`
- miss_cause_counts: `{"rerank_demotion": 1, "rerank_insufficient_promotion": 3, "upstream_missing": 2}`

### G (Intent-Aware Rerank + Mode Variants)
- summary: `reports/eval/questions_20260303T173944Z_summary.md`
- per-question: `reports/eval/questions_20260303T173944Z_per_question.jsonl`
- miss_cause_counts: `{"outside_pool": 2, "rerank_demotion": 2, "rerank_insufficient_promotion": 1, "upstream_missing": 2}`

## Targeted Attribution (q018/q019/q023)

### F Run (No Mode Variants)
| qid | pre_pool_rank | post_rank | delta_pre_pool_to_post | final_top8_hit | cause |
|---|---:|---:|---:|---|---|
| q018 | 1 | 1 | 0 | yes | retained_or_recovered |
| q019 | 5 | 5 | 0 | yes | retained_or_recovered |
| q023 | 17 | 8 | -9 | yes | retained_or_recovered |

### G Run (Mode Variants Enabled)
| qid | pre_pool_rank | post_rank | delta_pre_pool_to_post | final_top8_hit | cause |
|---|---:|---:|---:|---|---|
| q018 | 1 | 1 | 0 | yes | retained_or_recovered |
| q019 | 5 | 5 | 0 | yes | retained_or_recovered |
| q023 | 6 | 7 | 1 | yes | retained_or_recovered |

Interpretation:
- The intent-aware do-no-harm rerank recovered both key misses (`q019`, `q023`) and cleared recovery parity in run F.
- Mode-aware variants (run G) did not improve upstream-missing questions (`q003`, `q012`) and introduced a new top-8 demotion miss (`q009`, pre_pool=7 -> post=12).
- Current best default is intent-aware rerank with mode-aware variants disabled.

## Recommendation
- promote_defaults: `yes`, with `--no-mode-variants` (or equivalent default setting) for now.
- rationale:
  - Run F passes recovery gates (Recall@8 and nDCG@8).
  - Run F satisfies do-no-harm gate (`0` misses with `pre_pool<=8`).
  - Run G regresses nDCG@8 and violates do-no-harm.
  - Upstream missing (`q003`, `q012`) remains unsolved in both F and G and should be targeted separately.

## Next Steps
1. Keep intent-aware do-no-harm rerank as default.
2. Keep mode-aware variant templates behind a non-default flag until upstream-missing coverage improves without introducing demotions.
3. Add a targeted upstream-missing patch for `q003/q012` with stricter topic-anchored templates and run a dedicated ablation.

## Appendix
- command_log:
  - `D:\Softwares\anaconda\envs\eleven\python.exe -m eval.run --dataset eval/day4/questions.jsonl --outdir reports/eval --mode hybrid --backend faiss --k 8 --ks 1,3,5,8 --rerank-pool 40 --no-mode-variants`
  - `D:\Softwares\anaconda\envs\eleven\python.exe -m eval.run --dataset eval/day4/questions.jsonl --outdir reports/eval --mode hybrid --backend faiss --k 8 --ks 1,3,5,8 --rerank-pool 40`
- manifest_path: `data/processed/manifest.json`
- summary_artifacts:
  - `reports/eval/baseline_summary.md`
  - `reports/eval/docling+v1_summary.md`
  - `reports/eval/docling+v2_summary.md`
  - `reports/eval/questions_20260303T165754Z_summary.md`
  - `reports/eval/questions_20260303T165754Z_per_question.jsonl`
  - `reports/eval/questions_20260303T173852Z_summary.md`
  - `reports/eval/questions_20260303T173852Z_per_question.jsonl`
  - `reports/eval/questions_20260303T173944Z_summary.md`
  - `reports/eval/questions_20260303T173944Z_per_question.jsonl`
