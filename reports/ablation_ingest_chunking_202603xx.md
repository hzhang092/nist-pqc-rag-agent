# Ingestion + Chunking Ablation Report

## Run Metadata
- date_utc:
- dataset: `eval/day4/questions.jsonl`
- baseline_summary: `reports/eval/questions_20260301T202803Z_summary.md`
- baseline_recall_at_8: `0.5476`
- baseline_ndcg_at_8: `0.4286`

## Gate
- required_recall_at_8: `>= 0.5776`
- required_ndcg_at_8: `>= 0.4186`

## Config Matrix
- A: `llamaparse + v1` : baseline
- B: `docling + v1` : questions_20260302T213225Z_summary.md
- C: `llamaparse + v2`
- D: `docling + v2`

## Results
| config | recall@8 | ndcg@8 | pass_gate | notes | 
|---|---:|---:|---|---|
| A |  0.5476 | 0.3968 | 0.4286 |  |
| B | 0.4286 |0.2278  | 0.2708 |  |
| C |  |  |  |  |
| D |  |  |  |  |

## Recommendation
- promote_defaults:
- rationale:

## Appendix
- command_log:
- manifest_path:
- summary_artifacts:

what is pass_gate: whether the config meets the required recall and ndcg thresholds to be considered an improvement over the baseline.