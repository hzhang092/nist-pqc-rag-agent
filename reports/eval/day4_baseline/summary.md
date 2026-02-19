# Evaluation Summary

- generated_at_utc: 2026-02-19T23:16:39.494820+00:00
- dataset: eval\day4\questions.jsonl
- total_questions: 13
- answerable_questions: 12
- unanswerable_questions: 1
- labeled_answerable_questions: 12
- unlabeled_answerable_questions: 0

## Retrieval
- scoring_scope: answerable_with_non_empty_gold_only
- primary_k: 8
- Recall@k: 0.4167
- MRR@k: 0.2569
- nDCG@k: 0.3011

### Retrieval By K
- k1: recall=0.1667, mrr=0.1667, ndcg=0.1667
- k3: recall=0.3333, mrr=0.2361, ndcg=0.2609
- k5: recall=0.3750, mrr=0.2569, ndcg=0.2829
- k8: recall=0.4167, mrr=0.2569, ndcg=0.3011

## Answer
- enabled: False
- model_dependent: True
- note: Answer metrics are model-dependent and less stable than retrieval metrics; use retrieval metrics as primary regression signals.
- answer_evaluated: 0
- citation_presence_rate: n/a
- inline_citation_sentence_rate: n/a
- refusal_accuracy: n/a
