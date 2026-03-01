# Evaluation Summary

- generated_at_utc: 2026-03-01T02:30:49.683700+00:00
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
- MRR@k: 0.2986
- nDCG@k: 0.3319

### Retrieval By K
- k1: recall=0.2500, mrr=0.2500, ndcg=0.2500
- k3: recall=0.3333, mrr=0.2778, ndcg=0.2917
- k5: recall=0.3750, mrr=0.2986, ndcg=0.3137
- k8: recall=0.4167, mrr=0.2986, ndcg=0.3319

### Secondary Diagnostics
- near_page_tolerance: 1
- k1: strict=0.2500, doc_only=0.7500, near_page=0.4167
- k3: strict=0.3333, doc_only=0.8333, near_page=0.5000
- k5: strict=0.4167, doc_only=0.8333, near_page=0.5833
- k8: strict=0.4167, doc_only=0.8333, near_page=0.5833

## Answer
- enabled: False
- model_dependent: True
- note: Answer metrics are model-dependent and less stable than retrieval metrics; use retrieval metrics as primary regression signals.
- answer_evaluated: 0
- citation_presence_rate: n/a
- inline_citation_sentence_rate: n/a
- refusal_accuracy: n/a
