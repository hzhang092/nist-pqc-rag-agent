# Evaluation Summary

- generated_at_utc: 2026-03-03T03:31:26.866077+00:00
- run_id: questions_20260303T033126Z
- dataset: eval\day4\questions.jsonl
- total_questions: 24
- answerable_questions: 21
- unanswerable_questions: 3
- labeled_answerable_questions: 21
- unlabeled_answerable_questions: 0

## Retrieval
- scoring_scope: answerable_with_non_empty_gold_only
- primary_k: 8
- Recall@k: 0.4286
- MRR@k: 0.3175
- nDCG@k: 0.3295

### Retrieval By K
- k1: recall=0.1905, mrr=0.2381, ndcg=0.2381
- k3: recall=0.4286, mrr=0.3175, ndcg=0.3295
- k5: recall=0.4286, mrr=0.3175, ndcg=0.3295
- k8: recall=0.4286, mrr=0.3175, ndcg=0.3295

### Secondary Diagnostics
- near_page_tolerance: 1
- k1: strict=0.2381, doc_only=0.7619, near_page=0.2857
- k3: strict=0.4762, doc_only=0.9048, near_page=0.5238
- k5: strict=0.4762, doc_only=0.9048, near_page=0.5238
- k8: strict=0.4762, doc_only=0.9048, near_page=0.5238

### Metric Definitions
- Retrieval By K:
  Recall@k = average fraction of gold spans recovered per question; MRR@k = average reciprocal first strict-hit rank; nDCG@k = rank-aware gain over unique gold spans.
- Secondary Diagnostics:
  hit-rate style metrics (per-question success rate): at least one matching hit appears in top-k.
- strict:
  doc_id match + page overlap (same relevance rule as primary retrieval metrics).
- doc_only:
  doc_id must match; page overlap is ignored.
- near_page:
  doc_id match + page overlap with +-near_page_tolerance slack.
- Why numbers differ:
  Retrieval By K is span-coverage/rank quality; Secondary Diagnostics is question-level any-hit rate, so strict can be higher when questions have multiple gold spans.

### Questions Missing Gold In Top-k
- count: 11
- q002: why do we need the PQC standards | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p11-p11, NIST.FIPS.205:p11-p11 | top_hits=r1 NIST.IR.8547.ipd p9-p9; r2 NIST.IR.8547.ipd p12-p12; r3 NIST.IR.8545 p25-p25
- q003: What are the PQC standards | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p3-p3, NIST.FIPS.205:p3-p3 | top_hits=r1 NIST.IR.8547.ipd p24-p24; r2 NIST.IR.8547.ipd p12-p12; r3 NIST.IR.8547.ipd p23-p23
- q005: explain ML-DSA | gold=NIST.FIPS.204:p3-p4, NIST.FIPS.204:p11-p11 | top_hits=r1 NIST.FIPS.204 p19-p19; r2 NIST.FIPS.204 p8-p8; r3 NIST.FIPS.204 p28-p28
- q009: What are the difference between ML-KEM and ML-DSA? | gold=NIST.FIPS.203:p3-p3, NIST.FIPS.203:p21-p21, NIST.FIPS.204:p3-p4, NIST.FIPS.204:p19-p19 | top_hits=r1 NIST.SP.800-227 p43-p43; r2 NIST.FIPS.204 p8-p8; r3 NIST.FIPS.204 p8-p8
- q010: Define ML-KEM encapsulation key and where it’s used. | gold=NIST.FIPS.203:p4-p4, NIST.FIPS.203:p11-p11 | top_hits=r1 NIST.FIPS.203 p44-p44; r2 NIST.FIPS.203 p3-p3; r3 NIST.FIPS.203 p25-p25
- q012: How do ML-DSA and SLH-DSA differ in intended use-cases? | gold=NIST.FIPS.204:p5-p5, NIST.FIPS.205:p5-p5 | top_hits=r1 NIST.FIPS.204 p29-p29; r2 NIST.FIPS.205 p47-p47; r3 NIST.FIPS.205 p18-p18
- q014: What underlying mathematical hardness assumptions form the security basis for ML-KEM, ML-DSA, and SLH-DSA? | gold=NIST.FIPS.203:p10-p10, NIST.FIPS.204:p11-p11, NIST.FIPS.204:p11-p11 | top_hits=r1 NIST.FIPS.204 p7-p7; r2 NIST.IR.8545 p8-p8; r3 NIST.FIPS.204 p7-p7
- q018: Under NIST recommendations for key-encapsulation mechanisms, what specific components must be concatenated to create the MAC_Data string during key confirmation? | gold=NIST.SP.800-227:p29-p29 | top_hits=r1 NIST.SP.800-227 p56-p56; r2 NIST.SP.800-227 p1-p1; r3 NIST.SP.800-227 p7-p7
- q019: Which specific message authentication code (MAC) algorithms are approved for use during the key confirmation process of a KEM? | gold=NIST.SP.800-227:p11-p11 | top_hits=r1 NIST.SP.800-227 p56-p56; r2 NIST.SP.800-227 p59-p59; r3 NIST.SP.800-227 p28-p28
- q020: What are the required steps to construct a composite key-encapsulation mechanism (KEM) from two existing component KEMs? | gold=NIST.SP.800-227:p35-p36 | top_hits=r1 NIST.FIPS.203 p3-p3; r2 NIST.SP.800-227 p5-p5; r3 NIST.SP.800-227 p58-p58
- q023: What does MLWE mean? | gold=NIST.FIPS.203:p13-p13 | top_hits=r1 NIST.FIPS.203 p22-p22; r2 NIST.FIPS.204 p19-p19; r3 NIST.SP.800-227 p9-p9

## Answer
- enabled: False
- model_dependent: True
- note: Answer metrics are model-dependent and less stable than retrieval metrics; use retrieval metrics as primary regression signals.
- answer_evaluated: 0
- citation_presence_rate: n/a
- inline_citation_sentence_rate: n/a
- refusal_accuracy: n/a
