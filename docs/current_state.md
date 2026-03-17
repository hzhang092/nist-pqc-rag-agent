# Current State

As of 2026-03-16, this repo is a local-first technical RAG system over the NIST PQC corpus with:

- deterministic PDF ingestion, cleaning, chunking, and artifact versioning
- hybrid retrieval (FAISS + BM25 + RRF + conservative rerank)
- citation-enforced direct QA through `/ask`
- bounded LangGraph orchestration through `/ask-agent`
- FastAPI and Dockerized local serving

## Graph Runtime Status

The graph-lite subsystem now affects one real user-facing runtime flow:

- `/ask-agent` uses graph-assisted `analyze_query` for both definition and algorithm-style questions
- graph lookup can narrow `doc_ids`, add `required_anchors`, and pass candidate sections into planner retrieval as soft rerank priors
- trace output exposes both a compact graph lookup summary and the full structured graph lookup payload for debugging

This remains intentionally scoped:

- direct `/ask` is unchanged and remains the control path
- Neo4j is still a dev/debug/export surface, not a live serving dependency
- retrieval is still hybrid-first; graph section signals only bias reranking inside the bounded agent path
