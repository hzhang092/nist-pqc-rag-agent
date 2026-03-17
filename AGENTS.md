# AGENTS.md

## Mission
Evolve this repo from a strong local-first technical RAG system into a recruiter-legible AI engineering project for AI engineer / ML intern roles.

## Current truth
This repo already has:
- deterministic PDF ingestion and artifact versioning
- structure-aware chunking for technical NIST documents
- hybrid retrieval (FAISS + BM25 + RRF + conservative rerank)
- citation-enforced answer generation with refusal behavior
- bounded LangGraph agent flow
- FastAPI endpoints and Dockerized local serving
- graph-lite sidecar and narrow graph-assisted analyze-query support

## Do not misrepresent
- Do not describe Neo4j as a live runtime dependency unless code actually uses it in retrieval or answering.
- Do not claim cloud-native production deployment unless implemented and tested.
- Do not claim full KG-based RAG unless graph retrieval is actually wired into user-facing flows.

## High-priority future work
Prefer changes that improve hiring signal for AI engineer / ML intern roles:
1. measurable eval and regression infrastructure
2. stronger agent planning / query analysis
3. graph-assisted retrieval in one real user-facing flow
4. deployment and persistence improvements
5. observability, latency, failure analysis
6. safe, honest documentation and demoability

## Change rules
- Preserve deterministic artifacts and citation integrity.
- Keep changes bounded and testable.
- Update tests, eval artifacts, and docs with every meaningful behavior change.
- Prefer minimal, high-confidence edits over broad rewrites.

## Before finishing a task
- run relevant tests
- note tradeoffs
- state what was not implemented
- avoid overclaiming