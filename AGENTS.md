# AGENTS.md — nist-pqc-rag-agent (project rules for coding agents)

Codex should read `reports/project_overview.md` first and treat it as the source of truth for scope, milestones, and data contracts.

## Week-1 scope guardrails
- Do NOT widen document scope (keep optional PDFs optional) unless explicitly asked.
- Prioritize: deterministic ingestion → structure-aware chunking → hybrid retrieval + fusion → bounded LangGraph agent → eval harness.
- Any “improvement” must be backed by eval deltas (not vibes).

## Data contracts (do not break silently)
- Preserve page-level citations: every chunk must include `doc_id`, `start_page`, `end_page`.
- JSONL artifacts must be stable/deterministic (stable ordering, stable chunk IDs).

## Engineering preferences
- Keep retrieval backends swappable via a tiny interface (Retriever Protocol + ChunkHit schema).
- Prefer minimal, testable diffs. Add at least one sanity check when changing pipeline behavior.
- Avoid introducing heavy new dependencies unless essential.

## How to validate changes (choose what exists in-repo)
- If present, run: `python -m rag.search_faiss "ML-KEM key generation"` (or `python rag/search_faiss.py ...`)
- If present, run eval: `python -m eval.run` (or `python eval/run.py`)
- When unsure, inspect repo tree and existing entrypoints before assuming commands.

## Output expectations
- When answering PQC questions: no factual claims without citations.
- When coding: provide file-by-file patches and explain how to run/verify.
