---
name: rag-proj
description: >
  Use for planning, implementing, debugging, and evaluating the nist-pqc-rag-agent repo:
  ingestion/chunking over NIST PQC PDFs, hybrid retrieval (BM25+vector) with fusion,
  LangChain/LangGraph agent control, and an evaluation harness with measurable deltas.
---

## Purpose
This skill turns “build a RAG agent” into a week-long, recruiter-legible mini production system:
deterministic pipeline, citation-first answers, bounded agent loops, and eval-driven iteration.

## Always do first (no exceptions)
1) Read `project_overview.md` to anchor scope, milestones, and contracts.
2) Inspect current repo structure (don’t guess filenames).
3) Identify the *smallest* change that advances the current milestone.

## Non-negotiables (PQC PDFs are prickly)
- Preserve page numbers and page spans for citations.
- Chunk sizes should stay focused (~250–400 tokens, overlap OK). Avoid giant “muddy vectors”.
- Treat tables/algorithms/math blocks as first-class citizens: don’t destroy structure if avoidable.

## System design rules (employer-appealing)
- Retrieval must be swappable: app code depends on `Retriever` interface, not FAISS details.
- Default retrieval: hybrid BM25 + vector + Reciprocal Rank Fusion (RRF).
- Agent must be bounded: step budget + tool-call budget + stop rules.
- run scripts in codna environment "eleven"

## What to output for any request
When the user asks you to do something:
- A short “success criteria” statement for this change
- The exact files to create/modify
- Copy/paste-ready code blocks per file
- The exact commands to run to verify
- If the change claims improvement: include eval deltas (before/after)

## Common workflows

### A) Planning (Day-level)
- Map work to Week-1 milestones in `project_overview.md`
- Produce a checklist with acceptance criteria and the one metric that should move

### B) Implementing
- Make minimal diffs
- Add/adjust the smallest test or smoke check
- Keep JSONL schema stable

### C) Debugging
- Reproduce from logs
- Localize root cause
- Patch + add regression check

### D) Evaluation
- Run harness
- Report retrieval metrics + citation compliance
- Only then propose next improvement

## Handy commands (adapt to what exists in repo)
- Tree: `python .agents/skills/rag_proj/scripts/smoke.py --tree`
- Smoke checks: `python .agents/skills/rag_proj/scripts/smoke.py`
- Optional run mode: `python .agents/skills/rag_proj/scripts/smoke.py --run`
