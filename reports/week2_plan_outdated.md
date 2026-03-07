## New 1-week concrete plan (merged with your wishlist, aligned to the role)

### Scope guardrails (so it stays realistic)
- Stay within the project’s Week-1 architecture: **deterministic ingestion → structure-aware chunking → hybrid retrieval (BM25 + vector) + fusion (+ optional rerank) → citation-first generation → evaluation**, with LangGraph as a **bounded controller** (retrieve → assess → optional refine/retrieve → answer).   
- Avoid Week-1 non-goals like full knowledge graphs or perfect PDF parsing; prioritize measurable retrieval/citation reliability.   

---

## Deliverables you can demo in interviews
1) **Measured retrieval uplift** (ablation table showing strict/near-page recall improvement).  
2) **Agentic RAG with bounded behavior** (traceable retrieve→assess→refine loop, refusal reasons).   
3) **On-prem packaging**: local LLM + FastAPI endpoint + Docker runbook (matches job responsibilities and common tool stack).    

---

## Day-by-day plan

### Day 1 — Chunking upgrade (structure signals + recursive strategy)
**Goal:** improve localization for standards-style answers (algorithms/tables/sections) without breaking determinism/citations.

**Work**
- Add chunk metadata fields: `section_path` (best-effort), `block_type` ∈ {text, list, table, algorithm, code, math}.  
- Implement **recursive chunking**: split by detected headings/Algorithm/Table blocks first, then fall back to token-window splits (~250–400 tokens) with overlap. Keep `start_page/end_page` unchanged and stable.   
- Add “breadcrumb header” (Document > Section path) to chunk text for embeddings (do not change citation fields).

**Acceptance**
- Chunk IDs + page spans remain stable across runs; unit test asserts deterministic output ordering and metadata presence.

#### manual note: also add version tags to chunks if possible, to support version-aware retrieval in the future. 
---

### Day 2 — Keyword map query expansion (your #3) + “mode hints”
**Goal:** make exact identifiers win (e.g., `ML-KEM.KeyGen`, `Encaps`, `Decaps`) while keeping behavior deterministic.

**Work**
- Create a domain keyword map: `{concept → canonical tokens}` and integrate into deterministic query fusion.
- Extend query fusion with **operation expansions** and anchor expansions (Algorithm N / Table N / Section N). This matches your existing approach (regex token extraction + dot-name expansions + Algorithm variants) but makes it more explicit and controllable.   
- Add a `mode_hint` computed from the query (definition vs algorithm/table vs compare vs drafting) to bias retrieval variant generation.

**Acceptance**
- `tests/test_query_fusion.py`-style tests: variants are deterministic; keyword-map expansions appear when relevant; no explosion in variant count.

---

### Day 3 — Graph upgrade: bounded “query transformation” + evidence aggregation
This is your #2, but scoped to be reliable and testable.

**Work**
- Add a first LangGraph node `analyze_query` that outputs **structured fields**:
  - `canonical_query`, `mode_hint`, `required_anchors` (e.g., `ML-KEM.Decaps`, `Algorithm 2`), optional `filters` (doc_id/version tags if present).
- Keep it deterministic (temperature 0, strict schema). LangGraph’s role remains a bounded controller with explicit budgets and stop rules.    
- Implement “aggregation of retrieved files” as **evidence packets**:
  - group hits by `(doc_id, section_path)` and pick top groups; expand neighbors within same section to capture full algorithm/table blocks.

**Acceptance**
- Graph trace shows: `analyze_query → retrieve → assess_evidence → (optional refine/retrieve) → answer → verify/refuse` with bounded rounds.   
- Add/extend `tests/test_lc_graph.py` to lock down new behavior.

# manual note: can we add reranking model? if so, add it as an optional step after retrieval and before evidence aggregation. this would allow us to test whether reranking improves the quality of retrieved evidence and ultimately the final answer.

---

### Day 4 — Evidence refusal gate (your #5) with calibration, not a single raw threshold
**Goal:** refuse only when evidence is truly weak; avoid “arbitrary score threshold” pitfalls in hybrid+RRF.

**Work**
- In `assess_evidence`, compute a small set of signals:
  - minimum hits, anchor coverage (required tokens present), section diversity for compare queries, and a retrieval margin/entropy proxy.
- Make thresholds **intent-aware** (definition vs algorithm/table vs compare) and keep budgets explicit. The LangGraph report already points to “intent-aware thresholds” and better calibration as the next step.   
- Add a deterministic “citation repair” fallback: if generation returns zero citations, produce a short extractive answer from top evidence spans (still cited), else refuse.

**Acceptance**
- Fewer false refusals on answerable eval questions; refusal reason remains explicit and traceable.   

---

### Day 5 — Local LLM installation + on-prem serving path (your #4)
**Goal:** align directly with “deploy local/open-source LLMs” and production on-prem constraints.

**Work**
- Stand up one local serving option (pick one):
  - **vLLM** for GPU serving, or
  - **llama.cpp** for lightweight CPU/GGUF demos.
- Wire your RAG pipeline to the local endpoint; keep the rest of the system unchanged.
- Add simple perf counters: retrieval latency, rerank latency, generation latency.

**Acceptance**
- `rag.ask` works end-to-end with local LLM; reproducible run instructions.

(Interview relevance: directly maps to the job’s “local/open-source LLMs” and “Linux/Docker/perf tuning.” )

#### manual note: can we use Ollama for local serving? it supports GGUF and has a nice API. if so, add it as an optional path in addition to vLLM/llama.cpp.

---

### Day 6 — Evaluation improvements (your #6) + ablation report (the “proof”)
**Goal:** produce a recruiter-legible artifact: “here’s what I changed, here’s what improved.”

**Work**
- Run a small ablation grid (4–8 configs):
  1) baseline (current hybrid+fusion)
  2) + structure chunking
  3) + keyword map
  4) + evidence packet aggregation
  5) + (optional) rerank stage
- Report metrics: Recall@k/MRR/nDCG and strict/near-page overlap; plus citation compliance rate.
- Store outputs as `reports/eval/<date>/ablation.md`.

**Acceptance**
- One best config selected by strict/near-page recall while holding nDCG as a guardrail.

(Your project already emphasizes hybrid retrieval (FAISS + BM25), RRF fusion, deterministic query fusion, and determinism tests—this ablation report makes those improvements “real” and measurable.  )

---

### Day 7 — Internal tool surface + Docker packaging (job-aligned demo)
**Goal:** match “internal AI web tools” + “Docker deployment”.

**Work**
- Build a minimal FastAPI service:
  - `/ask` returns `{answer, citations, trace_summary}`
  - `/search` returns top hits + metadata for debugging
- Add a tiny Streamlit/Gradio UI (optional) for interactive demos.
- Dockerize: `docker compose up` to run `serve` + local LLM (if applicable).

**Acceptance**
- One-command demo that shows: prompt → retrieved evidence → cited answer or refusal.

(This aligns with the job’s “FastAPI / Streamlit / Gradio” and Docker requirements, and also matches common employer stack preferences like FAISS/BM25/Docker/HF.  )

---

## What from your wishlist is intentionally not emphasized
- A “large” LLM agent that freely rewrites queries is not prioritized; instead you’ll do **bounded, schema’d query analysis** + deterministic expansions so improvements show up in eval and remain reproducible.

If you want the highest interview ROI with limited time: **Day 1–3 (chunking + keyword map + bounded query analysis + evidence aggregation) + Day 6 (ablation report) + Day 5/7 (local LLM + FastAPI/Docker demo)**.