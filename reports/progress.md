# Day 1 — Data + indexing

Pick 5–15 PDFs (standards/specs)

Write ingestion + chunking + metadata

Build vector index + basic “search top-k”

1) download PQC files
2) ingest.py: LLamaparse, Implement PDF extraction with page-level control. problem: math symbols as unicode not latex
3) clean.py: Clean + normalize text !! better cleaning strategy for technical specifications (tables, algorithms, and math).
4) chunks
5) embed

# Day 2 — Basic RAG
Here’s a **Day 2 plan** that matches the repo’s Milestone 2 (“RAG answers with citations”) and the project rules (“no claim without evidence”, deterministic CLI) from `project_overview.md`. 

---

## Day 2 deliverable (what “done” means)
You can run:

```bash
python -m rag.ask "What does FIPS 203 specify for ML-KEM key generation?"
```
…and it prints:

- a concise answer
- **citations** with `{doc_id, start_page, end_page, chunk_id}`
- a refusal (`"not found in provided docs"`) when evidence is insufficient
- stable ordering across reruns

---

## 1) Preflight: verify Day 1 artifacts + metadata contracts
Before writing new logic, confirm the Day 1 pipeline produced what Day 2 needs:

- `chunks.jsonl` / `chunk_store.jsonl` contains:
  - `chunk_id`, `doc_id`, `start_page`, `end_page`, `text`
- FAISS artifacts exist and load:
  - `embeddings.npy`, `faiss.index`, `emb_meta.json` (or equivalent)
- A debug search still works:
  - `python -m rag.search "ML-KEM key generation"` returns plausible hits with correct page spans

If anything is missing, fix that first—Day 2 depends on reliable citations.

---

## 2) Centralize knobs in `rag/config.py` (single source of truth)
Add (or consolidate) config fields so everything is adjustable without touching code paths:

- Retrieval:
  - `VECTOR_BACKEND` (e.g., `"faiss"`)
  - `TOP_K`
- Answering / evidence:
  - `ASK_MAX_CONTEXT_CHUNKS`
  - (strongly recommended) `ASK_MAX_CONTEXT_CHARS` or `ASK_MAX_CONTEXT_TOKENS` to prevent prompt bloat
  - `ASK_MIN_EVIDENCE_HITS`
  - `ASK_REQUIRE_CITATIONS` (True)
- Determinism:
  - `LLM_TEMPERATURE = 0`
  - stable tie-break rules (implemented in code, but document them here)

Keep it boring. Boring is production.

---

## 3) Define the Day-2 “answer contract” (types + invariants)
Create a small, explicit schema in `rag/types.py`

- `Citation`:
  - `doc_id`, `start_page`, `end_page`, `chunk_id`
- `AnswerResult`:
  - `answer: str`
  - `citations: list[Citation]`
  - `notes: Optional[str]`

**Invariants to enforce:**
- If `answer != "not found in provided docs"` and `ASK_REQUIRE_CITATIONS=True`, then `citations` must be non-empty.
- Citation ordering is deterministic.

This contract becomes the API boundary for Day 3’s LangGraph agent.

---

## 4) Evidence selection: dedup + stable ordering + budget
Implement a single function that turns raw hits into a final evidence set:

### `select_evidence(hits, max_chunks, max_chars) -> list[ChunkHit]`
Rules:
- **Deduplicate** by `chunk_id`
- Sort deterministically:
  - primarily by `-score`
  - tie-break by `(doc_id, start_page, end_page, chunk_id)`  
  (this is the secret sauce for stable reruns)
- Trim to `ASK_MAX_CONTEXT_CHUNKS`
- Enforce a context budget (`ASK_MAX_CONTEXT_CHARS/TOKENS`) by stopping when exceeded

Why this matters for FIPS docs: they’re dense with tables/algorithms, and “just shove everything in” will produce random truncation effects.

---

## 5) Build an evidence “packet” that preserves citations
Create a prompt-ready evidence string that the generator can’t misunderstand.

### `format_evidence(evidence: list[ChunkHit]) -> (context_str, citation_map)`
Make each chunk look like:

- an ID you control (e.g., `c1`, `c2`, …)  
- doc + page range up front  
- then the chunk text

Example structure (conceptually):

- `c1 | FIPS_203 | p12–p13 | chunk_id=...`
  - `<text>`

Return:
- `context_str` for the prompt
- `citation_map` that maps `c# -> {doc_id,start_page,end_page,chunk_id}`

This makes it easy to require inline citation markers later.

---

## 6) Implement `build_cited_answer(...)` with a strict citation policy
### Inputs
- `question: str`
- `hits: list[ChunkHit]` (or `evidence: list[ChunkHit]`)
- config knobs

### Core logic
1) `evidence = select_evidence(...)`
2) If `len(evidence) < ASK_MIN_EVIDENCE_HITS`: **refuse**
3) Build the evidence packet + citation map
4) Generate an answer using one of these (pick the simplest that’s already wired in your repo):
   - **Option A (fastest path today):** LLM call that returns structured JSON
   - **Option B (fallback baseline):** extractive response = return top chunk(s) verbatim + citations (ugly but deterministic and grounded)

### Output format recommendation (high ROI)
Require the answer to include **inline citation markers** like `[c1]`, `[c2]`.  
Then your post-check can be mechanical, not vibes.

### Post-check (non-negotiable)
After generation:
- If `ASK_REQUIRE_CITATIONS=True`:
  - citations list must be non-empty
  - (recommended) every paragraph must contain at least one `[c#]`
  - every `[c#]` must exist in `citation_map`
- If checks fail: either
  - run a single “repair” pass (tell model to add citations only), or
  - refuse with `"not found in provided docs"` (safer and faster for Day 2)

This is exactly the “strict citation policy + post-check” described in the overview. 

---

## 7) Implement `rag/ask.py` CLI (end-to-end flow)
### CLI behavior
- Parse question from args (support quoted string)
- Load retriever via `get_retriever(VECTOR_BACKEND)`
- Retrieve `TOP_K`
- Call `build_cited_answer(question, hits)`
- Print:
  - answer text
  - citations as lines: `doc_id pX–pY chunk_id`

### Two debug flags that will save you hours
- `--show-evidence`: prints top evidence chunks with doc/page/chunk_id
- `--json`: prints the `AnswerResult` as JSON (great for tests + eval harness later)

---

## 8) Add tests: one unit test + one integration-ish sanity check
### Unit tests (pure, fast)
- Evidence selection determinism:
  - given a fixed list of `ChunkHit`s (with ties), output order is stable
- Refusal logic:
  - fewer than `ASK_MIN_EVIDENCE_HITS` → exact refusal answer + empty citations

### Optional integration sanity test (if artifacts available locally)
- Run retrieval on a known query and assert:
  - `citations` is non-empty
  - page spans look valid (start ≤ end, positive ints)

Keep the integration test guarded (e.g., skip if FAISS files missing) so CI doesn’t explode.

---

## 9) Validation run + captured output (for README + Day-2 notes)
Run a tiny “demo script” manually and paste outputs into docs:

- A query that should succeed (FIPS 203 keygen or encaps/decaps steps)
- A query that should refuse (something clearly not in your PDFs)

Save a snippet to:
- `README.md` (quick demo)
- or `notes/day2.md` / `docs/day2_samples.md`

This is recruiter candy: “here’s the CLI, here’s a cited answer.”

---

## 10) Day-2 acceptance checklist (print this at the end of your notes)
- `python -m rag.ask "...?"` works end-to-end
- Non-refusal answers include citations and pass post-check
- Insufficient evidence returns **exactly** `not found in provided docs` + `[]`
- Rerunning the same query yields stable citation ordering
- `--show-evidence` makes it easy to audit page spans quickly

---

This gets you a **citation-grounded CLI product** on Day 2, and it sets up Day 3 cleanly (LangGraph can treat `retrieve()` and `build_cited_answer()` as tools without rewriting your core logic).

day2 Retrieval quality improvements (high ROI, 1-week friendly)

Right now you’re doing pure dense vector search. For NIST FIPS PDFs, that’s always going to be a little chaotic because questions often hinge on exact symbols and section/algorithm identifiers (“ML-KEM.KeyGen”, “Algorithm 19”, “K-PKE.KeyGen”, parameter sets). Dense embeddings don’t love those. Your evidence list proves it.

The upgrade path that fits your project plan

Per project_overview.md, your best next move is:

BM25 (lexical) index

Hybrid retrieval (BM25 + vector)

Query fusion + RRF merge (RAG-Fusion style)

Optional rerank (stretch)

# Day 3 — Agent tools

Add tool calling: the agent decides when to retrieve more, when to compare, etc.

Add “citation enforcement” (refuse or ask to retrieve if evidence missing)

# Day 4 — Evaluation v1

Create labeled QA set (manual + maybe some synthetic generation + manual spot-check)

Implement retrieval metrics + a simple answer grading rubric

# Day 5 — Improve retrieval

Try hybrid retrieval or reranking

Tune chunk size/overlap, top-k

Record before/after metrics

# Day 6 — Docker + CI + cleanup

Dockerize, add GitHub Actions

Write a clean README with architecture diagram

# Day 7 — Portfolio packaging

Short write-up: “what failed, what worked, what metrics improved”

Add screenshots/GIF of demo

Add “Resume bullets” (below)
