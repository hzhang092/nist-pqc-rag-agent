# Day 2 Technical Summary — nist-pqc-rag-agent

### Goal shipped
Day 2 turned the Day 1 “index + retriever” foundation into a **citation-grounded, deterministic RAG answering flow** runnable from the CLI:

- `python -m rag.ask "<question>"` → **answer + inline citations** + structured citation objects
- `python -m rag.search "<query>"` → debug retrieval view (now using the shared retrieval pipeline)

The key product change is that answers are now **evidence-bound**: every sentence must end with an inline marker like **`[c1]`**, and those markers must match `citations[].key`. If the system can’t support the question with retrieved evidence, it refuses with a fixed contract:  
`"not found in provided docs"` and empty citations.

---

## What I built / changed

### 1) Centralized runtime knobs in `rag/config.py`
I added a single settings surface for Day 2 behavior, including:

- Retrieval knobs: backend selection (FAISS vs BM25 vs hybrid), `TOP_K`, candidate multipliers, RRF constants (e.g., `k0`), rerank toggles/pool sizes.
- Answering knobs: max evidence chunks, max context chars, minimum evidence hits, strict citation enforcement.
- LLM knobs: default Gemini model (`gemini-3-flash-preview`), temperature 0, plus debugging flags for `--json` and `--show-evidence`.

**Detailed mechanism implemented:**
- Implemented a frozen `Settings` dataclass with typed defaults and env parsing helpers: `_env_str`, `_env_int`, `_env_bool`, `_env_int_any`.
- Wired retrieval controls directly to env-backed fields (`RETRIEVAL_MODE`, `RETRIEVAL_QUERY_FUSION`, `RETRIEVAL_RRF_K0`, `RETRIEVAL_CANDIDATE_MULTIPLIER`, `RETRIEVAL_ENABLE_RERANK`, `RETRIEVAL_RERANK_POOL`).
- Wired evidence/answer controls (`ASK_MAX_CONTEXT_CHUNKS`, `ASK_MAX_CONTEXT_CHARS`, `ASK_MIN_EVIDENCE_HITS`, `ASK_REQUIRE_CITATIONS`, `ASK_INCLUDE_NEIGHBOR_CHUNKS`, `ASK_NEIGHBOR_WINDOW`).
- Added `validate_settings()` guardrails (allowed backend/mode checks, positive bound checks) invoked by CLI entrypoints before retrieval/generation.

**Why:** This makes the pipeline tunable without editing core logic, supports a “swappable backend” architecture, and keeps behavior reproducible across runs/environments.

---

### 2) Output contract + strict validation in `rag/types.py`
I defined a structured output contract:

- `Citation(key, doc_id, start_page, end_page, chunk_id)`
- `AnswerResult(answer, citations, notes?)`

And added validation utilities to enforce invariants:

- refusal ⇒ exact refusal string + `citations=[]`
- non-refusal ⇒ citations required
- optional but enabled: inline markers `[c#]` required and must correspond to real citation keys

**Detailed mechanism implemented:**
- Standardized refusal sentinel as `REFUSAL_TEXT = "not found in provided docs"`.
- Added regex-based marker extraction (`\[(c\d+)\]`) via `extract_citation_keys()`.
- Enforced page-span sanity per citation (`start_page/end_page > 0`, `start_page <= end_page`).
- Enforced refusal/non-refusal invariants in `validate_answer()`:
  - refusal must return empty citations,
  - non-refusal must include citations when required,
  - optional strict mode requires inline markers and rejects unknown keys.

**Why:** This makes “citation groundedness” testable and machine-checkable (for later eval harness + agent tool calls).

---

### 3) Citation-grounded answer builder in `rag/rag_answer.py`
This is the core Day 2 engine.

**Evidence selection**
- Deduplicate by `chunk_id`
- Deterministic sort key: `(-score, doc_id, start_page, end_page, chunk_id)`
- Enforce context budgets:
  - max chunks
  - max chars (guards against big tables/algorithms blowing up prompts)

**Evidence windowing**
- After selecting the main evidence chunks, I expanded context using **neighbor chunks** (prev/next in doc order) when available and within the context budget.
- This targets a classic FIPS failure mode: algorithms and tables often span chunk boundaries.

**Prompting policy**
- “Prompt specificity guard”: ban details not explicitly supported by evidence (e.g., algorithm numbers, byte lengths, variable names) unless present in evidence text.
- “Evidence-driven answering mode”: answers must be concise and sentence-level cited.

**Inline citation enforcement**
- Every sentence must contain at least one marker `[c#]`
- Markers must exist in the evidence-derived `key_to_citation` map
- If the model violates rules, the system refuses (safe default)

**Detailed mechanism implemented:**
- `select_evidence()`:
  - deduped raw hits by `chunk_id` while keeping max score,
  - sorted deterministically with `(-score, doc_id, start_page, end_page, chunk_id)`,
  - took primary hits, then optionally expanded with neighbor chunks using `chunk_id <-> vector_id` maps from `chunk_store.jsonl`,
  - enforced both chunk-count and character budgets (`ASK_MAX_CONTEXT_CHUNKS`, `ASK_MAX_CONTEXT_CHARS`).
- `build_context_and_citations()`:
  - assigned stable citation keys in order (`c1..cN`),
  - emitted context headers with `doc_id`, page span, and `chunk_id` for each evidence block.
- Prompt contract was explicit and strict:
  - 6 hard rules, including one citation per sentence and exact refusal text on insufficient evidence.
- `enforce_inline_citations()`:
  - normalized weak refusal variants to canonical refusal,
  - rejected answers with missing markers, unknown markers, or uncited sentences,
  - returned only cited keys actually used in text (sorted numerically).

**Why:** This turns the model into a *summarizer of evidence*, not a “freeform explainer,” which is essential for standards PDFs.

---


---

### 3.1) Bugfix: “Algorithm N” questions refusing despite correct evidence (SHAKE128example)

**Repro / symptom**

A concrete failure showed up with algorithm-style questions, e.g.:

- `python -m rag.ask "What are the steps in Algorithm 2 SHAKE128?" --k 5 --show-evidence`

Retrieval **did return** the correct chunk (`NIST.FIPS.203::p0028::c001`) containing the full Algorithm 2 pseudocode, but the system could still refuse with the fixed contract (`"not found in provided docs"`). This was initially confusing because `--show-evidence` truncates each chunk for readability, so the “top hits” preview didn’t visibly include the numbered steps even when the full chunk did.

**Debugging instrumentation added**

- Printed the **actual context block** sent to the LLM (the `[c1] …` evidence payload). This confirmed that the “retriever hits” list and the “context to LLM” list can differ because `select_evidence()` applies **dedup + neighbor expansion + context budgeting** (expected behavior).
- Printed **RAW model output** + parsed citation keys + validated keys to confirm whether the refusal was coming from retrieval, the model, or the validator.

**Root cause**

In `rag_answer.py`, the prompt assembly had a formatting bug: the Rule 6 line was concatenated directly into `Question:` (missing a newline), which reduced compliance with the citation/answering rules and increased refusals/low-quality outputs for algorithm-heavy evidence.

**Fix shipped**

Updated `rag/rag_answer.py` to make algorithm answers reliable:

- Fixed prompt formatting (explicit newline separation before `Question:`).
- Added evidence prettification for standards pseudocode (insert newlines before `1:`, `2:`, `for (...)` etc.) so algorithm steps are readable to the model.
- Added a deterministic safety net for Algorithm questions: if the model refuses but the evidence chunk contains numbered steps, extract and return the steps directly **with valid inline citations**, still passing the strict `enforce_inline_citations` contract.

**Detailed mechanism implemented:**
- `_prettify_evidence_text()` now inserts line breaks before numbered steps (`\d+:`) and `for (...)` blocks when algorithm-like patterns are detected.
- `_algorithm_fallback_answer()` triggers only for `Algorithm N` questions, finds evidence containing `Algorithm N` + step markers, extracts numbered steps with `_extract_algorithm_steps()`, and emits cited bullet lines (`- k: ... [c#].`).
- Fallback output is still validated by the same strict citation checker, so this is deterministic but contract-safe.

This turns “algorithm blocks + PDF layout weirdness” into a stable, testable path rather than a prompt-luck issue.

### 4) Gemini LLM integration (free tier) in `rag/llm/gemini.py`
I integrated Gemini via the `google-genai` SDK and set the default model to:

- `gemini-3-flash-preview`

This avoided the earlier quota pitfall where `gemini-2.5-pro` effectively had a free-tier quota limit of 0.

I also made the CLI output include the model name (in `--show-evidence` and/or `--json`) for reproducibility.

**Detailed mechanism implemented:**
- Loaded `.env` (when available), then resolved credentials via `GEMINI_API_KEY`; fails fast with a clear error if missing.
- Built a reusable `generate_fn(prompt)->text` wrapper around `google.genai.Client`.
- Passed `GenerateContentConfig(temperature=SETTINGS.LLM_TEMPERATURE)` for deterministic generation behavior.
- Added retry with exponential backoff for transient/free-tier errors (3 attempts: 0.5s, 1s, 2s).
- Exposed `get_model_name()` and surfaced the effective model name in CLI outputs/payloads.

**Why:** Day 2 needed a working free-tier LLM backend to complete the end-to-end “ask” flow.

---

## Retrieval quality improvements (major Day 2 work)

### 5) BM25 indexing + retriever
I added lexical retrieval because dense embeddings struggle with spec-heavy tokens:

- Built a BM25 index artifact: `data/processed/bm25.pkl`
- Implemented `rag/index_bm25.py` and `rag/retriever/bm25_retriever.py`
- Used a tokenizer that preserves “technical compounds”:
  - dot tokens (`ML-KEM.KeyGen`)
  - hyphen tokens (`ML-KEM-768`)
  - acronyms / constants (`SHAKE128`, `NTT`)

**Detailed mechanism implemented:**
- `rag/index_bm25.py` builds artifact from `chunk_store.jsonl` sorted by `vector_id` to keep deterministic doc ordering.
- Tokenizer regex keeps compound tokens (`[-._]`) and expands them into component parts (e.g., `ml-kem.keygen` plus `ml`, `kem`, `keygen`).
- For each chunk/doc:
  - computed term frequency (`tf`) and document length,
  - built postings lists as `[doc_idx, tf]`,
  - computed IDF with BM25-style smoothing: `log(1 + (N - df + 0.5)/(df + 0.5))`.
- Persisted a single `bm25.pkl` artifact containing params, vocab/idf, postings, doc lengths, and doc metadata (`chunk_id`, `doc_id`, `start_page`, `end_page`, `text`, `vector_id`).
- `BM25Retriever.search()` scores query terms with `k1/b` normalization and returns `ChunkHit` objects with full citation fields.

**Why:** Standards PDFs depend heavily on exact identifiers; lexical retrieval recovers precise sections that dense vectors often miss.

---

### 6) Hybrid retrieval + fusion in `rag/retrieve.py`
I implemented hybrid search combining:

- FAISS dense hits
- BM25 lexical hits

Then fused results using **Reciprocal Rank Fusion (RRF)**:

- `rrf_score += 1 / (k0 + rank)`
- Final stable sort by:
  - `-rrf_score`, then `(doc_id, start_page, chunk_id)`

**Detailed mechanism implemented:**
- Per variant, fetched from both retrievers with expanded depth:
  - `per_source_k = max(top_k * candidate_multiplier, top_k)`.
- Fused all ranking lists with `rrf_fuse()`:
  - score update rule `+ 1/(k0 + rank)` (rank is 1-indexed),
  - representative chunk per `chunk_id` selected with deterministic tie-break.
- Used stable fused ordering by `(-rrf_score, doc_id, start_page, chunk_id)`.
- Applied optional rerank on a larger fused pool (`fused_pool = max(top_k, rerank_pool)`) before truncating to final `top_k`.

**Why:** Hybrid retrieval improves recall of exact spec sections while retaining semantic matching.

---

### 7) Query fusion (generalized + deterministic)
I added deterministic query rewriting/expansion (“query fusion”) to improve retrieval for standards questions, including:

- Generalized rules across ML-KEM / ML-DSA / SLH-DSA style patterns
- Added `"Algorithm N"` as its own explicit variant when detected
- Included “dot-name” variants (e.g., `X.KeyGen`, `X.Sign`, `X.Verify`) when the query implies those operations

Then ran hybrid retrieval across variants and merged again via RRF.

**Detailed mechanism implemented:**
- Built variants with deterministic rules only (no LLM rewrite path), including:
  - technical-token extraction via regex (`[A-Za-z0-9]+(?:[-._][A-Za-z0-9]+)+`),
  - pattern-specific expansions like `ML-KEM key generation -> ML-KEM.KeyGen key generation`,
  - algorithm expansions (`Algorithm N`, `Algorithm N ML-KEM.KeyGen`) when detected.
- Performed stable de-dup while preserving insertion order, so variant ordering is reproducible run-to-run.

**Why:** This is a lightweight, deterministic version of RAG-Fusion: it increases recall of the right spec blocks without using an LLM for rewriting (keeps cost + nondeterminism down).

---

### 8) Cheap reranker toggle
After fusion, I added an optional lightweight rerank stage:
- prioritize hits containing **exact technical tokens** from the query
- then apply a lexical scoring signal (BM25-style score or BM25’s `score_text`)
- stable tie-breaking maintained

**Detailed mechanism implemented:**
- Extracted lowercase technical query tokens once, then flagged each hit by exact token presence in hit text.
- Scored each candidate with `BM25Retriever.score_text(query, hit.text)` using the same lexical statistics as indexed retrieval.
- Ranked by:
  - exact-token match flag (descending),
  - lexical score (descending),
  - deterministic tie-break (`doc_id`, `start_page`, `chunk_id`).
- Returned top `k` after reranking; disabling rerank cleanly falls back to pure RRF order.

**Why:** This nudges “actual algorithm text” above “overview/differences” sections, which is common when dense search returns semantically related but non-authoritative chunks.

---

## Tests added (what + why)

### Answer contract + citation correctness
- `tests/test_types.py`  
  Validates `AnswerResult`/`Citation` invariants (refusal rules, page span sanity).
- `tests/test_rag_answer.py`  
  Verifies:
  - refusal behavior when evidence insufficient
  - inline markers required
  - markers map to `citations[].key`
  - evidence selection is stable (dedup + deterministic sorting)

**Why:** Prevent regressions where answers “look cited” but violate your strict rules.

### Retrieval tests (determinism + correctness)
- `tests/test_bm25_index.py`  
  Checks BM25 artifact build and basic retriever behavior.
- `tests/test_query_fusion.py`  
  Ensures query variants are deterministic, deduped, and include key expansions.
- `tests/test_retrieve_rrf.py`  
  Tests RRF behavior and stable tie-breaking.
- `tests/test_retrieve_determinism.py`  
  Ensures:
  - query variants order stability
  - RRF stable output under ties
  - hybrid pipeline deterministic even if retrievers return different orders (with rerank enabled)

**Why:** Retrieval order instability causes citation key instability (`c1..cN`) which breaks reproducibility and makes evaluation impossible.

### Tooling sanity outputs
- `scripts/mini_retrieval_sanity.py` + `reports/mini_retrieval_sanity.{json,md}`  
  A small snapshot suite of fixed queries that records top hits (doc + pages), acting as a regression baseline.

**Why:** Simple “does retrieval still look sane” check without needing a full eval harness yet.

---

## End-of-Day 2 results
You now have:

- **Working RAG CLI** with strict inline citations + refusal contract (`rag.ask`)
- **Hybrid retrieval** (FAISS + BM25) with **RRF fusion**
- **Deterministic query fusion** rules tuned for technical spec language
- **Evidence windowing** to capture multi-chunk algorithm blocks
- **Cheap reranker** toggle to prioritize exact-match spec content
- A growing **test suite** that locks down determinism, citation validity, and retrieval behavior

---

## What needs to be done next (Day 3+)

### 1) Add an agent loop (LangGraph) that uses your stable tool boundaries
Now that retrieval and answer generation are clean functions, wrap them as tools:
- `retrieve(query) -> hits`
- `answer(question, hits) -> AnswerResult`

Then build a bounded agent loop:
- retrieve → assess evidence sufficiency → (optional) reformulate query → retrieve again → answer

This matches your project’s “agentic, tool-using RAG” direction and makes improvements measurable rather than prompt-only.

### 2) Add evaluation harness beyond sanity snapshots
Build a small eval set (10–30 questions) and score:
- citation compliance rate
- refusal correctness (should refuse when evidence truly absent)
- retrieval hit-rate for “gold pages”
- answer faithfulness via simple heuristics (e.g., claims per sentence must cite, no uncited numbers)

### 3) Improve faithfulness checks
Right now you enforce citation *format* strongly. Next step is checking citation *support* more directly:
- sentence-level evidence overlap heuristic
- “quote-backed” mode for critical claims (optionally include short quoted spans)
- optional lightweight verifier pass (still cheap model)

### 4) Backend swap readiness (vector DB later)
Your retriever interface is now strong enough to introduce:
- pgvector / Chroma backend behind the same interface
- the hybrid+fusion logic should remain unchanged (only the dense backend changes)

---

## File structure at end of Day 2
You ended Day 2 with a coherent repo layout including:
- `rag/` core pipeline (`config.py`, `types.py`, `rag_answer.py`, `retrieve.py`, `ask.py`, `search.py`)
- `rag/retriever/` swappable retrievers (FAISS, BM25, factory)
- `data/processed/` artifacts: FAISS index, embeddings, BM25 pickle, parsed pages/chunks
- `scripts/` reproducible utilities (`mini_retrieval_sanity.py`, chunking/cleaning)
- `tests/` unit tests for contracts + retrieval determinism and fusion correctness
- `reports/` day summaries + progress + sanity snapshots

That’s a strong Day 2. You’ve moved from “I have an index” to “I have an auditable, reproducible RAG system with hybrid retrieval and tests.”
