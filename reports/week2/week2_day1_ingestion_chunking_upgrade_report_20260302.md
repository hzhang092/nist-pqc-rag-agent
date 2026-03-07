# Week2 Day1 Upgrade Report: Ingestion + Chunking + Manifest Versioning

Date: 2026-03-02  
Repo: `nist-pqc-rag-agent`  
Scope anchor: `reports/project_overview.md` Week-1 guardrails (determinism, citation-preserving chunk metadata, measurable eval outcomes)

## 1) Executive Summary

This Day1 upgrade implemented three core capabilities:

1. Parser abstraction with dual backend support (`llamaparse` + `docling`) and deterministic ingestion controls.
2. Markdown-aware chunking v2 (behind `CHUNKER_VERSION`), while keeping v1 as default for safe rollout.
3. Manifest-based version awareness (`data/processed/manifest.json`) with retriever-time compatibility checks and artifact hash tracking.

System status after implementation:

- Backward compatibility preserved for existing Week-1 pipelines and contracts.
- Full test suite passes (`68 passed, 2 skipped`).
- Eval harness still runs end-to-end with unchanged baseline retrieval metrics.
- Manifest-aware runtime checks are active once artifacts are rebuilt.

## 2) Why This Upgrade Was Done

The previous pipeline worked, but had the following limitations:

1. Ingestion parser path was hard-coded to LlamaParse.
2. No formal mechanism to trace artifact provenance (parser/chunker/embed/index settings) or detect stale/mismatched artifacts.
3. Chunking logic was good for v1 but had no dedicated markdown-structure mode to leverage headings/tables/algorithm blocks from parser markdown output.

The Day1 upgrade addresses these without widening document scope or breaking page-level citation contracts.

## 3) High-Level Before/After Architecture

### Before
`raw_pdfs -> rag.ingest (LlamaParse only) -> pages.jsonl -> clean -> chunk(v1) -> embed -> faiss/bm25 -> retrieve/eval`

### After
`raw_pdfs -> rag.ingest (parser factory: llamaparse|docling) -> pages.jsonl(+optional markdown/parser_backend) -> clean -> chunk(v1|v2 markdown-aware) -> embed(+breadcrumb encoding) -> faiss/bm25(+manifest stages) -> retrievers(manifest compatibility checks) -> retrieve/eval`

Additional controls:

- `scripts/docling_preflight.py` for local Docling readiness.
- `scripts/check_artifacts.py` for rebuild-required detection via config hash comparison.

## 4) File-Level Change Inventory

### New modules

1. `rag/parsers/base.py`
2. `rag/parsers/factory.py`
3. `rag/parsers/llamaparse_backend.py`
4. `rag/parsers/docling_backend.py`
5. `rag/parsers/__init__.py`
6. `rag/versioning.py`
7. `scripts/docling_preflight.py`
8. `scripts/check_artifacts.py`
9. `tests/test_parser_factory.py`
10. `tests/test_ingest_contract.py`
11. `tests/test_chunk_markdown_structure.py`
12. `tests/test_chunk_determinism.py`
13. `tests/test_manifest_versioning.py`
14. `tests/test_retriever_manifest_guard.py`
15. `reports/ablation_ingest_chunking_202603xx.md`

### Updated modules

1. `rag/ingest.py`
2. `rag/chunk.py`
3. `rag/embed.py`
4. `rag/index_faiss.py`
5. `rag/index_bm25.py`
6. `rag/retriever/faiss_retriever.py`
7. `rag/retriever/bm25_retriever.py`
8. `rag/config.py`
9. `scripts/make_chunks.py`
10. `scripts/clean_pages.py`
11. `.env.example`
12. `README.md`
13. `pyproject.toml`
14. `requirements.txt`
15. `tests/test_embed_store_records.py`

## 5) Detailed Mechanism Changes

## 5.1 Parser Abstraction and Dual-Backend Ingestion

### 5.1.1 Contract introduced

`rag/parsers/base.py` defines:

- `ParsedPage` fields: `doc_id`, `source_path`, `page_number`, `text`, `markdown`, `parser_backend`.
- `ParserBackend` protocol:
  - `parse_pdf(path, expected_pages=...) -> list[ParsedPage]`
  - `backend_version() -> str`

Also introduced `markdown_to_text(markdown)` to provide deterministic markdown-to-plain fallback text normalization.

### 5.1.2 Factory selection

`rag/parsers/factory.py`:

- Reads `PARSER_BACKEND` from settings (`llamaparse` or `docling`).
- Instantiates backend class through small selector logic.

### 5.1.3 LlamaParse backend behavior

`rag/parsers/llamaparse_backend.py`:

- Uses lazy parser initialization (`_get_parser`) so backend can be instantiated without immediately requiring API key.
- Returns page-level rows including both `text` and `markdown` (from LlamaParse markdown output) and `parser_backend`.
- Stable sorting by `page_number`.

### 5.1.4 Docling backend behavior

`rag/parsers/docling_backend.py`:

- Instantiates a single `DocumentConverter` per backend instance.
- Parses deterministically with a bulk fast path (`page_range=(1, N)`), then exports page-scoped markdown via `export_to_markdown(page_no=p)`.
- Falls back to per-page conversion (`page_range=(p, p)`) only when bulk/page export fails, preserving one-record-per-page determinism.
- Produces:
  - `markdown`: Docling markdown export for that page.
  - `text`: deterministic plain text from `markdown_to_text`.
- Sanitizes known Docling artifact patterns in markdown/formula text before writing page records.
- Requires `expected_pages` to enforce page-aligned deterministic traversal.

### 5.1.5 Ingestion refactor behavior

`rag/ingest.py` now:

1. Reads and validates settings.
2. Sorts PDFs deterministically from `data/raw_pdfs`.
3. Computes true page count via `pypdf`.
4. Calls selected parser backend.
5. Normalizes/writes records with optional `markdown`.
6. Enforces page-count and page-number sanity (`PARSER_STRICT_PAGE_MATCH`).
7. Writes:
   - per-document debug JSON (`<doc>_parsed.json`)
   - unified `pages.jsonl`
8. Updates manifest stage `ingest` with parser backend/version and page stats.

## 5.2 Chunking v2 (Markdown-Aware) with Backward-Compatible Default

### 5.2.1 Strategy switch

`rag/config.py` introduces:

- `CHUNKER_VERSION` in `{v1, v2}`; default `v1`.

`scripts/make_chunks.py` now passes settings-driven chunker selection.

### 5.2.2 v1 behavior retained

Original block splitting and packing heuristics are preserved for `v1`:

- Split by blank-line groups.
- Preserve verbatim-ish blocks (table/code/math) line structure.
- Greedy packing with overlap and min/max char constraints.
- Stable chunk id format remains `doc::p####::c###`.

### 5.2.3 v2 markdown-structure mechanism

`rag/chunk.py` adds structured markdown parsing path:

1. Heading detection:
   - ATX style (`#`, `##`, ...)
   - Numeric headings (`1`, `1.2`, ...)
2. Algorithm block detection:
   - starts with `Algorithm`, `Input`, `Output`, `Require`, `Ensure`.
   - continuation rules for algorithm-like lines and code/math-like lines.
3. Table block detection:
   - markdown pipe table lines.
4. List block detection:
   - bullet/numbered list line patterns.
5. Paragraph fallback:
   - merges contiguous non-structure lines into prose block.

The v2 packer then groups `StructuredBlock` items with same packing policy (target/min/max/overlap) and computes:

- `block_type` (dominant class with preference for technical classes: algorithm/table/code/math when mixed with text).
- `section_path` (derived from heading stack, best effort).
- `chunker_version`.

### 5.2.4 Data contract compatibility

Required fields preserved:

- `chunk_id`, `doc_id`, `start_page`, `end_page`, `text`.

Optional enrichments added:

- `section_path`, `block_type`, `chunker_version`.

Page-level citation span contract remains preserved (`start_page`/`end_page` valid and page-bounded in current scope).

### 5.2.5 Source-selection logic

For v2:

- If page has non-empty `markdown`, use markdown-aware splitter.
- Otherwise fall back to v1 text-based splitting on `text_clean`.

## 5.3 Embedding Text Breadcrumb Mechanism

`rag/embed.py` adds `build_embedding_text(chunk)`:

- Embedding payload:
  - `"{doc_id} > {section_path}\n\n{text}"` when `section_path` exists.
  - otherwise unchanged `text`.

Critical separation:

- Encoded embedding text can include breadcrumb.
- Stored evidence text in `chunk_store.jsonl` remains original chunk `text` (no breadcrumb mutation), preserving downstream answer/citation readability and compatibility.

Additional metadata persisted:

- `section_path`
- `chunker_version`
- `embedding_text_breadcrumb: true` in `emb_meta.json` and manifest stage payload.

## 5.4 Manifest Versioning and Compatibility Checks

### 5.4.1 Manifest schema

`rag/versioning.py` introduces a deterministic manifest system:

- `schema_version`
- `generated_at_utc`
- `git_commit`
- `config_hash` (SHA-256 of stable JSON dump of current settings dataclass)
- `stages` map (per stage payloads)
- `artifact_hashes` map (SHA-256 per artifact path)

Current manifest path:

- `data/processed/manifest.json`

### 5.4.2 Stage updates integrated

Manifest stage updates now happen in:

1. `rag/ingest.py` -> `ingest`
2. `rag/chunk.py` -> `chunk`
3. `rag/embed.py` -> `embed`
4. `rag/index_faiss.py` -> `index_faiss`
5. `rag/index_bm25.py` -> `index_bm25`

### 5.4.3 Retriever runtime checks

`rag/retriever/faiss_retriever.py` and `rag/retriever/bm25_retriever.py` now call `ensure_manifest_compat`:

- FAISS checks embed `model_name` and `dim`.
- BM25 checks tokenizer name.
- If manifest missing: warning + continue (backward compatibility).
- If manifest present and mismatch detected: hard error (prevents silent stale-index usage).

### 5.4.4 Artifact check script

`scripts/check_artifacts.py`:

- compares `manifest.config_hash` with current computed settings hash.
- exits non-zero with `REBUILD NEEDED` if mismatch or missing manifest.

## 5.5 Docling Operational Safety

`scripts/docling_preflight.py` performs one-page Docling conversion over first raw PDF and reports:

- pass/fail
- backend version
- output lengths
- strict non-empty page-1 content gate (`markdown` and `text` must both be non-empty)
- actionable hint for Windows symlink privilege issue (`WinError 1314` class failure mode).

This supports the rollout policy of keeping default parser on `llamaparse` until environment is validated.

## 6) New Configuration Surface

Added environment settings:

1. `PARSER_BACKEND` (default `llamaparse`)
2. `PARSER_STRICT_PAGE_MATCH` (default `true`)
3. `CHUNKER_VERSION` (default `v1`)

These were documented in `.env.example` and README.

## 7) Validation and Test Evidence

## 7.1 Automated tests

Executed:

- `python -m pytest -q`

Result:

- `68 passed, 2 skipped`

New test coverage added for:

1. Parser factory selection and unknown backend handling.
2. Ingestion sorting and strict page-count guard.
3. Chunk v2 structure extraction and deterministic output.
4. Embedding breadcrumb behavior (encode-only, storage unchanged).
5. Manifest stage/hash behavior.
6. Retriever manifest compatibility guards.

## 7.2 Runtime checks and pipeline commands

Executed:

1. `python scripts/docling_preflight.py` -> passed on this machine.
2. `python scripts/check_artifacts.py` -> passed after artifact rebuild.
3. `python -m rag.search_faiss "ML-KEM key generation"` -> returned valid ranked hits.
4. `python -m eval.run --dataset eval/day4/questions.jsonl --mode hybrid --k 8` -> completed and wrote reports.

## 7.3 Eval metric status after upgrade wiring

Latest run (`questions_20260302T010135Z`) reports:

- Recall@8: `0.5476`
- nDCG@8: `0.4286`

Compared to baseline target gate:

- Required Recall@8 for pass: `>= 0.5776`
- Required nDCG@8 floor: `>= 0.4186`

Status:

- Gate not yet passed because retrieval uplift ablation has not yet been applied; Day1 implemented infrastructure and mechanisms, not final retrieval uplift tuning.

## 8) Data Contract Impact Assessment

## 8.1 pages.jsonl

Preserved required fields:

- `doc_id`, `source_path`, `page_number`, `text`

Added optional fields:

- `markdown`
- `parser_backend`

## 8.2 chunks.jsonl

Preserved required fields:

- `chunk_id`, `doc_id`, `start_page`, `end_page`, `text`

Added optional fields:

- `section_path`
- `block_type`
- `chunker_version`

No silent break introduced for existing consumers that only require legacy fields.

## 9) Mechanisms That Improve Reliability

1. Deterministic parser input ordering and explicit page validation.
2. Parser backend interface boundary (implementation can swap without changing ingestion contract).
3. Config hash + artifact hash manifest for reproducibility and stale-artifact detection.
4. Runtime retriever compatibility guards to fail fast on model/tokenizer mismatches.
5. Docling preflight script to detect environment blockers before full migration.

## 10) Known Tradeoffs / Current Limitations

1. Docling backend now uses a single-convert fast path with per-page fallback; throughput is improved, but full-document parsing remains expensive on CPU with formula/image enrichments enabled.
2. Chunking v2 exists but default remains v1 until controlled ablation demonstrates measurable uplift.
3. Manifest compatibility checks warn-and-continue if manifest is absent by design (for backward compatibility). Strict mode can be introduced later if desired.

## 11) Day1 Deliverable Conclusion

Day1 successfully delivered the architecture and mechanism upgrade requested:

1. Dual parser abstraction with deterministic ingestion guarantees.
2. Markdown-aware chunking v2 implementation with backward-compatible default behavior.
3. End-to-end manifest version awareness integrated across ingestion/chunk/embed/index/retrieval.

The system is now prepared for Day2+ ablations (`A/B/C/D` parser+chunker combinations) to drive measurable retrieval improvements while maintaining reproducibility and traceability.

## 12) Post-Day1 Addendum (2026-03-01): Docling GPU + Ingestion Contract Hardening

After the Day1 report snapshot, the following upgrades were applied and validated in-repo.

### 12.1 Docling explicit accelerator wiring (GPU-capable)

`rag/parsers/docling_backend.py` now explicitly configures Docling `AcceleratorOptions` through environment variables when building `PdfPipelineOptions`.

New env controls:

1. `DOCLING_DEVICE` (default `auto`; examples: `cuda`, `cuda:0`, `cpu`)
2. `DOCLING_CUDA_USE_FLASH_ATTENTION2` (default `false`)
3. `DOCLING_NUM_THREADS` (optional integer)

Impact:

- GPU usage is now first-class and visible in code (not only implicit auto-detection).
- Existing formula enrichment behavior is preserved.
- Deterministic output contract remains unchanged.

### 12.2 Deterministic per-page Docling failure fallback

`DoclingBackend.parse_pdf(...)` now catches per-page parsing exceptions and emits an empty markdown/text payload for that page with a warning.

Impact:

- A single failing page no longer aborts the entire document ingest.
- One-record-per-expected-page determinism is preserved.

### 12.3 Ingestion settings and page-coverage guard hardening

`rag/ingest.py` was tightened with two correctness fixes:

1. `load_dotenv()` now executes before importing `SETTINGS` so `.env` values reliably apply at runtime.
2. Strict page validation now enforces exact `1..N` page coverage (detects missing and duplicate page numbers), not just count/bounds.

Impact:

- Reduces silent misconfiguration risk.
- Improves ingest contract enforcement for citation-safe page mapping.

### 12.4 Validation evidence for addendum changes

Executed focused parser tests after patching:

- `python -m pytest tests/test_docling_backend.py tests/test_parser_factory.py -q`
- Result: `4 passed` (with existing upstream `llama-parse` deprecation warnings)

Additional contract tests (from previous hardening pass) remain present for strict coverage behavior in `tests/test_ingest_contract.py`.

### 12.5 Operational note for GPU rollout

GPU acceleration requires a CUDA-enabled PyTorch runtime. Current repo Docker image is intentionally CPU-pinned (for portability), so GPU Docling in containers requires a separate CUDA image/profile.

### 12.6 Post-addendum patch (2026-03-02): preflight gate + single-convert path + sanitizer + tests

After the initial addendum, a targeted parser hardening patch was applied.

#### 12.6.1 Preflight correctness fix

`scripts/docling_preflight.py` was tightened to eliminate false-positive pass conditions:

1. Probe exactly one page (`expected_pages=1`) for the page-1 readiness check.
2. Require exactly one parsed page.
3. Fail preflight if page-1 `markdown` or `text` is empty.
4. Ensure debug output directory exists before writing.

Impact:

- Prevents “preflight passed” outcomes when Docling returns empty payloads or only partial failures outside page 1.
- Makes preflight semantics match its operator-facing message (“page 1”).

#### 12.6.2 Docling single-convert fast path with deterministic fallback

`rag/parsers/docling_backend.py` now uses:

1. Primary path: one `convert(pdf, page_range=(1, expected_pages))` call per document.
2. Page export path: `export_to_markdown(page_no=...)` for deterministic page mapping.
3. Resilience path: if bulk/page export fails, fallback to per-page conversion for that page.

Impact:

- Reduces repeated converter invocation overhead while preserving page-level citation contracts.
- Retains one-output-row-per-expected-page determinism under partial page failures.

#### 12.6.3 Markdown/formula sanitization for Docling artifacts

`rag/parsers/docling_backend.py` now sanitizes known noisy patterns before downstream ingestion:

1. Removes `<!-- image -->` comments.
2. Removes Docling location tags and noisy location spans (e.g., `<loc_...>`, `<text><loc_...>...</text>`).
3. Removes malformed formula tags (e.g., `</formula` fragments).
4. Cleans formula strings before injection and removes unresolved `formula-not-decoded` placeholders.

Impact:

- Reduces parser-induced noise that can degrade chunking and retrieval quality.
- Keeps parser output deterministic and citation-safe.

#### 12.6.4 Validation evidence for this patch set

Executed after patching:

- `python -m pytest tests/test_docling_backend.py tests/test_docling_preflight.py -q`
  - Result: `4 passed`
- `python -m pytest tests/test_parser_factory.py tests/test_ingest_contract.py -q`
  - Result: `6 passed`
- `python scripts/docling_preflight.py`
  - Result: pass with non-empty page-1 output (`markdown_len=617`, `text_len=608`)

Quick local timing probe (same machine/env, non-benchmark):

- `DoclingBackend.parse_pdf(..., expected_pages=5)` observed at approximately `4.91s` after this patch.

### 12.7 Post-addendum patch (2026-03-02): adaptive GPU batching + CUDA allocator tuning (no CPU fallback)

To address Docling OOM failures on longer PDFs while preserving parsing quality and deterministic contracts, `rag/parsers/docling_backend.py` was upgraded with adaptive GPU batch control and CUDA memory tuning.

#### 12.7.1 Adaptive page-range batching in `parse_pdf`

The parser now processes documents in configurable page batches rather than one full-document convert call.

New behavior:

1. Start with `DOCLING_PAGE_BATCH_SIZE` (default `12`).
2. Convert page ranges in deterministic order (`start_page -> end_page`).
3. On OOM-like failures (`CUDA out of memory`, `std::bad_alloc`, `unable to allocate`), shrink batch size adaptively (halve until `DOCLING_MIN_PAGE_BATCH_SIZE`, default `1`).
4. Continue processing without changing device policy.

Impact:

- Eliminates the “single massive gulp” memory spike pattern.
- Preserves deterministic one-record-per-page output and stable page ordering.

#### 12.7.2 CUDA memory management hooks

To reduce fragmentation and improve allocator behavior under repeated convert calls:

1. Backend sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when unset and `DOCLING_SET_PYTORCH_ALLOC_CONF=1`.
2. Between batches, performs explicit cleanup (`gc.collect()`, `torch.cuda.empty_cache()`, best-effort `torch.cuda.ipc_collect()`).

Impact:

- Improves memory reuse across batches.
- Reduces cumulative allocator pressure in long ingestion runs.

#### 12.7.3 Explicit no-CPU-fallback policy

Per current GPU-quality/runtime preference, this patch does **not** introduce a CPU fallback path for OOM handling.

Behavior under pressure:

- First response is adaptive batch-size reduction on GPU.
- If failures persist at minimum batch size, page-level failure handling remains deterministic (empty page payload with warning) rather than device switching.

#### 12.7.4 Default tuning updates

Environment/default alignment updated for safer GPU operation:

1. `DOCLING_GENERATE_PAGE_IMAGES` default changed to `0` (memory reduction). (now changed to 1 )
2. Added config knobs to `.env` and `.env.example`:
   - `DOCLING_PAGE_BATCH_SIZE`
   - `DOCLING_MIN_PAGE_BATCH_SIZE`
   - `DOCLING_ADAPTIVE_BATCHING`
   - `DOCLING_SET_PYTORCH_ALLOC_CONF`
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

#### 12.7.5 Validation evidence for this patch set

Executed after patching:

- `python -m pytest tests/test_docling_backend.py -q`
  - Result: `4 passed`
- `python -m pytest tests/test_ingest_contract.py -q`
  - Result: `3 passed`

New coverage added:

- `test_docling_parse_pdf_adaptive_batch_shrinks_on_oom` verifies adaptive batch reduction and deterministic page coverage when encountering simulated CUDA OOM.

### 12.8 Post-addendum patch (2026-03-03): chunk-structure quality hardening + docling layout-noise fix

This patch addressed concrete chunk-quality regressions observed in `data/processed/chunks.jsonl` and parser-output artifacts.

#### 12.8.1 Issues fixed

1. False table detection from prose/math bars:
   - Absolute-value style text (e.g., `| str_i |`) and concatenation notation (e.g., `str||...||str_m`) were occasionally classified as `table`.
2. Over-aggressive math labeling:
   - Long prose lines containing a single comparison/equality symbol were sometimes labeled as math blocks.
3. Algorithm hierarchy drift:
   - Algorithm headings could form misleading nested paths across pages (including Docling artifact headings like `## -1 Algorithm 10 ...`).
4. Fenced pseudocode typing:
   - Fenced algorithm blocks could be typed as generic `code` instead of `algorithm`.
5. Docling layout-math noise (Issue 4 follow-up):
   - Additional suppression for alignment/layout token spam (`\\quad`, `&`, `\\\\`) was applied in markdown/formula sanitization to reduce retrieval noise.

#### 12.8.2 Mechanism updates

1. `rag/chunk.py`
   - Tightened markdown table-line heuristics to reject non-table bar patterns (`|x|`, `||`, `\\dots` contexts) while preserving real pipe-table detection.
   - Refined `looks_like_math_line` so single-symbol long prose remains `text` unless stronger math signals exist.
   - Added heading normalization for broken algorithm prefixes (`-1 Algorithm ...` -> `Algorithm ...`) and heading-stack cleanup so numeric section headings do not remain nested under prior algorithm headings.
   - Restored fenced algorithm detection (`algorithm` vs `code`) for pseudocode in fenced blocks.
   - Added hard chunk boundaries across strong-structure/section transitions to reduce structure bleed between adjacent chunks.

2. `rag/parsers/docling_backend.py`
   - Added stronger layout-noise collapsing and line-drop heuristics for alignment-token dominated formulas/lines.
   - Applied sanitizer consistently in both markdown-level and formula-level cleanup paths.

3. Regression tests
   - `tests/test_chunk_markdown_structure.py` expanded with cases for:
     - absolute-value bars not treated as tables,
     - concatenation bars not treated as tables,
     - single-symbol prose not mislabeled as math,
     - fenced algorithm blocks typed as `algorithm`,
     - algorithm heading sibling/pop behavior (no algorithm-under-algorithm chain drift).
   - `tests/test_docling_backend.py` includes explicit alignment-layout noise sanitizer coverage.

#### 12.8.3 Validation evidence

Executed after patching:

- `python -m pytest tests/test_chunk_markdown_structure.py tests/test_chunk_determinism.py tests/test_docling_backend.py -q`
  - Result: `15 passed`
- `python scripts/make_chunks.py`
  - Result: rebuilt `data/processed/chunks.jsonl` with `chunker_version=v2`.

Spot checks on rebuilt chunks:

1. `NIST.FIPS.203::p0028::c001` now remains a single `algorithm` chunk (heading+body preserved).
2. `NIST.FIPS.204::p0024::c001` is no longer mislabeled as `table` (now typed as `text`).
3. Section-path sanity check for `Algorithm > Algorithm` chain patterns reports `0` occurrences.

Data-contract status:

- Required citation fields remain intact (`doc_id`, `start_page`, `end_page`).
- Deterministic chunk-id/page-order behavior remains preserved.
