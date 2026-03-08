# Week 3 plan (revised on 2026-03-08)

## Why this revision
This revision keeps the original Week 3 direction but tightens it around the real remaining gaps shown by the current reports:

- Day 1 productization is already the highest-signal gap for the target ML engineering roles (`FastAPI`, `Docker`, `RAG`, `AI agents`, `NLP pipelines`, `knowledge graphs`, secure integration).  
- Retrieval is now in a better state after the Week 2 recovery work, and the current project overview already says answer reliability is stronger on retrieval than on answer-side recovery.  
- The compare-citation failure report shows an important answer-side failure mode: evidence can be sufficient while the answer still fails citation requirements. That specific compare bug is fixed, but the broader Week 3 target remains answer-side robustness across more cases.  
- Since the LangGraph path will gain an `analyze_query` node, direct retrieval heuristics such as automatic mode-hint inference and default query-variant expansion should not also fire blindly inside the graph path.

This plan stays aligned with `project_overview.md`:
- keep deterministic ingestion -> structure-aware chunking -> hybrid retrieval -> citation-first generation -> bounded LangGraph control
- do not expand into a full knowledge graph, full MLOps platform, or full LLM fine-tuning pipeline in Week 3
- prioritize measurable upgrades that improve both hiring signal and demo quality within one week

## Main Week 3 objective

Turn the NIST PQC RAG assistant into a recruiter-legible mini production system that demonstrates:
- deployable agentic RAG behavior
- stronger query analysis and evidence handling
- explicit citation-safe answering and refusal behavior
- measurable before/after retrieval and answer quality
- a small graph-like organization feature tied to standards documents

## Priorities for this week

### Priority 1 - Productize the project
Add a local/on-prem serving path, FastAPI surface, Docker packaging, and a clear runbook. This closes one of the largest gaps versus the job description.

### Priority 2 - Make the agent story stronger
Upgrade from mostly retrieval-centric behavior to explicit query analysis, graph-visible state, bounded refinement, and an explicit graph-serving path.

### Priority 3 - Make answer-side robustness visible
Focus Week 3 answer work on the realistic remaining gap: retrieval can be sufficient while the answer still fails citation requirements. Treat the compare-citation fix as evidence for the class of problem, not as the entire problem being finished.

### Priority 4 - Publish measurable proof
Create a clean ablation story with before/after metrics, not just scattered implementation reports.

### Priority 5 - Add a scoped KG-lite feature
Do not attempt a full knowledge graph. Instead, add a lightweight graph-like layer over documents, sections, algorithms, and terms so the project can partially demonstrate knowledge organization.

## Success criteria

By the end of Week 3, the project should be able to demonstrate:
- a local or on-prem style demo path with one-command startup
- FastAPI endpoints for asking questions and inspecting retrieved evidence
- two explicit serving modes:
  - `/ask` = stable direct QA path
  - `/ask-agent` = explicit LangGraph path
- a bounded LangGraph flow with explicit `analyze_query -> retrieve -> assess -> optional refine -> answer -> verify/refuse` trace
- stronger citation-safe answering, including deterministic fallback(s) when free-form generation fails citation checks
- a small ablation report showing concrete retrieval and/or answer-side improvement over baseline
- a graph-lite navigation feature over standards structure
- updated README / demo notes that explain architecture, tradeoffs, and remaining gaps honestly

## Cross-cutting implementation rule

When a request enters through the LangGraph path:
- disable default retrieval-side automatic `mode_hint` inference
- disable default retrieval-side query-variant expansion
- let the `analyze_query` node be the source of:
  - `canonical_query`
  - `mode_hint`
  - `required_anchors`
  - optional filters / constraints
  - any bounded query rewrites or expansions

Rationale:
- Week 2 recommended keeping mode-aware query-variant expansion disabled by default unless revalidated.
- Doubling graph-side query transformation with retrieval-side automatic transformation creates conflicting behavior and makes traces harder to interpret.
- The graph path should show deliberate agent reasoning; the direct `/ask` path can keep the simpler retrieval heuristics where appropriate.

## Day-by-day plan

### Day 1 - Deployment foundation: local LLM path + FastAPI skeleton

Goal:  
Close the biggest project-to-JD gap first by making the system look deployable rather than notebook-like.

Work:
- add a local model adapter path for one realistic serving option
  - preferred for one-week scope: local endpoint abstraction that can target OpenAI-compatible local serving
  - choose one practical backend for demo: vLLM if GPU access is available, otherwise llama.cpp / Ollama style local serving
- keep the rest of the direct QA pipeline unchanged behind a stable model interface
- scaffold FastAPI endpoints:
  - `/ask` -> stable direct QA path; returns answer, citations, refusal reason if any, trace summary
  - `/search` -> returns retrieved chunks and metadata for debugging
  - `/health` -> sanity check for interview/demo use
- add timing hooks for retrieval latency, rerank latency, and generation latency
- optionally reserve routing/service plumbing so `/ask-agent` can be added cleanly on Day 2 without changing the API shape again

Acceptance:
- the app can answer at least one end-to-end query through FastAPI
- model backend is configurable without changing pipeline code
- latency fields are visible in logs or trace output

Why this matters:  
This directly addresses the role's emphasis on AI systems that integrate into real infrastructure, not just experimental retrieval code.

### Day 2 - Query analysis node + explicit graph-serving path

Goal:  
Make the LangGraph controller look like a genuine agent pipeline instead of a thin retrieve-and-answer wrapper.

Work:
- add `analyze_query` as the first graph node
- produce structured outputs such as:
  - `canonical_query`
  - `mode_hint` (`definition | algorithm | compare | general`)
  - `required_anchors` (for example `Algorithm 2`, `ML-KEM.Decaps`, `Section 5.1`)
  - optional filters (doc_id or document family if inferable)
- add `/ask-agent` as the explicit LangGraph endpoint
- wire `/ask-agent` so the graph trace is externally visible through the API response / trace summary
- make graph-produced `mode_hint` and graph-produced rewrites actually flow through retrieve APIs, rather than being partial or best-effort
- keep this deterministic: schema-constrained output, temperature 0, bounded outputs
- for LangGraph-entered requests, disable retrieval-side automatic query variants and automatic mode-hint inference
- preserve direct `/ask` as the stable control path for comparison

Acceptance:
- graph trace clearly shows `analyze_query` before retrieval
- `/ask-agent` returns answer/refusal plus citations and trace summary
- unit tests verify deterministic structured analysis
- retrieval changes behavior for algorithm / definition / compare queries in a controlled, traceable way

Why this matters:  
The JD emphasizes AI agents and NLP pipelines. This creates a more credible agent story without sacrificing determinism, while keeping `/ask` as a stable baseline.

### Day 3 - Answer-side citation robustness (not “retrieval again”)

Goal:  
Solve the practical failure mode where retrieval is decent or sufficient, but answer generation still fails citation requirements.

Important scope note:
- the comparison-specific citation failure documented in the report is already fixed
- Day 3 should therefore focus on broadening and hardening the answer-side recovery pattern, not on re-fixing the exact same compare bug
- evidence packets are still useful only insofar as they support more reliable answer composition and citation-safe fallback

Work:
- keep raw retrieval ranking defaults stable unless a new eval-backed reason appears
- add a bounded answer-side citation-repair pass:
  - if generation returns unsupported claims, malformed citations, or zero citations, run one repair attempt with stricter citation formatting guidance
- generalize deterministic fallback behavior beyond the exact compare fix where justified:
  - short extractive / semi-extractive fallback for answerable cases
  - explicit refusal if evidence is still insufficient
- standardize citation parsing / enforcement behavior across answer and validation layers
- record refusal reason categories such as:
  - `no_strong_evidence`
  - `anchor_missing`
  - `comparison_evidence_one_sided`
  - `citation_generation_failed`
  - `citation_validation_failed`
- optionally add light evidence-packet grouping or neighbor expansion only if it directly improves citation-safe answer recovery on standards-style content

Acceptance:
- more answerable eval questions end with cited output instead of refusal
- refusal reasons are explicit and traceable
- Day 3 report clearly distinguishes:
  - retrieval failures
  - evidence sufficiency failures
  - answer/citation formatting failures

Why this matters:  
This matches the current project state more honestly: retrieval has improved, but answer-side robustness remains the more important Week 3 gap.

### Day 4 - Eval pass + recruiter-visible ablation report

Goal:  
Convert implementation work into measurable evidence.

Work:
- freeze a baseline config and run it on the same eval set used for comparison
- run a small, targeted ablation set rather than many scattered experiments
- recommended ablations:
  1. baseline hybrid retrieval + direct `/ask`
  2. + `/ask-agent` with `analyze_query`
  3. + answer-side citation-repair pass
  4. + any light evidence-packet / neighbor-expansion change (only if shipped)
  5. + graph-lite navigation signals (if ready)
- report:
  - Recall@k / MRR / nDCG
  - strict page overlap and near-page overlap
  - citation compliance / citation coverage
  - refusal rate and refusal reason breakdown
  - comparison of `/ask` vs `/ask-agent` behavior on representative prompts
- write a concise report under `reports/eval/<date>/ablation.md`
- summarize tradeoffs: what improved, what regressed, what remains open

Acceptance:
- one table clearly compares baseline vs upgraded configurations
- one best default configuration is selected with justification
- the report is concise enough to use in interviews and resume bullets

Why this matters:  
This is the proof artifact that turns technical work into hiring signal.

### Day 5 - Docker packaging + secure local demo path

Goal:  
Close the packaging story and improve the “usable internal AI tool” signal.

Work:
- add Dockerfile and docker-compose setup for the API service
- keep model weights out of the image; document mounted volumes / external local model runtime assumptions
- document environment variables and secrets handling cleanly
- add a runbook for local startup, index use, and common failure cases
- include a basic security-minded checklist suitable for this project's scope:
  - no secrets in code
  - pinned dependencies where practical
  - explicit offline/local mode assumptions
  - allowlisted document corpus
  - auditable trace logs
- test startup on a clean environment as closely as possible

Acceptance:
- `docker compose up` can start the main service path successfully
- the README/runbook is sufficient for someone else to reproduce the demo
- security and environment assumptions are documented explicitly

Why this matters:  
The target roles emphasize containerized AI apps, secure integration, and Linux-friendly engineering.

### Day 6 - KG-lite / standards navigation layer

Goal:  
Partially address the knowledge-organization side of the JD without derailing the week.

Work:
- extract a lightweight structured layer from the corpus:
  - Document
  - Section
  - Algorithm
  - Table
  - Term / symbol / operation
- record simple relations such as:
  - `defined_in`
  - `appears_in`
  - `near_algorithm`
  - `referenced_by_section`
  - `same_document_as`
- use this layer only where it gives clear value:
  - improve navigation for definition and algorithm questions
  - improve compare query evidence grouping
  - improve debug output / explainer diagrams
- keep implementation simple:
  - JSON or lightweight adjacency store first
  - do not require a full graph database in Week 3

Acceptance:
- one or two retrieval/answer flows demonstrably benefit from this graph-lite structure
- a simple artifact exists showing extracted entities/relations from the standards corpus
- the README explains that this is a scoped knowledge-organization layer, not a full KG

Why this matters:  
It gives you an honest partial story for knowledge graphs and intelligent data organization without overclaiming.

### Day 7 - Polish, docs, and interview-facing packaging

Goal:  
Make the work easy to explain, demo, and defend.

Work:
- update README and architecture notes to reflect the upgraded system
- add one system diagram showing:
  - ingest/chunk/index
  - `/ask` direct path
  - `/ask-agent` graph path
  - `analyze_query`
  - hybrid retrieval
  - answer/verify/refuse
  - API + local serving path
- write a short engineering decisions note covering:
  - why `/ask` and `/ask-agent` both exist
  - why graph-side query analysis disables retrieval-side auto variants / auto mode hints
  - why answer-side repair was prioritized over more retrieval tweaks
  - why Docker/FastAPI were prioritized over a full KG or fine-tuning
- prepare a short demo script with 4-5 representative questions:
  - definition question
  - algorithm question
  - compare question
  - direct `/ask` vs `/ask-agent` contrast
  - insufficient-evidence / refusal case

Acceptance:
- someone can understand the project quickly from the repo
- the architecture, tradeoffs, and outcomes are easy to communicate in an interview
