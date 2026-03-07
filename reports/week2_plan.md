# Week 2 plan (tailored to the Qorsa Machine Learning Engineering role)

## Why this version is different

The original Week 2 plan was strong on retrieval quality, but the role asks for more than better search. The JD emphasizes AI agents, RAG, NLP/data pipelines, Docker/Kubernetes, secure integration, knowledge graphs, LLM evaluation/fine-tuning familiarity, and production-minded engineering. Within one week, the best strategy is to push this project from a strong retrieval prototype toward a small deployable internal AI system.

This plan stays aligned with project_overview.md:
- keep deterministic ingestion -> structure-aware chunking -> hybrid retrieval -> citation-first generation -> bounded LangGraph control
- do not expand into a full knowledge graph, full MLOps platform, or full LLM fine-tuning pipeline in Week 2
- prioritize measurable upgrades that improve both hiring signal and demo quality

## Main Week 2 objective

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
Upgrade from mostly retrieval-centric behavior to explicit query analysis, mode-aware retrieval, evidence grouping, and bounded refinement.

### Priority 3 - Make answer-side robustness visible
Fix cases where evidence is retrieved but the answer still fails citation requirements. Add citation-repair / extractive fallback and explicit refusal reasons.

### Priority 4 - Publish measurable proof
Create a clean ablation story with before/after metrics, not just scattered implementation reports.

### Priority 5 - Add a scoped KG-lite feature
Do not attempt a full knowledge graph. Instead, add a lightweight graph-like layer over documents, sections, algorithms, and terms so the project can partially demonstrate knowledge organization.

### Success criteria

By the end of Week 2, the project should be able to demonstrate:
- a local or on-prem style demo path with one-command startup
- FastAPI endpoints for asking questions and inspecting retrieved evidence
- a bounded LangGraph flow with explicit analyze_query -> retrieve -> assess -> optional refine -> answer -> verify/refuse trace
- stronger citation-safe answering, including a deterministic fallback when free-form generation fails citation checks
- a small ablation report showing concrete retrieval and/or answer-side improvement over baseline
- a graph-lite navigation feature over standards structure
- updated README / demo notes that explain architecture, tradeoffs, and remaining gaps honestly

## Day-by-day plan

### Day 1 - Deployment foundation: local LLM path + FastAPI skeleton

Goal:
Close the biggest project-to-JD gap first by making the system look deployable rather than notebook-like.

Work:
- add a local model adapter path for one realistic serving option
  - preferred for one-week scope: local endpoint abstraction that can target OpenAI-compatible local serving
  - choose one practical backend for demo: vLLM if GPU access is available, otherwise llama.cpp / Ollama style local serving
- keep the rest of the RAG pipeline unchanged behind a stable model interface
- scaffold FastAPI endpoints:
  - /ask -> returns answer, citations, refusal reason if any, trace summary
  - /search -> returns retrieved chunks and metadata for debugging
  - /health -> sanity check for interview/demo use
- add timing hooks for retrieval latency, rerank latency, and generation latency

Acceptance:
- the app can answer at least one end-to-end query through FastAPI
- model backend is configurable without changing pipeline code
- latency fields are visible in logs or trace output

Why this matters:
This directly addresses the role's emphasis on AI systems that integrate into real infrastructure, not just experimental retrieval code.

### Day 2 - Query analysis node + mode-aware retrieval plumbing

Goal:
Make the LangGraph controller look like a genuine agent pipeline instead of a thin retrieve-and-answer wrapper.

Work:
- add analyze_query as the first graph node
- produce structured outputs such as:
  - canonical_query
  - mode_hint (definition | algorithm | compare | general)
  - required_anchors (for example Algorithm 2, ML-KEM.Decaps, Section 5.1)
  - optional filters (doc_id or document family if inferable)
- make mode_hint actually flow through retrieve APIs, rather than being partial or best-effort
- keep this deterministic: schema-constrained output, temperature 0, bounded variants
- strengthen query variants using the existing standards-aware logic:
  - acronym anchors
  - dot-name operation variants
  - Algorithm / Table / Section expansions

Acceptance:
- graph trace clearly shows analyze_query before retrieval
- unit tests verify deterministic structured analysis and bounded variant generation
- retrieval changes behavior for algorithm / definition / compare queries in a controlled way

Why this matters:
The JD emphasizes AI agents and NLP pipelines. This creates a more credible agent story without sacrificing determinism.

### Day 3 - Evidence packets + answer-side citation repair

Goal:
Solve the practical failure mode where retrieval is decent but answer generation still fails to produce a usable cited answer.

Work:
- group retrieved hits into evidence packets by document + section_path + nearby page/block context
- add neighbor expansion for standards-style content so algorithms, tables, and nearby explanatory text stay together
- update assess_evidence to use packet-level signals, not only raw top-k hits
- implement deterministic citation-repair fallback:
  - if generation returns unsupported claims or zero citations, produce a short extractive / semi-extractive answer from top evidence spans
  - if evidence is still insufficient, refuse with an explicit reason
- record refusal reason categories such as:
  - no strong evidence
  - anchor missing
  - comparison evidence one-sided
  - citation generation failed

Acceptance:
- more answerable eval questions end with cited output instead of refusal
- refusal reasons are explicit and traceable
- evidence packets are visible in trace/debug output

Why this matters:
This improves real answer reliability and shows production-minded handling of failure modes.

### Day 4 - Eval pass + recruiter-visible ablation report

Goal:
Convert implementation work into measurable evidence.

Work:
- freeze a baseline config and run it on the same eval set used for comparison
- run a small, targeted ablation set rather than many scattered experiments
- recommended ablations:
  1. baseline hybrid retrieval
  2. + query analysis / mode-aware variants
  3. + evidence packets / neighbor expansion
  4. + citation repair fallback
  5. + graph-lite navigation signals (if ready)
- report:
  - Recall@k / MRR / nDCG
  - strict page overlap and near-page overlap
  - citation compliance / citation coverage
  - refusal rate and refusal reason breakdown
- write a concise report under reports/eval/<date>/ablation.md
- summarize tradeoffs: what improved, what regressed, what remains open

Acceptance:
- one table clearly compares baseline vs upgraded configurations
- one best configuration is selected with justification
- the report is concise enough to use in interviews and resume bullets

Why this matters:
This is the proof artifact that turns technical work into hiring signal.

### Day 5 - Docker packaging + secure local demo path

Goal:
Close the packaging story and improve the "usable internal AI tool" signal.

Work:
- add Dockerfile and docker-compose setup for the API service
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
- docker compose up can start the main service path successfully
- the README/runbook is sufficient for someone else to reproduce the demo
- security and environment assumptions are documented explicitly

Why this matters:
The Qorsa role is security-adjacent and container-focused. Even a modest but clean Docker story is a major improvement.

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
  - defined_in
  - appears_in
  - near_algorithm
  - referenced_by_section
  - same_document_as
- use this layer only where it gives clear value:
  - improve navigation for definition and algorithm questions
  - improve compare query evidence grouping
  - improve debug output / explainer diagrams
- keep implementation simple:
  - JSON or lightweight adjacency store first
  - do not require a full graph database in Week 2

Acceptance:
- one or two retrieval flows demonstrably benefit from this graph-lite structure
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
  - analyze_query
  - hybrid retrieval
  - evidence packets
  - answer/verify/refuse
  - API + local serving path
- write a short "engineering decisions" note covering:
  - why no full KG yet
  - why deterministic query analysis instead of open-ended agent rewriting
  - why citation repair is needed
  - why Docker/FastAPI were prioritized over more retrieval tweaks
- prepare a short demo script with 3-5 representative questions:
  - definition question
  - algorithm question
  - compare question
  - insufficient-evidence / refusal case

Acceptance:
- someone can understand the project quickly from the repo
- the architecture, tradeoffs, and outcomes are easy to communicate in an interview

## What is intentionally not in Week 2

These are valuable, but they are not realistic or highest-ROI within one week:
- full knowledge graph construction
- serious LLM fine-tuning pipeline
- Kubernetes deployment beyond light mention or optional stretch
- GCP deployment
- full MLOps stack with orchestration tools
- major parser replacement or broad corpus expansion

### Stretch goals only if ahead of schedule

- add a tiny Streamlit or Gradio demo in front of FastAPI
- add pgvector as a second backend behind the retrieval interface
- run a tiny Neo4j prototype from the graph-lite exports
- add CI smoke tests for API + one retrieval query

##Skills this project will still not fully cover, and how to work toward them

1. Full LLM fine-tuning / embedding-model training
Why the project will still fall short:
- Week 2 can improve retrieval and answer quality, but it still will not honestly demonstrate substantial model fine-tuning or embedding training

How to work toward it:
- run one small LoRA/QLoRA experiment on a narrowly scoped task after Week 2
- or fine-tune a lightweight reranker / embedding model on relevance pairs derived from your eval set
- document the training data, objective, baseline, and before/after metrics
- use HuggingFace + PyTorch so the work maps directly to common ML engineering expectations

2. Real graph databases and query languages (Neo4j / Cypher / SQL)
Why the project will still fall short:
- graph-lite structure is useful, but it is not the same as using a true graph database or writing Cypher queries

How to work toward it:
- export the graph-lite entities/relations into Neo4j after Week 2
- write 5-10 concrete Cypher queries over standards entities
- add one small demo such as "find all sections mentioning ML-KEM.Decaps and related algorithms"
- separately add PostgreSQL + pgvector as a persistent retrieval backend to strengthen SQL relevance

3. Kubernetes / cloud-native deployment / GCP
Why the project will still fall short:
- Docker is realistic this week; Kubernetes and cloud deployment are not the best use of limited time

How to work toward it:
- deploy the FastAPI container once to a small GCP or local k3d/minikube environment
- write a minimal deployment manifest and short ops notes
- focus on one concrete deployment story rather than broad cloud coverage

4. Production MLOps depth
Why the project will still fall short:
- the project can show evaluation discipline and containerization, but not full production ML operations

How to work toward it:
- add CI to run tests and one smoke evaluation
- log retrieval/answer metrics over runs
- later integrate one workflow tool only if it serves a real need (for example Prefect for scheduled evaluation)
- do not add Kubeflow, Ray, Airflow, or Dask unless there is a real scaling reason

5. Security engineering integration beyond secure coding hygiene
Why the project will still fall short:
- this project can become security-aware, but it will not fully demonstrate collaboration with security engineers or enterprise security architecture

How to work toward it:
- add a short threat-model note covering prompt injection, unsafe evidence, secret handling, and trace retention
- document trust boundaries between corpus, retriever, and model
- add dependency review / pinning and a simple security checklist for the repo
- if possible, build one small feature around document allowlisting or evidence-source trust levels

6. Large-scale search infrastructure
Why the project will still fall short:
- the current corpus is small and standards-focused, so it will not prove billion-scale retrieval or distributed search

How to work toward it:
- later add a backend such as pgvector, OpenSearch, or Elasticsearch behind the same retrieval interface
- keep evaluation constant while swapping backends so the systems-design story is stronger
- frame the current project honestly as architecture-ready, not scale-proven

How to talk about this plan in interviews

The story should be:
"I already had a strong citation-grounded RAG core over technical standards. For Week 2, I prioritized the gaps that mattered most for ML engineering roles: deployment, stronger agent control, answer robustness, measurable ablations, and a lightweight knowledge-organization layer. I intentionally did not overreach into full fine-tuning, full knowledge graphs, or cloud orchestration in one week; instead, I scoped the project into a small deployable AI system with clear next steps toward those skills."
