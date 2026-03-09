"""
LangGraph agent state definitions.

What this module is for:
- Defines the `AgentState` TypedDict used by the LangGraph agent.
- Provides frozen dataclasses for evidence items, citations, query analysis, and routing plans.
- Supports serialization via `.to_dict()` methods for debugging, trace output, and eval artifacts.

How it is used:
- Imported by `rag.lc.graph` to define and manipulate agent state during execution.
- Used by `state_utils` to initialize and update state consistently.
- QueryAnalysis and Plan dataclasses enable structured routing decisions.
- EvidenceItem and Citation dataclasses preserve page-level citations (data contract).
- Enables structured debugging and evaluation via traceable state fields.

Key types:
- `AgentState`: Main TypedDict with fields for question, evidence, plan, answer, citations, and debug data.
- `QueryAnalysis`: Router input containing canonical query, mode hint, anchors, and doc scope.
- `Plan`: Router output specifying action (retrieve/compare/answer/refuse) and reasoning.
- `EvidenceItem`: Chunk with score, doc_id, page range, and text.
- `Citation`: Page-level reference with doc_id, page range, and chunk_id.

CLI flags:
- None. This is a library module (non-CLI) and is not executed via command-line flags.
"""

# rag/lc/state.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict
from dataclasses import dataclass, asdict


# -----------------------------
# Serializable evidence + citations
# -----------------------------

@dataclass(frozen=True)
class EvidenceItem:
    score: float
    chunk_id: str
    doc_id: str
    start_page: int
    end_page: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Citation:
    doc_id: str
    start_page: int
    end_page: int
    chunk_id: str
    key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Router decision ("plan")
# -----------------------------

Action = Literal[
    "retrieve",
    "compare",
    "resolve_definition",
    "summarize",
    "answer",
    "refuse",
]

ModeHint = Optional[Literal[
    "general",        # broad conceptual Qs
    "definition",     # "what is X", notation/terms
    "algorithm",      # "Algorithm 2", step lists, pseudocode
    "compare",        # side-by-side comparisons
]]


@dataclass(frozen=True)
class CompareTopics:
    topic_a: str
    topic_b: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class QueryAnalysis:
    original_query: str
    canonical_query: str
    mode_hint: Literal["general", "definition", "algorithm", "compare"]
    rewrite_needed: bool
    protected_spans: List[str]
    required_anchors: List[str]
    sparse_query: str
    dense_query: str
    subqueries: List[str]
    confidence: float
    compare_topics: Optional[CompareTopics] = None
    doc_ids: Optional[List[str]] = None
    doc_family: Optional[str] = None
    analysis_notes: Optional[str] = None
    answer_prompt_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["doc_ids"] = list(self.doc_ids or [])
        payload["protected_spans"] = list(self.protected_spans or [])
        payload["required_anchors"] = list(self.required_anchors or [])
        payload["subqueries"] = list(self.subqueries or [])
        payload["answer_prompt_question"] = self.answer_prompt_question or self.original_query
        return payload


@dataclass(frozen=True)
class Plan:
    action: Action
    reason: str

    # For retrieval-like actions
    query: Optional[str] = None

    # For compare/summarize/etc.
    args: Dict[str, Any] = None

    # Optional hint for standards-y queries (helps routing + retrieval mode)
    mode_hint: ModeHint = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["args"] is None:
            d["args"] = {}
        return d


# -----------------------------
# LangGraph state (keep it dict-like)
# -----------------------------

class AgentState(TypedDict, total=False):
    # Input
    question: str
    original_query: str
    canonical_query: str
    mode_hint: str
    rewrite_needed: bool
    protected_spans: List[str]
    required_anchors: List[str]
    sparse_query: str
    dense_query: str
    subqueries: List[str]
    confidence: float
    compare_topics: Optional[Dict[str, str]]
    doc_ids: List[str]
    doc_family: str
    analysis_notes: str
    answer_prompt_question: str
    query_analysis: Dict[str, Any]
    request_k: int

    # Router output
    plan: Dict[str, Any]  # Plan.to_dict()

    # Evidence + generation
    evidence: List[Dict[str, Any]]     # EvidenceItem.to_dict()
    draft_answer: str
    final_answer: str
    citations: List[Dict[str, Any]]    # Citation.to_dict()

    # Budgets / counters
    tool_calls: int
    steps: int
    retrieval_round: int

    # Debugging / provenance
    trace: List[Dict[str, Any]]
    errors: List[str]
    evidence_sufficient: bool
    stop_reason: str
    refusal_reason: str
    last_retrieval_stats: Dict[str, Any]
    timing_ms: Dict[str, float]
