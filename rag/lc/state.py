"""
LangGraph agent state definitions.

What this module is for:
- Defines the `AgentState` structure used by the LangGraph agent.
- Provides dataclasses for evidence items, citations, and routing plans.
- Supports serialization of state components for debugging and trace output.

How it is used:
- Imported by `rag.lc.graph` to define and manipulate agent state during execution.
- Used by `state_utils` to initialize and update state consistently.
- Enables structured debugging and evaluation via traceable state fields.

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
    "symbolic",       # parameter names, symbols, section numbers
]]


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
