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
    citations: List[Dict[str, Any]]    # Citation.to_dict()

    # Budgets / counters
    tool_calls: int
    steps: int

    # Debugging / provenance
    trace: List[Dict[str, Any]]
    errors: List[str]
