from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    doc_id: str | None
    start_page: int | None
    end_page: int | None
    display_name: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    type: str
    source_id: str
    target_id: str
    doc_id: str | None
    start_page: int | None
    end_page: int | None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)