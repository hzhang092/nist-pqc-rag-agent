# rag/lc/trace.py
from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _slugify(text: str, max_len: int = 80) -> str:
    s = text.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:max_len] if s else "question"


def _safe(obj: Any) -> Any:
    """JSON-safe conversion for odd objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_trace(
    state: Dict[str, Any],
    out_dir: str = "runs/agent",
    filename_prefix: Optional[str] = None,
    truncate_evidence_chars: int = 800,
) -> str:
    """
    Writes the final AgentState to a JSON file.
    Returns the filepath as a string.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    q = state.get("question", "")
    slug = _slugify(q)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix = filename_prefix or "agent"
    file = out_path / f"{prefix}_{ts}_{slug}.json"

    # Make a copy and truncate evidence text to keep trace readable
    payload = json.loads(json.dumps(state, default=_safe))  # deep copy + json-safe
    ev = payload.get("evidence", []) or []
    for e in ev:
        if isinstance(e, dict) and "text" in e and isinstance(e["text"], str):
            if truncate_evidence_chars and len(e["text"]) > truncate_evidence_chars:
                e["text"] = e["text"][:truncate_evidence_chars] + "…(truncated)"

    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(file)
