# rag/config.py
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"Invalid int for {name}={v!r}") from e


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Retrieval backend (swappable) ---
    VECTOR_BACKEND: str = _env_str("VECTOR_BACKEND", "faiss")
    TOP_K: int = _env_int("TOP_K", 8)

    # --- Answering / evidence policy ---
    ASK_MAX_CONTEXT_CHUNKS: int = _env_int("ASK_MAX_CONTEXT_CHUNKS", 6)
    # FIPS chunks can be dense (algorithms/tables). This prevents runaway prompt length.
    ASK_MAX_CONTEXT_CHARS: int = _env_int("ASK_MAX_CONTEXT_CHARS", 12000)

    ASK_MIN_EVIDENCE_HITS: int = _env_int("ASK_MIN_EVIDENCE_HITS", 2)
    ASK_REQUIRE_CITATIONS: bool = _env_bool("ASK_REQUIRE_CITATIONS", True)

    # --- Debug / output ergonomics ---
    ASK_SHOW_EVIDENCE_DEFAULT: bool = _env_bool("ASK_SHOW_EVIDENCE_DEFAULT", False)
    ASK_JSON_DEFAULT: bool = _env_bool("ASK_JSON_DEFAULT", False)

    # --- Determinism knobs (for LLM calls later) ---
    LLM_TEMPERATURE: float = float(_env_str("LLM_TEMPERATURE", "0.0"))


SETTINGS = Settings()


def validate_settings() -> None:
    allowed_backends = {"faiss", "pgvector", "chroma"}
    if SETTINGS.VECTOR_BACKEND not in allowed_backends:
        raise ValueError(
            f"VECTOR_BACKEND must be one of {sorted(allowed_backends)}, "
            f"got {SETTINGS.VECTOR_BACKEND!r}"
        )
    if SETTINGS.TOP_K <= 0:
        raise ValueError("TOP_K must be > 0")
    if SETTINGS.ASK_MAX_CONTEXT_CHUNKS <= 0:
        raise ValueError("ASK_MAX_CONTEXT_CHUNKS must be > 0")
    if SETTINGS.ASK_MAX_CONTEXT_CHARS <= 0:
        raise ValueError("ASK_MAX_CONTEXT_CHARS must be > 0")
    if SETTINGS.ASK_MIN_EVIDENCE_HITS < 0:
        raise ValueError("ASK_MIN_EVIDENCE_HITS must be >= 0")
