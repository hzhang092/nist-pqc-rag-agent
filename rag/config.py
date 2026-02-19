"""
Configuration management for the RAG application.

This module centralizes all application settings, loading them from environment
variables with sensible defaults. It uses a frozen dataclass to provide
immutable, type-hinted access to configuration values throughout the application.

This approach allows for easy configuration in different environments (e.g.,
development, testing, production) without changing the code.

Key components:
- Helper functions (`_env_str`, `_env_int`, `_env_bool`) to safely read and
  parse environment variables.
- A `Settings` dataclass that defines all configurable parameters with their
  default values.
- A `validate_settings` function to ensure that the configured values are
  valid and consistent.

Environment variables (quick reference):

Retrieval:
- VECTOR_BACKEND (default: faiss): base-mode backend.
- TOP_K (default: 8): final number of hits returned.
- RETRIEVAL_MODE (default: hybrid): base or hybrid.
- RETRIEVAL_QUERY_FUSION (default: true): enable deterministic query variants.
- RETRIEVAL_RRF_K0 (default: 60): RRF constant in 1/(k0 + rank).
- RETRIEVAL_CANDIDATE_MULTIPLIER (default: 4): candidate expansion factor before fusion.
- RETRIEVAL_ENABLE_RERANK (default: true): enable lightweight lexical rerank.
- RETRIEVAL_RERANK_POOL (default: 40): fused pool size considered before rerank truncation.

Answering:
- ASK_MAX_CONTEXT_CHUNKS (default: 6): max evidence chunks sent to generator.
- ASK_MAX_CONTEXT_CHARS (default: 12000): max combined evidence text length.
- ASK_MIN_EVIDENCE_HITS (default: 2): minimum hits required before answering.
- ASK_REQUIRE_CITATIONS (default: true): enforce citation-required answer contract.
- ASK_INCLUDE_NEIGHBOR_CHUNKS (default: true): include nearby chunks from same doc.
- ASK_NEIGHBOR_WINDOW (default: 1): neighbor distance in vector_id space.

CLI defaults:
- ASK_SHOW_EVIDENCE_DEFAULT (default: false): default behavior for `rag.ask --show-evidence`.
- ASK_JSON_DEFAULT (default: false): default behavior for `rag.ask --json`.

LLM determinism:
- LLM_TEMPERATURE (default: 0.0): generation temperature.
"""
# rag/config.py
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    """Reads a string environment variable, returning a default if not set."""
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _env_int(name: str, default: int) -> int:
    """Reads an integer environment variable, returning a default if not set."""
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"Invalid int for {name}={v!r}") from e


def _env_int_any(names: tuple[str, ...], default: int) -> int:
    """Reads the first available integer environment variable from a list."""
    for name in names:
        v = os.getenv(name)
        if v is None or v.strip() == "":
            continue
        try:
            return int(v)
        except ValueError as e:
            raise ValueError(f"Invalid int for {name}={v!r}") from e
    return default


def _env_bool(name: str, default: bool) -> bool:
    """Reads a boolean environment variable, returning a default if not set."""
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """
    Immutable dataclass holding all application settings.

    Values are loaded from environment variables via the `_env_*` helper
    functions, providing a single source of truth for configuration.
    """
    # --- Retrieval backend (swappable) ---
    # Specifies which vector search backend to use (e.g., 'faiss', 'pgvector').
    VECTOR_BACKEND: str = _env_str("VECTOR_BACKEND", "faiss")
    # The default number of top results to retrieve from the vector store.
    TOP_K: int = _env_int("TOP_K", 8)
    # Retrieval strategy: "base" (single backend) or "hybrid" (faiss+bm25+fusion).
    RETRIEVAL_MODE: str = _env_str("RETRIEVAL_MODE", "hybrid")
    # Enable deterministic query variants and second-stage fusion.
    RETRIEVAL_QUERY_FUSION: bool = _env_bool("RETRIEVAL_QUERY_FUSION", True)
    # Reciprocal Rank Fusion constant.
    RETRIEVAL_RRF_K0: int = _env_int("RETRIEVAL_RRF_K0", 60)
    # Candidate expansion factor before fusion.
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = _env_int("RETRIEVAL_CANDIDATE_MULTIPLIER", 4)
    # Lightweight lexical rerank over fused candidates.
    RETRIEVAL_ENABLE_RERANK: bool = _env_bool("RETRIEVAL_ENABLE_RERANK", True)
    # Number of fused candidates to consider before rerank truncation.
    RETRIEVAL_RERANK_POOL: int = _env_int("RETRIEVAL_RERANK_POOL", 40)

    # --- Answering / evidence policy ---
    # The maximum number of retrieved chunks to include in the context for the LLM.
    ASK_MAX_CONTEXT_CHUNKS: int = _env_int("ASK_MAX_CONTEXT_CHUNKS", 6)
    # The maximum total characters for all context chunks to prevent overly long prompts.
    # FIPS chunks can be dense (algorithms/tables). This prevents runaway prompt length.
    ASK_MAX_CONTEXT_CHARS: int = _env_int("ASK_MAX_CONTEXT_CHARS", 12000)

    # The minimum number of retrieved chunks required to attempt an answer.
    ASK_MIN_EVIDENCE_HITS: int = _env_int("ASK_MIN_EVIDENCE_HITS", 2)
    # If True, the model will be instructed to only use the provided citations.
    ASK_REQUIRE_CITATIONS: bool = _env_bool("ASK_REQUIRE_CITATIONS", True)
    # Include adjacent chunks from the same document for algorithm spillover.
    ASK_INCLUDE_NEIGHBOR_CHUNKS: bool = _env_bool("ASK_INCLUDE_NEIGHBOR_CHUNKS", True)
    # Neighbor distance in vector_id space (1 means +/- 1).
    ASK_NEIGHBOR_WINDOW: int = _env_int("ASK_NEIGHBOR_WINDOW", 1)

    # --- Debug / output ergonomics ---
    # Whether to show the retrieved evidence chunks by default when asking a question.
    ASK_SHOW_EVIDENCE_DEFAULT: bool = _env_bool("ASK_SHOW_EVIDENCE_DEFAULT", False)
    # Whether to output the answer in JSON format by default.
    ASK_JSON_DEFAULT: bool = _env_bool("ASK_JSON_DEFAULT", False)

    # --- Determinism knobs (for LLM calls later) ---
    # The temperature for the LLM, controlling randomness (0.0 is deterministic).
    LLM_TEMPERATURE: float = float(_env_str("LLM_TEMPERATURE", "0.0"))

    # --- Agent loop bounds / stop rules ---
    # Total LangGraph node transitions allowed per request.
    AGENT_MAX_STEPS: int = _env_int("AGENT_MAX_STEPS", 8)
    # Maximum retrieval tool calls allowed per request.
    AGENT_MAX_TOOL_CALLS: int = _env_int("AGENT_MAX_TOOL_CALLS", 3)
    # Maximum retrieve-assess rounds before forced stop/refusal.
    AGENT_MAX_RETRIEVAL_ROUNDS: int = _env_int("AGENT_MAX_RETRIEVAL_ROUNDS", 2)
    # Minimum unique evidence chunks required before answer generation.
    AGENT_MIN_EVIDENCE_HITS: int = _env_int_any(
        ("AGENT_MIN_EVIDENCE_HITS", "ASK_MIN_EVIDENCE_HITS"),
        2,
    )


SETTINGS = Settings()


def validate_settings() -> None:
    """
    Performs validation checks on the loaded settings.

    Raises:
        ValueError: If any of the settings have invalid or inconsistent values.
    """
    allowed_backends = {"faiss", "bm25", "pgvector", "chroma"}
    if SETTINGS.VECTOR_BACKEND not in allowed_backends:
        raise ValueError(
            f"VECTOR_BACKEND must be one of {sorted(allowed_backends)}, "
            f"got {SETTINGS.VECTOR_BACKEND!r}"
        )
    if SETTINGS.TOP_K <= 0:
        raise ValueError("TOP_K must be > 0")
    allowed_modes = {"base", "hybrid"}
    if SETTINGS.RETRIEVAL_MODE not in allowed_modes:
        raise ValueError(
            f"RETRIEVAL_MODE must be one of {sorted(allowed_modes)}, "
            f"got {SETTINGS.RETRIEVAL_MODE!r}"
        )
    if SETTINGS.RETRIEVAL_RRF_K0 <= 0:
        raise ValueError("RETRIEVAL_RRF_K0 must be > 0")
    if SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER <= 0:
        raise ValueError("RETRIEVAL_CANDIDATE_MULTIPLIER must be > 0")
    if SETTINGS.RETRIEVAL_RERANK_POOL <= 0:
        raise ValueError("RETRIEVAL_RERANK_POOL must be > 0")
    if SETTINGS.ASK_MAX_CONTEXT_CHUNKS <= 0:
        raise ValueError("ASK_MAX_CONTEXT_CHUNKS must be > 0")
    if SETTINGS.ASK_MAX_CONTEXT_CHARS <= 0:
        raise ValueError("ASK_MAX_CONTEXT_CHARS must be > 0")
    if SETTINGS.ASK_MIN_EVIDENCE_HITS < 0:
        raise ValueError("ASK_MIN_EVIDENCE_HITS must be >= 0")
    if SETTINGS.ASK_NEIGHBOR_WINDOW < 0:
        raise ValueError("ASK_NEIGHBOR_WINDOW must be >= 0")
    if SETTINGS.AGENT_MAX_STEPS <= 0:
        raise ValueError("AGENT_MAX_STEPS must be > 0")
    if SETTINGS.AGENT_MAX_TOOL_CALLS <= 0:
        raise ValueError("AGENT_MAX_TOOL_CALLS must be > 0")
    if SETTINGS.AGENT_MAX_RETRIEVAL_ROUNDS <= 0:
        raise ValueError("AGENT_MAX_RETRIEVAL_ROUNDS must be > 0")
    if SETTINGS.AGENT_MIN_EVIDENCE_HITS < 0:
        raise ValueError("AGENT_MIN_EVIDENCE_HITS must be >= 0")
