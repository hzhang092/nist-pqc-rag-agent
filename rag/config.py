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

    # --- Debug / output ergonomics ---
    # Whether to show the retrieved evidence chunks by default when asking a question.
    ASK_SHOW_EVIDENCE_DEFAULT: bool = _env_bool("ASK_SHOW_EVIDENCE_DEFAULT", False)
    # Whether to output the answer in JSON format by default.
    ASK_JSON_DEFAULT: bool = _env_bool("ASK_JSON_DEFAULT", False)

    # --- Determinism knobs (for LLM calls later) ---
    # The temperature for the LLM, controlling randomness (0.0 is deterministic).
    LLM_TEMPERATURE: float = float(_env_str("LLM_TEMPERATURE", "0.0"))


SETTINGS = Settings()


def validate_settings() -> None:
    """
    Performs validation checks on the loaded settings.

    Raises:
        ValueError: If any of the settings have invalid or inconsistent values.
    """
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
