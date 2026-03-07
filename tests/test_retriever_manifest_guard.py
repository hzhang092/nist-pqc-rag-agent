from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.versioning import ensure_manifest_compat


def _write_manifest(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "generated_at_utc": "2026-03-01T00:00:00+00:00",
        "git_commit": "abc",
        "config_hash": "xyz",
        "stages": {
            "embed": {"model_name": "m1", "dim": 768},
            "index_bm25": {"tokenizer": "regex_compound_v1"},
        },
        "artifact_hashes": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_guard_passes_when_expected_matches(tmp_path: Path):
    _write_manifest(tmp_path / "manifest.json")
    ensure_manifest_compat(
        processed_dir=tmp_path,
        expected_embed_model="m1",
        expected_embed_dim=768,
        expected_bm25_tokenizer="regex_compound_v1",
    )


def test_manifest_guard_raises_on_embed_model_mismatch(tmp_path: Path):
    _write_manifest(tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="model mismatch"):
        ensure_manifest_compat(processed_dir=tmp_path, expected_embed_model="m2")


def test_manifest_guard_raises_on_bm25_tokenizer_mismatch(tmp_path: Path):
    _write_manifest(tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        ensure_manifest_compat(processed_dir=tmp_path, expected_bm25_tokenizer="other_tok")
