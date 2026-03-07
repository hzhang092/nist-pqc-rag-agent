from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rag.config import SETTINGS


SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = Path("data/processed/manifest.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def compute_settings_hash() -> str:
    payload = _stable_json_dumps(asdict(SETTINGS)).encode("utf-8")
    return _sha256_bytes(payload)


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        commit = out.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _artifact_key(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return path.as_posix()
    try:
        rel = path.resolve().relative_to(base_dir.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def compute_artifact_hashes(
    artifact_paths: Iterable[str | Path],
    *,
    base_dir: str | Path | None = "data/processed",
) -> dict[str, str]:
    base = Path(base_dir) if base_dir is not None else None
    out: dict[str, str] = {}
    for raw in artifact_paths:
        p = Path(raw)
        digest = file_sha256(p)
        if digest is None:
            continue
        out[_artifact_key(p, base)] = digest
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def build_base_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "git_commit": get_git_commit(),
        "config_hash": compute_settings_hash(),
        "stages": {},
        "artifact_hashes": {},
    }


def load_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any] | None:
    path = Path(manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any], manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json_dumps(manifest) + "\n", encoding="utf-8")


def update_manifest(
    *,
    stage_name: str,
    stage_payload: dict[str, Any],
    artifact_paths: Iterable[str | Path] | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path) or build_base_manifest()

    manifest["schema_version"] = SCHEMA_VERSION
    manifest["generated_at_utc"] = _utc_now_iso()
    manifest["git_commit"] = get_git_commit()
    manifest["config_hash"] = compute_settings_hash()

    stages = manifest.setdefault("stages", {})
    stages[stage_name] = stage_payload
    manifest["stages"] = dict(sorted(stages.items(), key=lambda kv: kv[0]))

    artifact_hashes = dict(manifest.get("artifact_hashes", {}))
    if artifact_paths:
        merged = compute_artifact_hashes(artifact_paths, base_dir=Path(manifest_path).parent)
        artifact_hashes.update(merged)
    manifest["artifact_hashes"] = dict(sorted(artifact_hashes.items(), key=lambda kv: kv[0]))

    write_manifest(manifest, manifest_path=manifest_path)
    return manifest


def ensure_manifest_compat(
    *,
    processed_dir: str | Path = "data/processed",
    expected_embed_model: str | None = None,
    expected_embed_dim: int | None = None,
    expected_bm25_tokenizer: str | None = None,
) -> None:
    manifest_path = Path(processed_dir) / "manifest.json"
    manifest = load_manifest(manifest_path)
    if manifest is None:
        warnings.warn(
            f"Manifest not found at {manifest_path}. Compatibility checks skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    stages = manifest.get("stages", {})
    embed_stage = stages.get("embed", {})
    bm25_stage = stages.get("index_bm25", {})

    if expected_embed_model and embed_stage.get("model_name"):
        if str(embed_stage["model_name"]) != str(expected_embed_model):
            raise ValueError(
                "Manifest/embed model mismatch: "
                f"manifest={embed_stage['model_name']!r} expected={expected_embed_model!r}"
            )

    if expected_embed_dim is not None and embed_stage.get("dim") is not None:
        if int(embed_stage["dim"]) != int(expected_embed_dim):
            raise ValueError(
                "Manifest/embed dim mismatch: "
                f"manifest={embed_stage['dim']!r} expected={expected_embed_dim!r}"
            )

    if expected_bm25_tokenizer and bm25_stage.get("tokenizer"):
        if str(bm25_stage["tokenizer"]) != str(expected_bm25_tokenizer):
            raise ValueError(
                "Manifest/BM25 tokenizer mismatch: "
                f"manifest={bm25_stage['tokenizer']!r} expected={expected_bm25_tokenizer!r}"
            )
