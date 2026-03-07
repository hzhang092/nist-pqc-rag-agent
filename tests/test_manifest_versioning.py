from __future__ import annotations

from pathlib import Path

from rag.versioning import load_manifest, update_manifest


def test_update_manifest_records_stages_and_hashes(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    update_manifest(
        stage_name="ingest",
        stage_payload={"parser_backend": "llamaparse"},
        artifact_paths=[artifact],
        manifest_path=manifest_path,
    )
    update_manifest(
        stage_name="chunk",
        stage_payload={"chunker_version": "v2"},
        artifact_paths=[],
        manifest_path=manifest_path,
    )

    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.get("schema_version") == 1
    assert "config_hash" in manifest
    assert "ingest" in manifest.get("stages", {})
    assert "chunk" in manifest.get("stages", {})
    assert "artifact.txt" in manifest.get("artifact_hashes", {})
