#!/usr/bin/env python3
"""
Lightweight repo smoke checks for nist-pqc-rag-agent.
Non-destructive by default. Pass --run to execute pipeline commands if found.

Usage:
  python .agents/skills/rag_proj/scripts/smoke.py
  python .agents/skills/rag_proj/scripts/smoke.py --tree
  python .agents/skills/rag_proj/scripts/smoke.py --run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(20):
        if (cur / "project_overview.md").exists() or (cur / ".git").exists() or (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def print_tree(root: Path) -> None:
    # tiny tree printer (no external deps)
    max_depth = 4
    ignore = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules"}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        if any(part in ignore for part in rel.parts):
            continue
        prefix = "  " * (len(rel.parts) - 1)
        print(f"{prefix}- {rel}{'/' if path.is_dir() else ''}")


def exists_any(root: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    return None


def run_cmd(cmd: list[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="store_true", help="Print a shallow repo tree")
    ap.add_argument("--run", action="store_true", help="Attempt to run pipeline entrypoints if present")
    args = ap.parse_args()

    repo = find_repo_root(Path.cwd())
    print(f"Repo root: {repo}")

    if args.tree:
        print_tree(repo)
        return 0

    # Check for expected artifacts (adapt if your filenames differ)
    processed = repo / "data" / "processed"
    expected = [
        processed / "chunks.jsonl",
        processed / "chunk_store.jsonl",
        processed / "faiss.index",
        processed / "embeddings.npy",
    ]
    present = [p for p in expected if p.exists()]
    missing = [p for p in expected if not p.exists()]

    print("\nArtifacts:")
    for p in present:
        print(f"  ✅ {p.relative_to(repo)}")
    for p in missing:
        print(f"  ❌ {p.relative_to(repo)}")

    # Detect likely entrypoints (support v1/v2 naming)
    ingest = exists_any(repo, ["rag/ingest.py", "rag/ingest_v2.py"])
    chunk = exists_any(repo, ["rag/chunk.py"])
    index = exists_any(repo, ["rag/index_faiss.py"])
    search = exists_any(repo, ["rag/search_faiss.py"])

    print("\nEntrypoints found:")
    for name, p in [("ingest", ingest), ("chunk", chunk), ("index_faiss", index), ("search_faiss", search)]:
        print(f"  {name:12s}: {p.relative_to(repo) if p else '—'}")

    if not args.run:
        print("\nTip: re-run with --run to try executing detected entrypoints.")
        return 0

    # Try to run what exists (best-effort, don’t assume package layout)
    if ingest:
        run_cmd([sys.executable, str(ingest.relative_to(repo))], repo)
    if chunk:
        run_cmd([sys.executable, str(chunk.relative_to(repo))], repo)
    if index:
        run_cmd([sys.executable, str(index.relative_to(repo))], repo)
    if search:
        run_cmd([sys.executable, str(search.relative_to(repo)), "ML-KEM key generation"], repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
