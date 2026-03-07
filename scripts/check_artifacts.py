from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.versioning import compute_settings_hash, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="python scripts/check_artifacts.py")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    manifest_path = processed_dir / "manifest.json"

    manifest = load_manifest(manifest_path)
    if manifest is None:
        print(f"[REBUILD NEEDED] missing manifest: {manifest_path}")
        return 1

    manifest_hash = str(manifest.get("config_hash", "") or "")
    current_hash = compute_settings_hash()

    if manifest_hash != current_hash:
        print("[REBUILD NEEDED] config hash mismatch")
        print(f"  manifest: {manifest_hash}")
        print(f"  current : {current_hash}")
        return 1

    print("[OK] artifact config hash matches current settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
