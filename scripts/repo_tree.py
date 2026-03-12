#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_IGNORES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/repo_tree.py",
        description="Print a deterministic tree of the repository contents.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Directory to render. Defaults to the repository root.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth to print relative to root. Omit for full depth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to write the tree output to.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Directory or file name to ignore. Repeatable.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories except explicit ignores.",
    )
    return parser.parse_args()


def should_skip(path: Path, root: Path, ignore_names: set[str], include_hidden: bool) -> bool:
    if path == root:
        return False

    rel = path.relative_to(root)
    if any(part in ignore_names for part in rel.parts):
        return True

    if include_hidden:
        return False

    return any(part.startswith(".") for part in rel.parts)


def iter_children(directory: Path, root: Path, ignore_names: set[str], include_hidden: bool) -> list[Path]:
    children = [
        child
        for child in directory.iterdir()
        if not should_skip(child, root=root, ignore_names=ignore_names, include_hidden=include_hidden)
    ]
    return sorted(children, key=lambda child: (not child.is_dir(), child.name.lower(), child.name))


def render_tree(root: Path, max_depth: int | None, ignore_names: set[str], include_hidden: bool) -> str:
    lines = [f"{root.name}/"]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if max_depth is not None and depth >= max_depth:
            return

        children = iter_children(directory, root=root, ignore_names=ignore_names, include_hidden=include_hidden)
        last_index = len(children) - 1
        for index, child in enumerate(children):
            branch = "`-- " if index == last_index else "|-- "
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{branch}{child.name}{suffix}")
            if child.is_dir():
                extension = "    " if index == last_index else "|   "
                walk(child, prefix + extension, depth + 1)

    walk(root, prefix="", depth=0)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Path is not a directory: {root}")
    if args.max_depth is not None and args.max_depth < 0:
        raise SystemExit("--max-depth must be >= 0")

    ignore_names = DEFAULT_IGNORES | set(args.ignore)
    tree = render_tree(
        root=root,
        max_depth=args.max_depth,
        ignore_names=ignore_names,
        include_hidden=args.include_hidden,
    )

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(tree, encoding="utf-8")

    print(tree, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())