"""
LangGraph agent CLI entrypoint.

What this script is for:
- Runs the bounded tool-using agent graph (`rag.lc.graph.run_agent`) for one question.
- Prints either a human-readable answer with citations or full AgentState JSON.
- Optionally writes a trace artifact (`rag.lc.trace.write_trace`) for debugging/eval.

Usage examples:
    python -m rag.agent.ask "What are the steps in Algorithm 2 SHAKE128?"
    python -m rag.agent.ask "Compare ML-DSA and SLH-DSA" --json
    python -m rag.agent.ask "ML-KEM key generation" --out-dir runs/agent/day3
    python -m rag.agent.ask "What is ML-KEM?" --no-trace

Flags:
    question (positional)
        User question string sent to the agent.
    --no-trace
        Do not write trace JSON after execution.
    --out-dir PATH
        Directory where trace JSON is written when tracing is enabled.
        Default: runs/agent
    --json
        Print full AgentState JSON to stdout instead of formatted answer/citations.
"""
# rag/agent/ask.py
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from rag.lc.graph import run_agent
from rag.lc.trace import write_trace


def _print_answer(state: Dict[str, Any]) -> None:
    ans = (state.get("final_answer") or state.get("draft_answer") or "").strip()
    print(ans)
    print()

    cits = state.get("citations", []) or []
    if cits:
        print("Citations:")
        for c in cits:
            key = c.get("key")
            doc = c.get("doc_id")
            sp = c.get("start_page")
            ep = c.get("end_page")
            cid = c.get("chunk_id", "")
            prefix = f"[{key}] " if key else ""
            print(f"- {prefix}{doc} p{sp}–{ep} (chunk={cid})")
    else:
        print("Citations: (none)")


def main():
    ap = argparse.ArgumentParser(description="Ask the LangGraph agent (tool-using RAG).")
    ap.add_argument("question", type=str, help="User question")
    ap.add_argument("--no-trace", action="store_true", help="Do not write trace JSON")
    ap.add_argument("--out-dir", type=str, default="runs/agent", help="Trace output directory")
    ap.add_argument("--json", action="store_true", help="Print full AgentState as JSON to stdout")
    args = ap.parse_args()

    state = run_agent(args.question)

    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        _print_answer(state)

    if not args.no_trace:
        path = write_trace(state, out_dir=args.out_dir)
        print()
        print(f"Trace saved: {path}")


if __name__ == "__main__":
    main()
