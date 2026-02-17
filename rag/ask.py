"""
Command-line interface for the question-answering system.

This script serves as the main entry point for interacting with the RAG system.
It orchestrates the entire process:
1.  Parses command-line arguments for the user's question and other options
    (e.g., showing evidence, JSON output).
2.  Initializes the specified retriever backend.
3.  Performs a search to find relevant document chunks (evidence).
4.  Optionally displays the retrieved evidence.
5.  Initializes the language model generation function.
6.  Calls `build_cited_answer` to generate a validated, citation-grounded answer.
7.  Formats and prints the final answer and citations to the console in either
    human-readable or JSON format.

How to use:
    python -m rag.ask "What does FIPS 203 say about ML-KEM key generation?"
    python -m rag.ask "Compare ML-DSA and SLH-DSA use-cases" --show-evidence
    python -m rag.ask "What is Algorithm 19?" --mode hybrid --k 8 --json
    python -m rag.ask "KeyGen details" --mode base --backend bm25 --no-query-fusion

Used by:
    - Direct CLI entrypoint for end-to-end retrieval + generation.
    - Calls `rag.retrieve.retrieve` for evidence retrieval.
    - Calls `rag.rag_answer.build_cited_answer` for citation-grounded answer synthesis.
    - Uses `rag.llm.gemini.make_generate_fn` for model invocation.

Usage:
    python -m rag.ask "Your question here" [--show-evidence] [--json]
"""
# rag/ask.py
from __future__ import annotations

import argparse
import json

from rag.config import SETTINGS, validate_settings
from rag.llm.gemini import get_model_name, make_generate_fn
from rag.rag_answer import build_cited_answer
from rag.retrieve import retrieve
from rag.types import AnswerResult


def _format_citations(result: AnswerResult) -> str:
    """Formats the list of citations into a human-readable string."""
    lines = []
    for c in result.citations:
        lines.append(f"[{c.key}] {c.doc_id} p{c.start_page}-p{c.end_page} chunk_id={c.chunk_id}")
    return "\n".join(lines)


def main():
    """
    Parses arguments, retrieves evidence, generates a cited answer, and prints it.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.ask")
    parser.add_argument("question", nargs="+", help="Question text (wrap in quotes recommended).")
    parser.add_argument("--show-evidence", action="store_true", default=SETTINGS.ASK_SHOW_EVIDENCE_DEFAULT)
    parser.add_argument("--json", dest="as_json", action="store_true", default=SETTINGS.ASK_JSON_DEFAULT)
    parser.add_argument("--k", type=int, default=None, help="Override TOP_K for this run.")
    parser.add_argument("--backend", type=str, default=None, help="Override VECTOR_BACKEND for this run.")
    parser.add_argument(
        "--mode",
        type=str,
        default=SETTINGS.RETRIEVAL_MODE,
        choices=["base", "hybrid"],
        help="Retrieval mode: base backend or hybrid (faiss+bm25).",
    )
    parser.add_argument(
        "--no-query-fusion",
        action="store_true",
        help="Disable deterministic query variant fusion.",
    )
    args = parser.parse_args()

    validate_settings()

    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit("Empty question.")

    backend = args.backend or SETTINGS.VECTOR_BACKEND
    k = args.k or SETTINGS.TOP_K
    model_name = get_model_name()

    hits = retrieve(
        query=question,
        k=k,
        mode=args.mode,
        backend=backend,
        use_query_fusion=not args.no_query_fusion,
    )

    if args.show_evidence:
        print(f"\n=== Model ===\n{model_name}")
        print("\n=== Evidence (top hits) ===")
        for i, h in enumerate(hits, start=1):
            preview = (h.text or "").strip().replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            print(f"{i:02d}. score={h.score:.4f} {h.doc_id} p{h.start_page}-p{h.end_page} chunk_id={h.chunk_id}")
            print(f"    {preview}")

    generate_fn = make_generate_fn()
    result = build_cited_answer(question=question, hits=hits, generate_fn=generate_fn)

    if args.as_json:
        payload = result.to_dict()
        payload["model"] = model_name
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Citations ===")
    if result.citations:
        print(_format_citations(result))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
