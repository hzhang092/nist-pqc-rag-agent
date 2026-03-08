"""
Command-line interface for the question-answering system.

This script serves as the main entry point for interacting with the RAG system.
It orchestrates the entire process:
1.  Parses command-line arguments for the user's question and other options
    (e.g., showing evidence, JSON output, retrieval mode).
2.  Calls `rag.service.ask_question` which handles:
    - Retriever backend initialization
    - Evidence retrieval (with optional hybrid fusion and query expansion)
    - LLM-based answer generation with citation grounding
3.  Formats and prints the final answer, citations, and timing to the console
    in either human-readable or JSON format.

How to use:
    python -m rag.ask "What does FIPS 203 say about ML-KEM key generation?"
    python -m rag.ask "Compare ML-DSA and SLH-DSA use-cases" --show-evidence
    python -m rag.ask "What is Algorithm 19?" --mode hybrid --k 8 --json
    python -m rag.ask "KeyGen details" --mode base --backend bm25 --no-query-fusion
    python -m rag.ask "ML-KEM.Decaps" --candidate-multiplier 6 --k0 55 --rerank-pool 40
    python -m rag.ask "Algorithm 19" --no-rerank --save-json results/answer.json

Flags:
    --show-evidence
        Print retrieved evidence snippets before answer generation.
    --json
        Print machine-readable JSON output instead of formatted text.
    --no-evidence
        In --json mode, omit raw retrieved evidence from the payload.
    --save-json PATH
        In --json mode, also write the JSON payload to PATH.
    --k
        Override final number of retrieved hits (default: SETTINGS.TOP_K).
    --backend
        Override vector backend used by base mode (default: SETTINGS.VECTOR_BACKEND).
    --mode
        Retrieval mode: "base" (single backend) or "hybrid" (faiss+bm25 fusion).
    --no-query-fusion
        Disable deterministic query variant expansion before retrieval.
    --k0
        Reciprocal Rank Fusion constant; controls rank sensitivity (default: SETTINGS.RETRIEVAL_RRF_K0).
    --candidate-multiplier
        Candidate expansion factor before fusion (default: SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER).
    --rerank-pool
        Number of fused candidates considered before rerank truncates to k (default: SETTINGS.RETRIEVAL_RERANK_POOL).
    --no-rerank
        Disable lightweight lexical rerank over fused candidates.

Architecture:
    - Calls `rag.service.ask_question` for orchestration
    - Uses `rag.config.SETTINGS` for default configuration
    - Returns payload with answer, citations, evidence, timing, and trace data
    - All page-level citations preserved per data contract (doc_id, start_page, end_page)

Output format:
    Human-readable: Formatted answer, citations, and timing
    JSON: Complete payload including retrieval metadata and optional evidence
"""

# rag/ask.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import SETTINGS, validate_settings
from rag.service import ask_question


def _format_citations(citations: list[dict]) -> str:
    """Formats the list of citations into a human-readable string."""
    lines = []
    for c in citations:
        lines.append(
            f"[{c['key']}] {c['doc_id']} p{c['start_page']}-p{c['end_page']} chunk_id={c['chunk_id']}"
        )
    return "\n".join(lines)


def main():
    """
    Parses arguments, retrieves evidence, generates a cited answer, and prints it.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.ask")
    parser.add_argument("question", nargs="+", help="Question text (wrap in quotes recommended).")
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        default=SETTINGS.ASK_SHOW_EVIDENCE_DEFAULT,
        help="Print retrieved evidence snippets before answer generation.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=SETTINGS.ASK_JSON_DEFAULT,
        help="Print output as JSON.",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="In --json mode, omit retrieved evidence from the output payload.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="If set, write the JSON payload to this path (in addition to printing).",
    )
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
    parser.add_argument("--k0", type=int, default=SETTINGS.RETRIEVAL_RRF_K0, help="RRF constant (1/(k0+rank)).")
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=SETTINGS.RETRIEVAL_CANDIDATE_MULTIPLIER,
        help="Per-source candidate expansion before fusion (k * multiplier).",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=SETTINGS.RETRIEVAL_RERANK_POOL,
        help="Number of fused candidates considered before final rerank truncation.",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Disable lightweight lexical reranking.")
    args = parser.parse_args()

    validate_settings()

    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit("Empty question.")

    backend = args.backend or SETTINGS.VECTOR_BACKEND
    k = args.k or SETTINGS.TOP_K
    payload = ask_question(
        question=question,
        k=k,
        mode=args.mode,
        vector_backend=backend,
        use_query_fusion=not args.no_query_fusion,
        candidate_multiplier=args.candidate_multiplier,
        k0=args.k0,
        enable_rerank=not args.no_rerank,
        rerank_pool=args.rerank_pool,
    )
    model_name = str(payload["llm_model"])

    if args.show_evidence:
        print(f"\n=== Model ===\n{payload['llm_backend']} / {model_name}")
        print("\n=== Evidence (top hits) ===")
        for i, h in enumerate(payload["evidence"], start=1):
            print(
                f"{i:02d}. score={h['score']:.4f} {h['doc_id']} "
                f"p{h['start_page']}-p{h['end_page']} chunk_id={h['chunk_id']}"
            )
            print(f"    {h['preview_text']}")

    if args.as_json:
        json_payload = {
            "answer": payload["answer"],
            "citations": payload["citations"],
            "refusal_reason": payload["refusal_reason"],
            "trace_summary": payload["trace_summary"],
            "timing_ms": payload["timing_ms"],
            "model": model_name,
            "llm_backend": payload["llm_backend"],
            "question": question,
            "retrieval": payload["retrieval"],
        }
        if not args.no_evidence:
            json_payload["evidence"] = payload["evidence"]
        print(json.dumps(json_payload, ensure_ascii=False, indent=2))
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(
                json.dumps(json_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return

    print("\n=== Answer ===")
    print(payload["answer"])

    print("\n=== Citations ===")
    if payload["citations"]:
        print(_format_citations(payload["citations"]))
    else:
        print("(none)")

    print("\n=== Timing (ms) ===")
    print(json.dumps(payload["timing_ms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
