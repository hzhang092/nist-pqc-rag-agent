# rag/ask.py
from __future__ import annotations

import argparse
import json

from rag.config import SETTINGS, validate_settings
from rag.llm.gemini import make_generate_fn
from rag.rag_answer import build_cited_answer
from rag.retriever.factory import get_retriever  # assumes you have this from Day 1
from rag.types import AnswerResult


def _format_citations(result: AnswerResult) -> str:
    lines = []
    for c in result.citations:
        lines.append(f"[{c.key}] {c.doc_id} p{c.start_page}-p{c.end_page} chunk_id={c.chunk_id}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(prog="python -m rag.ask")
    parser.add_argument("question", nargs="+", help="Question text (wrap in quotes recommended).")
    parser.add_argument("--show-evidence", action="store_true", default=SETTINGS.ASK_SHOW_EVIDENCE_DEFAULT)
    parser.add_argument("--json", dest="as_json", action="store_true", default=SETTINGS.ASK_JSON_DEFAULT)
    parser.add_argument("--k", type=int, default=None, help="Override TOP_K for this run.")
    parser.add_argument("--backend", type=str, default=None, help="Override VECTOR_BACKEND for this run.")
    args = parser.parse_args()

    validate_settings()

    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit("Empty question.")

    backend = args.backend or SETTINGS.VECTOR_BACKEND
    k = args.k or SETTINGS.TOP_K

    retriever = get_retriever(backend)
    hits = retriever.search(question, k=k)

    if args.show_evidence:
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
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
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
