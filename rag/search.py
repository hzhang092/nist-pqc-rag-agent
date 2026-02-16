# rag/search.py 
# backend-agnostic
import sys
from rag.retriever.factory import get_retriever

TOP_K = 5
BACKEND = "faiss"  # later: "pgvector"

def main():
    qtext = " ".join(sys.argv[1:]).strip()
    if not qtext:
        print('Usage: python -m rag.search "your question"')
        sys.exit(1)

    retriever = get_retriever(BACKEND)
    hits = retriever.search(qtext, k=TOP_K)

    print(f"\nQuery: {qtext}\n")
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] score={h.score:.4f}  {h.doc_id}  p{h.start_page}-p{h.end_page}  ({h.chunk_id})")
        if h.text:
            preview = h.text[:300].replace("\n", " ")
            print(f"    {preview}...")
        print()

if __name__ == "__main__":
    main()
