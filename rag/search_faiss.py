# rag/search_faiss.py
import sys, json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

OUT_DIR = Path("data/processed")
META_PATH = OUT_DIR / "emb_meta.json"
INDEX_PATH = OUT_DIR / "faiss.index"
STORE_PATH = OUT_DIR / "chunk_store.jsonl"

TOP_K = 5

def load_store():
    store = {}
    with STORE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            store[int(rec["vector_id"])] = rec
    return store

def main():
    qtext = " ".join(sys.argv[1:]).strip()
    if not qtext:
        print('Usage: python -m rag.search_faiss "your question"')
        sys.exit(1)

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model_name = meta["model_name"]
    dim = int(meta["dim"])

    store = load_store()
    model = SentenceTransformer(model_name)

    q = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    assert q.shape[1] == dim

    index = faiss.read_index(str(INDEX_PATH))
    scores, ids = index.search(q, TOP_K)  # scores are inner products (cosine if normalized)

    print(f"\nQuery: {qtext}\n")
    for rank, (vid, score) in enumerate(zip(ids[0], scores[0]), start=1):
        rec = store[int(vid)]
        print(f"[{rank}] score={float(score):.4f}  {rec['doc_id']}  p{rec['start_page']}-p{rec['end_page']}  ({rec['chunk_id']})")
        # Optional preview if you stored text:
        if "text" in rec:
            preview = rec["text"][:300].replace("\n", " ")
            print(f"    {preview}...")
        print()

if __name__ == "__main__":
    main()
