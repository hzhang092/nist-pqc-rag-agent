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
CANDIDATES_K = 25          # retrieve more, then dedupe down
MAX_HITS_PER_PAGE = 1      # tighten to 1; set 2 if you want slightly more density

def load_store():
    store = {}
    with STORE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            store[int(rec["vector_id"])] = rec
    return store

def page_key(rec):
    # Dedupe at page-span level; you can change to (doc_id, page_number) if you prefer
    return (rec.get("doc_id"), rec.get("start_page"), rec.get("end_page"))

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
    scores, ids = index.search(q, CANDIDATES_K)  # search more, then dedupe

    # Dedupe logic
    kept = []
    page_counts = {}  # key -> count
    for vid, score in zip(ids[0], scores[0]):
        if vid < 0:
            continue  # FAISS uses -1 for missing sometimes
        rec = store.get(int(vid))
        if rec is None:
            continue

        key = page_key(rec)
        page_counts[key] = page_counts.get(key, 0) + 1
        if page_counts[key] > MAX_HITS_PER_PAGE:
            continue

        kept.append((int(vid), float(score), rec))
        if len(kept) >= TOP_K:
            break

    print(f"\nQuery: {qtext}\n")
    for rank, (vid, score, rec) in enumerate(kept, start=1):
        print(
            f"[{rank}] score={score:.4f}  {rec['doc_id']}  "
            f"p{rec['start_page']}-p{rec['end_page']}  ({rec['chunk_id']})"
        )
        if "text" in rec:
            preview = rec["text"][:300].replace("\n", " ")
            print(f"    {preview}...")
        print()

    if len(kept) < TOP_K:
        print(f"[WARN] Only {len(kept)} unique results after dedupe (try raising CANDIDATES_K).")

if __name__ == "__main__":
    main()
