# rag/index_faiss.py (build + save index)
from pathlib import Path
import json
import numpy as np
import faiss

OUT_DIR = Path("data/processed")
EMB_PATH = OUT_DIR / "embeddings.npy"
META_PATH = OUT_DIR / "emb_meta.json"
INDEX_PATH = OUT_DIR / "faiss.index"

def main():
    assert EMB_PATH.exists(), f"Missing {EMB_PATH}"
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    dim = int(meta["dim"])

    embs = np.load(EMB_PATH).astype("float32")
    assert embs.shape[1] == dim

    # If you didn't normalize during encoding, do it here:
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized vectors
    index.add(embs)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[OK] saved {INDEX_PATH}  vectors={index.ntotal} dim={dim}")

if __name__ == "__main__":
    main()
