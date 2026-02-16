# rag/embed.py
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import orjson
    def dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    def dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)

from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64

EMB_NPY = OUT_DIR / "embeddings.npy"
STORE_JSONL = OUT_DIR / "chunk_store.jsonl"
META_JSON = OUT_DIR / "emb_meta.json"

def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks

def main():
    assert CHUNKS_PATH.exists(), f"Missing {CHUNKS_PATH}"

    chunks = load_chunks(CHUNKS_PATH)
    assert chunks, "No chunks loaded."

    # Build vector_id mapping (int IDs are required by most ANN indexes)
    texts = []
    store_records = []
    for vid, ch in enumerate(chunks):
        text = ch["text"].strip()
        if not text:
            continue

        texts.append(text)

        store_records.append({
            "vector_id": vid,
            "chunk_id": ch.get("chunk_id"),
            "doc_id": ch.get("doc_id"),
            "start_page": ch.get("start_page", ch.get("page_number")),
            "end_page": ch.get("end_page", ch.get("page_number")),
            "page_number": ch.get("page_number"),
            "char_len": ch.get("char_len"),
            "approx_tokens": ch.get("approx_tokens"),
            "text": ch["text"]
        })

    print(f"Loaded {len(texts)} non-empty chunks")

    model = SentenceTransformer(MODEL_NAME)

    # Encode (normalize so cosine similarity works cleanly)
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(EMB_NPY, embs)

    with STORE_JSONL.open("w", encoding="utf-8") as f:
        for rec in store_records:
            f.write(dumps(rec) + "\n")

    meta = {
        "model_name": MODEL_NAME,
        "num_vectors": int(embs.shape[0]),
        "dim": int(embs.shape[1]),
        "normalized": True,
        "chunks_path": str(CHUNKS_PATH),
        "embeddings_path": str(EMB_NPY),
        "store_path": str(STORE_JSONL),
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] embeddings: {EMB_NPY}  shape={embs.shape}")
    print(f"[OK] store:      {STORE_JSONL}")
    print(f"[OK] meta:       {META_JSON}")

if __name__ == "__main__":
    main()
