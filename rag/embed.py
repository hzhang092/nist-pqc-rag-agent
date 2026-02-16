"""
Embedding Generation Script.

This script is responsible for generating vector embeddings for the document
chunks created by the chunking process. It uses a sentence-transformer model
to convert the text of each chunk into a high-dimensional vector.

The key steps are:
1.  Load the document chunks from `chunks.jsonl`.
2.  Initialize a specified sentence-transformer model (e.g., BAAI/bge-base-en-v1.5).
3.  Generate embeddings for all chunk texts in batches. The embeddings are
    normalized to facilitate cosine similarity calculations in the retrieval step.
4.  Save the generated embeddings as a NumPy array (`.npy` file).
5.  Create a "chunk store" (`.jsonl` file) that maps integer vector IDs to the
    original chunk metadata and text. This is essential for retrieving the
    source content after a vector search.
6.  Write a metadata file (`.json`) that contains information about the
    embedding process, including the model name, vector dimensions, and paths
    to the generated files.

This script is a crucial part of the ingestion pipeline, preparing the data for
efficient similarity search by a vector index.
"""
# rag/embed.py
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import orjson
    def dumps(obj) -> str:
        """
        Serializes a Python object to a JSON-formatted string.
        Uses the highly optimized `orjson` if available, otherwise falls back
        to the standard `json` library.
        """
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    def dumps(obj) -> str:
        """
        Serializes a Python object to a JSON-formatted string.
        Uses the standard `json` library.
        """
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
    """Loads document chunks from a JSONL file."""
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks

def main():
    """
    Main function to generate and save embeddings.

    This function orchestrates the entire embedding generation process:
    - Loads chunks.
    - Prepares text data and metadata for storage.
    - Initializes and runs the sentence-transformer model.
    - Saves the embeddings, chunk store, and metadata to disk.
    """
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
