# rag-proj skill references

Source of truth:
- `project_overview.md` (repo root)

Key design constraints:
- Preserve page spans for citations (start_page/end_page on every chunk).
- Keep retrieval backend swappable (Retriever interface + config selection).
- Default retrieval is hybrid (BM25 + vector) with fusion (RRF).

If you’re debugging:
- Start from logs + smallest repro command.
- Verify artifacts under `data/processed/`.
- Confirm chunk sizes aren’t huge (focused chunks win retrieval).

If you’re improving:
- Run eval, record baseline JSON, then change one thing and compare deltas.
