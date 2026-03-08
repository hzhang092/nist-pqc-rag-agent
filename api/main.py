from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rag.config import SETTINGS
from rag.service import ask_question, health_status, search_query


app = FastAPI(title="nist-pqc-rag-agent", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1)


@app.get("/health")
async def get_health() -> dict:
    try:
        return health_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/search")
async def get_search(
    q: str = Query(..., min_length=1),
    k: int = Query(default=SETTINGS.TOP_K, ge=1),
) -> dict:
    try:
        return search_query(q, k=k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
async def post_ask(body: AskRequest) -> dict:
    try:
        payload = ask_question(body.question, k=body.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": payload["answer"],
        "citations": payload["citations"],
        "refusal_reason": payload["refusal_reason"],
        "trace_summary": payload["trace_summary"],
        "timing_ms": payload["timing_ms"],
    }
