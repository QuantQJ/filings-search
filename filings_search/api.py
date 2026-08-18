"""FastAPI surface: /search, /ask, /chunk, /companies, /health."""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import config
from .search import search, get_chunk, neighbors, companies
from .index import stats
from .agent import ask

app = FastAPI(title="filings-search", version="0.1.0",
              description="Hybrid BM25+kNN retrieval and agentic Q&A over SEC 10-K filings (OpenSearch).")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    model: str | None = None
    max_turns: int = Field(12, ge=1, le=25)


@app.get("/health")
def health():
    return {"ok": True, "index": stats(), "embed_backend": config.EMBED_BACKEND, "agent_model": config.AGENT_MODEL}


@app.get("/companies")
def list_companies():
    return companies()


@app.get("/search")
def search_ep(q: str = Query(..., min_length=2), mode: str = Query("hybrid", pattern="^(hybrid|bm25|knn)$"),
              k: int = Query(10, ge=1, le=50), ticker: str | None = None, item: str | None = None,
              fiscal_year: int | None = None):
    hits = search(q, mode=mode, k=k, ticker=ticker, item=item, fiscal_year=fiscal_year)
    for h in hits:
        h.pop("embedding", None)
    return {"query": q, "mode": mode, "hits": hits}


@app.get("/chunk/{chunk_id}")
def chunk_ep(chunk_id: str, expand: int = Query(0, ge=0, le=3)):
    if expand:
        hits = neighbors(chunk_id, span=expand)
        if not hits:
            raise HTTPException(404, "chunk not found")
        return {"chunks": hits}
    d = get_chunk(chunk_id)
    if not d:
        raise HTTPException(404, "chunk not found")
    return d


@app.post("/ask")
def ask_ep(req: AskRequest):
    return ask(req.question, model=req.model, max_turns=req.max_turns)
