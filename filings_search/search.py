"""Retrieval: BM25, kNN, and hybrid (reciprocal rank fusion) with metadata filters."""
from . import config
from .index import client
from .embed import embed

_SRC = ["chunk_id", "ticker", "cik", "company", "form", "accession", "filed",
        "fiscal_year", "item", "item_title", "chunk_idx", "url", "text", "n_tokens"]


def _filters(ticker=None, cik=None, item=None, fiscal_year=None, items=None) -> list[dict]:
    f = []
    if ticker:
        f.append({"term": {"ticker": ticker.upper()}})
    if cik:
        f.append({"term": {"cik": str(cik).zfill(10)}})
    if item:
        f.append({"term": {"item": item.upper()}})
    if items:
        f.append({"terms": {"item": [i.upper() for i in items]}})
    if fiscal_year:
        f.append({"term": {"fiscal_year": int(fiscal_year)}})
    return f


def _hits(resp) -> list[dict]:
    out = []
    for h in resp["hits"]["hits"]:
        d = h["_source"]
        d["score"] = h["_score"]
        out.append(d)
    return out


def bm25(query: str, k: int = 10, **filt) -> list[dict]:
    body = {
        "size": k, "_source": _SRC,
        "query": {"bool": {
            "must": [{"multi_match": {"query": query, "fields": ["text^3", "item_title", "company"],
                                      "type": "best_fields", "operator": "or"}}],
            "filter": _filters(**filt),
        }},
    }
    return _hits(client().search(index=config.INDEX, body=body))


def knn(query: str, k: int = 10, **filt) -> list[dict]:
    vec = embed([query], kind="query")[0]
    knn_q = {"vector": vec, "k": max(k * 4, 50)}
    f = _filters(**filt)
    if f:
        knn_q["filter"] = {"bool": {"filter": f}}
    body = {"size": k, "_source": _SRC, "query": {"knn": {"embedding": knn_q}}}
    return _hits(client().search(index=config.INDEX, body=body))


def rrf(lists: list[list[dict]], k: int = 10, c: int = 60) -> list[dict]:
    scores, docs = {}, {}
    for lst in lists:
        for rank, d in enumerate(lst):
            cid = d["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + rank + 1)
            docs.setdefault(cid, d)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    out = []
    for cid, s in ranked:
        d = dict(docs[cid]); d["score"] = s
        out.append(d)
    return out


def hybrid(query: str, k: int = 10, depth: int = 30, **filt) -> list[dict]:
    return rrf([bm25(query, depth, **filt), knn(query, depth, **filt)], k=k)


def search(query: str, mode: str = "hybrid", k: int = 10, **filt) -> list[dict]:
    if mode == "bm25":
        return bm25(query, k, **filt)
    if mode == "knn":
        return knn(query, k, **filt)
    return hybrid(query, k, **filt)


def get_chunk(chunk_id: str) -> dict | None:
    r = client().search(index=config.INDEX, body={"size": 1, "_source": _SRC,
                                                   "query": {"term": {"chunk_id": chunk_id}}})
    hits = _hits(r)
    return hits[0] if hits else None


def neighbors(chunk_id: str, span: int = 1) -> list[dict]:
    """Adjacent chunks in the same filing/item (for context expansion)."""
    d = get_chunk(chunk_id)
    if not d:
        return []
    lo, hi = d["chunk_idx"] - span, d["chunk_idx"] + span
    r = client().search(index=config.INDEX, body={"size": 2 * span + 1, "_source": _SRC,
        "query": {"bool": {"filter": [{"term": {"accession": d["accession"]}}, {"term": {"item": d["item"]}},
                                      {"range": {"chunk_idx": {"gte": lo, "lte": hi}}}]}},
        "sort": [{"chunk_idx": "asc"}]})
    return _hits(r)


def companies() -> list[dict]:
    r = client().search(index=config.INDEX, body={"size": 0, "aggs": {"t": {"terms": {"field": "ticker", "size": 500},
        "aggs": {"c": {"terms": {"field": "company.kw", "size": 1}}, "fy": {"terms": {"field": "fiscal_year", "size": 20}}}}}})
    out = []
    for b in r["aggregations"]["t"]["buckets"]:
        out.append({"ticker": b["key"], "company": (b["c"]["buckets"] or [{"key": ""}])[0]["key"],
                    "fiscal_years": sorted(x["key"] for x in b["fy"]["buckets"]), "chunks": b["doc_count"]})
    return out
