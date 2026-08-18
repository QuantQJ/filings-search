"""OpenSearch index management + bulk indexing."""
from opensearchpy import OpenSearch, helpers
from . import config

_os = None


def client() -> OpenSearch:
    global _os
    if _os is None:
        _os = OpenSearch(hosts=[config.OS_URL], timeout=60, max_retries=3, retry_on_timeout=True)
    return _os


MAPPING = {
    "settings": {
        "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {"analyzer": {"fin_en": {"type": "english"}}},
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "ticker": {"type": "keyword"},
            "cik": {"type": "keyword"},
            "company": {"type": "text", "fields": {"kw": {"type": "keyword"}}},
            "form": {"type": "keyword"},
            "accession": {"type": "keyword"},
            "filed": {"type": "date"},
            "fiscal_year": {"type": "integer"},
            "item": {"type": "keyword"},
            "item_title": {"type": "text", "analyzer": "fin_en"},
            "chunk_idx": {"type": "integer"},
            "url": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "fin_en"},
            "n_tokens": {"type": "integer"},
            "embedding": {
                "type": "knn_vector", "dimension": config.EMBED_DIM,
                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene",
                           "parameters": {"ef_construction": 128, "m": 16}},
            },
        }
    },
}


def ensure_index(recreate: bool = False):
    c = client()
    if recreate and c.indices.exists(index=config.INDEX):
        c.indices.delete(index=config.INDEX)
    if not c.indices.exists(index=config.INDEX):
        c.indices.create(index=config.INDEX, body=MAPPING)


def bulk_index(docs: list[dict]):
    actions = ({"_index": config.INDEX, "_id": d["chunk_id"], "_source": d} for d in docs)
    ok, errs = helpers.bulk(client(), actions, chunk_size=200, raise_on_error=False)
    return ok, errs


def stats() -> dict:
    c = client()
    if not c.indices.exists(index=config.INDEX):
        return {"index": config.INDEX, "exists": False}
    cnt = c.count(index=config.INDEX)["count"]
    agg = c.search(index=config.INDEX, body={"size": 0, "aggs": {
        "tickers": {"terms": {"field": "ticker", "size": 100}},
        "filings": {"cardinality": {"field": "accession"}}}})
    return {"index": config.INDEX, "exists": True, "chunks": cnt,
            "filings": agg["aggregations"]["filings"]["value"],
            "tickers": [b["key"] for b in agg["aggregations"]["tickers"]["buckets"]]}
