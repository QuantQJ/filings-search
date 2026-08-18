#!/usr/bin/env python3
"""Ingest 10-K filings into OpenSearch.

    python ingest.py AAPL MSFT NVDA --filings 1
    python ingest.py --recreate AAPL
"""
import argparse
import sys
import time

from filings_search import config
from filings_search.edgar import fetch_filings
from filings_search.parse import split_items
from filings_search.chunk import chunk_text, chunk_id, n_tokens
from filings_search.embed import embed
from filings_search.index import ensure_index, bulk_index, stats


def ingest_ticker(ticker: str, n_filings: int) -> int:
    total = 0
    for rec in fetch_filings(ticker, n=n_filings):
        f = rec["filing"]
        sections = split_items(rec["text"])
        docs = []
        for sec in sections:
            pieces = chunk_text(sec["text"], config.CHUNK_TOKENS, config.CHUNK_OVERLAP)
            for i, p in enumerate(pieces):
                docs.append({
                    "chunk_id": chunk_id(f["accession"], sec["item"], i),
                    "ticker": f["ticker"], "cik": f["cik"], "company": f["company"], "form": f["form"],
                    "accession": f["accession"], "filed": f["filed"], "fiscal_year": f["fiscal_year"],
                    "item": sec["item"], "item_title": sec["item_title"], "chunk_idx": i,
                    "url": f["url"], "text": p, "n_tokens": n_tokens(p),
                })
        print(f"  {f['ticker']} FY{f['fiscal_year']} {f['accession']}: {len(sections)} items, {len(docs)} chunks "
              f"[{', '.join(s['item'] for s in sections)}]", flush=True)
        t0 = time.time()
        for i in range(0, len(docs), 128):
            batch = docs[i:i + 128]
            vecs = embed([d["text"] for d in batch], kind="document")
            for d, v in zip(batch, vecs):
                d["embedding"] = v
            bulk_index(batch)
            print(f"    indexed {min(i+128, len(docs))}/{len(docs)}  ({time.time()-t0:.0f}s)", end="\r", flush=True)
        print()
        total += len(docs)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--filings", type=int, default=1, help="how many most-recent 10-Ks per ticker")
    ap.add_argument("--recreate", action="store_true", help="drop and recreate the index first")
    a = ap.parse_args()
    ensure_index(recreate=a.recreate)
    grand = 0
    for t in a.tickers:
        print(f"== {t}")
        try:
            grand += ingest_ticker(t, a.filings)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {t}: {e}", file=sys.stderr)
    from filings_search.index import client
    client().indices.refresh(index=config.INDEX)
    print(f"done: {grand} chunks this run; index = {stats()}")


if __name__ == "__main__":
    main()
