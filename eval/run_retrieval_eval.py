#!/usr/bin/env python3
"""Retrieval eval: hit@1 / hit@5 / MRR@10 for bm25 vs knn vs hybrid on a labeled query set.

A hit = result whose (ticker, item) matches the label (item=None → ticker only) AND
whose text contains the `must` phrase (case-insensitive). Runs unfiltered (the agent
may filter, but this measures raw ranking quality) and with a ticker filter.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filings_search.search import search  # noqa: E402

QS = json.loads((Path(__file__).parent / "queries.json").read_text())
MODES = ["bm25", "knn", "hybrid"]


def relevant(h, q):
    if h["ticker"] != q["ticker"]:
        return False
    if q["item"] and h["item"] != q["item"]:
        return False
    return q["must"].lower() in h["text"].lower()


def evaluate(filtered: bool):
    rows = {m: {"hit1": 0, "hit5": 0, "mrr": 0.0, "lat": 0.0} for m in MODES}
    for q in QS:
        for m in MODES:
            t0 = time.time()
            hits = search(q["q"], mode=m, k=10, ticker=q["ticker"] if filtered else None)
            rows[m]["lat"] += time.time() - t0
            rank = next((i + 1 for i, h in enumerate(hits) if relevant(h, q)), None)
            if rank:
                rows[m]["mrr"] += 1.0 / rank
                rows[m]["hit1"] += rank == 1
                rows[m]["hit5"] += rank <= 5
    n = len(QS)
    out = {}
    for m in MODES:
        r = rows[m]
        out[m] = {"hit@1": round(r["hit1"] / n, 3), "hit@5": round(r["hit5"] / n, 3),
                  "MRR@10": round(r["mrr"] / n, 3), "avg_ms": round(1000 * r["lat"] / n, 1)}
    return out


if __name__ == "__main__":
    res = {"n_queries": len(QS), "unfiltered": evaluate(False), "ticker_filtered": evaluate(True)}
    print(json.dumps(res, indent=2))
    (Path(__file__).parent / "retrieval_results.json").write_text(json.dumps(res, indent=2))
    print("\n| setting | mode | hit@1 | hit@5 | MRR@10 | avg ms |\n|---|---|---|---|---|---|")
    for setting in ("unfiltered", "ticker_filtered"):
        for m in MODES:
            r = res[setting][m]
            print(f"| {setting} | {m} | {r['hit@1']} | {r['hit@5']} | {r['MRR@10']} | {r['avg_ms']} |")
