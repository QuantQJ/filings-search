# filings-search

Hybrid (BM25 + kNN) **agentic retrieval** over SEC 10-K filings, on **OpenSearch**, with a
Claude tool-use agent that plans its own searches and answers with per-claim citations and a
hard numeric-grounding check. FastAPI surface + TypeScript MCP tool server. Built as the
OpenSearch successor to my Qdrant pipeline; the cite-or-abstain grounding is ported from my
12-domain provenance wrapper.

```
EDGAR ──► connector ──► section splitter ──► chunker ──► embeddings ──► OpenSearch (BM25 + HNSW kNN)
(SEC API)  (ticker→CIK,   (Item 1/1A/1C/7/8…)  (450 tok,    (nomic-embed-text            │
            10-K list,     order-constrained,    60 overlap)  local via Ollama, or        │
            iXBRL strip)   x-ref filtered)                     OpenAI)                     ▼
                                                                          hybrid search (RRF) + filters
                                                                                       │
                                                       ┌───────────────────────────────┼─────────────────────┐
                                                       ▼                               ▼                     ▼
                                              FastAPI /search /ask           TS MCP server            eval harness
                                              /chunk /companies         (search_filings, ask_filings)  (P@k/MRR, LLM-judge)
                                                       ▲
                                              Claude agent (tool loop):
                                              resolve_company → search_filings (item/ticker/FY filters, hybrid|bm25|knn)
                                              → expand_chunk → answer with [c:chunk_id] cites → numeric grounding check
```

## What's in it

| Layer | File | Notes |
|---|---|---|
| Connector | `filings_search/edgar.py` | SEC ticker map + browse-edgar fallback (handles successor-shell CIKs), `data.sec.gov` submissions, primary-doc fetch, iXBRL/HTML strip, disk cache |
| Parsing | `filings_search/parse.py` | 10-K Item splitter: title-verified headings, cross-reference filter, canonical-order + longest-span selection (defeats ToC rows) |
| Chunking | `filings_search/chunk.py` | paragraph-respecting, token-bounded (450/60 overlap), deterministic chunk ids `sha1(accession|item|idx)` |
| Embeddings | `filings_search/embed.py` | `nomic-embed-text` (768-d) on local Ollama by default; OpenAI `text-embedding-3-small` fallback |
| Index | `filings_search/index.py` | OpenSearch 2.19 mapping: english analyzer BM25 field + `knn_vector` (lucene HNSW, cosine) + keyword metadata (ticker, cik, item, fiscal_year, accession…) |
| Retrieval | `filings_search/search.py` | `bm25`, `knn`, `hybrid` (client-side reciprocal-rank fusion), metadata filters, neighbor expansion |
| Agent | `filings_search/agent.py` | Claude (`claude-opus-5`) tool loop; tools: `list_indexed_companies`, `resolve_company`, `search_filings`, `expand_chunk`; answer must cite `[c:id]`; every figure in the answer must appear in a cited chunk or the response is flagged `grounded_numbers=false` |
| API | `filings_search/api.py` | FastAPI: `GET /search`, `POST /ask`, `GET /chunk/{id}?expand=`, `GET /companies`, `GET /health` |
| MCP | `mcp/src/server.ts` | TypeScript stdio MCP server exposing `list_companies`, `search_filings`, `get_chunk`, `ask_filings` |
| Eval | `eval/` | `run_retrieval_eval.py` (hit@1/hit@5/MRR@10 by mode, labeled queries), `run_grounding_eval.py` (numeric grounding + LLM-as-judge citation support) |

## Run it

```bash
docker compose up -d                       # OpenSearch 2.19 (knn + neural plugins), :9200
ollama pull nomic-embed-text               # local embeddings
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python ingest.py --recreate AAPL MSFT NVDA JPM XOM WMT TSLA JNJ   # ~2 min, 8 filings, ~2.1k chunks
./run_api.sh                               # FastAPI on :8801
./.venv/bin/python -m filings_search.agent "What does NVIDIA disclose about export controls to China?"
./.venv/bin/python eval/run_retrieval_eval.py
./.venv/bin/python eval/run_grounding_eval.py
```

MCP (Claude Desktop / Claude Code / Cursor):
```json
{"mcpServers": {"filings-search": {"command": "node", "args": ["/ABS/PATH/filings-search/mcp/dist/server.js"],
                                    "env": {"FILINGS_API_URL": "http://127.0.0.1:8801"}}}}
```

Config via env: `FS_OPENSEARCH_URL`, `FS_INDEX`, `FS_EMBED_BACKEND=ollama|openai`, `FS_AGENT_MODEL`, `FS_JUDGE_MODEL`, `SEC_USER_AGENT`. `ANTHROPIC_API_KEY` (or `~/.env`) for the agent/judge.

## Results (2026-08-18, 8 filings / 2,101 chunks, 28 labeled queries)

Retrieval — see `eval/retrieval_results.json`:

| setting | mode | hit@1 | hit@5 | MRR@10 | avg ms |
|---|---|---|---|---|---|
| unfiltered | bm25 | 0.571 | 0.857 | 0.686 | 5.7 |
| unfiltered | knn | 0.679 | 0.857 | 0.759 | 29.7 |
| unfiltered | hybrid | 0.679 | 0.857 | 0.759 | 37.9 |
| ticker_filtered | bm25 | 0.679 | 0.929 | 0.772 | 3.8 |
| ticker_filtered | knn | 0.679 | 0.893 | 0.779 | 26.6 |
| ticker_filtered | **hybrid** | **0.714** | **0.929** | **0.812** | 34.9 |

Agent grounding — 10 analyst questions, `claude-opus-5` agent + `claude-opus-5` judge (see `eval/grounding_results.json`):

| metric | value |
|---|---|
| numeric-grounded rate (every figure appears in a cited chunk) | **10/10 = 1.00** |
| LLM-judge verdict: grounded / partially grounded / ungrounded | **9 / 1 / 0** |
| mean fraction of claims supported by a cited chunk | **0.968** |
| avg citations per answer | 9.7 |
| avg tool calls per answer (agent-chosen searches/expansions) | 7.6 |
| avg latency | 41 s (13 s simple → 59 s multi-Item) |
| tokens for the 10-question run | 563k in / 23k out |

The one `partially_grounded` (Walmart tariffs, 0.87) was the agent summarizing a mitigation
that the cited chunk states more narrowly — the judge caught it; that is what the judge is for.

## Design notes / honest limitations

- **Agentic ≠ fixed RAG.** The model chooses company, Item, mode, and how many rounds; a fixed
  top-k pipeline gets no second chance. The tool descriptions carry the "when to use" guidance
  (item map, bm25-for-figures, widen-if-empty).
- **Grounding is strict on purpose.** A derived rounding ("$99,779M" → "~$99.8B") is flagged as
  ungrounded; analysts want the figure as filed. Loosen with a tolerance if you disagree.
- **10-K structure quirks are real, not parser bugs**: JPM and XOM are "wrapper" 10-Ks whose
  MD&A/financials sit in a back-of-book Financial Section (labeled under the last Item); NVIDIA
  files statements under Item 15. Eval labels for those are ticker-only. A follow-up is
  F-page detection (`Consolidated Statements of …` headings) to relabel as Item 8.
- **RRF is client-side** — transparent and easy to reason about; OpenSearch's `hybrid` query
  + normalization pipeline is the in-cluster alternative. No cross-encoder re-ranker yet.
- **Single filing per company** in this run; `--filings N` pulls prior years (fiscal_year filter
  already in the mapping and tools).
- No auth, no rate limits, single-node OpenSearch — this is a working vertical, not a deployment.
