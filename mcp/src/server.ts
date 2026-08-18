#!/usr/bin/env node
/**
 * filings-search MCP server (stdio).
 * Exposes hybrid BM25+kNN search over SEC 10-K filings and a grounded /ask agent
 * as MCP tools, so any MCP-capable agent (Claude Desktop, Claude Code, Cursor …)
 * can plan and iterate its own retrieval over the index.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const API = process.env.FILINGS_API_URL ?? "http://127.0.0.1:8801";

async function get(path: string, params: Record<string, string | number | undefined> = {}) {
  const url = new URL(path, API);
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json();
}
async function post(path: string, body: unknown) {
  const r = await fetch(new URL(path, API), { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json();
}
const text = (v: unknown) => ({ content: [{ type: "text" as const, text: typeof v === "string" ? v : JSON.stringify(v, null, 2) }] });

const server = new McpServer({ name: "filings-search", version: "0.1.0" });

server.tool("list_companies", "List companies (ticker, name, fiscal years, chunk counts) present in the 10-K index. Call first to see what is searchable.",
  {}, async () => text(await get("/companies")));

server.tool("search_filings",
  "Search SEC 10-K chunks. mode 'hybrid' (default) fuses BM25 + vector via reciprocal rank fusion; 'bm25' for exact figures/phrases; 'knn' for conceptual questions. Filter by ticker, 10-K item (1, 1A, 1C, 3, 5, 7, 7A, 8, 9A ...) and fiscal_year. Returns chunk_id, metadata, url and text — cite chunk_id in answers.",
  { query: z.string(), ticker: z.string().optional(), item: z.string().optional(), fiscal_year: z.number().int().optional(),
    mode: z.enum(["hybrid", "bm25", "knn"]).optional(), k: z.number().int().min(1).max(30).optional() },
  async ({ query, ticker, item, fiscal_year, mode, k }) => {
    const r = await get("/search", { q: query, ticker, item, fiscal_year, mode: mode ?? "hybrid", k: k ?? 8 });
    const lines = (r.hits as any[]).map(h => `[c:${h.chunk_id}] ${h.ticker} FY${h.fiscal_year} Item ${h.item} (${h.item_title}) score=${Number(h.score).toFixed(4)}\n${h.text}`);
    return text(lines.join("\n\n") || "(no results)");
  });

server.tool("get_chunk", "Fetch one chunk by id, optionally with neighbouring chunks (expand=1..3) from the same filing/Item for context.",
  { chunk_id: z.string(), expand: z.number().int().min(0).max(3).optional() },
  async ({ chunk_id, expand }) => text(await get(`/chunk/${chunk_id}`, { expand: expand ?? 0 })));

server.tool("ask_filings",
  "Run the grounded research agent: it resolves the company, plans searches across Items, iterates, and answers with [c:chunk_id] citations plus a numeric-grounding verdict. Use for analyst-style questions ('What are NVIDIA's largest customer concentration risks?').",
  { question: z.string(), max_turns: z.number().int().min(1).max(25).optional() },
  async ({ question, max_turns }) => text(await post("/ask", { question, max_turns: max_turns ?? 12 })));

const transport = new StdioServerTransport();
await server.connect(transport);
