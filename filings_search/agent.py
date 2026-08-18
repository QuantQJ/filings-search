"""Agentic retrieval over the filings index with Claude tool use + citation grounding.

The agent (not a fixed pipeline) decides which company, which Item, and how many
search rounds — that is the "agentic retrieval" contract. Every numeric figure in
the answer must appear in a cited chunk, else the answer is marked ungrounded.
"""
import json
import re
import time

import anthropic

from . import config
from .edgar import search_companies
from .search import search, get_chunk, neighbors, companies as indexed_companies

SYSTEM = """You are a research agent for institutional equity analysts. You answer questions strictly from SEC 10-K filings retrieved with your tools.

How to work:
- Resolve the company first if the user names one (resolve_company). Only companies returned by list_indexed_companies are searchable.
- Plan your retrieval: pick the Item that most likely holds the answer (1=Business, 1A=Risk Factors, 1C=Cybersecurity, 3=Legal, 5=Equity/buybacks, 7=MD&A, 7A=Market risk, 8=Financial statements & notes, 9A=Controls). Search with an item filter first; widen (drop the filter, rephrase, try another Item) if the results don't answer the question. Use expand_chunk when a hit is cut off mid-passage.
- Iterate: 2–5 focused searches beat one broad one. Stop when you can answer or when the filings clearly don't contain it.

How to answer:
- Cite every factual claim with the chunk id in square brackets, e.g. [c:3f9a1b2c4d5e6f70]. Use only ids returned by your tools.
- Quote numbers exactly as they appear in the cited text (same units and rounding). Never compute or infer figures that are not in a cited chunk; if you must derive something, show the cited inputs.
- If the filings do not contain the answer, say so plainly and cite what you checked. Do not fill gaps from memory.
- Be concise: lead with the answer, then supporting detail. Plain prose; no headers.
"""

TOOLS = [
    {"name": "list_indexed_companies", "description": "List companies (ticker, name, fiscal years) available in the filings index. Call this first when unsure whether a company is covered.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "resolve_company", "description": "Resolve a company name or ticker to SEC ticker/CIK candidates (entity resolution over the SEC ticker map). Call when the user names a company by name rather than ticker.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "Company name or ticker, e.g. 'Exxon' or 'JPM'"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "search_filings", "description": "Search 10-K chunks. Default mode 'hybrid' fuses BM25 keyword and vector search (best for most questions); 'bm25' for exact phrases/figures/defined terms; 'knn' for conceptual questions. Filter by ticker/item/fiscal_year to focus. Returns chunk ids, metadata and text.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "ticker": {"type": "string", "description": "Ticker filter, e.g. AAPL"},
         "item": {"type": "string", "description": "10-K Item filter: 1, 1A, 1C, 3, 5, 7, 7A, 8, 9A ..."},
         "fiscal_year": {"type": "integer"},
         "mode": {"type": "string", "enum": ["hybrid", "bm25", "knn"]},
         "k": {"type": "integer", "description": "results to return (default 6, max 15)"}},
      "required": ["query"], "additionalProperties": False}},
    {"name": "expand_chunk", "description": "Return the chunk before and after a given chunk id (same filing and Item) to read a passage that was cut off.",
     "input_schema": {"type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"], "additionalProperties": False}},
]

_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _fmt(hits: list[dict], max_chars: int = 1400) -> str:
    out = []
    for h in hits:
        t = h["text"]
        if len(t) > max_chars:
            t = t[:max_chars] + " …"
        out.append(f"[c:{h['chunk_id']}] {h['ticker']} FY{h['fiscal_year']} Item {h['item']} ({h['item_title']}) chunk {h['chunk_idx']}\n{t}")
    return "\n\n".join(out) if out else "(no results)"


def run_tool(name: str, inp: dict, seen: dict) -> str:
    if name == "list_indexed_companies":
        return json.dumps(indexed_companies())
    if name == "resolve_company":
        return json.dumps(search_companies(inp["query"]))
    if name == "search_filings":
        k = min(int(inp.get("k") or 6), 15)
        hits = search(inp["query"], mode=inp.get("mode") or "hybrid", k=k,
                      ticker=inp.get("ticker"), item=inp.get("item"), fiscal_year=inp.get("fiscal_year"))
        for h in hits:
            seen[h["chunk_id"]] = h
        return _fmt(hits)
    if name == "expand_chunk":
        hits = neighbors(inp["chunk_id"], span=1)
        for h in hits:
            seen[h["chunk_id"]] = h
        return _fmt(hits, max_chars=2400)
    return f"unknown tool {name}"


_NUM = re.compile(r"(?<![\w.])\$?\(?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|percent|billion|million|thousand|bn|mm|m\b|b\b)?", re.I)


def _figures(s: str) -> list[str]:
    figs = []
    for m in _NUM.finditer(s):
        raw = m.group(0).strip()
        core = re.sub(r"[^\d.]", "", raw)
        if not core or core in {".", ""}:
            continue
        # skip years, item numbers, footnote-ish small ints, and chunk ids
        if re.fullmatch(r"(19|20)\d\d", core) or (core.isdigit() and int(core) < 13 and "%" not in raw and "$" not in raw):
            continue
        figs.append(core)
    return figs


def numbers_grounded(answer: str, cited_texts: list[str]) -> tuple[bool, list[str]]:
    """Every figure in the answer must appear (as digits) in some cited chunk."""
    body = re.sub(r"\[c:[0-9a-f]+\]", " ", answer)
    src = " ".join(cited_texts)
    src_digits = re.sub(r"[,\s]", "", src)
    missing = []
    for f in set(_figures(body)):
        f_nc = f.replace(",", "")
        if f_nc in src_digits or f_nc.rstrip("0").rstrip(".") in src_digits:
            continue
        missing.append(f)
    return (not missing), missing


def ask(question: str, model: str | None = None, max_turns: int = 12, verbose: bool = False) -> dict:
    model = model or config.AGENT_MODEL
    messages = [{"role": "user", "content": question}]
    seen: dict[str, dict] = {}
    trace = []
    t0 = time.time()
    usage = {"input_tokens": 0, "output_tokens": 0}
    final_text = ""
    for turn in range(max_turns):
        with client().messages.stream(
            model=model, max_tokens=8000, system=SYSTEM, tools=TOOLS, messages=messages,
        ) as stream:
            resp = stream.get_final_message()
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            final_text = "(refused)"
            break
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        texts = [b.text for b in resp.content if b.type == "text"]
        if not tool_uses:
            final_text = "\n".join(texts).strip()
            break
        results = []
        for tu in tool_uses:
            out = run_tool(tu.name, tu.input, seen)
            trace.append({"tool": tu.name, "input": tu.input, "chars": len(out)})
            if verbose:
                print(f"  ↳ {tu.name} {json.dumps(tu.input)[:120]} → {len(out)} chars", flush=True)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
        messages.append({"role": "user", "content": results})
    cited_ids = re.findall(r"\[c:([0-9a-f]{16})\]", final_text)
    citations, texts = [], []
    for cid in dict.fromkeys(cited_ids):
        d = seen.get(cid) or get_chunk(cid)
        if not d:
            continue
        texts.append(d["text"])
        citations.append({"chunk_id": cid, "ticker": d["ticker"], "company": d["company"], "fiscal_year": d["fiscal_year"],
                          "item": d["item"], "item_title": d["item_title"], "url": d["url"], "snippet": d["text"][:300]})
    grounded, missing = numbers_grounded(final_text, texts)
    return {"question": question, "answer": final_text, "citations": citations,
            "grounded_numbers": grounded, "ungrounded_figures": missing,
            "n_citations": len(citations), "tool_calls": trace, "model": model,
            "usage": usage, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What were Apple's total net sales in fiscal 2025 and how did iPhone revenue change?"
    r = ask(q, verbose=True)
    print(json.dumps({k: v for k, v in r.items() if k != "citations"}, indent=2))
    for c in r["citations"]:
        print(f"[c:{c['chunk_id']}] {c['ticker']} FY{c['fiscal_year']} Item {c['item']} — {c['url']}")
