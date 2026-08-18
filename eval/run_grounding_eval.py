#!/usr/bin/env python3
"""Agent grounding eval: run /ask on analyst questions, then (a) numeric grounding
check and (b) LLM-as-judge citation support — does each factual claim have a cited
chunk that actually supports it? Reports grounded rate, unsupported claims, cost.
"""
import json
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filings_search import config  # noqa: E402
from filings_search.agent import ask  # noqa: E402

QUESTIONS = [
    "What were Apple's total net sales in fiscal 2025 and how did iPhone net sales change year over year?",
    "What does NVIDIA disclose about export controls affecting sales to China, and which products are affected?",
    "How much did Microsoft say Azure and other cloud services revenue grew, and what drove it?",
    "What are JPMorgan's main drivers of net interest income according to its MD&A?",
    "What is Walmart's disclosed exposure to tariffs and how does it say it will respond?",
    "Summarize Tesla's disclosures about regulatory credit revenue.",
    "What does Johnson & Johnson say about the status of talc litigation?",
    "What were Exxon Mobil's proved reserves and how did they change?",
    "Which of the indexed companies discuss dependence on a small number of suppliers as a risk factor?",
    "Does Apple disclose the size of its remaining share repurchase authorization? Quote the figure.",
]

JUDGE_SYS = """You are a strict citation auditor for financial research answers. You will get an answer with [c:id] citations and the full text of every cited chunk. For each factual claim in the answer, decide whether at least one cited chunk directly supports it (same facts, same numbers, no extrapolation). Return JSON only:
{"claims":[{"claim":"...","supported":true|false,"cited_ids":["..."],"note":"..."}],"answer_supported_fraction":0.0-1.0,"hallucinated_numbers":["..."],"verdict":"grounded"|"partially_grounded"|"ungrounded"}"""


def judge(client, result):
    cites = "\n\n".join(f"[c:{c['chunk_id']}] ({c['ticker']} FY{c['fiscal_year']} Item {c['item']})\n{c['snippet']}" for c in result["citations"])
    # snippets are 300 chars; pull the full text
    from filings_search.search import get_chunk
    full = []
    for c in result["citations"]:
        d = get_chunk(c["chunk_id"])
        full.append(f"[c:{c['chunk_id']}] ({c['ticker']} FY{c['fiscal_year']} Item {c['item']})\n{d['text'] if d else c['snippet']}")
    user = f"QUESTION:\n{result['question']}\n\nANSWER:\n{result['answer']}\n\nCITED CHUNKS:\n" + "\n\n".join(full)
    resp = client.messages.create(model=config.JUDGE_MODEL, max_tokens=4000, system=JUDGE_SYS,
                                  messages=[{"role": "user", "content": user}],
                                  output_config={"format": {"type": "json_schema", "schema": {
                                      "type": "object", "properties": {
                                          "claims": {"type": "array", "items": {"type": "object", "properties": {
                                              "claim": {"type": "string"}, "supported": {"type": "boolean"},
                                              "cited_ids": {"type": "array", "items": {"type": "string"}}, "note": {"type": "string"}},
                                              "required": ["claim", "supported", "cited_ids", "note"], "additionalProperties": False}},
                                          "answer_supported_fraction": {"type": "number"},
                                          "hallucinated_numbers": {"type": "array", "items": {"type": "string"}},
                                          "verdict": {"type": "string", "enum": ["grounded", "partially_grounded", "ungrounded"]}},
                                      "required": ["claims", "answer_supported_fraction", "hallucinated_numbers", "verdict"],
                                      "additionalProperties": False}}})
    if resp.stop_reason == "refusal":
        return {"verdict": "judge_refused", "claims": [], "answer_supported_fraction": 0, "hallucinated_numbers": []}
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def main():
    client = anthropic.Anthropic()
    out = []
    t0 = time.time()
    for q in QUESTIONS:
        print(f"\n== {q}", flush=True)
        r = ask(q, verbose=True)
        j = judge(client, r)
        print(f"   → {r['n_citations']} citations, numeric_grounded={r['grounded_numbers']} missing={r['ungrounded_figures']}, "
              f"judge={j['verdict']} ({j['answer_supported_fraction']:.2f}), {r['seconds']}s, "
              f"{r['usage']['input_tokens']}in/{r['usage']['output_tokens']}out", flush=True)
        out.append({"question": q, "answer": r["answer"], "citations": r["citations"], "numeric_grounded": r["grounded_numbers"],
                    "ungrounded_figures": r["ungrounded_figures"], "tool_calls": r["tool_calls"], "usage": r["usage"],
                    "seconds": r["seconds"], "judge": j})
    n = len(out)
    summary = {
        "n": n, "model": config.AGENT_MODEL, "judge_model": config.JUDGE_MODEL,
        "numeric_grounded_rate": round(sum(o["numeric_grounded"] for o in out) / n, 3),
        "judge_grounded_rate": round(sum(o["judge"]["verdict"] == "grounded" for o in out) / n, 3),
        "judge_partial_rate": round(sum(o["judge"]["verdict"] == "partially_grounded" for o in out) / n, 3),
        "mean_supported_fraction": round(sum(o["judge"]["answer_supported_fraction"] for o in out) / n, 3),
        "avg_citations": round(sum(len(o["citations"]) for o in out) / n, 2),
        "avg_tool_calls": round(sum(len(o["tool_calls"]) for o in out) / n, 2),
        "avg_seconds": round(sum(o["seconds"] for o in out) / n, 1),
        "total_input_tokens": sum(o["usage"]["input_tokens"] for o in out),
        "total_output_tokens": sum(o["usage"]["output_tokens"] for o in out),
        "wall_seconds": round(time.time() - t0, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    (Path(__file__).parent / "grounding_results.json").write_text(json.dumps({"summary": summary, "runs": out}, indent=2))


if __name__ == "__main__":
    main()
