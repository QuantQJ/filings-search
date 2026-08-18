"""Embeddings: local Ollama (nomic-embed-text) by default, OpenAI fallback."""
import httpx
from . import config

_client = httpx.Client(timeout=120)


def embed(texts: list[str], kind: str = "document") -> list[list[float]]:
    if not texts:
        return []
    if config.EMBED_BACKEND == "ollama":
        prefix = "search_document: " if kind == "document" else "search_query: "
        out = []
        for i in range(0, len(texts), 32):
            batch = [prefix + t for t in texts[i:i + 32]]
            r = _client.post(f"{config.OLLAMA_URL}/api/embed",
                             json={"model": config.OLLAMA_EMBED_MODEL, "input": batch})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out
    from openai import OpenAI
    oc = OpenAI()
    out = []
    for i in range(0, len(texts), 256):
        resp = oc.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts[i:i + 256])
        out.extend([d.embedding for d in resp.data])
    return out
