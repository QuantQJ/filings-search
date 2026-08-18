import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def load_env(path=Path.home() / ".env"):
    """Load KEY=VALUE lines from ~/.env into os.environ (no override)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


load_env()

OS_URL = os.environ.get("FS_OPENSEARCH_URL", "http://localhost:9200")
INDEX = os.environ.get("FS_INDEX", "filings_chunks")
EMBED_BACKEND = os.environ.get("FS_EMBED_BACKEND", "ollama")  # ollama | openai
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("FS_OLLAMA_EMBED_MODEL", "nomic-embed-text")
OPENAI_EMBED_MODEL = os.environ.get("FS_OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("FS_EMBED_DIM", "768" if EMBED_BACKEND == "ollama" else "1536"))
CHUNK_TOKENS = int(os.environ.get("FS_CHUNK_TOKENS", "450"))
CHUNK_OVERLAP = int(os.environ.get("FS_CHUNK_OVERLAP", "60"))
AGENT_MODEL = os.environ.get("FS_AGENT_MODEL", "claude-opus-5")
JUDGE_MODEL = os.environ.get("FS_JUDGE_MODEL", "claude-opus-5")
SEC_UA = os.environ.get("SEC_USER_AGENT", "Constant Systems filings-search qj@constantqj.com")
