"""SEC EDGAR connector: ticker -> CIK candidates -> 10-K filings -> primary document text.

Adapted from ~/finsense-filings/finsense_filings.py (same author). Handles the
successor-shell problem (ticker map points at a holdco that never filed a 10-K).
"""
import html as html_lib
import json
import re
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import RAW, SEC_UA

_TICKER_MAP = None


def _get(url: str, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def ticker_map() -> dict:
    global _TICKER_MAP
    if _TICKER_MAP is None:
        cache = RAW / "company_tickers.json"
        if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
            _TICKER_MAP = json.loads(cache.read_text())
        else:
            _TICKER_MAP = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
            cache.write_text(json.dumps(_TICKER_MAP))
    return _TICKER_MAP


def resolve_cik_candidates(ticker: str) -> list[tuple[str, str]]:
    t = ticker.upper()
    out: list[tuple[str, str]] = []
    for row in ticker_map().values():
        if row["ticker"].upper() == t:
            out.append((str(row["cik_str"]).zfill(10), row["title"]))
    try:
        atom = _get(
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={t}&type=10-K&count=1&output=atom"
        ).decode("utf-8", errors="ignore")
        m = re.search(r"CIK=(\d{4,})", atom)
        n = re.search(r"(?s)<conformed-name>(.*?)</conformed-name>", atom)
        if m:
            cik = m.group(1).zfill(10)
            if cik not in [c for c, _ in out]:
                out.append((cik, n.group(1).strip() if n else t))
    except Exception:
        pass
    if not out:
        raise LookupError(f"Ticker {ticker!r} not found in SEC ticker map.")
    return out


def search_companies(query: str, limit: int = 8) -> list[dict]:
    """Fuzzy company-name / ticker lookup over the SEC ticker map (entity resolution helper)."""
    q = query.strip().lower()
    hits = []
    for row in ticker_map().values():
        name = row["title"].lower()
        tick = row["ticker"].lower()
        score = 0
        if q == tick:
            score = 100
        elif q == name:
            score = 90
        elif name.startswith(q):
            score = 70
        elif q in name:
            score = 50
        elif tick.startswith(q):
            score = 40
        if score:
            hits.append((score, {"ticker": row["ticker"], "cik": str(row["cik_str"]).zfill(10), "name": row["title"]}))
    hits.sort(key=lambda x: (-x[0], x[1]["name"]))
    return [h for _, h in hits[:limit]]


@dataclass
class Filing:
    ticker: str
    cik: str
    company: str
    form: str
    accession: str
    filed: str
    period: str
    fiscal_year: int
    url: str


def list_10k(cik: str, ticker: str, company: str, n: int = 1) -> list[Filing]:
    sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    blocks = [sub["filings"]["recent"]]
    for extra in sub["filings"].get("files", []):
        if len(blocks) > 3:
            break
        try:
            blocks.append(json.loads(_get(f"https://data.sec.gov/submissions/{extra['name']}")))
        except Exception:
            continue
    out: list[Filing] = []
    for recent in blocks:
        rows = zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"],
                   recent["filingDate"], recent.get("reportDate", [""] * len(recent["form"])))
        for form, acc, doc, date, period in rows:
            if form != "10-K":
                continue
            acc_nodash = acc.replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}"
            fy = int((period or date)[:4])
            out.append(Filing(ticker.upper(), cik, company, form, acc, date, period or "", fy, f"{base}/{doc}"))
            if len(out) >= n:
                return out
    if not out:
        raise LookupError(f"No 10-K found for CIK {cik}.")
    return out


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?is)<ix:header.*?</ix:header>", " ", html)  # inline-XBRL hidden header
    html = re.sub(r"(?i)</(p|div|br|tr|li|h[1-6]|table)>", "\n", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</t[dh]>", " \t ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = html_lib.unescape(text)
    text = text.replace("\xa0", " ").replace("•", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def fetch_filings(ticker: str, n: int = 1) -> list[dict]:
    """Return list of {filing: Filing-dict, text: str}; cached on disk per accession."""
    results = []
    filings: list[Filing] = []
    for cik, name in resolve_cik_candidates(ticker):
        try:
            filings = list_10k(cik, ticker, name, n=n)
            break
        except LookupError:
            continue
    if not filings:
        raise LookupError(f"No 10-K found for {ticker}.")
    for f in filings:
        cache = RAW / f"{f.ticker}_{f.accession}.json"
        if cache.exists():
            results.append(json.loads(cache.read_text()))
            continue
        html = _get(f.url).decode("utf-8", errors="ignore")
        rec = {"filing": asdict(f), "text": strip_html(html)}
        cache.write_text(json.dumps(rec))
        results.append(rec)
        time.sleep(0.3)  # SEC fair-use
    return results
