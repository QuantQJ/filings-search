"""10-K section (Item) splitter over stripped text.

Finds Item headings whose following text matches the canonical title, then keeps
for each item the candidate whose span to the next heading is longest — this
discards table-of-contents hits (short spans) in favour of the body heading.
"""
import re

ITEMS = [
    ("1", "Business"), ("1A", "Risk Factors"), ("1B", "Unresolved Staff Comments"),
    ("1C", "Cybersecurity"), ("2", "Properties"), ("3", "Legal Proceedings"),
    ("4", "Mine Safety Disclosures"), ("5", "Market for Registrant"),
    ("6", "Reserved"), ("7", "Management's Discussion and Analysis"),
    ("7A", "Quantitative and Qualitative Disclosures About Market Risk"),
    ("8", "Financial Statements and Supplementary Data"),
    ("9", "Changes in and Disagreements"), ("9A", "Controls and Procedures"),
    ("9B", "Other Information"), ("9C", "Disclosure Regarding Foreign Jurisdictions"),
    ("10", "Directors, Executive Officers"), ("11", "Executive Compensation"),
    ("12", "Security Ownership"), ("13", "Certain Relationships"),
    ("14", "Principal Accountant Fees"), ("15", "Exhibit"), ("16", "Form 10-K Summary"),
]
TITLE = {k: v for k, v in ITEMS}
ORDER = {k: i for i, (k, _) in enumerate(ITEMS)}


def _loose(s: str) -> str:
    # tolerate spaces injected inside words by tag stripping, and curly quotes
    parts = []
    for ch in s:
        if ch == "'":
            parts.append(r"[’']")
        elif ch == " ":
            parts.append(r"\s+")
        else:
            parts.append(re.escape(ch) + r"\s*")
    return "".join(parts)


_ITEM_RE = re.compile(r"(?i)\bitem\s*(1A|1B|1C|7A|9A|9B|9C|1[0-6]|[1-9])\s*[\.:\-–—]?\s*", re.M)


_XREF = re.compile(r"(?i)(?:see|in|under|refer\s+to|within|described\s+in|discussed\s+in|included\s+in|of|and|,)\s*(?:\"|\u201c)?\s*$")


def split_items(text: str) -> list[dict]:
    cands = []
    for m in _ITEM_RE.finditer(text):
        item = m.group(1).upper()
        title = TITLE.get(item)
        if not title:
            continue
        window = text[m.end(): m.end() + 160]
        if not re.match(r"\s*" + _loose(title[:22]), window, re.I):
            continue
        # skip cross-references like "see Item 1A. Risk Factors" / "in Part II, Item 7"
        lead = text[max(0, m.start() - 40): m.start()]
        if _XREF.search(lead):
            continue
        cands.append((m.start(), item))
    if not cands:
        return [{"item": "FULL", "item_title": "Full document", "text": text}]
    cands.sort()
    spans = []
    for i, (pos, item) in enumerate(cands):
        end = cands[i + 1][0] if i + 1 < len(cands) else len(text)
        spans.append((item, pos, end))
    # choose in canonical Item order, each after the previous choice, longest span wins
    chosen, prev = [], -1
    for item, _ in ITEMS:
        opts = [(pos, end) for it, pos, end in spans if it == item and pos > prev]
        if not opts:
            continue
        pos, end = max(opts, key=lambda pe: pe[1] - pe[0])
        chosen.append((item, pos))
        prev = pos
    out = []
    for i, (item, pos) in enumerate(chosen):
        end = chosen[i + 1][1] if i + 1 < len(chosen) else len(text)
        body = text[pos:end].strip()
        if len(body) < 200:
            continue
        out.append({"item": item, "item_title": TITLE[item], "text": body})
    return out or [{"item": "FULL", "item_title": "Full document", "text": text}]
