"""Token-bounded chunking that respects paragraph boundaries."""
import hashlib
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def n_tokens(s: str) -> int:
    return len(_enc.encode(s, disallowed_special=()))


def chunk_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur, cur_tok = [], [], 0
    for p in paras:
        pt = n_tokens(p)
        if pt > max_tokens:  # hard-split huge paragraph
            ids = _enc.encode(p, disallowed_special=())
            for i in range(0, len(ids), max_tokens - overlap):
                piece = _enc.decode(ids[i:i + max_tokens])
                if cur:
                    chunks.append("\n".join(cur)); cur, cur_tok = [], 0
                chunks.append(piece)
            continue
        if cur_tok + pt > max_tokens and cur:
            chunks.append("\n".join(cur))
            # overlap: carry trailing paragraphs up to `overlap` tokens
            carry, ct = [], 0
            for q in reversed(cur):
                qt = n_tokens(q)
                if ct + qt > overlap:
                    break
                carry.insert(0, q); ct += qt
            cur, cur_tok = carry, ct
        cur.append(p); cur_tok += pt
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_id(accession: str, item: str, idx: int) -> str:
    return hashlib.sha1(f"{accession}|{item}|{idx}".encode()).hexdigest()[:16]
