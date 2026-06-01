import math
import re
from collections import Counter

from app.schemas.jobfit import Evidence

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}|[\u4e00-\u9fff]{2,}")


def chunk_text(text: str, source: str, chunk_size: int = 700, overlap: int = 120) -> list[Evidence]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []

    chunks: list[Evidence] = []
    start = 0
    chunk_id = 1
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        window = normalized[start:end]
        if end < len(normalized):
            last_break = max(window.rfind("\n"), window.rfind("。"), window.rfind("."))
            if last_break > chunk_size * 0.55:
                end = start + last_break + 1
                window = normalized[start:end]

        chunks.append(Evidence(source=source, chunk_id=chunk_id, text=window.strip(), score=0))
        chunk_id += 1
        if end >= len(normalized):
            break
        start = max(0, end - overlap)

    return chunks


def retrieve_evidence(
    query: str,
    chunks: list[Evidence],
    top_k: int = 8,
) -> list[Evidence]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]

    scored = []
    for chunk in chunks:
        score = _score(query_tokens, _tokenize(chunk.text))
        if score > 0:
            scored.append(chunk.model_copy(update={"score": round(score, 4)}))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def _tokenize(text: str) -> Counter[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return Counter(tokens)


def _score(query_tokens: Counter[str], doc_tokens: Counter[str]) -> float:
    if not doc_tokens:
        return 0

    overlap = set(query_tokens) & set(doc_tokens)
    lexical = sum(min(query_tokens[token], doc_tokens[token]) for token in overlap)
    if lexical == 0:
        return 0

    query_norm = math.sqrt(sum(count * count for count in query_tokens.values()))
    doc_norm = math.sqrt(sum(count * count for count in doc_tokens.values()))
    cosine = lexical / max(query_norm * doc_norm, 1)
    coverage = len(overlap) / max(len(query_tokens), 1)
    return cosine * 0.7 + coverage * 0.3
