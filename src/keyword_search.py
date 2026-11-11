import re
from rank_bm25 import BM25Okapi

from src.vector_store import get_user_chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def keyword_search(query: str, user_id: str, top_k: int = 10) -> list[dict]:
    records = get_user_chunks(user_id)
    documents = records["documents"]
    metadatas = records["metadatas"]
    ids = records["ids"]

    if not documents:
        return []

    valid = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        tokens = tokenize(text)
        if tokens:
            valid.append((chunk_id, text, metadata, tokens))

    query_tokens = tokenize(query)
    if not valid or not query_tokens:
        return []

    bm25 = BM25Okapi([item[3] for item in valid])
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []
    for index in ranked_indices:
        if scores[index] <= 0:
            continue
        chunk_id, text, metadata, _ = valid[index]
        results.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata,
            "score": float(scores[index]),
        })
        if len(results) >= top_k:
            break

    return results
