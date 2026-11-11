from src.vector_store import search_vectors
from src.keyword_search import keyword_search


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60,
                            top_k: int = 5) -> list[dict]:
    scores = {}
    chunk_by_id = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id[chunk_id] = result

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    fused = []
    for chunk_id, score in ranked[:top_k]:
        item = dict(chunk_by_id[chunk_id])
        item["rrf_score"] = score
        fused.append(item)
    return fused


def hybrid_search(query: str, user_id: str, top_k: int = 5) -> list[dict]:
    vector_results = search_vectors(query, user_id, top_k=10)
    keyword_results = keyword_search(query, user_id, top_k=10)
    return reciprocal_rank_fusion(
        [vector_results, keyword_results],
        k=60,
        top_k=top_k,
    )
