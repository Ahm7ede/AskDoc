import requests

from src.config import OLLAMA_URL, EMBEDDING_MODEL


def generate_embeddings(texts: list[str] | str) -> list[list[float]]:
    """Generate document embeddings with Ollama's /api/embed endpoint."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []

    # Nomic recommends task prefixes to distinguish documents from queries.
    inputs = [f"search_document: {text}" for text in texts]

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": inputs,
                "truncate": False,
            },
            timeout=180,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama for embeddings. Make sure Ollama is running "
            f"and '{EMBEDDING_MODEL}' is installed. Details: {exc}"
        ) from exc

    embeddings = response.json().get("embeddings", [])
    if len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an unexpected number of embeddings.")
    return embeddings


def generate_query_embedding(query: str) -> list[float]:
    """Generate a query embedding in the same embedding space."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": [f"search_query: {query}"],
                "truncate": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama for query embeddings. "
            f"Check that Ollama is running. Details: {exc}"
        ) from exc

    embeddings = response.json().get("embeddings", [])
    if not embeddings:
        raise RuntimeError("Ollama did not return a query embedding.")
    return embeddings[0]
