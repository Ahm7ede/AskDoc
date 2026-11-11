from pathlib import Path

import chromadb

from src.config import CHROMA_DIR
from src.embeddings import generate_embeddings, generate_query_embedding

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name="personal_documents",
    metadata={"hnsw:space": "cosine"},
)


def _clean_metadata(metadata: dict) -> dict:
    # Chroma metadata values must be supported scalar types; omit None.
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }


def store_chunks(chunks: list[dict], batch_size: int = 32) -> None:
    if not chunks:
        return

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = generate_embeddings(texts)

        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_clean_metadata(item["metadata"]) for item in batch],
        )


def search_vectors(query: str, user_id: str, top_k: int = 10) -> list[dict]:
    total = collection.count()
    if total == 0:
        return []

    query_embedding = generate_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        where={"user_id": str(user_id)},
        include=["documents", "metadatas", "distances"],
    )

    output = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, chunk_id in enumerate(ids):
        output.append({
            "id": chunk_id,
            "text": docs[i],
            "metadata": metas[i],
            "distance": distances[i],
        })
    return output


def get_user_chunks(user_id: str) -> dict:
    return collection.get(
        where={"user_id": str(user_id)},
        include=["documents", "metadatas"],
    )


def get_user_documents(user_id: str) -> dict:
    records = get_user_chunks(user_id)
    docs = {}
    for metadata in records["metadatas"]:
        docs[metadata["doc_id"]] = {
            "filename": metadata.get("filename", "Unknown"),
            "file_hash": metadata.get("file_hash"),
            "file_path": metadata.get("file_path"),
        }
    return docs


def find_duplicate_document(user_id: str, file_hash: str) -> bool:
    records = get_user_chunks(user_id)
    return any(
        metadata.get("file_hash") == file_hash
        for metadata in records["metadatas"]
    )


def delete_document(doc_id: str, user_id: str) -> int:
    # Filter by user first, then match doc_id. This prevents cross-user deletion.
    records = collection.get(
        where={"user_id": str(user_id)},
        include=["metadatas"],
    )

    ids_to_delete = []
    paths_to_delete = set()

    for index, metadata in enumerate(records["metadatas"]):
        if metadata.get("doc_id") == doc_id:
            ids_to_delete.append(records["ids"][index])
            if metadata.get("file_path"):
                paths_to_delete.add(metadata["file_path"])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    for raw_path in paths_to_delete:
        path = Path(raw_path).resolve()
        # Only delete files inside this user's upload directory.
        user_upload_dir = (Path("data/uploads") / str(user_id)).resolve()
        if path.is_relative_to(user_upload_dir) and path.is_file():
            path.unlink()

    return len(ids_to_delete)
