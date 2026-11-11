import hashlib

from src.document_loader import save_uploaded_file, load_document
from src.text_splitter import TextSplitter
from src.vector_store import store_chunks, find_duplicate_document
from src.hybrid_search import hybrid_search
from src.llm import generate_answer
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def ingest_document(uploaded_file, user_id: str) -> dict:
    # Hash before saving so duplicate uploads can be rejected.
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if find_duplicate_document(user_id, file_hash):
        raise ValueError("You have already uploaded this exact document.")

    file_info = save_uploaded_file(uploaded_file, user_id)

    try:
        documents = load_document(file_info)
        splitter = TextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks were created from this document.")

        store_chunks(chunks)

    except Exception:
        # Avoid leaving an orphaned original file if processing fails.
        from pathlib import Path
        Path(file_info["file_path"]).unlink(missing_ok=True)
        raise

    return {
        "doc_id": file_info["doc_id"],
        "filename": file_info["filename"],
        "total_sections": len(documents),
        "total_chunks": len(chunks),
    }


def ask_question(question: str, user_id: str, top_k: int = 5) -> dict:
    retrieved = hybrid_search(question, user_id, top_k=top_k)
    answer = generate_answer(question, retrieved)
    return {"answer": answer, "sources": retrieved}
