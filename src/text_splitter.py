import uuid
import tiktoken

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextSplitter:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> list[str]:
        token_ids = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids))
            chunk_text = self.tokenizer.decode(token_ids[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(token_ids):
                break
            start = end - self.chunk_overlap

        return chunks

    def split_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []

        for document in documents:
            for index, text in enumerate(self.split_text(document["text"])):
                chunk_id = str(uuid.uuid4())
                metadata = {
                    **document["metadata"],
                    "chunk_index": index,
                    "chunk_id": chunk_id,
                }
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                })

        return all_chunks
