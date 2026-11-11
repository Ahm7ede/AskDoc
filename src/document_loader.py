import hashlib
import uuid
from pathlib import Path

import pymupdf
from docx import Document

from src.config import MAX_FILE_SIZE, SUPPORTED_EXTENSIONS, UPLOAD_DIR


def save_uploaded_file(uploaded_file, user_id: str) -> dict:
    """Validate and save a Streamlit UploadedFile."""
    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("File exceeds the 20 MB upload limit.")

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    doc_id = str(uuid.uuid4())

    user_dir = UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    file_path = user_dir / f"{doc_id}{extension}"
    file_path.write_bytes(file_bytes)

    return {
        "doc_id": doc_id,
        "filename": original_name,
        "file_path": str(file_path),
        "file_hash": file_hash,
        "user_id": str(user_id),
        "file_type": extension,
    }


def extract_pdf(file_path: str) -> list[dict]:
    pages = []
    with pymupdf.open(file_path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "page_number": page_number})
    return pages


def extract_docx(file_path: str) -> list[dict]:
    document = Document(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    text = "\n".join(part for part in paragraphs if part).strip()
    if not text:
        raise ValueError("No readable text found in the DOCX file.")
    return [{"text": text, "page_number": None}]


def extract_txt(file_path: str) -> list[dict]:
    # utf-8-sig also handles UTF-8 files with a byte-order mark.
    text = Path(file_path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("No readable text found in the TXT file.")
    return [{"text": text, "page_number": None}]


def load_document(file_info: dict) -> list[dict]:
    file_type = file_info["file_type"]
    path = file_info["file_path"]

    if file_type == ".pdf":
        extracted = extract_pdf(path)
    elif file_type == ".docx":
        extracted = extract_docx(path)
    elif file_type == ".txt":
        extracted = extract_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    if not extracted:
        raise ValueError(
            "No extractable text found. This may be a scanned PDF; OCR is not included."
        )

    results = []
    for item in extracted:
        results.append({
            "text": item["text"],
            "metadata": {
                "doc_id": file_info["doc_id"],
                "filename": file_info["filename"],
                "file_type": file_type,
                "page_number": item["page_number"],
                "file_hash": file_info["file_hash"],
                "user_id": str(file_info["user_id"]),
                "file_path": file_info["file_path"],
            },
        })
    return results
