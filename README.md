# Personal RAG — Ollama + ChromaDB + BM25

A local document question-answering application using:
- Streamlit UI
- PyMuPDF, python-docx and TXT extraction
- Token-based chunking
- Ollama `nomic-embed-text` embeddings
- ChromaDB persistent vector storage
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF)
- Ollama `llama3.2:3b` answer generation

## 1. Requirements

- Python 3.10+
- Ollama installed and running
- Enough RAM to run the selected local models

Install Ollama: https://ollama.com/download

Pull models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## 2. Set up Python

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows CMD:

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Run

Make sure Ollama is running, then:

```bash
streamlit run app.py
```

## 4. Important notes

- This is a local learning/demo project, not a production multi-user deployment.
- `USER_ID` in `app.py` is hardcoded for demonstration. Add real authentication and enforce user authorization before exposing it to multiple users.
- Scanned PDFs need OCR; this version extracts selectable PDF text.
- DOCX page numbers are not available through this extraction method.
- BM25 is rebuilt from the user's chunks for each query. This is simple but can become slow for large collections.
- The app uses Nomic's `search_document:` and `search_query:` embedding prefixes.
- The RRF ranking combines ranks, not raw vector distances and BM25 scores.
- Treat uploaded files as untrusted input. For public deployment, add content validation, malware scanning, rate limits, and resource limits.
- The included text splitter uses `cl100k_base` tokenization as an approximation; it is not necessarily the exact tokenizer of the Ollama models.
