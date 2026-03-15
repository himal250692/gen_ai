# gen_ai

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Run end to end via CLI

1. Index one or more PDF files:
   ```bash
   python -m app.cli index path/to/doc1.pdf path/to/doc2.pdf
   ```
2. Ask a question against the saved FAISS index:
   ```bash
   python -m app.cli ask "What are the key points?"
   ```

### Run end to end via FastAPI

1. Start the API server:
   ```bash
   uvicorn app.api:app --reload
   ```
2. Index PDFs:
   ```bash
   curl -X POST http://127.0.0.1:8000/index \
     -H "Content-Type: application/json" \
     -d '{"pdf_paths": ["path/to/doc1.pdf", "path/to/doc2.pdf"]}'
   ```
3. Ask a question:
   ```bash
   curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the key points?"}'
   ```

Both interfaces persist data in `data/index.faiss` and `data/metadata.json` by default.
