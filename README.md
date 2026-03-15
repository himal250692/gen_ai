# gen_ai

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Run end to end via CLI

1. Index PDFs from a directory (defaults to `data/pdfs`):
   ```bash
   python -m app.cli index --input-dir data/pdfs --domain-tag finance
   ```
2. Ask a question against the saved FAISS index:
   ```bash
   python -m app.cli ask "What are the key points?" \
     --top-k 5 \
     --filter domain_tag=finance \
     --filter page_min=2 \
     --include-passages
   ```

The `ask` command now returns:
- final `answer` text from a configured LLM (`echo` by default, `openai` optionally),
- explicit `citations` (`filename`, `page_number`, `chunk_id`),
- optional raw `passages` when `--include-passages` is enabled.

### Run end to end via FastAPI

1. Start the API server:
   ```bash
   uvicorn app.api:app --reload
   ```
2. Index PDFs:
   ```bash
   curl -X POST http://127.0.0.1:8000/index \
     -H "Content-Type: application/json" \
     -d '{"input_dir": "data/pdfs", "domain_tag": "finance"}'
   ```
3. Ask a question:
   ```bash
   curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What are the key points?",
       "top_k": 4,
       "filters": {"domain_tag": "finance"},
       "include_passages": true,
       "llm_provider": "echo"
     }'
   ```

## Query module

`app/rag.py` provides retrieval + generation flow:

1. Accept question + retrieval options (`top_k`, metadata filters).
2. Retrieve top-k chunks from vector store.
3. Build a grounded prompt that includes chunk text and source identifiers.
4. Call an LLM through LangChain (`echo` local deterministic model or OpenAI model).
5. Return answer + explicit citations + optional passages.

## Vector layer module

`app/vectorstore.py` provides a reusable vector retrieval layer with:

- Configurable Sentence Transformers embedding model.
- Indexing from the chunk output produced by `app.indexing.build_index`.
- Two storage backends:
  - FAISS index persisted on disk (`data/index.faiss`) plus metadata JSON (`data/metadata.json`).
  - Qdrant collection with payload metadata.
- Retrieval API: `search(query, top_k, filters=None)` returning `chunk_text`, `score`, and `metadata`.

Example:

```python
from app.indexing import IndexingConfig
from app.vectorstore import VectorStore, VectorStoreConfig

store = VectorStore(
    VectorStoreConfig(
        backend="faiss",  # or "qdrant"
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
)

store.index_directory(
    input_dir="data/pdfs",
    indexing_config=IndexingConfig(domain_tag="finance"),
)

results = store.search(
    query="quarterly revenue outlook",
    top_k=5,
    filters={
        "filename": "q1_report.pdf",
        "domain_tag": "finance",
        "page_min": 2,
        "page_max": 8,
    },
)
```

Supported filter keys include metadata fields such as `filename`, `domain_tag`, `source_path`, `chunk_index`, exact `page_number`, and page ranges via `page_min`/`page_max`.

Indexing persists:
- Vector index: `data/index.faiss`
- Chunk metadata: `data/metadata.json`
- Intermediate extraction/chunk trace: `data/indexing_trace.json`
