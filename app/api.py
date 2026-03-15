from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core import ask_question, index_directory
from app.indexing import IndexingConfig

app = FastAPI(title="Gen AI PDF QA API", version="0.1.0")


class IndexRequest(BaseModel):
    input_dir: str = Field("data/pdfs", description="Directory that contains PDF files.")
    glob_pattern: str = "*.pdf"
    domain_tag: str = "general"
    chunk_size: int = 800
    chunk_overlap: int = 100
    trace_path: str = "data/indexing_trace.json"
    index_path: str = "data/index.faiss"
    metadata_path: str = "data/metadata.json"


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    index_path: str = "data/index.faiss"
    metadata_path: str = "data/metadata.json"


@app.post("/index")
def index_endpoint(payload: IndexRequest) -> dict[str, object]:
    try:
        config = IndexingConfig(
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            glob_pattern=payload.glob_pattern,
            domain_tag=payload.domain_tag,
            trace_path=payload.trace_path,
        )
        stats = index_directory(payload.input_dir, config, payload.index_path, payload.metadata_path)
        return {"status": "indexed", **stats}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask_endpoint(payload: AskRequest) -> dict[str, object]:
    try:
        return ask_question(
            payload.question,
            index_path=payload.index_path,
            metadata_path=payload.metadata_path,
            top_k=payload.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
