from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from app.indexing import IndexingConfig
from app.rag import QueryOptions, query_index
from app.vectorstore import DEFAULT_MODEL, VectorStore, VectorStoreConfig


def index_directory(
    input_dir: str = "data/pdfs",
    config: IndexingConfig | None = None,
    index_path: str = "data/index.faiss",
    metadata_path: str = "data/metadata.json",
    model_name: str = DEFAULT_MODEL,
) -> dict[str, int]:
    store = VectorStore(
        VectorStoreConfig(
            backend="faiss",
            model_name=model_name,
            faiss_index_path=index_path,
            faiss_metadata_path=metadata_path,
        )
    )
    return store.index_directory(input_dir=input_dir, indexing_config=config)


def index_pdfs(
    pdf_paths: Iterable[str],
    index_path: str = "data/index.faiss",
    metadata_path: str = "data/metadata.json",
    model_name: str = DEFAULT_MODEL,
) -> dict[str, int]:
    pdf_list = list(pdf_paths)
    if not pdf_list:
        raise ValueError("No PDF paths supplied.")

    return index_directory(
        input_dir=str(Path(pdf_list[0]).parent),
        config=IndexingConfig(glob_pattern="*.pdf"),
        index_path=index_path,
        metadata_path=metadata_path,
        model_name=model_name,
    )


def ask_question(
    question: str,
    index_path: str = "data/index.faiss",
    metadata_path: str = "data/metadata.json",
    top_k: int = 3,
    model_name: str = DEFAULT_MODEL,
    filters: dict[str, Any] | None = None,
    include_passages: bool = False,
    llm_provider: str = "echo",
    llm_model: str = "gpt-4o-mini",
) -> dict[str, object]:
    response = query_index(
        question=question,
        options=QueryOptions(
            top_k=top_k,
            filters=filters,
            include_passages=include_passages,
            llm_provider=llm_provider,
            llm_model=llm_model,
        ),
        index_path=index_path,
        metadata_path=metadata_path,
        model_name=model_name,
    )
    return response.to_dict()
