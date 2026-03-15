from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.indexing import IndexingConfig
from app.vectorstore import DEFAULT_MODEL, VectorStore, VectorStoreConfig


@dataclass
class SearchResult:
    chunk: str
    source: str
    score: float
    metadata: dict[str, Any]


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
) -> dict[str, object]:
    store = VectorStore(
        VectorStoreConfig(
            backend="faiss",
            model_name=model_name,
            faiss_index_path=index_path,
            faiss_metadata_path=metadata_path,
        )
    )
    hits = store.search(question, top_k=top_k, filters=filters)

    results: list[SearchResult] = []
    for hit in hits:
        metadata = hit["metadata"]
        source = f"{metadata['filename']}:page{metadata['page_number']}"
        results.append(
            SearchResult(
                chunk=hit["chunk_text"],
                source=source,
                score=hit["score"],
                metadata=metadata,
            )
        )

    answer = (
        "Top matching context snippets:\n\n"
        + "\n\n".join(f"[{r.source}] {r.chunk}" for r in results)
        if results
        else "No relevant context found."
    )

    return {
        "question": question,
        "answer": answer,
        "results": [r.__dict__ for r in results],
    }
