from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
from sentence_transformers import SentenceTransformer

from app.indexing import IndexingConfig, build_index

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    chunk: str
    source: str
    score: float



def _load_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)



def index_directory(
    input_dir: str = "data/pdfs",
    config: IndexingConfig | None = None,
    index_path: str = "data/index.faiss",
    metadata_path: str = "data/metadata.json",
    model_name: str = DEFAULT_MODEL,
) -> dict[str, int]:
    chunk_records = build_index(input_dir=input_dir, config=config)

    if not chunk_records:
        raise ValueError("No text chunks found in the provided PDFs.")

    embedder = _load_embedder(model_name)
    vectors = embedder.encode([record["chunk_text"] for record in chunk_records], normalize_embeddings=True)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    Path(metadata_path).write_text(json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "documents": len({record["metadata"]["filename"] for record in chunk_records}),
        "chunks": len(chunk_records),
    }



def index_pdfs(
    pdf_paths: Iterable[str],
    index_path: str = "data/index.faiss",
    metadata_path: str = "data/metadata.json",
    model_name: str = DEFAULT_MODEL,
) -> dict[str, int]:
    pdf_list = [Path(path) for path in pdf_paths]
    if not pdf_list:
        raise ValueError("No PDF paths supplied.")

    input_dir = str(pdf_list[0].parent)
    config = IndexingConfig(glob_pattern="*.pdf")
    return index_directory(
        input_dir=input_dir,
        config=config,
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
) -> dict[str, object]:
    index_file = Path(index_path)
    metadata_file = Path(metadata_path)

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Index files not found. Run indexing first.")

    index = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    embedder = _load_embedder(model_name)
    query_vec = embedder.encode([question], normalize_embeddings=True)

    distances, indices = index.search(query_vec, top_k)

    results: list[SearchResult] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        source = f"{entry['metadata']['filename']}:page{entry['metadata']['page_number']}"
        results.append(SearchResult(chunk=entry["chunk_text"], source=source, score=float(score)))

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
