from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    chunk: str
    source: str
    score: float


def _load_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf_chunks(pdf_path: Path) -> list[dict[str, str]]:
    reader = PdfReader(str(pdf_path))
    chunks: list[dict[str, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in _chunk_text(text):
            chunks.append({"text": chunk, "source": f"{pdf_path.name}:page{page_number}"})
    return chunks


def index_pdfs(
    pdf_paths: Iterable[str],
    index_path: str = "data/index.faiss",
    metadata_path: str = "data/metadata.json",
    model_name: str = DEFAULT_MODEL,
) -> dict[str, int]:
    docs: list[dict[str, str]] = []
    for path in pdf_paths:
        docs.extend(_extract_pdf_chunks(Path(path)))

    if not docs:
        raise ValueError("No text chunks found in the provided PDFs.")

    embedder = _load_embedder(model_name)
    vectors = embedder.encode([d["text"] for d in docs], normalize_embeddings=True)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with Path(metadata_path).open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    return {"documents": len(set(Path(p).name for p in pdf_paths)), "chunks": len(docs)}


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
        results.append(SearchResult(chunk=entry["text"], source=entry["source"], score=float(score)))

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
