from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import faiss
from sentence_transformers import SentenceTransformer

from app.indexing import IndexingConfig, build_index

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class VectorStoreConfig:
    backend: Literal["faiss", "qdrant"] = "faiss"
    model_name: str = DEFAULT_MODEL
    faiss_index_path: str = "data/index.faiss"
    faiss_metadata_path: str = "data/metadata.json"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pdf_chunks"


class VectorStore:
    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self.config = config or VectorStoreConfig()
        self._embedder: SentenceTransformer | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.config.model_name)
        return self._embedder

    def index_directory(self, input_dir: str = "data/pdfs", indexing_config: IndexingConfig | None = None) -> dict[str, int]:
        chunk_records = build_index(input_dir=input_dir, config=indexing_config)
        self.index_chunks(chunk_records)
        return {
            "documents": len({record["metadata"]["filename"] for record in chunk_records}),
            "chunks": len(chunk_records),
        }

    def index_chunks(self, chunk_records: list[dict[str, Any]]) -> None:
        if not chunk_records:
            raise ValueError("No chunks were provided for vector indexing.")

        vectors = self.embedder.encode(
            [record["chunk_text"] for record in chunk_records],
            normalize_embeddings=True,
        )

        if self.config.backend == "faiss":
            self._index_to_faiss(vectors, chunk_records)
            return

        self._index_to_qdrant(vectors, chunk_records)

    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vec = self.embedder.encode([query], normalize_embeddings=True)

        if self.config.backend == "faiss":
            return self._search_faiss(query_vec, top_k, filters)

        return self._search_qdrant(query_vec, top_k, filters)

    def _index_to_faiss(self, vectors: Any, chunk_records: list[dict[str, Any]]) -> None:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        index_path = Path(self.config.faiss_index_path)
        metadata_path = Path(self.config.faiss_metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path))
        metadata_path.write_text(json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8")

    def _search_faiss(self, query_vec: Any, top_k: int, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        index_path = Path(self.config.faiss_index_path)
        metadata_path = Path(self.config.faiss_metadata_path)
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("FAISS index files were not found. Index documents first.")

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        search_k = min(len(metadata), max(top_k * 5, top_k))
        scores, indices = index.search(query_vec, search_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            record = metadata[idx]
            if not _metadata_matches(record["metadata"], filters):
                continue
            metadata = dict(record["metadata"])
            metadata.setdefault("chunk_id", record.get("chunk_id"))
            results.append(
                {
                    "chunk_text": record["chunk_text"],
                    "score": float(score),
                    "metadata": metadata,
                }
            )
            if len(results) == top_k:
                break

        return results

    def _index_to_qdrant(self, vectors: Any, chunk_records: list[dict[str, Any]]) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qdrant_models
        except ImportError as exc:
            raise ImportError("qdrant-client is required for the qdrant backend.") from exc

        client = QdrantClient(url=self.config.qdrant_url)
        vector_size = vectors.shape[1]
        client.recreate_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE),
        )

        points = []
        for idx, (vector, record) in enumerate(zip(vectors, chunk_records, strict=True)):
            points.append(
                qdrant_models.PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={
                        "chunk_text": record["chunk_text"],
                        "chunk_id": record.get("chunk_id"),
                        **record["metadata"],
                    },
                )
            )

        client.upsert(collection_name=self.config.qdrant_collection, points=points, wait=True)

    def _search_qdrant(self, query_vec: Any, top_k: int, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qdrant_models
        except ImportError as exc:
            raise ImportError("qdrant-client is required for the qdrant backend.") from exc

        client = QdrantClient(url=self.config.qdrant_url)
        query_filter = _build_qdrant_filter(qdrant_models, filters)
        points = client.search(
            collection_name=self.config.qdrant_collection,
            query_vector=query_vec[0].tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        results: list[dict[str, Any]] = []
        for point in points:
            payload = dict(point.payload or {})
            chunk_text = payload.pop("chunk_text", "")
            results.append(
                {
                    "chunk_text": chunk_text,
                    "score": float(point.score),
                    "metadata": payload,
                }
            )
        return results


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True

    page_min = filters.get("page_min")
    page_max = filters.get("page_max")
    page_number = metadata.get("page_number")
    if page_min is not None and (page_number is None or page_number < page_min):
        return False
    if page_max is not None and (page_number is None or page_number > page_max):
        return False

    for key, value in filters.items():
        if key in {"page_min", "page_max"}:
            continue
        if metadata.get(key) != value:
            return False
    return True


def _build_qdrant_filter(qdrant_models: Any, filters: dict[str, Any] | None) -> Any:
    if not filters:
        return None

    conditions: list[Any] = []
    page_min = filters.get("page_min")
    page_max = filters.get("page_max")

    if page_min is not None or page_max is not None:
        conditions.append(
            qdrant_models.FieldCondition(
                key="page_number",
                range=qdrant_models.Range(gte=page_min, lte=page_max),
            )
        )

    for key, value in filters.items():
        if key in {"page_min", "page_max"}:
            continue
        conditions.append(
            qdrant_models.FieldCondition(key=key, match=qdrant_models.MatchValue(value=value))
        )

    return qdrant_models.Filter(must=conditions)
