from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.language_models.llms import LLM

from app.vectorstore import DEFAULT_MODEL, VectorStore, VectorStoreConfig


class EchoCitationLLM(LLM):
    """Deterministic local LLM used when no external provider is configured."""

    @property
    def _llm_type(self) -> str:
        return "echo-citation"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any | None = None, **kwargs: Any) -> str:
        citation_ids = re.findall(r"\[(\d+)\]", prompt)
        unique_ids: list[str] = []
        for cid in citation_ids:
            if cid not in unique_ids:
                unique_ids.append(cid)

        if not unique_ids:
            return "I could not find relevant indexed passages to answer your question."

        joined = ", ".join(f"[{cid}]" for cid in unique_ids[:3])
        return (
            "Based on the retrieved passages, here is a grounded response. "
            f"Please validate details against citations {joined}."
        )


@dataclass
class QueryOptions:
    top_k: int = 3
    filters: dict[str, Any] | None = None
    include_passages: bool = False
    llm_provider: str = "echo"
    llm_model: str = "gpt-4o-mini"


@dataclass
class Citation:
    filename: str
    page_number: int | None
    chunk_id: str


@dataclass
class QueryResponse:
    question: str
    answer: str
    citations: list[Citation]
    passages: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
        }
        if self.passages is not None:
            payload["passages"] = self.passages
        return payload


def _build_llm(provider: str, model: str) -> LLM:
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("langchain-openai is required for llm_provider='openai'.") from exc
        return ChatOpenAI(model=model, temperature=0)
    if provider == "echo":
        return EchoCitationLLM()
    raise ValueError("Unsupported llm_provider. Use 'echo' or 'openai'.")


def _format_prompt(question: str, retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return (
            "You are a retrieval QA assistant.\n"
            "No context was retrieved. Reply that no grounded answer can be produced."
        )

    blocks: list[str] = []
    for idx, hit in enumerate(retrieved, start=1):
        metadata = hit["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"filename: {metadata.get('filename', 'unknown')}",
                    f"page_number: {metadata.get('page_number', 'unknown')}",
                    f"chunk_id: {metadata.get('chunk_id', 'unknown')}",
                    f"score: {hit['score']:.4f}",
                    f"text: {hit['chunk_text']}",
                ]
            )
        )

    context = "\n\n".join(blocks)
    return (
        "You are a retrieval QA assistant. Answer only using the retrieved chunks below. "
        "Include bracket citations like [1], [2] that reference chunk ids.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved chunks:\n{context}\n\n"
        "Return a concise answer with citations."
    )


def query_index(
    question: str,
    options: QueryOptions | None = None,
    index_path: str = "data/index.faiss",
    metadata_path: str = "data/metadata.json",
    model_name: str = DEFAULT_MODEL,
) -> QueryResponse:
    effective_options = options or QueryOptions()

    store = VectorStore(
        VectorStoreConfig(
            backend="faiss",
            model_name=model_name,
            faiss_index_path=index_path,
            faiss_metadata_path=metadata_path,
        )
    )

    retrieved = store.search(
        question,
        top_k=effective_options.top_k,
        filters=effective_options.filters,
    )

    prompt = _format_prompt(question, retrieved)
    llm = _build_llm(effective_options.llm_provider, effective_options.llm_model)
    answer = llm.invoke(prompt)

    citations: list[Citation] = []
    for hit in retrieved:
        metadata = hit["metadata"]
        citations.append(
            Citation(
                filename=str(metadata.get("filename", "unknown")),
                page_number=metadata.get("page_number"),
                chunk_id=str(metadata.get("chunk_id", "unknown")),
            )
        )

    passages = None
    if effective_options.include_passages:
        passages = retrieved

    return QueryResponse(
        question=question,
        answer=str(answer),
        citations=citations,
        passages=passages,
    )


def parse_filter_args(filters: list[str] | None) -> dict[str, Any] | None:
    if not filters:
        return None

    parsed: dict[str, Any] = {}
    for item in filters:
        if "=" not in item:
            raise ValueError(f"Invalid filter '{item}'. Expected key=value format.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = raw_value.strip()

        if not key:
            raise ValueError(f"Invalid filter '{item}'. Key cannot be empty.")

        if value.isdigit():
            parsed[key] = int(value)
        else:
            try:
                parsed[key] = float(value)
            except ValueError:
                if value.lower() in {"true", "false"}:
                    parsed[key] = value.lower() == "true"
                elif value.startswith("{") or value.startswith("["):
                    parsed[key] = json.loads(value)
                else:
                    parsed[key] = value

    return parsed
