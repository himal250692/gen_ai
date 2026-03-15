from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class IndexingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100
    glob_pattern: str = "*.pdf"
    domain_tag: str = "general"
    trace_path: str = "data/indexing_trace.json"


@dataclass
class ChunkRecord:
    chunk_text: str
    chunk_id: str
    metadata: dict[str, Any]



def _discover_pdfs(input_dir: str, glob_pattern: str) -> list[Path]:
    base = Path(input_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted([path for path in base.glob(glob_pattern) if path.is_file()])



def _extract_page_text(pdf_path: Path, domain_tag: str) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(
            {
                "filename": pdf_path.name,
                "page_number": page_number,
                "source_path": str(pdf_path),
                "domain_tag": domain_tag,
                "text": text,
            }
        )

    return pages



def _split_pages(pages: list[dict[str, Any]], chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[ChunkRecord] = []

    for page in pages:
        if not page["text"]:
            continue

        split_texts = splitter.split_text(page["text"])
        for chunk_index, chunk_text in enumerate(split_texts, start=1):
            chunk_id = (
                f"{page['filename']}:p{page['page_number']}:"
                f"{chunk_index}:{hashlib.sha1((page['source_path'] + chunk_text).encode('utf-8')).hexdigest()[:12]}"
            )
            chunks.append(
                ChunkRecord(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    metadata={
                        "filename": page["filename"],
                        "page_number": page["page_number"],
                        "source_path": page["source_path"],
                        "domain_tag": page["domain_tag"],
                        "chunk_index": chunk_index,
                    },
                )
            )

    return chunks



def _persist_trace(trace_path: str, payload: dict[str, Any]) -> None:
    trace_file = Path(trace_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def build_index(input_dir: str = "data/pdfs", config: IndexingConfig | None = None) -> list[dict[str, Any]]:
    effective_config = config or IndexingConfig()
    pdf_paths = _discover_pdfs(input_dir, effective_config.glob_pattern)

    all_pages: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        all_pages.extend(_extract_page_text(pdf_path, effective_config.domain_tag))

    chunks = _split_pages(
        all_pages,
        chunk_size=effective_config.chunk_size,
        chunk_overlap=effective_config.chunk_overlap,
    )

    chunk_payload = [asdict(chunk) for chunk in chunks]
    _persist_trace(
        effective_config.trace_path,
        {
            "input_dir": input_dir,
            "config": asdict(effective_config),
            "documents": [str(path) for path in pdf_paths],
            "pages": all_pages,
            "chunks": chunk_payload,
        },
    )

    return chunk_payload
