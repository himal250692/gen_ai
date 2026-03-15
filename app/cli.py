from __future__ import annotations

import argparse
import json

from app.core import ask_question, index_directory
from app.indexing import IndexingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index PDFs and ask questions over the indexed content.")
    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", help="Index PDF files from a directory.")
    index_cmd.add_argument("--input-dir", default="data/pdfs", help="Directory containing PDF files.")
    index_cmd.add_argument("--glob", default="*.pdf", help="Glob pattern used to discover PDF files.")
    index_cmd.add_argument("--domain-tag", default="general", help="Domain tag persisted in chunk metadata.")
    index_cmd.add_argument("--chunk-size", type=int, default=800)
    index_cmd.add_argument("--chunk-overlap", type=int, default=100)
    index_cmd.add_argument("--trace-path", default="data/indexing_trace.json")
    index_cmd.add_argument("--index-path", default="data/index.faiss")
    index_cmd.add_argument("--metadata-path", default="data/metadata.json")

    ask_cmd = sub.add_parser("ask", help="Query an existing index.")
    ask_cmd.add_argument("question", help="Question to ask.")
    ask_cmd.add_argument("--index-path", default="data/index.faiss")
    ask_cmd.add_argument("--metadata-path", default="data/metadata.json")
    ask_cmd.add_argument("--top-k", type=int, default=3)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        config = IndexingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            glob_pattern=args.glob,
            domain_tag=args.domain_tag,
            trace_path=args.trace_path,
        )
        stats = index_directory(
            input_dir=args.input_dir,
            config=config,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
        print(json.dumps({"status": "indexed", **stats}, indent=2))
    elif args.command == "ask":
        response = ask_question(
            args.question,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            top_k=args.top_k,
        )
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
