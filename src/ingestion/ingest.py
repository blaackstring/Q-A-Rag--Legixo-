"""Ingestion pipeline: load -> chunk -> embed -> upsert into Pinecone.

Runnable both as a function (`run_ingest`) and as a CLI command
(`python -m src.ingestion.ingest`), so reviewers can load the corpus easily.
"""

import argparse
from dotenv import load_dotenv
from ..common.config import get_settings
from .loaders import load_corpus
from .chunking import split_documents
from ..common.vectorstore import upsert_chunks, delete_namespace, index_stats


def run_ingest(corpus_dir: str | None = None, reset: bool = False) -> dict:
    """Load files, split, embed, and upsert vectors to Pinecone.

    If `reset` is True, delete the namespace first so stale chunks from an old
    corpus are removed.
    """
    settings = get_settings()
    corpus_dir = corpus_dir or settings.corpus_dir

    if reset:
        delete_namespace()  # wipe old vectors -> start clean

    docs = load_corpus(corpus_dir)                       # read files recursively
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)  # split
    count = upsert_chunks(chunks)                        # embed + save

    stats = index_stats()
    return {
        "files_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_upserted": count,
        "pinecone": stats,
    }


def main() -> None:
    """CLI entry point so ingestion works without starting the web server."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest the corpus into Pinecone.")
    parser.add_argument("--corpus-dir", default=None, help="Folder holding the documents (default: from .env)")
    parser.add_argument("--reset", action="store_true", help="Delete the namespace before ingesting")
    args = parser.parse_args()

    report = run_ingest(corpus_dir=args.corpus_dir, reset=args.reset)
    print(f"Ingest done: {report}")  # single-line summary for humans


if __name__ == "__main__":
    main()