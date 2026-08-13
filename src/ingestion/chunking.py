"""Chunking: split loaded documents into small pieces ready for embeddings.

Uses LangChain's RecursiveCharacterTextSplitter: it splits on markdown /
paragraph / line boundaries first, then falls back to character splits, so
chunks keep readable text. Every chunk gets a deterministic `chunk_id` plus
its `source` file path so we can cite exactly where an answer came from.
"""

import hashlib
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_chunk_id(source: str, text: str) -> str:
    """Stable id for a chunk: md5 of (source + text).

    Deterministic ids mean re-running ingest overwrites the same vectors
    instead of duplicating them.
    """
    raw = f"{source}::{text}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 80) -> list:
    """Split every loaded Document into chunks with ids + source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,              # count by characters, not tokens
        separators=["\n## ", "\n---\n", "\n\n", "\n", " ", ""],
        add_start_index=False,
    )
    chunks: list = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name  # basename for clean citations
        pieces = splitter.split_text(doc.page_content)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_id": make_chunk_id(source, piece),
                    "source": source,
                    "text": piece,
                    "score": 0.0,
                    "relevant": False,
                }
            )
    return chunks