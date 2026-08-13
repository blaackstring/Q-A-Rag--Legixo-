"""Embeddings: turns text (chunks and questions) into vectors.

Uses OpenAi text-embedding-3-smal` (1536 dims). Kept in one module so the
rest of the app never cares which provider is used.
"""

from langchain_openai import OpenAIEmbeddings
embedder: OpenAIEmbeddings | None = None


def get_embedder() -> OpenAIEmbeddings:
    """Return the shared embedder (created once)."""
    global embedder
    if embedder is None:
        embedder = OpenAIEmbeddings (model= "text-embedding-3-small")
    return embedder


def embed_texts(texts: list) -> list:
    """Embed a list of strings -> list of float vectors."""
    return get_embedder().embed_documents(texts)


def embed_query(text: str) -> list:
    """Embed one query into a single float vector for Pinecone search."""
    return get_embedder().embed_query(text)