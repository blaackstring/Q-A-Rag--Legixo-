"""Pinecone vector store: the real vector index for search + ingestion.

All Pinecon -specific calls live here (create index, upsert, query, dlete)
"""

import time
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from .config import Settings, get_settings

_client: Optional[Pinecone] = None
_index = None


def get_client(settings: Settings | None = None) -> Pinecone:
    """Return the shared Pinecone client (created once per process)."""
    global _client
    if _client is None:
        settings = settings or get_settings()
        _client = Pinecone(api_key=settings.pinecone_api_key)
    return _client


def get_or_create_index(settings: Settings | None = None):
    """Return the index, creating it first if it does not exist.
    """
    global _index
    settings = settings or get_settings()
    client = get_client(settings)

    if _index is None:
        existing = client.list_indexes().names()
        if settings.pinecone_index_name not in existing:
            client.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_env),
            )
            wait_for_index_ready(client, settings.pinecone_index_name)
        _index = client.Index(settings.pinecone_index_name)
    return _index


def wait_for_index_ready(client: Pinecone, index_name: str, timeout: int = 60) -> None:
    """Poll index status until READY (or raise after timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        desc = client.describe_index(index_name)
        if desc.status and desc.status.state == "Ready":
            return
        time.sleep(2)

    raise TimeoutError(f"Pinecone index '{index_name}' not ready after {timeout}s")


def upsert_chunks(chunks: list, namespace: str | None = None) -> int:
    """Embed every chunk, upsert its vector into Pinecone; return count.
    """
    from .embeddings import embed_texts

    index = get_or_create_index()
    namespace = namespace or get_settings().pinecone_namespace

    texts = [c["text"] for c in chunks]
    vectors = []
    for chunk, vector in zip(chunks, embed_texts(texts)):
        vectors.append(
            (
                chunk["chunk_id"],
                vector,
                {"chunk_id": chunk["chunk_id"], "source": chunk["source"], "text": chunk["text"]},
            )
        )
# batches of 100 to avoid huge payloads.
    for batch_start in range(0, len(vectors), 100):
        batch = vectors[batch_start : batch_start + 100]
        index.upsert(vectors=batch, namespace=namespace)
    return len(vectors)


def query_chunks(vector: list, top_k: int, namespace: str | None = None, include_text: bool = True) -> list:
    """Search Pinecone with a query vector; return the top_k chunk dicts."""
    index = get_or_create_index()
    namespace = namespace or get_settings().pinecone_namespace
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    chunks = []
    for hit in result.get("matches", []):
        meta = hit.get("metadata", {})
        chunks.append(
            {
                "chunk_id": meta.get("chunk_id", hit.get("id", "")),
                "source": meta.get("source", "unknown"),
                "text": meta.get("text", ""),
                "score": hit.get("score", 0.0),
                "relevant": False,
            }
        )
    return chunks


def delete_namespace(namespace: str | None = None) -> None:
    """Delete every vector in the namespace (used by `--reset` ingest)."""
    index = get_or_create_index()
    namespace = namespace or get_settings().pinecone_namespace
    index.delete(delete_all=True, namespace=namespace)


def index_stats(namespace: str | None = None) -> dict:
    """Return Pinecone stats (index name, total vectors, namespaced count)."""
    index = get_or_create_index()
    namespace = namespace or get_settings().pinecone_namespace
    raw = index.describe_index_stats()
    return {
        "index_name": get_settings().pinecone_index_name,
        "total_vectors": raw.get("total_vector_count", 0),
        "namespace_vectors": raw.get("namespaces", {}).get(namespace, {}).get("vector_count", 0),
    }