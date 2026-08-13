"""Central config: loads env vars once for the whole app."""

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

# Load .env into os.environ (won't override real env vars).
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable object holding every config value."""
    openai_api_key: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_env: str
    pinecone_index_name: str
    pinecone_dimension: int
    pinecone_namespace: str
    pinecone_metric: str

    corpus_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_steps: int
    answer_temperature: float

    # Server
    api_host: str
    api_port: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read all settings from env vars (with safe defaults)."""
    config = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_env=os.getenv("PINECONE_ENV", "us-east-1"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "legixo-docs"),
        pinecone_dimension=int(os.getenv("PINECONE_DIMENSION", "1536")),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "takehome"),
        pinecone_metric=os.getenv("PINECONE_INDEX_METRIC", "cosine"),
        corpus_dir=os.getenv("CORPUS_DIR", "./data/raw"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "5")),
        max_steps=int(os.getenv("MAX_STEPS", "3")),
        answer_temperature=float(os.getenv("ANSWER_TEMPERATURE", "0.0")),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )
    return config