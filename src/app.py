"""FastAPI app: the HTTP surface of the project.

Exposes the Q&A endpoint (`POST /ask`) plus an ingestion route (`POST /ingest`)
so reviewers never need a REPL - everything runs over HTTP or `python -m src.ingest`.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common.config import get_settings
from .graph import ask_question
from .ingestion.ingest import run_ingest
from .common.vectorstore import index_stats

app = FastAPI()


# Request/response schemas
class AskRequest(BaseModel):
    """Body accepted by POST /ask: a question plus optional tuning knobs."""

    question: str = Field(min_length=1, max_length=2000, description="The user's question")
    top_k: int = Field(default=None, ge=1, le=20, description="Chunks to fetch from Pinecone")
    max_steps: int = Field(default=None, ge=1, le=5, description="Max retrieval loops")
    trace: bool = Field(default=True, description="Include the node trace in the response")


class IngestRequest(BaseModel):
    """Body accepted by POST /ingest: corpus folder (optional) and reset flag."""

    corpus_dir: str = Field(default=None, description="Folder holding documents (default: .env CORPUS_DIR)")
    reset: bool = Field(default=False, description="Delete the namespace before ingesting")


# Routes

@app.get("/")
def root() -> dict:
    """Return a tiny API map so the server is self-documenting at a glance."""
    return {
        "app": "Legixo take-home Q&A API",
        "endpoints": {
            "ask": "POST /ask  {question, top_k?, max_steps?, trace?}",
            "ingest": "POST /ingest  {corpus_dir?, reset?}",
            "health": "GET /health",
            "stats": "GET /stats",
        },
    }


@app.get("/health")
def health() -> dict:
    """Return service liveness plus a Pinecone connectivity check."""
    try:
        return {"status": "ok", "pinecone": index_stats()}
    except Exception as exc:  # noqa: BLE001 - surface any provider problem
        raise HTTPException(status_code=503, detail=f"pinecone unavailable: {exc}")


@app.get("/stats")
def stats() -> dict:
    """Return current Pinecone index stats (vector counts per namespace)."""
    return index_stats()


@app.post("/ask")
def ask(body: AskRequest) -> dict:
    """Answer one question through the LangGraph pipeline (retrieve/grade/answer)."""
    try:
        result = ask_question(
            question=body.question,
            top_k=body.top_k,
            max_steps=body.max_steps,
            trace=body.trace,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}")
    return result


@app.post("/ingest")
def ingest(body: IngestRequest) -> dict:
    """Load, chunk, embed and upsert the corpus into Pinecone (via /ingest)."""
    try:
        report = run_ingest(corpus_dir=body.corpus_dir, reset=body.reset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ingest error: {exc}")
    return report


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.api:app", host=settings.api_host, port=settings.api_port, reload=True)


if __name__ == "__main__":
    main()