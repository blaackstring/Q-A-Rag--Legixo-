# Legal Document Q&A API (RAG)

**FLOW**

```
START → ingestt->retrieve → grade → [answer | rewrite→retrieve (loop) | refuse]
```

---

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
copy .env.example .env            # Windows
# cp .env.example .env            # Mac/Linux
```

Fill `.env` with your keys:

| Var | From |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com |
| `PINECONE_API_KEY` | app.pinecone.io → API Keys |

The Pinecone index is **auto-created** on first ingest. If you make it manually: **1536 dims, cosine, aws/us-east-1**.

---

## Run

```bash
# 1. Load documents into Pinecone
python -m src.ingest    OR can do request to url/ingest API          # or: python -m src.ingest --reset (wipe first)

# 2. Start server
uv run uvicorn src.app:app --reload --port 8000             # → http://localhost:8000/docs (Swagger)
```

---

## API

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What notice period applies to the Bluecrest agreement?"}'
```

Optional fields: `top_k` (1–20), `max_steps` (1–5), `trace` (bool).

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'
# {"reset": true}  →  delete namespace first
```

### `GET /health` · `GET /stats`

---

## Structure

```
src/
  app.py              FastAPI routes
  graph.py            LangGraph pipeline + branch router
  state.py            shared graph state
  nodes/              retrieve · grade · rewrite · answer · refuse
  common/             config · llm · embeddings · vectorstore (Pinecone)
  ingestion/          load · chunk · embed · save
data/raw/             sample documents
```

## Notes

- Corpus is fictional legal notes — use your own keys only.
- `python -m src.ingest` twice is safe (same chunk ids → overwrite, no dupes).
- Questions the docs can't answer → `status: not_found`, no made-up reply.
