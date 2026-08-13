"""Retrieve node: find candidate chunks in Pinecone for the current question.

Embeds the (possibly rewritten) question and searches the vector index; each
matching chunk gets a score so later nodes can judge quality.
"""

from ..common.embeddings import embed_query
from ..common.vectorstore import query_chunks
from ..state import log_trace


def retrieve_node(state: dict) -> dict:
    """Embed the active question and pull `top_k` candidate chunks from Pinecone."""
    question = state.get("rewritten_question") or state["question"]
    vector = embed_query(question)
    chunks = query_chunks(vector, top_k=state["top_k"], include_text=True)
    trace_update = log_trace(
        state,
        node="retrieve",
        detail=f"retrieved {len(chunks)} chunks for: {question[:80]}",
        metadata={"top_k": state["top_k"], "scores": [round(c["score"], 3) for c in chunks]},
    )
    # `step: 1` + the reducer bumps the loop counter exactly once per visit.
    return {"chunks": chunks, "step": 1, **trace_update}