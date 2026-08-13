"""Answer node: write the final grounded answer with citations from accepted chunks.
"""

from ..common.llm import run_prompt
from ..state import log_trace


def answer_node(state: dict) -> dict:
    """Build the answer text from accepted chunks and attach real citations."""
    chunks = state["relevant_chunks"]
    question = state["question"]

    sources_text = "\n\n".join(
        f"[{i + 1}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks)
    )
    system = (
        "You are a legal-document assistant. Answer the question using ONLY the "
        "provided excerpts. NEVER invent facts, dates, or numbers. If the excerpts "
        "do not fully answer the question, say what is not covered. Cite excerpts "
        "by their bracketed numbers the  source of document."
    )
    answer = run_prompt(
        system=system,
        user=f"QUESTION:\n{question}\n\nEXCERPTS:\n{sources_text}",
        temperature=0.0,
    )

    # Citations come straight from the accepted chunks - never hallucinated.
    citations = [
        {
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "text": c["text"],
            "score": round(c["score"], 4),
        }
        for c in chunks
    ]
    trace_update = log_trace(
        state,
        node="answer",
        detail=f"wrote answer with {len(citations)} citation(s)",
        metadata={"citation_sources": list({c['source'] for c in citations})},
    )
    return {"answer": answer, "citations": citations, "status": "answered", **trace_update}