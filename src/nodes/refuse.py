"""Refuse node: honest "not found in the documents" outcome for the bad path.

Reached either because no chunk was ever relevant or because we exceeded the
loop budget. We never hallucinate an answer; a clear refusal is correct for
out-of-corpus questions.
"""

from ..state import log_trace


def refuse_node(state: dict) -> dict:
    """Mark the run as not_found and set a readable refusal answer."""
    message = (
        "I could not find relevant information in the available documents. "
    )
    trace_update = log_trace(
        state,
        node="refuse",
        detail=f"no relevant chunks found after {state['step']} step(s); refusing",
        metadata={"max_steps": state["max_steps"]},
    )
    return {"answer": message, "citations": [], "status": "not_found", **trace_update}