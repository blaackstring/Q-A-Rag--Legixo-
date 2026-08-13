"""Rewrite node: bad-path branch that re-phrases the question and loops back.

When the grader rejects everything, we rewrite the query (e.g. broader wording)
and go back to `retrieve`, giving weak searches a second chance. This makes the
graph non-linear without spinning forever.
"""

from ..common.llm import run_prompt
from ..state import log_trace


def rewrite_node(state: dict) -> dict:
    """Produce a rewritten question tuned for retrieval and update the state."""
    original = state["question"]
    rewritten = run_prompt(
        system=(
            "You are a retrieval query rewriter. Rewrite the user's question so a "
            "semantic search over LEGAL documents (briefs, contracts, statutes, "
            "notes) finds the most relevant passages. Keep the same facts and "
            "entities. Reply with the rewritten query only."
        ),
        user=f"Original question: {original}",
        temperature=0.2,
    )
    trace_update = log_trace(
        state,
        node="rewrite",
        detail=f"rewrote question (step={state['step']})",
        metadata={"before": original, "after": rewritten},
    )
    return {"rewritten_question": rewritten, **trace_update}