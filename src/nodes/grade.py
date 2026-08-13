"""Grade node: decides whether each retrieved chunk can answer the question.

Asks the LLM a per-chunk yesno and only passes accepted chunks forward.
"""

from ..common.llm import run_prompt
from ..state import log_trace


def grade_chunk(chunk: dict, question: str) -> bool:
    """Return True if the LLM judges the chunk could support the question.

    The model replies with exactly YES or NO; anything else counts as "no help"
    (safer than a wrong yes).
    """
    prompt = (
        "You decide if a document excerpt can help answer the user's question.\n"
        "Answer with exactly YES or NO. Do not add anything else.\n"
        f"QUESTION: {question}\n"
        f"EXCERPT: {chunk['text'][:1000]}\n"
        "Is this excerpt relevant to answering the question?"
    )
    verdict = run_prompt(
        system="You are a strict relevance checker. Reply only YES or NO.",
        user=prompt,
        temperature=0.0,
    )
    return verdict.upper().startswith("YES")


def grade_node(state: dict) -> dict:
    """Mark every chunk as relevant or not; keep only the accepted ones.
    """
    question = state["question"]
    accepted = []

    for chunk in state["chunks"]:
        ok = grade_chunk(chunk, question)
        chunk["relevant"]=ok
        if ok:
            accepted.append(chunk)

    trace_update = log_trace(
        state,
        node= "grade",
        detail=f"accepted {len(accepted)}/{len(state['chunks'])} chunks",
        metadata={"chunk_ids  ": [c["chunk_id"] for c in accepted]},
    )
    return {"relevant_chunks": accepted, "relevant_count": len(accepted), **trace_update}