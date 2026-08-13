"""Shared LangGraph state: the dict that flows through every node.

Keeps one `trace` list so reviewers can see, for any request, exactly which
node ran when and what it decided (used by the optional `trace` in /ask).
"""

from typing import Annotated, Any, TypedDict, Literal


# Reducers: how LangGraph merges a node's return value into the shared state.

def bump_step(current: int, update: int) -> int:
    """Reducer for counters: the node sets `step=1` and we add it."""
    return current + update


def append_entries(current: list, update: list) -> list:
    """Reducer for lists: append new entries after existing ones."""
    return [*current, *update]


def append_chunks(current: list, update: list) -> list:
    """Reducer for chunks: keep only fully-formed chunk dicts."""
    return current + [c for c in update if isinstance(c, dict)]


# ---------------------------------------------------------------------------
# TypedDict shapes (used only for documentation and mypy, never enforced).
# ---------------------------------------------------------------------------
class ChunkDict(TypedDict, total=False):
    """A single retrieved chunk: text plus the metadata needed to cite it."""

    chunk_id: str       # deterministic id: md5(source + text)
    source: str         # source file name (shown in citations)
    text: str           # the chunk content (shown in citations)
    score: float        # Pinecone cosine similarity in [0, 1]
    relevant: bool      # grader verdict (True = good enough to cite)


class TraceStep(TypedDict, total=False):
    """One entry in the run trace: which node ran and what it decided."""

    node: str
    detail: str
    metadata: dict


# ---------------------------------------------------------------------------
# The state TypedDict every LangGraph node reads and (partially) writes.
# ---------------------------------------------------------------------------
class AskState(TypedDict, total=False):
    """Full state for one /ask question as it walks through the graph."""

    # --- Input ---
    question: str                                  # user question to answer

    # --- Retrieval ---
    top_k: int                                     # how many chunks to fetch
    rewritten_question: str                        # bad path: re-phrased query
    chunks: Annotated[list, append_chunks]         # raw chunks from every loop

    # --- Grading / branch ---
    relevant_chunks: Annotated[list, append_chunks]  # chunks grader accepted
    relevant_count: int                            # number of accepted chunks

    # --- Answer ---
    answer: str                                    # final grounded answer text
    citations: Annotated[list, append_entries]     # dicts returned by /ask API
    status: Literal["answered", "not_found", "error"]  # end-of-run outcome

    # --- Loop guards (cannot spin forever) ---
    max_steps: int                                 # hard cap on retrieve attempts
    step: Annotated[int, bump_step]                # how many retrieve loops done

    # --- Telemetry ---
    trace: Annotated[list, append_entries]         # ordered node event log
    error: str                                     # message on unexpected failure


def initial_state(question: str, top_k: int = 5, max_steps: int = 3) -> dict:
    """Produce a fresh, empty state dict with sensible defaults for each run.

    Every /ask request starts from here so no stale field leaks across calls.
    """
    return {
        "question": question,
        "top_k": top_k,
        "rewritten_question": question,
        "chunks": [],
        "relevant_chunks": [],
        "relevant_count": 0,
        "answer": "",
        "citations": [],
        "status": "error",
        "max_steps": max_steps,
        "step": 0,
        "trace": [],
        "error": "",
    }


def log_trace(state: dict, node: str, detail: str, metadata: dict | None = None) -> dict:
    """Build the trace update a node returns; central place for trace format."""
    entry: dict[str, Any] = {"node": node, "detail": detail}
    if metadata:
        entry["metadata"] = metadata
    return {"trace": [entry]}