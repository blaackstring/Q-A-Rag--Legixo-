"""LangGraph orchestration: the StateGraph that wires the five nodes together.

Graph layout (see docs/langgraph.md for the ASCII diagram):

    START -> retrieve -> grade -> (branch)
                                   |---> answer -> END       (good path)
                                   |---> rewrite -> retrieve  (bad path, loop)
                                   |---> refuse -> END       (gave up / max steps)

The grader/rewrite/refuse router is a real branch in the graph, and the step
counter plus `max_steps` make it impossible for the loop to spin forever.
"""

from functools import lru_cache

from langgraph.graph import StateGraph, START, END

from .common.config import get_settings
from .state import AskState, initial_state
from .common.vectorstore import index_stats
from .nodes.retrieve import retrieve_node
from .nodes.grade import grade_node
from .nodes.rewrite import rewrite_node
from .nodes.answer import answer_node
from .nodes.refuse import refuse_node


def route_after_grade(state: dict) -> str:
    """Decide the branch after grading: answer, re-query, or give up.

    Good path: at least one relevant chunk exists -> write an answer.
    Bad path--< nothing relevant yet but loop budget left -> rewrite and retry.
    Give up: loop budget exhausted -> refus.
    """
    if state.get("relevant_count", 0) > 0:
        return "answer"
    if state.get("step", 0) >= state.get("max_steps", 3):
        return "refuse"
    return "rewrite"


def build_graph():

    
    """Construct and compile the QA StateGraph, returning the runnable graph."""
    builder = StateGraph(AskState)

    builder.add_node("retrieve", retrieve_node)   # search Pinecone for candidates
    builder.add_node("grade", grade_node)         # check candidate relevance
    builder.add_node("rewrite", rewrite_node)     # bad path: rephrase the query
    builder.add_node("answer", answer_node)       # grounded answer + citations
    builder.add_node("refuse", refuse_node)       # honest not-found reply

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"},
    )
    builder.add_edge("rewrite", "retrieve")  # loop back until budget runs out

    builder.add_edge("answer", END)

    builder.add_edge("refuse", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Return the compiled graph once, reused for every /ask request."""
    return build_graph()


def ask_question(question: str, top_k: int | None = None, max_steps: int | None = None,
                 trace: bool = True) -> dict:
    """Run one question through the graph and return a clean result dict.
    """
    settings = get_settings()
    state = initial_state(
        question=question,
        top_k=top_k or settings.top_k,
        max_steps=max_steps or settings.max_steps,
    )
    final_state = get_graph().invoke(state, config={"recursion_limit": 50})

    payload = {
        "question": question,
        "answer": final_state.get("answer", ""),
        "status": final_state.get("status", "error"),
        "citations": final_state.get("citations", []),
        "steps": final_state.get("step", 0),
        "index_stats": index_stats(),
    }
    if trace:
        payload["trace"] = final_state.get("trace", [])
    return payload