"""LM helpers: shared ChatOpenAI instance plus prompt utilities.

"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
_llm: ChatOpenAI | None = None


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return the shared model with the requested temperature."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    _llm.temperature = temperature
    return _llm


def run_prompt(system: str, user: str, temperature: float = 0.0) -> str:
    """Run a (system, user) prompt through the LLM and return its text."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("user", user)]
    )
    chain = prompt | get_llm(temperature) | StrOutputParser()
    return chain.invoke({"input": user}).strip()