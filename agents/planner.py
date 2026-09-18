from langchain_core.prompts import ChatPromptTemplate

from agents.common import get_llm
from state.state import ResearchPlan


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research planning specialist. Decompose the user's question into """
            """3-5 distinct, answerable research tasks. Keep every field concise and propose """
            """at most 3 focused search queries per task and at most 5 scope_notes. Do not """
            """answer or research the question. Return only the requested structured plan.""",
        ),
        ("human", "Research question: {question}"),
    ]
)


def plan_research(question: str) -> ResearchPlan:
    if not question.strip():
        raise ValueError("A research question is required")
    planner = PLANNER_PROMPT | get_llm().with_structured_output(ResearchPlan)
    return planner.invoke({"question": question.strip()})
