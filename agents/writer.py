from langchain_core.prompts import ChatPromptTemplate

from agents.common import get_llm
from state.state import Analysis, CriticDecision, ResearchFindings, ResearchPlan, Source


WRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a careful research report writer. Write a clear, useful report using only """
            """the supplied evidence and analysis. Keep the report under 900 words. Do not browse """
            """or introduce outside """
            """facts. Explain uncertainty, distinguish evidence from conclusions, and cite sources """
            """inline using markdown links with the provided URLs. Include a brief methodology note.""",
        ),
        (
            "human",
            "Question: {question}\nPlan: {plan}\nFindings: {findings}\nAnalysis: {analysis}\n"
            "Review: {review}\nSources: {sources}",
        ),
    ]
)


def write_report(
    question: str,
    plan: ResearchPlan,
    findings: ResearchFindings,
    analysis: Analysis,
    review: CriticDecision,
    sources: list[Source],
) -> str:
    writer = WRITER_PROMPT | get_llm(temperature=0.2, max_tokens=1800)
    response = writer.invoke(
        {
            "question": question,
            "plan": plan.model_dump_json(exclude_none=True),
            "findings": findings.model_dump_json(exclude_none=True),
            "analysis": analysis.model_dump_json(exclude_none=True),
            "review": review.model_dump_json(exclude_none=True),
            "sources": [{"title": source.title, "url": source.url} for source in sources[:8]],
        }
    )
    return response.content
