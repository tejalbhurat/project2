from langchain_core.prompts import ChatPromptTemplate

from agents.common import get_llm
from state.state import Analysis, CriticDecision, ResearchFindings, ResearchPlan, Source


CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a rigorous research reviewer. Return ONLY valid JSON matching """
            """CriticDecision. Choose PASS only when the important tasks are answered and the """
            """analysis is grounded. Otherwise choose NEEDS_MORE_RESEARCH. Use at most 3 items """
            """in each list, keep rationale under 100 words, and use short strings. Required """
            """keys are decision, rationale, missing_questions, recommended_queries, and """
            """source_concerns. Use double quotes, valid JSON, and no Markdown.""",
        ),
        (
            "human",
            "Compact plan:\n{plan}\n\nFindings:\n{findings}\n\nAnalysis:\n{analysis}\n\n"
            "Sources:\n{sources}\n\nReturn the JSON object now.",
        ),
    ]
)


def critique(
    plan: ResearchPlan,
    findings: ResearchFindings,
    analysis: Analysis,
    sources: list[Source],
) -> CriticDecision:
    critic = CRITIC_PROMPT | get_llm(max_tokens=1200).with_structured_output(
        CriticDecision, method="json_mode"
    )
    return critic.invoke(
        {
            "plan": {
                "question": plan.question,
                "objective": plan.objective[:240],
                "tasks": [
                    {"question": task.question, "priority": task.priority}
                    for task in plan.tasks[:3]
                ],
            },
            "findings": findings.model_dump(
                include={"findings", "unanswered_questions"}
            ),
            "analysis": analysis.model_dump(exclude_none=True),
            "sources": [{"title": source.title, "url": source.url} for source in sources[:6]],
        }
    )
