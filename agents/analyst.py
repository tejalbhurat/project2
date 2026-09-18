from langchain_core.prompts import ChatPromptTemplate

from agents.common import get_llm
from state.state import Analysis, ResearchFindings, ResearchPlan, Source


ANALYST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a concise research analyst. Organize the supplied findings against the """
            """compact plan. Return ONLY valid JSON matching Analysis. Use no more than 5 items """
            """in key_findings, comparisons, and conclusions; no more than 3 contradictions and """
            """4 gaps. Keep each item under 40 words. Do not search or write the final report.""",
        ),
        (
            "human",
            "Compact plan:\n{plan}\n\nFindings:\n{findings}\n\nReturn the JSON object now.",
        ),
    ]
)


def analyze(plan: ResearchPlan, findings: ResearchFindings, sources: list[Source]) -> Analysis:
    analyst = ANALYST_PROMPT | get_llm(max_tokens=1400).with_structured_output(Analysis)
    return analyst.invoke(
        {
            "plan": {
                "question": plan.question,
                "objective": plan.objective[:240],
                "tasks": [
                    {"question": task.question, "priority": task.priority}
                    for task in plan.tasks[:3]
                ],
            },
            "findings": findings.model_dump_json(exclude_none=True),
        }
    )
