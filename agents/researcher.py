import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from agents.common import get_llm
from state.state import CriticDecision, ResearchFindings, ResearchPlan, Source
from tools.web_search import web_search

MAX_SOURCES = 6
MAX_QUERIES = 4
MAX_SNIPPET_CHARS = 450


RESEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an evidence-focused web researcher. Return ONLY a valid JSON object matching """
            """ResearchFindings. Produce 5-8 non-duplicative findings. For each finding: use one """
            """concise claim, 1-2 sentence summarized evidence, a confidence of high/medium/low, """
            """and up to 3 relevant source_urls. Every source_urls item MUST be one plain HTTP(S) """
            """URL only. Never use Markdown links, brackets, parentheses, citations, labels, or """
            """explanations inside source_urls. Do not copy article text, HTML, or long snippets. """
            """Use valid JSON with double quotes, escaped strings, no trailing commas, and all """
            """required fields. Summarize rather than quote. Do not invent facts or sources.""",
        ),
        (
            "human",
            "Research plan:\n{plan}\n\nCompact search evidence:\n{results}\n\n"
            "Critic gaps:\n{feedback}\n\nReturn the JSON object now.",
        ),
    ]
)


def _search_queries(
    plan: ResearchPlan, feedback: CriticDecision | None
) -> list[tuple[str, str]]:
    queries = [(task.question, task.question) for task in plan.tasks]
    for task in plan.tasks:
        queries.extend((task.question, query) for query in task.search_queries)
    if feedback:
        queries.extend(("critic feedback", query) for query in feedback.recommended_queries)
        queries.extend(("missing question", query) for query in feedback.missing_questions)
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for task_name, query in queries:
        normalized = query.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append((task_name, query))
    return unique


def research(
    plan: ResearchPlan,
    previous_sources: list[Source] | None = None,
    feedback: CriticDecision | None = None,
) -> tuple[ResearchFindings, list[Source]]:
    collected_sources = list(previous_sources or [])
    raw_results: list[dict[str, Any]] = []
    existing_urls = {source.url for source in collected_sources}

    for task_name, query in _search_queries(plan, feedback)[:MAX_QUERIES]:
        results = web_search.invoke({"query": query, "max_results": 3})
        for result in results:
            if len(collected_sources) >= MAX_SOURCES and str(result.get("url", "")).strip() not in existing_urls:
                continue
            url = str(result.get("url", "")).strip()
            if not url:
                continue
            title = str(result.get("title", "Untitled source"))
            snippet = str(result.get("content", result.get("snippet", "")))[:MAX_SNIPPET_CHARS]
            raw_results.append({"task": task_name, "title": title[:180], "url": url, "evidence": snippet})
            if url not in existing_urls:
                collected_sources.append(Source(title=title[:180], url=url))
                existing_urls.add(url)
            if len(raw_results) >= MAX_SOURCES:
                break
        if len(raw_results) >= MAX_SOURCES:
            break

    if not raw_results:
        raise RuntimeError("Web search returned no usable results")

    researcher = RESEARCH_PROMPT | get_llm(max_tokens=1200).with_structured_output(
        ResearchFindings, method="json_mode"
    )
    findings = researcher.invoke(
        {
            "plan": plan.model_dump_json(exclude_none=True),
            "results": json.dumps(raw_results, separators=(",", ":")),
            "feedback": (
                feedback.model_dump_json(indent=2)
                if feedback
                else "No previous critic feedback."
            ),
        }
    )
    return findings.model_copy(update={"findings": findings.findings[:8]}), collected_sources[:MAX_SOURCES]
