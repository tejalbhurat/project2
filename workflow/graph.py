import json
import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.analyst import analyze
from agents.critic import critique
from agents.planner import plan_research
from agents.researcher import research
from agents.writer import write_report
from state.state import ResearchState

# One research pass keeps the free-tier workflow deterministic. The Critic still
# records gaps, while the Writer reports uncertainty instead of burning a retry.
MAX_ITERATIONS = 1
logger = logging.getLogger("research_workflow")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _log(state: ResearchState, message: str) -> list[str]:
    return [*state.get("progress_log", []), message]


def _state_size(value: object) -> int:
    return len(json.dumps(value, default=str, separators=(",", ":")))


def _compact_research_plan(plan):
    """Keep only the highest-signal plan fields for the Researcher prompt."""
    tasks = [
        task.model_copy(
            update={
                "question": task.question[:240],
                "rationale": task.rationale[:120],
                "search_queries": task.search_queries[:1],
            }
        )
        for task in plan.tasks[:3]
    ]
    return plan.model_copy(
        update={"objective": plan.objective[:240], "tasks": tasks, "scope_notes": []}
    )


def planner_node(state: ResearchState) -> dict:
    logger.info("[Planner] BEFORE question_chars=%d", len(state["user_question"]))
    plan = plan_research(state["user_question"])
    logger.info("[Planner] AFTER output_chars=%d tasks=%d", _state_size(plan.model_dump()), len(plan.tasks))
    return {"research_plan": plan, "progress_log": _log(state, "Planning research")}


def researcher_node(state: ResearchState) -> dict:
    iteration = state.get("iteration_count", 0) + 1
    feedback = state.get("critic_feedback")
    logger.info(
        "[Researcher] BEFORE iteration=%d plan_chars=%d feedback_chars=%d",
        iteration,
        _state_size(state["research_plan"].model_dump()),
        _state_size(feedback.model_dump()) if feedback else 0,
    )
    findings, sources = research(
        _compact_research_plan(state["research_plan"]), state.get("sources", []), feedback
    )
    logger.info(
        "[Researcher] AFTER findings_chars=%d findings=%d sources=%d",
        _state_size(findings.model_dump()), len(findings.findings), len(sources),
    )
    return {
        "research_findings": findings,
        "sources": sources,
        "iteration_count": iteration,
        "progress_log": _log(state, f"Searching the web (iteration {iteration})"),
    }


def analyst_node(state: ResearchState) -> dict:
    logger.info(
        "[Analyst] BEFORE plan_chars=%d findings_chars=%d sources_chars=%d",
        _state_size(state["research_plan"].model_dump()),
        _state_size(state["research_findings"].model_dump()),
        _state_size([source.model_dump() for source in state.get("sources", [])[:8]]),
    )
    analysis = analyze(
        state["research_plan"], state["research_findings"], state.get("sources", [])
    )
    logger.info("[Analyst] AFTER output_chars=%d", _state_size(analysis.model_dump()))
    return {"analysis": analysis, "progress_log": _log(state, "Analyzing findings")}


def critic_node(state: ResearchState) -> dict:
    logger.info(
        "[Critic] BEFORE plan_chars=%d findings_chars=%d analysis_chars=%d sources_chars=%d",
        _state_size(state["research_plan"].model_dump()),
        _state_size(state["research_findings"].model_dump()),
        _state_size(state["analysis"].model_dump()),
        _state_size([source.model_dump() for source in state.get("sources", [])[:8]]),
    )
    review = critique(
        state["research_plan"],
        state["research_findings"],
        state["analysis"],
        state.get("sources", []),
    )
    logger.info("[Critic] AFTER output_chars=%d decision=%s", _state_size(review.model_dump()), review.decision)
    decision = review.decision
    if state.get("iteration_count", 0) >= state.get("max_iterations", MAX_ITERATIONS):
        decision = "PASS"
        review = review.model_copy(
            update={
                "decision": "PASS",
                "rationale": (
                    f"Maximum of {state.get('max_iterations', MAX_ITERATIONS)} research "
                    "iterations reached. The report will clearly state remaining uncertainty."
                ),
            }
        )
    return {
        "critic_feedback": review,
        "critic_decision": decision,
        "progress_log": _log(state, f"Critical review: {decision}"),
    }


def writer_node(state: ResearchState) -> dict:
    logger.info(
        "[Writer] BEFORE question_chars=%d plan_chars=%d findings_chars=%d analysis_chars=%d review_chars=%d sources_chars=%d",
        len(state["user_question"]),
        _state_size(state["research_plan"].model_dump()),
        _state_size(state["research_findings"].model_dump()),
        _state_size(state["analysis"].model_dump()),
        _state_size(state["critic_feedback"].model_dump()),
        _state_size([source.model_dump() for source in state.get("sources", [])[:8]]),
    )
    report = write_report(
        state["user_question"],
        state["research_plan"],
        state["research_findings"],
        state["analysis"],
        state["critic_feedback"],
        state.get("sources", []),
    )
    logger.info("[Writer] AFTER report_chars=%d", len(report))
    return {"final_report": report, "progress_log": _log(state, "Writing final report")}


def route_after_critic(state: ResearchState) -> Literal["writer", "researcher"]:
    findings = state.get("research_findings")
    if (
        state["critic_decision"] == "NEEDS_MORE_RESEARCH"
        and state.get("iteration_count", 0) < state.get("max_iterations", MAX_ITERATIONS)
        and findings is not None
        and findings.findings
    ):
        return "researcher"
    return "writer"


def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "critic")
    workflow.add_conditional_edges(
        "critic", route_after_critic, {"writer": "writer", "researcher": "researcher"}
    )
    workflow.add_edge("writer", END)
    return workflow.compile()
