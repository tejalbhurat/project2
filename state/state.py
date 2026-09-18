import re
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ResearchTask(BaseModel):
    question: str = Field(max_length=240)
    rationale: str = Field(max_length=240)
    priority: Literal["high", "medium", "low"] = "medium"
    search_queries: list[str] = Field(default_factory=list, max_length=3)


class ResearchPlan(BaseModel):
    question: str
    objective: str = Field(max_length=500)
    tasks: list[ResearchTask] = Field(max_length=5)
    scope_notes: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("scope_notes", mode="before")
    @classmethod
    def keep_scope_notes_bounded(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [text for item in value[:6] if (text := str(item).strip()[:240])]


class Source(BaseModel):
    title: str = Field(max_length=180)
    url: str


class Finding(BaseModel):
    claim: str = Field(max_length=260)
    evidence: str = Field(max_length=420)
    source_urls: list[str] = Field(default_factory=list, max_length=3)
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("source_urls", mode="before")
    @classmethod
    def normalize_source_urls(cls, value: object) -> list[str]:
        """Keep only unique plain HTTP(S) URLs, stripping accidental Markdown."""
        values = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            match = re.search(r"https?://[^\s<>\]\)\"']+", str(item))
            if not match:
                continue
            url = match.group(0).rstrip(".,;:")
            if url not in seen:
                seen.add(url)
                normalized.append(url)
        return normalized[:3]


class ResearchFindings(BaseModel):
    findings: list[Finding] = Field(default_factory=list, max_length=8)
    unanswered_questions: list[str] = Field(default_factory=list, max_length=3)


class Analysis(BaseModel):
    key_findings: list[str] = Field(default_factory=list, max_length=5)
    comparisons: list[str] = Field(default_factory=list, max_length=5)
    contradictions: list[str] = Field(default_factory=list, max_length=3)
    gaps: list[str] = Field(default_factory=list, max_length=4)
    conclusions: list[str] = Field(default_factory=list, max_length=5)

    @field_validator(
        "key_findings", "comparisons", "contradictions", "gaps", "conclusions",
        mode="before",
    )
    @classmethod
    def keep_concise_items(cls, value: object, info: ValidationInfo) -> list[str]:
        if not isinstance(value, list):
            return []
        limits = {
            "key_findings": 5,
            "comparisons": 5,
            "contradictions": 3,
            "gaps": 4,
            "conclusions": 5,
        }
        limit = limits[info.field_name]
        return [str(item)[:320] for item in value[:limit]]


class CriticDecision(BaseModel):
    decision: Literal["PASS", "NEEDS_MORE_RESEARCH"]
    rationale: str = Field(max_length=1200)
    missing_questions: list[str] = Field(default_factory=list, max_length=3)
    recommended_queries: list[str] = Field(default_factory=list, max_length=3)
    source_concerns: list[str] = Field(default_factory=list, max_length=3)

    @field_validator(
        "missing_questions", "recommended_queries", "source_concerns", mode="before"
    )
    @classmethod
    def keep_short_review_lists(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:240] for item in value[:3]]


class ResearchState(TypedDict, total=False):
    user_question: str
    research_plan: ResearchPlan
    research_findings: ResearchFindings
    sources: list[Source]
    analysis: Analysis
    critic_feedback: CriticDecision
    critic_decision: Literal["PASS", "NEEDS_MORE_RESEARCH"]
    iteration_count: int
    max_iterations: int
    final_report: str
    progress_log: list[str]
