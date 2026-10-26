from enum import Enum

from pydantic import BaseModel, Field


class Maturity(str, Enum):
    EMERGING = "emerging"
    GROWING = "growing"
    MATURE = "mature"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PlanOutput(BaseModel):
    sub_questions: list[str] = Field(
        description="Focused, self-contained, searchable sub-questions for the research topic"
    )


class TechOption(BaseModel):
    name: str = Field(description="Name of the candidate technology or approach")
    summary: str = Field(description="One or two sentences on what it is and what it is good at")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    maturity: Maturity = Maturity.UNKNOWN
    source_ids: list[int] = Field(
        default_factory=list, description="Numbers of the sources supporting this option"
    )


class ComparisonEntry(BaseModel):
    criterion: str = Field(description="Comparison criterion, for example scalability or cost")
    option: str = Field(description="Name of the option being assessed, matching an option name")
    assessment: str = Field(description="Short evidence-based assessment of the option on this criterion")
    source_ids: list[int] = Field(default_factory=list)


class Risk(BaseModel):
    description: str
    severity: Severity = Severity.MEDIUM
    mitigation: str = ""


class Recommendation(BaseModel):
    choice: str = Field(description="The recommended option or approach")
    rationale: str = Field(description="Why this choice follows from the comparison")
    confidence: Confidence = Confidence.MEDIUM
    next_steps: list[str] = Field(default_factory=list)


class RDBrief(BaseModel):
    title: str
    problem_statement: str = Field(description="The decision or question being researched")
    executive_summary: str
    landscape: str = Field(description="Overview of the current technology landscape")
    options: list[TechOption] = Field(min_length=1)
    comparison: list[ComparisonEntry] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    recommendation: Recommendation
    poc_plan: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=list, description="What could not be verified from the available sources"
    )


class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The technology topic or requirement to evaluate",
        json_schema_extra={
            "examples": ["Which vector database should we use for 1M documents with hybrid search?"]
        },
    )


class SourceResponse(BaseModel):
    id: int
    title: str
    url: str
    source: str
    relevance_score: float


class ResearchResponse(BaseModel):
    query: str
    sub_questions: list[str]
    sources: list[SourceResponse]
    brief: RDBrief
    report: str
    source_count: int
    duration_seconds: float
