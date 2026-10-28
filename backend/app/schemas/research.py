import re
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_str_list(value: Any) -> Any:
    """Turns a bare string into a list of lines so slightly off model output still validates."""
    if value is None:
        return []
    if isinstance(value, str):
        lines = [line.strip(" -*\u2022\t") for line in value.splitlines() if line.strip()]
        return lines or []
    return value


def _coerce_int_list(value: Any) -> Any:
    """Extracts integers from lists such as [1, "2", "source 3"] and drops anything else."""
    if value is None:
        return []
    if isinstance(value, (int, str)):
        value = [value]
    if not isinstance(value, list):
        return value
    numbers: list[int] = []
    for item in value:
        match = re.search(r"\d+", str(item))
        if match:
            numbers.append(int(match.group()))
    return numbers


def _lowercase(value: Any) -> Any:
    """Lowercases enum input such as 'Mature' or 'HIGH'."""
    return value.strip().lower() if isinstance(value, str) else value


StrList = Annotated[list[str], BeforeValidator(_coerce_str_list)]
IntList = Annotated[list[int], BeforeValidator(_coerce_int_list)]


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
    strengths: StrList = Field(default_factory=list)
    weaknesses: StrList = Field(default_factory=list)
    maturity: Annotated[Maturity, BeforeValidator(_lowercase)] = Maturity.UNKNOWN
    source_ids: IntList = Field(
        default_factory=list, description="Numbers of the sources supporting this option"
    )


class ComparisonEntry(BaseModel):
    criterion: str = Field(description="Comparison criterion, for example scalability or cost")
    option: str = Field(description="Name of the option being assessed, matching an option name")
    assessment: str = Field(description="Short evidence-based assessment of the option on this criterion")
    source_ids: IntList = Field(default_factory=list)


class Risk(BaseModel):
    description: str
    severity: Annotated[Severity, BeforeValidator(_lowercase)] = Severity.MEDIUM
    mitigation: str = ""


class Recommendation(BaseModel):
    choice: str = Field(description="The recommended option or approach")
    rationale: str = Field(description="Why this choice follows from the comparison")
    confidence: Annotated[Confidence, BeforeValidator(_lowercase)] = Confidence.MEDIUM
    next_steps: StrList = Field(default_factory=list)


class RDBrief(BaseModel):
    title: str
    problem_statement: str = Field(description="The decision or question being researched")
    executive_summary: str
    landscape: str = Field(description="Overview of the current technology landscape")
    options: list[TechOption] = Field(min_length=1)
    comparison: list[ComparisonEntry] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    recommendation: Recommendation
    poc_plan: StrList = Field(default_factory=list)
    limitations: StrList = Field(
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