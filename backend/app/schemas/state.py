import operator
from typing import Annotated, TypedDict

from app.schemas.research import RDBrief


class SearchResult(TypedDict):
    sub_question: str
    source: str
    title: str
    url: str
    content: str
    relevance_score: float


class ResearchState(TypedDict):
    query: str
    sub_questions: list[str]
    search_results: Annotated[list[SearchResult], operator.add]
    sources: list[SearchResult]
    brief: RDBrief | None
    report: str
