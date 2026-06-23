"""
prompts.py — all LLM prompt templates in one place.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPT: Why separate prompts from graph logic?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prompts are the "brain" of each node — they control what the LLM does.
Keeping them here means:
  - You can tune a prompt without touching graph logic
  - Easy to compare prompt versions
  - Interviewers can see your prompt engineering skills clearly

Each prompt uses Python's str.format() style with {variable} placeholders.
The graph node fills in the variables at runtime from the current state.
"""

# ── Planner Graph ─────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are an expert research planner. Your job is to decompose 
a complex research query into focused sub-questions that together will fully answer 
the original query.

Rules:
- Generate exactly 4 sub-questions
- Each sub-question must be self-contained and searchable
- Cover different angles: definition, comparison, use cases, limitations
- Be specific — vague questions return vague results
- Output ONLY valid JSON, no explanation, no markdown fences

Output format:
{{"sub_questions": ["question 1", "question 2", "question 3", "question 4"]}}"""

PLANNER_USER = """Research query: {query}

Generate 4 focused sub-questions to fully research this topic."""


# ── Research Graph ────────────────────────────────────────────────────

SEARCH_QUERY_SYSTEM = """You are a search query optimizer.
Given a research sub-question, generate the best possible search query for it.
Output ONLY the search query string, nothing else. No quotes, no explanation."""

SEARCH_QUERY_USER = """Sub-question: {sub_question}

Generate the optimal web search query for this sub-question."""


# ── Writer Graph ──────────────────────────────────────────────────────

WRITER_SYSTEM = """You are a research analyst. Write a clear, concise markdown report.
Structure: Executive Summary → Key Findings (bullets) → Conclusion → Sources.
Only use facts from provided sources. Be brief and factual."""


WRITER_USER = """Query: {query}

Sources:
{sources_text}

Write a concise research report (300-400 words max) answering the query.
Use only information from the sources. Cite inline as [Title](URL)."""