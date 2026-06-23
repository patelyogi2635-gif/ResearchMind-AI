"""
llm.py — single source of truth for the LLM instance.

Why a separate file?
All 3 LangGraph graphs (Planner, Research, Writer) need an LLM.
Instead of each graph creating its own instance, they all import
get_llm() from here. Change the model in one place → affects everything.

Groq is free at: https://console.groq.com/
Signup → API Keys → Create key. No credit card needed.

Available free models on Groq:
  llama-3.3-70b-versatile  ← best quality, use this (default)
  llama-3.1-8b-instant     ← fastest, use for simple tasks
  mixtral-8x7b-32768       ← large context window (32k tokens)
  gemma2-9b-it             ← Google's model, good for structured output
"""

from functools import lru_cache
from langchain_groq import ChatGroq
from app.config import get_settings


@lru_cache()
def get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Returns a cached ChatGroq instance.

    temperature=0.3 is a good default for research tasks:
    - 0.0 = fully deterministic (good for structured JSON output)
    - 0.3 = mostly focused but some creativity (good for research + writing)
    - 0.7+ = more creative (good for brainstorming, not for factual tasks)
    """
    settings = get_settings()
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
    )


def get_fast_llm() -> ChatGroq:
    """
    Returns the 8b model for simple/fast tasks like query classification.
    Useful in Phase 2 for the Planner graph's sub-question decomposition.
    """
    settings = get_settings()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=settings.groq_api_key,
        temperature=0.1,
    )