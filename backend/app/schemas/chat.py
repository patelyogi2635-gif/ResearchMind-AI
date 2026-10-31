from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

KNOWN_CATEGORIES = frozenset({"none", "prompt_injection", "malicious_use", "harmful_content", "other"})


def _normalize_intent(value: Any) -> Any:
    """Normalizes intent text such as 'Off-Topic' to 'off_topic'."""
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_").replace(" ", "_")
    return value


def _coerce_category(value: Any) -> Any:
    """Maps a missing category to 'none' and an unknown one to 'other'."""
    if value is None:
        return "none"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in KNOWN_CATEGORIES else "other"


def _coerce_reply(value: Any) -> Any:
    """Turns a null reply into an empty string."""
    return "" if value is None else value


class Intent(str, Enum):
    GREETING = "greeting"
    RESEARCH = "research"
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"
    UNCLEAR = "unclear"


class UnsafeCategory(str, Enum):
    NONE = "none"
    PROMPT_INJECTION = "prompt_injection"
    MALICIOUS_USE = "malicious_use"
    HARMFUL_CONTENT = "harmful_content"
    OTHER = "other"


class RouteDecision(BaseModel):
    intent: Annotated[Intent, BeforeValidator(_normalize_intent)] = Field(
        description="One of: greeting, research, off_topic, unsafe, unclear"
    )
    unsafe_category: Annotated[UnsafeCategory, BeforeValidator(_coerce_category)] = Field(
        default=UnsafeCategory.NONE,
        description="Only for unsafe: prompt_injection, malicious_use, harmful_content or other. Otherwise none",
    )
    reply: Annotated[str, BeforeValidator(_coerce_reply)] = Field(
        default="",
        description="Short friendly reply for greeting, off_topic and unclear. Empty for research and unsafe",
    )


class ChatReply(BaseModel):
    intent: Intent
    text: str