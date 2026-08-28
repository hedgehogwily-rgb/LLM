import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator


SUMMARY_MAX_WORDS = 25
FINAL_ANSWER_MAX_SENTENCES = 4


class RequestType(str, Enum):
    support = "support"
    feedback = "feedback"
    complaint = "complaint"
    sales = "sales"
    general_question = "general_question"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def count_sentences(text: str) -> int:
    # Не считаем точки в нумерации шагов ("1.", "2.") концом предложения.
    normalized = re.sub(r"(?<=\d)\.(?=\s)", " ", text.strip())
    parts = [p for p in re.split(r"[.!?…]+", normalized) if p.strip()]
    return max(len(parts), 1) if text.strip() else 0


class ClassificationResult(BaseModel):
    category: RequestType
    intent: str = Field(..., min_length=1, description="Краткое описание намерения пользователя")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("intent")
    @classmethod
    def strip_intent(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("intent не должен быть пустым")
        return value


class StructuredAnswer(BaseModel):
    summary: str = Field(..., min_length=1)
    category: RequestType
    sentiment: Sentiment
    key_points: list[str] = Field(..., min_length=3, max_length=3)
    final_answer: str = Field(..., min_length=1)

    @field_validator("summary", "final_answer")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("поле не должно быть пустым")
        return value

    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, value: str) -> str:
        words = count_words(value)
        if words > SUMMARY_MAX_WORDS:
            raise ValueError(
                f"summary слишком длинное: {words} слов, максимум {SUMMARY_MAX_WORDS}"
            )
        return value

    @field_validator("final_answer")
    @classmethod
    def validate_final_answer_length(cls, value: str) -> str:
        sentences = count_sentences(value)
        if sentences > FINAL_ANSWER_MAX_SENTENCES:
            raise ValueError(
                f"final_answer слишком длинный: {sentences} предложений, "
                f"максимум {FINAL_ANSWER_MAX_SENTENCES}"
            )
        return value

    @field_validator("key_points")
    @classmethod
    def validate_key_points(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("каждый key point должен быть непустой строкой")
        if len(cleaned) != 3:
            raise ValueError("нужно ровно 3 key points")
        return cleaned


class RoutedAnswer(StructuredAnswer):
    intent: str = Field(..., min_length=1)
    prompt_used: str = Field(..., min_length=1)

    @field_validator("intent", "prompt_used")
    @classmethod
    def strip_routing_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("поле не должно быть пустым")
        return value
