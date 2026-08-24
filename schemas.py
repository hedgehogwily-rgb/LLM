from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


Category = Literal["tech", "health", "education", "lifestyle", "other"]


class AnalysisResult(BaseModel):
    """Структурированный ответ модели после валидации."""

    summary: str = Field(..., min_length=1, description="Краткое резюме текста")
    category: Category
    sentiment: Sentiment
    key_points: list[str] = Field(..., min_length=3, max_length=3)
    final_answer: str = Field(..., min_length=1, description="Короткий полезный ответ")

    @field_validator("summary", "final_answer")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("поле не должно быть пустым")
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
