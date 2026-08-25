from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RequestType(str, Enum):
    support = "support"
    feedback = "feedback"
    complaint = "complaint"
    sales = "sales"
    general_question = "general_question"


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


class RoutedAnswer(BaseModel):
    category: RequestType
    intent: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    key_points: list[str] = Field(..., min_length=3, max_length=3)
    final_answer: str = Field(..., min_length=1)
    prompt_used: str = Field(..., min_length=1)

    @field_validator("intent", "summary", "final_answer", "prompt_used")
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
