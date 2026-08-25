import json
import os
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_BASE,
    build_classify_user_prompt,
    build_response_user_prompt,
)
from router import get_style_instructions, route
from schemas import ClassificationResult, RequestType, RoutedAnswer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Понятная ошибка, если JSON от модели сломан или не проходит схему."""


class ResponseBody(BaseModel):
    summary: str = Field(..., min_length=1)
    key_points: list[str] = Field(..., min_length=3, max_length=3)
    final_answer: str = Field(..., min_length=1)

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


def _parse_json_response(raw: str | None, model: type[T]) -> T:
    if not raw or not raw.strip():
        raise StructuredOutputError("Модель вернула пустой ответ вместо JSON.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:200].replace("\n", " ")
        raise StructuredOutputError(
            f"Модель вернула невалидный JSON: {exc.msg} (позиция {exc.pos}). "
            f"Фрагмент ответа: {preview!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise StructuredOutputError(
            f"Ожидался JSON-объект, получен тип {type(payload).__name__}."
        )

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise StructuredOutputError(
            f"JSON не прошёл валидацию схемы: {details}. Получено: {payload}"
        ) from exc


def _chat_json(system: str, user: str) -> str | None:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def classify(text: str) -> ClassificationResult:
    raw = _chat_json(CLASSIFY_SYSTEM_PROMPT, build_classify_user_prompt(text))
    return _parse_json_response(raw, ClassificationResult)


def generate_routed_answer(
    text: str,
    classification: ClassificationResult,
    category: RequestType | None = None,
) -> RoutedAnswer:
    """Генерирует ответ с промптом, выбранным роутером по категории."""
    selected = category or classification.category
    prompt_key = route(selected)
    style_instructions = get_style_instructions(selected)

    raw = _chat_json(
        RESPONSE_SYSTEM_BASE,
        build_response_user_prompt(
            text=text,
            category=selected,
            intent=classification.intent,
            style_instructions=style_instructions,
        ),
    )
    body = _parse_json_response(raw, ResponseBody)
    return RoutedAnswer(
        category=selected,
        intent=classification.intent,
        summary=body.summary,
        key_points=body.key_points,
        final_answer=body.final_answer,
        prompt_used=prompt_key,
    )
