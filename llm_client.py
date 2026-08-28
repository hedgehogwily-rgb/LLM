import json
import os
from typing import TypeVar

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_BASE,
    build_classify_user_prompt,
    build_response_user_prompt,
)
from router import get_style_instructions, route
from schemas import ClassificationResult, RequestType, RoutedAnswer, StructuredAnswer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

T = TypeVar("T", bound=BaseModel)


class PipelineError(Exception):
    """Ошибка пайплайна: API, сеть, пустой ответ или сломанный JSON/схема."""


class StructuredOutputError(PipelineError):
    """JSON от модели сломан или не проходит схему."""


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


def _chat_json(system: str, user: str) -> str:
    try:
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
    except openai.AuthenticationError as exc:
        raise PipelineError(
            "Ошибка API: неверный или отсутствующий OPENAI_API_KEY."
        ) from exc
    except openai.RateLimitError as exc:
        raise PipelineError(
            "Ошибка API: превышен rate limit или квота. Попробуйте позже."
        ) from exc
    except openai.APIConnectionError as exc:
        raise PipelineError(
            f"Ошибка API: нет соединения с OpenAI ({exc})."
        ) from exc
    except openai.APIStatusError as exc:
        raise PipelineError(
            f"Ошибка API: HTTP {exc.status_code} — {exc.message}"
        ) from exc
    except openai.APIError as exc:
        raise PipelineError(f"Ошибка API OpenAI: {exc}") from exc

    if not response.choices:
        raise PipelineError("Ошибка модели: пустой список choices в ответе API.")

    content = response.choices[0].message.content
    if content is None:
        raise PipelineError("Ошибка модели: content в ответе равен null.")
    return content


def classify(text: str) -> ClassificationResult:
    raw = _chat_json(CLASSIFY_SYSTEM_PROMPT, build_classify_user_prompt(text))
    return _parse_json_response(raw, ClassificationResult)


def generate_routed_answer(
    text: str,
    classification: ClassificationResult,
    category: RequestType | None = None,
) -> RoutedAnswer:
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
    body = _parse_json_response(raw, StructuredAnswer)

    if body.category != selected:
        raise StructuredOutputError(
            f"category в structured output ({body.category.value}) "
            f"не совпадает с routed category ({selected.value})."
        )

    return RoutedAnswer(
        summary=body.summary,
        category=body.category,
        sentiment=body.sentiment,
        key_points=body.key_points,
        final_answer=body.final_answer,
        intent=classification.intent,
        prompt_used=prompt_key,
    )
