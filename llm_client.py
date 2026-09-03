import json
import os
from typing import TypeVar

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from prompts import (
    BUILD_FIELDS_SYSTEM,
    CLASSIFY_SYSTEM_PROMPT,
    EXTRACT_MEANING_SYSTEM,
    RESPONSE_SYSTEM_BASE,
    SELF_CHECK_SYSTEM,
    build_classify_user_prompt,
    build_extract_meaning_prompt,
    build_fields_prompt,
    build_response_user_prompt,
    build_self_check_prompt,
)
from router import get_style_instructions, route
from schemas import (
    ChainResult,
    ChainStepLog,
    ClassificationResult,
    FieldsResult,
    MeaningResult,
    RequestType,
    RoutedAnswer,
    SelfCheckResult,
    StructuredAnswer,
)

load_dotenv()

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None


class PipelineError(Exception):
    """Ошибка пайплайна: API, сеть, пустой ответ или сломанный JSON/схема."""


class StructuredOutputError(PipelineError):
    """JSON от модели сломан или не проходит схему."""


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise PipelineError(
            "OPENAI_API_KEY не задан или пустой. "
            "Добавьте его в .env или переменную окружения."
        )
    _client = OpenAI(api_key=api_key)
    return _client


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
        response = _get_client().chat.completions.create(
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
            "Ошибка API: неверный OPENAI_API_KEY."
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


# ─── Day 5: multi-step chain ───


def extract_meaning(text: str) -> MeaningResult:
    raw = _chat_json(EXTRACT_MEANING_SYSTEM, build_extract_meaning_prompt(text))
    return _parse_json_response(raw, MeaningResult)


def build_fields(
    text: str,
    meaning: MeaningResult,
    classification: ClassificationResult,
) -> FieldsResult:
    raw = _chat_json(
        BUILD_FIELDS_SYSTEM,
        build_fields_prompt(
            text=text,
            meaning=meaning.core_meaning,
            category=classification.category.value,
            intent=classification.intent,
        ),
    )
    return _parse_json_response(raw, FieldsResult)


def self_check(text: str, answer: RoutedAnswer) -> SelfCheckResult:
    raw = _chat_json(
        SELF_CHECK_SYSTEM,
        build_self_check_prompt(
            text=text,
            summary=answer.summary,
            category=answer.category.value,
            sentiment=answer.sentiment.value,
            key_points=json.dumps(answer.key_points, ensure_ascii=False),
            final_answer=answer.final_answer,
        ),
    )
    return _parse_json_response(raw, SelfCheckResult)


def run_chain(text: str) -> ChainResult:
    steps: list[ChainStepLog] = []

    # Шаг 1: extract meaning
    meaning = extract_meaning(text)
    steps.append(ChainStepLog(
        step=1, name="extract_meaning",
        input_summary=text[:80] + ("…" if len(text) > 80 else ""),
        output_summary=meaning.core_meaning,
    ))

    # Шаг 2: classify (использует исходный текст)
    classification = classify(text)
    steps.append(ChainStepLog(
        step=2, name="classify",
        input_summary=text[:80] + ("…" if len(text) > 80 else ""),
        output_summary=f"{classification.category.value} ({classification.confidence:.2f})",
    ))

    # Шаг 3: build structured fields (использует meaning + classification)
    fields = build_fields(text, meaning, classification)
    steps.append(ChainStepLog(
        step=3, name="build_fields",
        input_summary=f"meaning + {classification.category.value}",
        output_summary=f"summary={fields.summary[:50]}…",
    ))

    # Шаг 4: generate final answer (использует classification + fields)
    answer = generate_routed_answer(text, classification)
    # Подменяем summary/key_points/sentiment из fields (шаг 3), чтобы цепочка
    # была связной: answer опирается на то, что построил build_fields.
    answer = RoutedAnswer(
        summary=fields.summary,
        category=fields.category,
        sentiment=fields.sentiment,
        key_points=fields.key_points,
        final_answer=answer.final_answer,
        intent=classification.intent,
        prompt_used=answer.prompt_used,
    )
    steps.append(ChainStepLog(
        step=4, name="generate_final_answer",
        input_summary=f"category={classification.category.value}, intent={classification.intent}",
        output_summary=answer.final_answer[:60] + "…",
    ))

    # Шаг 5: self-check (проверяет итоговый ответ против исходного текста)
    check = self_check(text, answer)
    steps.append(ChainStepLog(
        step=5, name="self_check",
        input_summary="original_text + final_answer",
        output_summary=check.verdict,
    ))

    return ChainResult(
        meaning=meaning,
        classification=classification,
        fields=fields,
        answer=answer,
        self_check=check,
        steps_log=steps,
    )
