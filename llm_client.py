import json
import logging
import os
import re
import time
from typing import TypeVar

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from prompts import (
    BUILD_FIELDS_SYSTEM,
    CLASSIFY_SYSTEM_PROMPT,
    EXTRACT_MEANING_SYSTEM,
    FALLBACK_SYSTEM,
    FINAL_ANSWER_SYSTEM,
    RESPONSE_SYSTEM_BASE,
    SELF_CHECK_SYSTEM,
    build_classify_fallback_prompt,
    build_classify_user_prompt,
    build_extract_meaning_fallback_prompt,
    build_extract_meaning_prompt,
    build_fields_fallback_prompt,
    build_fields_prompt,
    build_final_answer_fallback_prompt,
    build_final_answer_prompt,
    build_response_user_prompt,
    build_self_check_fallback_prompt,
    build_self_check_prompt,
)
from router import get_style_instructions, route
from schemas import (
    FINAL_ANSWER_MAX_SENTENCES,
    SUMMARY_MAX_WORDS,
    ChainResult,
    ChainStepLog,
    ClassificationResult,
    FieldsResult,
    FinalAnswerBody,
    MeaningResult,
    RequestType,
    RoutedAnswer,
    SelfCheckResult,
    Sentiment,
    StructuredAnswer,
    count_sentences,
)

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None

MAX_API_RETRIES = 3
RETRY_BASE_DELAY_SEC = 1.0


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


def _is_transient_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


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


def _chat_json_once(system: str, user: str) -> str:
    """Один вызов API без retry (внутренний)."""
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
    except openai.RateLimitError:
        raise
    except openai.APIConnectionError:
        raise
    except openai.APIStatusError as exc:
        if exc.status_code >= 500:
            raise
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


def _chat_json(system: str, user: str) -> str:
    """Вызов API с retry при временных ошибках."""
    last_exc: BaseException | None = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return _chat_json_once(system, user)
        except (openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError) as exc:
            if not _is_transient_api_error(exc):
                if isinstance(exc, openai.APIStatusError):
                    raise PipelineError(
                        f"Ошибка API: HTTP {exc.status_code} — {exc.message}"
                    ) from exc
                raise
            last_exc = exc
            if attempt >= MAX_API_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            logger.warning(
                "Временная ошибка API (попытка %s/%s): %s. Повтор через %.1fs",
                attempt, MAX_API_RETRIES, exc, delay,
            )
            time.sleep(delay)

    if isinstance(last_exc, openai.RateLimitError):
        raise PipelineError(
            "Ошибка API: превышен rate limit или квота после retry."
        ) from last_exc
    if isinstance(last_exc, openai.APIConnectionError):
        raise PipelineError(
            f"Ошибка API: нет соединения с OpenAI после retry ({last_exc})."
        ) from last_exc
    if isinstance(last_exc, openai.APIStatusError):
        raise PipelineError(
            f"Ошибка API: HTTP {last_exc.status_code} — {last_exc.message} (после retry)"
        ) from last_exc
    raise PipelineError(f"Ошибка API после retry: {last_exc}") from last_exc


def _chat_json_validated(
    system: str,
    user: str,
    model: type[T],
    *,
    fallback_system: str | None = None,
    fallback_user: str | None = None,
) -> tuple[T, bool]:
    """API + parse; при плохом JSON/схеме — один повтор с fallback-промптом.

    Returns:
        (parsed_model, fallback_used)
    """
    try:
        raw = _chat_json(system, user)
        return _parse_json_response(raw, model), False
    except StructuredOutputError as primary_exc:
        if not fallback_system or not fallback_user:
            raise
        logger.warning(
            "Плохой ответ модели, пробуем fallback-промпт: %s", primary_exc
        )
        try:
            raw_fb = _chat_json(fallback_system, fallback_user)
            parsed = _parse_json_response(raw_fb, model)
            logger.info("Fallback-промпт успешен для схемы %s", model.__name__)
            return parsed, True
        except StructuredOutputError as fallback_exc:
            logger.error(
                "Fallback-промпт тоже не помог: %s (первичная ошибка: %s)",
                fallback_exc, primary_exc,
            )
            raise StructuredOutputError(
                f"Основной и fallback ответы невалидны. "
                f"Основной: {primary_exc}. Fallback: {fallback_exc}"
            ) from fallback_exc


def classify(text: str, meaning: MeaningResult) -> tuple[ClassificationResult, bool]:
    """Шаг 2: классификация на основе meaning из шага 1."""
    return _chat_json_validated(
        CLASSIFY_SYSTEM_PROMPT,
        build_classify_user_prompt(
            text=text,
            core_meaning=meaning.core_meaning,
            language=meaning.language,
            tone=meaning.tone,
            key_entities=json.dumps(meaning.key_entities, ensure_ascii=False),
        ),
        ClassificationResult,
        fallback_system=FALLBACK_SYSTEM,
        fallback_user=build_classify_fallback_prompt(text, meaning.core_meaning),
    )


def generate_routed_answer(
    text: str,
    classification: ClassificationResult,
    category: RequestType | None = None,
) -> RoutedAnswer:
    """Day 4 standalone: один вызов без цепочки (smoke/legacy)."""
    selected = category or classification.category
    prompt_key = route(selected)
    style_instructions = get_style_instructions(selected)

    body, _ = _chat_json_validated(
        RESPONSE_SYSTEM_BASE,
        build_response_user_prompt(
            text=text,
            category=selected,
            intent=classification.intent,
            style_instructions=style_instructions,
        ),
        StructuredAnswer,
    )

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


# ─── Day 5/6: multi-step chain with guardrails ───


def extract_meaning(text: str) -> tuple[MeaningResult, bool]:
    return _chat_json_validated(
        EXTRACT_MEANING_SYSTEM,
        build_extract_meaning_prompt(text),
        MeaningResult,
        fallback_system=FALLBACK_SYSTEM,
        fallback_user=build_extract_meaning_fallback_prompt(text),
    )


def build_fields(
    text: str,
    meaning: MeaningResult,
    classification: ClassificationResult,
) -> tuple[FieldsResult, bool]:
    fields, used_fallback = _chat_json_validated(
        BUILD_FIELDS_SYSTEM,
        build_fields_prompt(
            text=text,
            meaning=meaning.core_meaning,
            category=classification.category.value,
            intent=classification.intent,
        ),
        FieldsResult,
        fallback_system=FALLBACK_SYSTEM,
        fallback_user=build_fields_fallback_prompt(
            text=text,
            meaning=meaning.core_meaning,
            category=classification.category.value,
            intent=classification.intent,
        ),
    )

    if fields.category != classification.category:
        raise StructuredOutputError(
            f"fields.category ({fields.category.value}) не совпадает с "
            f"classification.category ({classification.category.value})."
        )
    return fields, used_fallback


def generate_final_answer(
    text: str,
    meaning: MeaningResult,
    classification: ClassificationResult,
    fields: FieldsResult,
) -> tuple[RoutedAnswer, bool]:
    """Шаг 4: final_answer строится из meaning + classification + fields."""
    selected = classification.category
    prompt_key = route(selected)
    style_instructions = get_style_instructions(selected)
    key_points_json = json.dumps(fields.key_points, ensure_ascii=False)

    body, used_fallback = _chat_json_validated(
        FINAL_ANSWER_SYSTEM,
        build_final_answer_prompt(
            text=text,
            core_meaning=meaning.core_meaning,
            language=meaning.language,
            tone=meaning.tone,
            key_entities=json.dumps(meaning.key_entities, ensure_ascii=False),
            category=selected.value,
            intent=classification.intent,
            summary=fields.summary,
            sentiment=fields.sentiment.value,
            key_points=key_points_json,
            style_instructions=style_instructions,
        ),
        FinalAnswerBody,
        fallback_system=FALLBACK_SYSTEM,
        fallback_user=build_final_answer_fallback_prompt(
            text=text,
            summary=fields.summary,
            key_points=key_points_json,
            intent=classification.intent,
        ),
    )

    return RoutedAnswer(
        summary=fields.summary,
        category=fields.category,
        sentiment=fields.sentiment,
        key_points=fields.key_points,
        final_answer=body.final_answer,
        intent=classification.intent,
        prompt_used=prompt_key,
    ), used_fallback


def self_check(text: str, answer: RoutedAnswer) -> tuple[SelfCheckResult, bool]:
    return _chat_json_validated(
        SELF_CHECK_SYSTEM,
        build_self_check_prompt(
            text=text,
            summary=answer.summary,
            category=answer.category.value,
            sentiment=answer.sentiment.value,
            key_points=json.dumps(answer.key_points, ensure_ascii=False),
            final_answer=answer.final_answer,
        ),
        SelfCheckResult,
        fallback_system=FALLBACK_SYSTEM,
        fallback_user=build_self_check_fallback_prompt(text, answer.final_answer),
    )


def _truncate_summary(text: str) -> str:
    words = text.split()
    if len(words) <= SUMMARY_MAX_WORDS:
        return text.strip() or "Краткое содержание недоступно."
    return " ".join(words[:SUMMARY_MAX_WORDS])


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"(?<=\d)\.(?=\s)", " ", text.strip())
    return [p.strip() for p in re.split(r"[.!?…]+", normalized) if p.strip()]


def _join_sentences(parts: list[str]) -> str:
    if not parts:
        return "Не удалось сгенерировать полный ответ."
    return ". ".join(parts).rstrip(".") + "."


def _safe_degraded_final_answer(summary: str) -> str:
    prefix = "Не удалось сгенерировать полный ответ"
    budget = max(FINAL_ANSWER_MAX_SENTENCES - 1, 0)
    summary_parts = _split_sentences(summary)[:budget]
    if summary_parts:
        candidate = _join_sentences([prefix, *summary_parts])
    else:
        candidate = prefix + "."

    while count_sentences(candidate) > FINAL_ANSWER_MAX_SENTENCES:
        parts = _split_sentences(candidate)
        if len(parts) <= 1:
            return prefix + "."
        candidate = _join_sentences(parts[:-1])
    return candidate


def _degraded_meaning(text: str) -> MeaningResult:
    return MeaningResult(
        core_meaning=_truncate_summary(text)[:200] or "Смысл не извлечён.",
        language="unknown",
        tone="neutral",
        key_entities=[],
    )


def _degraded_classification(meaning: MeaningResult) -> ClassificationResult:
    return ClassificationResult(
        category=RequestType.general_question,
        intent=meaning.core_meaning[:120] or "не удалось классифицировать",
        confidence=0.0,
    )


def _degraded_fields(
    meaning: MeaningResult,
    classification: ClassificationResult,
) -> FieldsResult:
    point = meaning.core_meaning or "детали недоступны"
    return FieldsResult(
        summary=_truncate_summary(meaning.core_meaning),
        category=classification.category,
        sentiment=Sentiment.neutral,
        key_points=[point[:80], "Частичный результат (degraded)", "Повторная генерация не удалась"],
    )


def _degraded_answer(
    classification: ClassificationResult,
    fields: FieldsResult,
) -> RoutedAnswer:
    final_answer = _safe_degraded_final_answer(fields.summary)
    try:
        return RoutedAnswer(
            summary=fields.summary,
            category=fields.category,
            sentiment=fields.sentiment,
            key_points=fields.key_points,
            final_answer=final_answer,
            intent=classification.intent,
            prompt_used=route(classification.category),
        )
    except ValidationError:
        logger.error(
            "degraded RoutedAnswer не прошёл валидацию; "
            "возвращаем минимальный stub"
        )
        return RoutedAnswer(
            summary=_truncate_summary(fields.summary),
            category=fields.category,
            sentiment=fields.sentiment,
            key_points=fields.key_points,
            final_answer="Не удалось сгенерировать полный ответ.",
            intent=(classification.intent or "degraded")[:200],
            prompt_used=route(classification.category),
        )


def _degraded_self_check(errors: list[str]) -> SelfCheckResult:
    return SelfCheckResult(
        is_consistent=False,
        details_preserved=False,
        issues=list(errors) or ["pipeline degraded"],
        verdict="fail: degraded",
    )


def run_chain(text: str) -> ChainResult:
    """5-шаговая цепочка с retry/fallback; при сбое шага — degraded результат."""
    steps: list[ChainStepLog] = []
    fallback_used = False
    degraded = False
    errors: list[str] = []

    meaning: MeaningResult | None = None
    classification: ClassificationResult | None = None
    fields: FieldsResult | None = None
    answer: RoutedAnswer | None = None
    check: SelfCheckResult | None = None

    # Шаг 1
    try:
        meaning, fb = extract_meaning(text)
        fallback_used = fallback_used or fb
        steps.append(ChainStepLog(
            step=1, name="extract_meaning",
            input_summary=text[:80] + ("…" if len(text) > 80 else ""),
            output_summary=meaning.core_meaning,
        ))
    except PipelineError as exc:
        logger.error("Шаг extract_meaning провален: %s → degraded stub", exc)
        errors.append(f"extract_meaning: {exc}")
        degraded = True
        meaning = _degraded_meaning(text)
        steps.append(ChainStepLog(
            step=1, name="extract_meaning",
            input_summary=text[:80] + ("…" if len(text) > 80 else ""),
            output_summary=f"DEGRADED: {meaning.core_meaning}",
        ))

    # Шаг 2
    try:
        classification, fb = classify(text, meaning)
        fallback_used = fallback_used or fb
        steps.append(ChainStepLog(
            step=2, name="classify",
            input_summary=f"meaning: {meaning.core_meaning[:60]}…",
            output_summary=f"{classification.category.value} ({classification.confidence:.2f})",
        ))
    except PipelineError as exc:
        logger.error("Шаг classify провален: %s → degraded stub", exc)
        errors.append(f"classify: {exc}")
        degraded = True
        classification = _degraded_classification(meaning)
        steps.append(ChainStepLog(
            step=2, name="classify",
            input_summary=f"meaning: {meaning.core_meaning[:60]}…",
            output_summary=f"DEGRADED: {classification.category.value}",
        ))

    # Шаг 3
    try:
        fields, fb = build_fields(text, meaning, classification)
        fallback_used = fallback_used or fb
        steps.append(ChainStepLog(
            step=3, name="build_fields",
            input_summary=f"meaning + {classification.category.value}",
            output_summary=f"summary={fields.summary[:50]}…",
        ))
    except PipelineError as exc:
        logger.error("Шаг build_fields провален: %s → degraded stub", exc)
        errors.append(f"build_fields: {exc}")
        degraded = True
        fields = _degraded_fields(meaning, classification)
        steps.append(ChainStepLog(
            step=3, name="build_fields",
            input_summary=f"meaning + {classification.category.value}",
            output_summary=f"DEGRADED: {fields.summary[:50]}",
        ))

    # Шаг 4
    try:
        answer, fb = generate_final_answer(text, meaning, classification, fields)
        fallback_used = fallback_used or fb
        steps.append(ChainStepLog(
            step=4, name="generate_final_answer",
            input_summary=(
                f"fields.summary + fields.key_points + {classification.category.value}"
            ),
            output_summary=answer.final_answer[:60] + "…",
        ))
    except PipelineError as exc:
        logger.error("Шаг generate_final_answer провален: %s → degraded stub", exc)
        errors.append(f"generate_final_answer: {exc}")
        degraded = True
        answer = _degraded_answer(classification, fields)
        steps.append(ChainStepLog(
            step=4, name="generate_final_answer",
            input_summary=f"fields + {classification.category.value}",
            output_summary=f"DEGRADED: {answer.final_answer[:60]}",
        ))

    # Шаг 5
    try:
        check, fb = self_check(text, answer)
        fallback_used = fallback_used or fb
        steps.append(ChainStepLog(
            step=5, name="self_check",
            input_summary="original_text + answer from steps 1–4",
            output_summary=check.verdict,
        ))
    except PipelineError as exc:
        logger.error("Шаг self_check провален: %s → degraded stub", exc)
        errors.append(f"self_check: {exc}")
        degraded = True
        check = _degraded_self_check(errors)
        steps.append(ChainStepLog(
            step=5, name="self_check",
            input_summary="original_text + answer",
            output_summary=check.verdict,
        ))

    if degraded:
        logger.warning(
            "Цепочка завершена в degraded-режиме. errors=%s", errors
        )

    return ChainResult(
        meaning=meaning,
        classification=classification,
        fields=fields,
        answer=answer,
        self_check=check,
        steps_log=steps,
        fallback_used=fallback_used,
        degraded=degraded,
        errors=errors,
    )
