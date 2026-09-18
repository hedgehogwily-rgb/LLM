"""Негативные сценарии Day 6: retry / fallback / degraded без реального API."""

from __future__ import annotations

import json
import logging
import unittest
from unittest.mock import patch

import openai

import llm_client
from llm_client import (
    StructuredOutputError,
    _chat_json,
    _chat_json_validated,
    _degraded_answer,
    _degraded_fields,
    _parse_json_response,
    _safe_degraded_final_answer,
    _truncate_summary,
    run_chain,
)
from schemas import (
    FINAL_ANSWER_MAX_SENTENCES,
    SUMMARY_MAX_WORDS,
    ClassificationResult,
    FieldsResult,
    MeaningResult,
    RequestType,
    Sentiment,
    count_sentences,
    count_words,
)

logging.basicConfig(level=logging.INFO)


class FakeMessage:
    def __init__(self, content: str | None):
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str | None):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self._side_effects:
            raise RuntimeError("Неожиданный лишний вызов API")
        effect = self._side_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return FakeResponse(effect)


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeClient:
    def __init__(self, side_effects):
        self.completions = FakeCompletions(side_effects)
        self.chat = FakeChat(self.completions)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=None)


class GuardrailsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Сбрасываем кэш клиента между тестами.
        llm_client._client = None

    def test_text_instead_of_json_triggers_fallback_then_degraded(self) -> None:
        """Сценарий 1: модель вернула текст вместо JSON → fallback → при повторном фейле degraded."""
        meaning_ok = json.dumps({
            "core_meaning": "Пользователь жалуется на биллинг",
            "language": "ru",
            "tone": "angry",
            "key_entities": ["биллинг"],
        }, ensure_ascii=False)
        classify_ok = json.dumps({
            "category": "complaint",
            "intent": "Жалоба на биллинг",
            "confidence": 0.9,
        }, ensure_ascii=False)
        fields_ok = json.dumps({
            "summary": "Жалоба на ошибочный биллинг",
            "category": "complaint",
            "sentiment": "negative",
            "key_points": ["Ошибочный счёт", "Нужен возврат", "Ждёт ответа"],
        }, ensure_ascii=False)
        # final_answer: primary = plain text, fallback = also bad → degraded
        self_check_ok = json.dumps({
            "is_consistent": True,
            "details_preserved": True,
            "issues": [],
            "verdict": "pass",
        })

        fake = FakeClient([
            meaning_ok,
            classify_ok,
            fields_ok,
            "это не json вообще",          # final primary
            "снова невалидный текст",      # final fallback
            self_check_ok,
        ])

        with patch.object(llm_client, "_get_client", return_value=fake):
            with patch.object(llm_client, "time") as mock_time:
                mock_time.sleep = lambda *_: None
                chain = run_chain("Мне списали лишние деньги за подписку")

        self.assertTrue(chain.degraded)
        self.assertTrue(any("generate_final_answer" in e for e in chain.errors))
        self.assertIn("Не удалось сгенерировать полный ответ", chain.answer.final_answer)
        self.assertEqual(chain.answer.category.value, "complaint")
        # Программа не упала — вернула usable ChainResult.
        self.assertIsNotNone(chain.meaning)
        self.assertIsNotNone(chain.fields)

    def test_missing_keys_triggers_fallback(self) -> None:
        """Сценарий 2: часть ключей отсутствует → валидация ловит → fallback успешен."""
        primary = '{"summary": "ok"}'  # нет обязательных полей MeaningResult
        fallback = json.dumps({
            "core_meaning": "Краткий смысл текста",
            "language": "ru",
            "tone": "neutral",
            "key_entities": [],
        }, ensure_ascii=False)

        fake = FakeClient([primary, fallback])

        with patch.object(llm_client, "_get_client", return_value=fake):
            result, used_fallback = _chat_json_validated(
                "system",
                "user",
                MeaningResult,
                fallback_system="fallback system",
                fallback_user="fallback user",
            )

        self.assertTrue(used_fallback)
        self.assertEqual(result.core_meaning, "Краткий смысл текста")
        self.assertEqual(fake.completions.calls, 2)

    def test_transient_api_error_is_retried(self) -> None:
        """Сценарий 3: API временно недоступно → retry → успешный ответ."""
        ok_payload = json.dumps({
            "core_meaning": "Вопрос про тариф",
            "language": "ru",
            "tone": "neutral",
            "key_entities": [],
        }, ensure_ascii=False)

        fake = FakeClient([
            _connection_error(),
            _connection_error(),
            ok_payload,
        ])

        with patch.object(llm_client, "_get_client", return_value=fake):
            with patch.object(llm_client.time, "sleep", return_value=None):
                raw = _chat_json("system", "user")

        self.assertEqual(json.loads(raw)["core_meaning"], "Вопрос про тариф")
        self.assertEqual(fake.completions.calls, 3)

        parsed = _parse_json_response(raw, MeaningResult)
        self.assertEqual(parsed.language, "ru")

    def test_parse_rejects_plain_text(self) -> None:
        with self.assertRaises(StructuredOutputError):
            _parse_json_response("это не json", MeaningResult)

    def test_degraded_answer_respects_sentence_limit(self) -> None:
        summary = "Один. Два. Три. Четыре."
        fields = FieldsResult(
            summary=summary,
            category=RequestType.complaint,
            sentiment=Sentiment.negative,
            key_points=["a", "b", "c"],
        )
        classification = ClassificationResult(
            category=RequestType.complaint,
            intent="тест",
            confidence=0.0,
        )

        answer = _degraded_answer(classification, fields)

        self.assertLessEqual(
            count_sentences(answer.final_answer),
            FINAL_ANSWER_MAX_SENTENCES,
        )
        self.assertTrue(answer.final_answer.startswith("Не удалось"))
        safe = _safe_degraded_final_answer(summary)
        self.assertLessEqual(count_sentences(safe), FINAL_ANSWER_MAX_SENTENCES)
        self.assertEqual(answer.final_answer, safe)

    def test_truncate_summary_matches_count_words_for_hyphens(self) -> None:
        long_text = " ".join(["слово-слово"] * 20)
        self.assertGreater(count_words(long_text), SUMMARY_MAX_WORDS)
        self.assertEqual(len(long_text.split()), 20)

        truncated = _truncate_summary(long_text)
        self.assertLessEqual(count_words(truncated), SUMMARY_MAX_WORDS)

        meaning = MeaningResult(
            core_meaning=long_text,
            language="ru",
            tone="neutral",
            key_entities=[],
        )
        classification = ClassificationResult(
            category=RequestType.general_question,
            intent="тест",
            confidence=0.0,
        )
        fields = _degraded_fields(meaning, classification)
        self.assertLessEqual(count_words(fields.summary), SUMMARY_MAX_WORDS)


if __name__ == "__main__":
    unittest.main()
