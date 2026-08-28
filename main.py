import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from llm_client import PipelineError, classify, generate_routed_answer
from schemas import RequestType, RoutedAnswer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TEXTS_DIR = Path("texts")
EXPECTED_PATTERN = re.compile(
    r"^(?P<label>support|feedback|complaint|sales|general)_.+\.txt$"
)


def expected_category_from_filename(filename: str) -> str | None:
    match = EXPECTED_PATTERN.match(filename)
    if not match:
        return None
    label = match.group("label")
    return "general_question" if label == "general" else label


def load_texts(directory: Path = TEXTS_DIR) -> list[tuple[str, str]]:
    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"Нет .txt файлов в {directory}")
    return [(path.name, path.read_text(encoding="utf-8").strip()) for path in paths]


def log_result(
    index: int,
    filename: str,
    expected: str | None,
    result: RoutedAnswer,
    confidence: float,
) -> None:
    match_mark = ""
    if expected:
        match_mark = " OK" if expected == result.category.value else " MISS"
    logger.info("=" * 60)
    logger.info("Текст #%s (%s)%s", index, filename, match_mark)
    logger.info(
        "expected=%s | predicted=%s | sentiment=%s | prompt_used=%s | confidence=%.2f",
        expected,
        result.category.value,
        result.sentiment.value,
        result.prompt_used,
        confidence,
    )
    logger.info("Intent: %s", result.intent)
    logger.info("Summary: %s", result.summary)
    logger.info("Key points:")
    for i, point in enumerate(result.key_points, start=1):
        logger.info("  %s. %s", i, point)
    logger.info("Final answer: %s", result.final_answer)


def build_report(rows: list[dict]) -> dict:
    by_category: dict[str, list[str]] = defaultdict(list)
    answers_by_sentiment: dict[str, list[str]] = defaultdict(list)
    prompt_usage: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    correct = 0
    labeled = 0

    for row in rows:
        result = row["result"]
        category = result["category"]
        sentiment = result["sentiment"]
        by_category[category].append(row["file"])
        sentiment_counts[sentiment] += 1
        answers_by_sentiment[sentiment].append(result["final_answer"])
        prompt_usage[result["prompt_used"]] += 1
        expected = row.get("expected_category")
        if expected:
            labeled += 1
            if expected == category:
                correct += 1

    positive_files = [
        row["file"] for row in rows if row["result"]["sentiment"] == "positive"
    ]
    # Позитивные feedback/sales можно подсветить отдельно от жалоб.
    priority_followup = [
        row["file"]
        for row in rows
        if row["result"]["sentiment"] == "negative"
        and row["result"]["category"] in {"complaint", "support"}
    ]

    return {
        "total": len(rows),
        "by_category": dict(by_category),
        "sentiment_counts": dict(sentiment_counts),
        "positive_files": positive_files,
        "priority_followup": priority_followup,
        "answers_by_sentiment": dict(answers_by_sentiment),
        "prompt_usage": dict(prompt_usage),
        "classification_accuracy": {
            "labeled": labeled,
            "correct": correct,
            "accuracy": round(correct / labeled, 3) if labeled else None,
        },
        "routing_proof": [
            {
                "file": row["file"],
                "category": row["result"]["category"],
                "sentiment": row["result"]["sentiment"],
                "prompt_used": row["result"]["prompt_used"],
                "final_answer": row["result"]["final_answer"],
            }
            for row in rows
        ],
    }


def run_routing_smoke_check(sample_text: str) -> list[dict]:
    demos = []
    try:
        classification = classify(sample_text)
        for forced in (RequestType.complaint, RequestType.sales, RequestType.support):
            answer = generate_routed_answer(sample_text, classification, category=forced)
            demos.append(
                {
                    "forced_category": forced.value,
                    "prompt_used": answer.prompt_used,
                    "sentiment": answer.sentiment.value,
                    "final_answer": answer.final_answer,
                }
            )
            logger.info(
                "Smoke | forced=%s | prompt=%s | sentiment=%s | answer=%s",
                forced.value,
                answer.prompt_used,
                answer.sentiment.value,
                answer.final_answer,
            )
    except PipelineError as exc:
        logger.error("Ошибка пайплайна в smoke-check: %s", exc)
        demos.append({"error": str(exc)})
    return demos


def main() -> None:
    texts = load_texts()
    if len(texts) < 8:
        raise ValueError("Нужно минимум 8 тестовых текстов в папке texts/")

    results: list[dict] = []
    errors: list[dict] = []

    for i, (filename, text) in enumerate(texts, start=1):
        expected = expected_category_from_filename(filename)
        try:
            classification = classify(text)
            answer = generate_routed_answer(text, classification)
            log_result(i, filename, expected, answer, classification.confidence)
            results.append(
                {
                    "file": filename,
                    "text": text,
                    "expected_category": expected,
                    "classification": classification.model_dump(mode="json"),
                    "result": answer.model_dump(mode="json"),
                }
            )
        except PipelineError as exc:
            logger.error("Ошибка пайплайна для %s: %s", filename, exc)
            errors.append({"file": filename, "error": str(exc)})

    report = build_report(results)
    logger.info("=" * 60)
    logger.info("Отчёт routing")
    logger.info("Успешно: %s | ошибок: %s", report["total"], len(errors))
    logger.info("По категориям: %s", report["by_category"])
    logger.info("Тональность: %s", report["sentiment_counts"])
    logger.info("Позитивные тексты: %s", report["positive_files"])
    logger.info("Приоритет на follow-up (negative complaint/support): %s", report["priority_followup"])
    logger.info("Использование промптов: %s", report["prompt_usage"])
    logger.info("Accuracy классификации: %s", report["classification_accuracy"])

    smoke = []
    if results:
        logger.info("=" * 60)
        logger.info("Smoke-check: один текст, разные forced categories")
        smoke = run_routing_smoke_check(results[0]["text"])

    output_path = Path("results.json")
    output_path.write_text(
        json.dumps(
            {
                "results": results,
                "errors": errors,
                "report": report,
                "routing_smoke_check": smoke,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Результаты сохранены в %s", output_path)

    if errors and not results:
        raise SystemExit("Все примеры завершились ошибкой валидации JSON.")


if __name__ == "__main__":
    main()
