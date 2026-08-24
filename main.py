import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from llm_client import StructuredOutputError, generate_summary
from prompts import DEFAULT_VARIANT
from schemas import AnalysisResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TEXTS_DIR = Path("texts")


def load_texts(directory: Path = TEXTS_DIR) -> list[tuple[str, str]]:
    paths = sorted(directory.glob("text*.txt"))
    if not paths:
        raise FileNotFoundError(f"Нет файлов text*.txt в {directory}")
    return [(path.name, path.read_text(encoding="utf-8").strip()) for path in paths]


def log_result(index: int, filename: str, result: AnalysisResult) -> None:
    logger.info("=" * 60)
    logger.info("Текст #%s (%s)", index, filename)
    logger.info("Category: %s | Sentiment: %s", result.category, result.sentiment.value)
    logger.info("Summary: %s", result.summary)
    logger.info("Key points:")
    for i, point in enumerate(result.key_points, start=1):
        logger.info("  %s. %s", i, point)
    logger.info("Final answer: %s", result.final_answer)


def build_report(rows: list[tuple[str, AnalysisResult]]) -> dict:
    """Используем поля schema: группировка, тональность, ответы по категориям."""
    by_category: dict[str, list[str]] = defaultdict(list)
    sentiments: Counter[str] = Counter()
    answers_by_category: dict[str, list[str]] = defaultdict(list)

    for filename, result in rows:
        by_category[result.category].append(filename)
        sentiments[result.sentiment.value] += 1
        answers_by_category[result.category].append(result.final_answer)

    positive_files = [
        filename for filename, result in rows if result.sentiment.value == "positive"
    ]

    return {
        "total": len(rows),
        "by_category": dict(by_category),
        "sentiment_counts": dict(sentiments),
        "positive_files": positive_files,
        "answers_by_category": dict(answers_by_category),
    }


def main() -> None:
    texts = load_texts()
    if len(texts) < 5:
        raise ValueError("Нужно минимум 5 тестовых текстов в папке texts/")

    results: list[dict] = []
    parsed_rows: list[tuple[str, AnalysisResult]] = []
    errors: list[dict] = []

    for i, (filename, text) in enumerate(texts, start=1):
        try:
            result = generate_summary(text, variant=DEFAULT_VARIANT)
            log_result(i, filename, result)
            parsed_rows.append((filename, result))
            results.append(
                {
                    "file": filename,
                    "text": text,
                    "variant": DEFAULT_VARIANT,
                    "result": result.model_dump(mode="json"),
                }
            )
        except StructuredOutputError as exc:
            logger.error("Ошибка structured output для %s: %s", filename, exc)
            errors.append({"file": filename, "error": str(exc)})

    report = build_report(parsed_rows)

    logger.info("=" * 60)
    logger.info("Отчёт по structured fields")
    logger.info("Всего успешно: %s | ошибок: %s", report["total"], len(errors))
    logger.info("По категориям: %s", report["by_category"])
    logger.info("Тональность: %s", report["sentiment_counts"])
    logger.info("Позитивные тексты: %s", report["positive_files"])
    for category, answers in report["answers_by_category"].items():
        logger.info("Final answers [%s]: %s", category, answers)

    output_path = Path("results.json")
    output_path.write_text(
        json.dumps(
            {"results": results, "errors": errors, "report": report},
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
