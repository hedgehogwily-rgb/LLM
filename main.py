import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from llm_client import PipelineError, run_chain
from schemas import ChainResult, RequestType

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


def log_chain_steps(chain: ChainResult) -> None:
    for step in chain.steps_log:
        logger.info(
            "  [Шаг %d] %s | вход: %s | выход: %s",
            step.step, step.name, step.input_summary, step.output_summary,
        )


def log_result(
    index: int,
    filename: str,
    expected: str | None,
    chain: ChainResult,
) -> None:
    answer = chain.answer
    classification = chain.classification
    check = chain.self_check

    match_mark = ""
    if expected:
        match_mark = " OK" if expected == answer.category.value else " MISS"

    logger.info("=" * 60)
    logger.info("Текст #%s (%s)%s", index, filename, match_mark)
    logger.info(
        "expected=%s | predicted=%s | sentiment=%s | confidence=%.2f",
        expected, answer.category.value, answer.sentiment.value,
        classification.confidence,
    )

    logger.info("--- Шаги цепочки ---")
    log_chain_steps(chain)

    logger.info("--- Meaning ---")
    logger.info("  core: %s", chain.meaning.core_meaning)
    logger.info("  tone: %s | lang: %s | entities: %s",
                chain.meaning.tone, chain.meaning.language,
                chain.meaning.key_entities)

    logger.info("--- Fields ---")
    logger.info("  summary: %s", chain.fields.summary)
    logger.info("  key_points: %s", chain.fields.key_points)

    logger.info("--- Final answer ---")
    logger.info("  prompt_used: %s | intent: %s", answer.prompt_used, answer.intent)
    logger.info("  %s", answer.final_answer)

    logger.info("--- Self-check ---")
    logger.info(
        "  consistent=%s | details_preserved=%s | verdict=%s",
        check.is_consistent, check.details_preserved, check.verdict,
    )
    if check.issues:
        for issue in check.issues:
            logger.info("  ⚠ %s", issue)


def build_report(rows: list[dict]) -> dict:
    by_category: dict[str, list[str]] = defaultdict(list)
    sentiment_counts: Counter[str] = Counter()
    prompt_usage: Counter[str] = Counter()
    self_check_pass = 0
    correct = 0
    labeled = 0

    for row in rows:
        result = row["answer"]
        category = result["category"]
        sentiment = result["sentiment"]
        by_category[category].append(row["file"])
        sentiment_counts[sentiment] += 1
        prompt_usage[result["prompt_used"]] += 1

        if row["self_check"]["verdict"].startswith("pass"):
            self_check_pass += 1

        expected = row.get("expected_category")
        if expected:
            labeled += 1
            if expected == category:
                correct += 1

    return {
        "total": len(rows),
        "by_category": dict(by_category),
        "sentiment_counts": dict(sentiment_counts),
        "prompt_usage": dict(prompt_usage),
        "self_check_passed": self_check_pass,
        "self_check_total": len(rows),
        "classification_accuracy": {
            "labeled": labeled,
            "correct": correct,
            "accuracy": round(correct / labeled, 3) if labeled else None,
        },
    }


def main() -> None:
    texts = load_texts()
    if len(texts) < 5:
        raise ValueError("Нужно минимум 5 тестовых текстов в папке texts/")

    results: list[dict] = []
    errors: list[dict] = []

    for i, (filename, text) in enumerate(texts, start=1):
        expected = expected_category_from_filename(filename)
        try:
            chain = run_chain(text)
            log_result(i, filename, expected, chain)
            results.append({
                "file": filename,
                "text": text,
                "expected_category": expected,
                "meaning": chain.meaning.model_dump(mode="json"),
                "classification": chain.classification.model_dump(mode="json"),
                "fields": chain.fields.model_dump(mode="json"),
                "answer": chain.answer.model_dump(mode="json"),
                "self_check": chain.self_check.model_dump(mode="json"),
                "steps_log": [s.model_dump(mode="json") for s in chain.steps_log],
            })
        except PipelineError as exc:
            logger.error("Ошибка пайплайна для %s: %s", filename, exc)
            errors.append({"file": filename, "error": str(exc)})

    report = build_report(results)
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ")
    logger.info("Успешно: %s | ошибок: %s", report["total"], len(errors))
    logger.info("По категориям: %s", report["by_category"])
    logger.info("Тональность: %s", report["sentiment_counts"])
    logger.info("Промпты: %s", report["prompt_usage"])
    logger.info(
        "Self-check: %s/%s passed",
        report["self_check_passed"], report["self_check_total"],
    )
    logger.info("Accuracy классификации: %s", report["classification_accuracy"])

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
        raise SystemExit("Все примеры завершились ошибкой.")


if __name__ == "__main__":
    main()
