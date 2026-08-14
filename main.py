import json
import logging
import re
from pathlib import Path

from llm_client import generate_summary
from prompts import (
    DEFAULT_VARIANT,
    HELPFUL_RESPONSE_MAX_SENTENCES,
    KEY_POINTS_COUNT,
    PROMPT_VARIANTS,
    SUMMARY_MAX_WORDS,
)

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


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def count_sentences(text: str) -> int:
    parts = [p for p in re.split(r"[.!?…]+", text.strip()) if p.strip()]
    return max(len(parts), 1) if text.strip() else 0


def evaluate_result(result: dict) -> dict:
    summary = result.get("summary", "")
    key_points = result.get("key_points", [])
    helpful = result.get("helpful_response", "")

    summary_words = count_words(summary)
    helpful_sentences = count_sentences(helpful)
    points_ok = isinstance(key_points, list) and len(key_points) == KEY_POINTS_COUNT
    summary_ok = summary_words <= SUMMARY_MAX_WORDS
    helpful_ok = helpful_sentences <= HELPFUL_RESPONSE_MAX_SENTENCES

    checks = {
        "summary_words": summary_words,
        "summary_ok": summary_ok,
        "key_points_count": len(key_points) if isinstance(key_points, list) else 0,
        "key_points_ok": points_ok,
        "helpful_sentences": helpful_sentences,
        "helpful_ok": helpful_ok,
    }
    checks["score"] = int(summary_ok) + int(points_ok) + int(helpful_ok)
    return checks


def log_result(index: int, filename: str, variant: str, text: str, result: dict, checks: dict) -> None:
    logger.info("=" * 60)
    logger.info("Текст #%s (%s) | variant=%s | score=%s/3", index, filename, variant, checks["score"])
    logger.info("Summary (%s words): %s", checks["summary_words"], result.get("summary", ""))
    logger.info("Key points (%s):", checks["key_points_count"])
    for i, point in enumerate(result.get("key_points", []), start=1):
        logger.info("  %s. %s", i, point)
    logger.info("Helpful response: %s", result.get("helpful_response", ""))


def compare_variants(texts: list[tuple[str, str]]) -> dict:
    comparison = {"variants": {}, "totals": {}}

    for variant in PROMPT_VARIANTS:
        variant_rows = []
        total_score = 0
        for i, (filename, text) in enumerate(texts, start=1):
            result = generate_summary(text, variant=variant)
            checks = evaluate_result(result)
            total_score += checks["score"]
            log_result(i, filename, variant, text, result, checks)
            variant_rows.append(
                {
                    "file": filename,
                    "text": text,
                    "result": result,
                    "checks": checks,
                }
            )
        comparison["variants"][variant] = variant_rows
        comparison["totals"][variant] = {
            "score_sum": total_score,
            "max_score": len(texts) * 3,
            "avg_score": round(total_score / (len(texts) * 3), 3) if texts else 0.0,
        }
        logger.info(
            "Итог %s: %s/%s",
            variant,
            total_score,
            len(texts) * 3,
        )

    # При равном score предпочитаем DEFAULT_VARIANT (более структурированный промпт).
    best_variant = max(
        comparison["totals"],
        key=lambda name: (
            comparison["totals"][name]["score_sum"],
            1 if name == DEFAULT_VARIANT else 0,
        ),
    )
    comparison["best_variant"] = best_variant
    comparison["default_variant"] = DEFAULT_VARIANT
    return comparison


def run_best(texts: list[tuple[str, str]], variant: str) -> list[dict]:
    results = []
    for i, (filename, text) in enumerate(texts, start=1):
        result = generate_summary(text, variant=variant)
        checks = evaluate_result(result)
        log_result(i, filename, variant, text, result, checks)
        results.append(
            {
                "file": filename,
                "text": text,
                "variant": variant,
                "result": result,
                "checks": checks,
            }
        )
    return results


def main() -> None:
    texts = load_texts()
    if len(texts) < 3:
        raise ValueError("Нужно минимум 3 тестовых текста в папке texts/")

    logger.info("Сравниваем варианты промптов: %s", ", ".join(PROMPT_VARIANTS))
    comparison = compare_variants(texts)

    comparison_path = Path("comparison.json")
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Сравнение сохранено в %s", comparison_path)

    best_variant = comparison["best_variant"]
    logger.info("Лучший по метрикам: %s", best_variant)

    final_results = run_best(texts, variant=best_variant)
    results_path = Path("results.json")
    results_path.write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Финальные результаты (%s) сохранены в %s", best_variant, results_path)


if __name__ == "__main__":
    main()
