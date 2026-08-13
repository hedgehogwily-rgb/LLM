import json
import logging
from pathlib import Path

from llm_client import generate_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TEXTS_DIR = Path("texts")
TEXT_FILES = [
    TEXTS_DIR / "text1.txt",
    TEXTS_DIR / "text2.txt",
    TEXTS_DIR / "text3.txt",
]


def load_texts(paths: list[Path] = TEXT_FILES) -> list[tuple[str, str]]:
    texts = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл: {path}")
        texts.append((path.name, path.read_text(encoding="utf-8").strip()))
    return texts


def log_result(index: int, filename: str, text: str, result: dict) -> None:
    logger.info("=" * 60)
    logger.info("Текст #%s (%s)", index, filename)
    logger.info("=" * 60)
    logger.info("Исходный текст:\n%s", text)
    logger.info("Summary: %s", result.get("summary", ""))
    logger.info("Key points:")
    for i, point in enumerate(result.get("key_points", []), start=1):
        logger.info("  %s. %s", i, point)
    logger.info("Helpful response: %s", result.get("helpful_response", ""))


def main() -> None:
    texts = load_texts()
    if len(texts) < 3:
        raise ValueError("Нужно минимум 3 тестовых текста в папке texts/")

    results = []
    for i, (filename, text) in enumerate(texts, start=1):
        result = generate_summary(text)
        log_result(i, filename, text, result)
        results.append({"file": filename, "text": text, "result": result})

    output_path = Path("results.json")
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Результаты сохранены в %s", output_path)


if __name__ == "__main__":
    main()
