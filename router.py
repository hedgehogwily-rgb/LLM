"""Явный routing в коде: категория → ключ/инструкции промпта."""

from prompts import RESPONSE_PROMPTS
from schemas import RequestType


def route(category: RequestType) -> str:
    if category not in RESPONSE_PROMPTS:
        raise KeyError(f"Нет response-промпта для категории: {category}")
    return category.value


def get_style_instructions(category: RequestType) -> str:
    try:
        return RESPONSE_PROMPTS[category]
    except KeyError as exc:
        raise KeyError(f"Нет response-промпта для категории: {category}") from exc
