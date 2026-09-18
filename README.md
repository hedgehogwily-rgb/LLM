# LLM Text Pipeline

Учебный mini-product: принимает текст пользователя, анализирует смысл, классифицирует запрос, строит structured JSON и итоговый ответ. Пайплайн устойчив к сбоям API и плохим ответам модели (retry, fallback-промпт, degraded-режим).

**Требования:** Python 3.10+

## Что делает проект

Полный цикл обработки текста (5 шагов):

1. `extract_meaning` — суть, язык, тон
2. `classify` — категория + intent
3. `build_fields` — summary, sentiment, key_points
4. `generate_final_answer` — ответ в стиле категории (routing в коде)
5. `self_check` — проверка согласованности с исходным текстом

Категории: `support`, `feedback`, `complaint`, `sales`, `general_question`.

## Структура проекта

```text
LLM/
  main.py              # CLI entrypoint
  llm_client.py        # API, цепочка, retry/fallback/degraded
  prompts.py           # system/user и fallback-промпты
  schemas.py           # Pydantic-схемы и лимиты
  router.py            # категория → style prompt
  test_guardrails.py   # негативные тесты Day 6
  requirements.txt
  .env.example
  texts/               # входные тексты (10 штук)
  demo/
    SCENARIOS.md       # описание 5+ демо-сценариев
    sample_output.json # пример structured JSON
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API-ключ

```bash
cp .env.example .env
```

Откройте `.env` и укажите ключ:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Файл `.env` не коммитится (уже в `.gitignore`).

## Запуск

```bash
# Все тексты из texts/
python main.py

# Пять демо-сценариев (разные категории)
python main.py --demo

# Один файл
python main.py --file texts/support_login.txt

# Произвольный текст
python main.py --text "Не могу войти, ошибка 500"

# Другой путь для JSON-результата
python main.py --demo --out demo_results.json
```

Результат сохраняется в `results.json` (или путь из `--out`).

## Негативные тесты (без реального API)

```bash
python -m test_guardrails
```

## Примеры входов и выходов

| Файл | Категория | Что показывает |
|---|---|---|
| `texts/complaint_billing.txt` | complaint | эмпатичный ответ на жалобу |
| `texts/support_login.txt` | support | пошаговый troubleshooting |
| `texts/sales_pricing.txt` | sales | короткий продающий ответ + CTA |
| `texts/feedback_ui.txt` | feedback | благодарность за отзыв |
| `texts/general_temperature.txt` | general_question | нейтральный информативный ответ |

Подробнее: [demo/SCENARIOS.md](demo/SCENARIOS.md).  
Пример structured JSON: [demo/sample_output.json](demo/sample_output.json).

### Фрагмент выхода

```json
{
  "summary": "Пользователь жалуется на двойное списание за подписку...",
  "category": "complaint",
  "sentiment": "negative",
  "key_points": ["...", "...", "..."],
  "final_answer": "Приносим извинения...",
  "fallback_used": false,
  "degraded": false
}
```

## Как устроена устойчивость

- **Retry** — временные ошибки API (сеть, rate limit, 5xx)
- **Fallback-промпт** — если JSON/схема сломаны
- **Degraded** — частичный usable-результат, если шаг всё равно упал

В отчёте смотрите поля `fallback_used`, `degraded`, `errors`.
