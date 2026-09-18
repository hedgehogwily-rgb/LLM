# Demo-сценарии

Минимум 5 демонстрационных входов из папки [`texts/`](../texts/).  
Запуск: `python main.py --demo`

| # | Файл | Ожидаемая категория | Что показать |
|---|---|---|---|
| 1 | `complaint_billing.txt` | `complaint` | Эмпатичный ответ на жалобу: извинение + конкретный next step |
| 2 | `support_login.txt` | `support` | Структурированный troubleshooting (нумерованные шаги) |
| 3 | `sales_pricing.txt` | `sales` | Короткий продающий ответ с call-to-action |
| 4 | `feedback_ui.txt` | `feedback` | Благодарность за отзыв и отражение ключевого пункта |
| 5 | `general_temperature.txt` | `general_question` | Нейтральный информативный ответ без продаж |

## Как смотреть результат

1. Запустите `python main.py --demo`
2. В логах пройдите шаги цепочки 1→5 для каждого файла
3. Откройте `results.json` (или путь из `--out`)
4. Сверьте образец структуры с [`sample_output.json`](sample_output.json)

## Полный набор входов

В `texts/` лежит 10 файлов (по 2 на категорию).  
Батч всех файлов: `python main.py`
