import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from prompts import DEFAULT_VARIANT, build_user_prompt, get_system_prompt
from schemas import AnalysisResult

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class StructuredOutputError(Exception):
    """Понятная ошибка, если JSON от модели сломан или не проходит схему."""


def generate_summary(text: str, variant: str = DEFAULT_VARIANT) -> AnalysisResult:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": get_system_prompt(variant)},
            {"role": "user", "content": build_user_prompt(text, variant)},
        ],
        temperature=0.5,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
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
        return AnalysisResult.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise StructuredOutputError(
            f"JSON не прошёл валидацию схемы: {details}. Получено: {payload}"
        ) from exc
