import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from prompts import DEFAULT_VARIANT, build_user_prompt, get_system_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_summary(text: str, variant: str = DEFAULT_VARIANT) -> dict:
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

    return json.loads(response.choices[0].message.content)
