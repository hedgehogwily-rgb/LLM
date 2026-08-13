import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from prompts import build_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_summary(text: str) -> dict:
    prompt = build_prompt(text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can summarize and analyze text. Always reply with valid JSON only. All text values must be in the same language as the input text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
