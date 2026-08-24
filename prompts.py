"""Промпты отделены от пайплайна: system + user template + варианты формулировок."""

SUMMARY_MAX_WORDS = 25
KEY_POINTS_COUNT = 3
FINAL_ANSWER_MAX_SENTENCES = 2

CATEGORIES = "tech | health | education | lifestyle | other"
SENTIMENTS = "positive | neutral | negative"

SYSTEM_PROMPTS = {
    "v1_basic": (
        "You are a helpful assistant. "
        "Reply with valid JSON only. "
        "Use the same language as the input text for summary, key_points and final_answer. "
        "category and sentiment must be English enum values."
    ),
    "v2_strict": (
        "You are a careful text analyst. "
        "Always reply with valid JSON only. "
        "Follow length and enum limits strictly. "
        "Use the same language as the input text for text fields. "
        "Do not add extra fields."
    ),
    "v3_structured": (
        "You are a precise text analyst. "
        "Always reply with valid JSON only and no markdown. "
        "Obey all constraints and enums exactly. "
        "Use the same language as the input text for summary, key_points and final_answer. "
        "Prefer clarity and concrete wording over vague statements."
    ),
}

USER_TEMPLATES = {
    "v1_basic": (
        "Analyze the text and return JSON with fields:\n"
        '- "summary": short summary\n'
        f'- "category": one of [{CATEGORIES}]\n'
        f'- "sentiment": one of [{SENTIMENTS}]\n'
        '- "key_points": array of key points\n'
        '- "final_answer": short helpful response\n\n'
        "Text:\n{text}"
    ),
    "v2_strict": (
        "Return JSON with exactly these fields:\n"
        f'- "summary": max {SUMMARY_MAX_WORDS} words\n'
        f'- "category": one of [{CATEGORIES}]\n'
        f'- "sentiment": one of [{SENTIMENTS}]\n'
        f'- "key_points": exactly {KEY_POINTS_COUNT} short ideas\n'
        f'- "final_answer": at most {FINAL_ANSWER_MAX_SENTENCES} sentences\n\n'
        "Do not exceed these limits.\n\n"
        "Text:\n{text}"
    ),
    "v3_structured": (
        "Task: analyze the text and return ONLY this JSON schema:\n"
        "{{\n"
        '  "summary": string,\n'
        f'  "category": "{CATEGORIES}",\n'
        f'  "sentiment": "{SENTIMENTS}",\n'
        '  "key_points": [string, string, string],\n'
        '  "final_answer": string\n'
        "}}\n\n"
        "Constraints:\n"
        f"1) summary <= {SUMMARY_MAX_WORDS} words\n"
        f"2) key_points length must be exactly {KEY_POINTS_COUNT}\n"
        f"3) final_answer <= {FINAL_ANSWER_MAX_SENTENCES} short sentences\n"
        "4) no extra keys, no markdown, no commentary\n"
        "5) each key point must be specific and non-overlapping\n\n"
        "Text:\n{text}"
    ),
}

DEFAULT_VARIANT = "v3_structured"
PROMPT_VARIANTS = tuple(SYSTEM_PROMPTS.keys())


def get_system_prompt(variant: str = DEFAULT_VARIANT) -> str:
    if variant not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt variant: {variant}. Available: {PROMPT_VARIANTS}")
    return SYSTEM_PROMPTS[variant]


def build_user_prompt(text: str, variant: str = DEFAULT_VARIANT) -> str:
    if variant not in USER_TEMPLATES:
        raise ValueError(f"Unknown prompt variant: {variant}. Available: {PROMPT_VARIANTS}")
    return USER_TEMPLATES[variant].format(text=text)
