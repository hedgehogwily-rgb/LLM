from schemas import FINAL_ANSWER_MAX_SENTENCES, RequestType, SUMMARY_MAX_WORDS

KEY_POINTS_COUNT = 3
REQUEST_TYPES = " | ".join(item.value for item in RequestType)
SENTIMENTS = "positive | neutral | negative"

CLASSIFY_SYSTEM_PROMPT = (
    "You are a request classifier for a customer-facing assistant. "
    "Reply with valid JSON only and no markdown. "
    "Pick exactly one category from the allowed enum. "
    "intent must be a short phrase in the same language as the input text. "
    "confidence is a float from 0 to 1."
)

CLASSIFY_USER_TEMPLATE = (
    "Classify the user request and return ONLY this JSON schema:\n"
    "{{\n"
    f'  "category": "{REQUEST_TYPES}",\n'
    '  "intent": string,\n'
    '  "confidence": number\n'
    "}}\n\n"
    "Category meanings:\n"
    "- support: technical help, bugs, access issues\n"
    "- feedback: praise or constructive product feedback\n"
    "- complaint: dissatisfaction, anger, service failure\n"
    "- sales: buying, pricing, plans, upgrades\n"
    "- general_question: informational question without clear sales/support intent\n\n"
    "Text:\n{text}"
)

RESPONSE_SYSTEM_BASE = (
    "You are a helpful assistant. "
    "Always reply with valid JSON only and no markdown. "
    "Critical: summary, key_points and final_answer MUST be written in the same language "
    "as the input text (if the input is Russian, answer in Russian; if English, in English). "
    "Never switch language. "
    "category and sentiment must be English enum values. "
    "Do not add extra fields."
)

# Явные инструкции по стилю ответа — выбираются роутером в коде.
RESPONSE_PROMPTS: dict[RequestType, str] = {
    RequestType.complaint: (
        "Style: empathetic complaint handling. "
        "Acknowledge the problem, apologize briefly, stay calm, and propose one concrete next step. "
        "Do not sound defensive or salesy."
    ),
    RequestType.sales: (
        "Style: short sales response. "
        "Highlight value clearly, keep it concise, and end with a clear call to action."
    ),
    RequestType.support: (
        "Style: structured technical support. "
        "Give a clear numbered troubleshooting path of exactly 3 short steps. "
        "Keep the whole final_answer within 4 sentences. "
        "Ask for missing diagnostic details only inside those steps, not as an extra paragraph."
    ),
    RequestType.feedback: (
        "Style: grateful feedback handling. "
        "Thank the user, reflect the key point, and say what you will take into account."
    ),
    RequestType.general_question: (
        "Style: clear informative answer. "
        "Answer directly, stay neutral, and avoid sales pressure."
    ),
}

RESPONSE_USER_TEMPLATE = (
    "Category: {category}\n"
    "Intent: {intent}\n"
    "Style instructions:\n{style_instructions}\n\n"
    "Return ONLY this JSON schema:\n"
    "{{\n"
    '  "summary": string,\n'
    f'  "category": "{REQUEST_TYPES}",\n'
    f'  "sentiment": "{SENTIMENTS}",\n'
    '  "key_points": [string, string, string],\n'
    '  "final_answer": string\n'
    "}}\n\n"
    "Constraints:\n"
    f"1) category MUST be exactly \"{{category}}\"\n"
    f"2) summary <= {SUMMARY_MAX_WORDS} words\n"
    f"3) key_points length must be exactly {KEY_POINTS_COUNT}\n"
    f"4) final_answer <= {FINAL_ANSWER_MAX_SENTENCES} sentences\n"
    "5) final_answer must clearly follow the style instructions\n"
    "6) summary, key_points, final_answer language = input text language\n"
    "7) no extra keys, no markdown\n\n"
    "Text:\n{text}"
)


EXTRACT_MEANING_SYSTEM = (
    "You are a text analyst. Extract the core meaning, language, tone, and key entities "
    "from the input text. Reply with valid JSON only, no markdown."
)

EXTRACT_MEANING_USER_TEMPLATE = (
    "Analyze the following text and return ONLY this JSON schema:\n"
    "{{\n"
    '  "core_meaning": string (1-2 sentences, same language as text),\n'
    '  "language": string (e.g. "ru", "en"),\n'
    '  "tone": string (formal | informal | angry | grateful | neutral),\n'
    '  "key_entities": [string, ...] (names, products, companies mentioned)\n'
    "}}\n\n"
    "Text:\n{text}"
)

BUILD_FIELDS_SYSTEM = (
    "You are a structured data builder. "
    "Using the provided meaning analysis and classification, build structured output fields. "
    "Reply with valid JSON only, no markdown. "
    "summary, key_points MUST be in the same language as the original text."
)

BUILD_FIELDS_USER_TEMPLATE = (
    "Original text:\n{text}\n\n"
    "Meaning analysis:\n{meaning}\n\n"
    "Classification:\n  category: {category}\n  intent: {intent}\n\n"
    "Return ONLY this JSON schema:\n"
    "{{\n"
    '  "summary": string,\n'
    f'  "category": "{REQUEST_TYPES}",\n'
    f'  "sentiment": "{SENTIMENTS}",\n'
    '  "key_points": [string, string, string]\n'
    "}}\n\n"
    "Constraints:\n"
    f"1) summary <= {SUMMARY_MAX_WORDS} words\n"
    f"2) key_points must be exactly {KEY_POINTS_COUNT} items\n"
    "3) category MUST be exactly \"{category}\"\n"
    "4) language of summary and key_points = language of original text\n"
)

SELF_CHECK_SYSTEM = (
    "You are a quality-assurance reviewer. Compare the original text with the generated "
    "response and check for consistency and completeness. Reply with valid JSON only."
)

SELF_CHECK_USER_TEMPLATE = (
    "Original text:\n{text}\n\n"
    "Generated response:\n"
    "  summary: {summary}\n"
    "  category: {category}\n"
    "  sentiment: {sentiment}\n"
    "  key_points: {key_points}\n"
    "  final_answer: {final_answer}\n\n"
    "Check:\n"
    "1. Does the response contradict the original text?\n"
    "2. Are important details from the original text missing?\n\n"
    "Return ONLY this JSON:\n"
    "{{\n"
    '  "is_consistent": boolean,\n'
    '  "details_preserved": boolean,\n'
    '  "issues": [string, ...] (empty list if all good),\n'
    '  "verdict": string ("pass" or "fail: <reason>")\n'
    "}}"
)


def build_extract_meaning_prompt(text: str) -> str:
    return EXTRACT_MEANING_USER_TEMPLATE.format(text=text)


def build_fields_prompt(
    text: str, meaning: str, category: str, intent: str,
) -> str:
    return BUILD_FIELDS_USER_TEMPLATE.format(
        text=text, meaning=meaning, category=category, intent=intent,
    )


def build_self_check_prompt(
    text: str, summary: str, category: str, sentiment: str,
    key_points: str, final_answer: str,
) -> str:
    return SELF_CHECK_USER_TEMPLATE.format(
        text=text, summary=summary, category=category,
        sentiment=sentiment, key_points=key_points, final_answer=final_answer,
    )


def build_classify_user_prompt(text: str) -> str:
    return CLASSIFY_USER_TEMPLATE.format(text=text)


def build_response_user_prompt(
    text: str,
    category: RequestType,
    intent: str,
    style_instructions: str,
) -> str:
    return RESPONSE_USER_TEMPLATE.format(
        text=text,
        category=category.value,
        intent=intent,
        style_instructions=style_instructions,
    )
