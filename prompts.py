from schemas import RequestType

KEY_POINTS_COUNT = 3
REQUEST_TYPES = " | ".join(item.value for item in RequestType)

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
    "Use the same language as the input text for summary, key_points and final_answer. "
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
        "Give a clear numbered troubleshooting path (2–3 steps) and ask for missing diagnostic details if needed."
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
    '  "key_points": [string, string, string],\n'
    '  "final_answer": string\n'
    "}}\n\n"
    f"Constraints:\n"
    f"1) key_points length must be exactly {KEY_POINTS_COUNT}\n"
    "2) summary is short\n"
    "3) final_answer must clearly follow the style instructions\n"
    "4) no extra keys, no markdown\n\n"
    "Text:\n{text}"
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
