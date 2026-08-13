def build_prompt(text: str) -> str:
    return (
        "Analyze the text and return JSON with fields:\n"
        '- "summary": short summary\n'
        '- "key_points": array of exactly 3 key points\n'
        '- "helpful_response": short helpful response\n\n'
        "Answer must be in JSON format only. "
        "All text values must be in the same language as the input text.\n\n"
        f"Text:\n{text}"
    )
