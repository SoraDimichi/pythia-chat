from config import AI_NAME


def system_prefix(username="User"):
    return (
        f"Below is a conversation between {username} and {AI_NAME}. "
        f"{AI_NAME} answers questions directly and concisely. "
        f"{AI_NAME} never breaks character or writes as {username}.\n\n"
    )


def extract_text(content):
    """Normalize Gradio 6.x message format (str | dict | list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content[0]["text"] if content else ""
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def build_prompt(history, mode, username="User"):
    """Build the full prompt string from chat history based on mode.

    Modes:
        Raw     — last user message only, no formatting
        Chat    — turn labels (username/AI) with system prompt
        Trained — same prompt format as Chat (model selection handled in model.py)
    """
    if mode == "Raw":
        parts = [
            extract_text(msg["content"])
            for msg in history
            if msg["role"] == "user"
        ]
        return parts[-1] if parts else ""

    lines = [system_prefix(username)]

    for msg in history:
        text = extract_text(msg["content"])
        if msg["role"] == "user":
            lines.append(f"{username}: {text}")
        elif msg["role"] == "assistant":
            lines.append(f"{AI_NAME}: {text}")
    lines.append(f"{AI_NAME}:")
    return "\n".join(lines)


def strip_stop_string(response, text, stop_markers):
    """Check if any stop marker appears in the combined response + new text.

    Returns (text_to_append, was_stopped).
    """
    combined = response + text
    for marker in stop_markers:
        pos = combined.find(marker)
        if pos != -1:
            return combined[:pos][len(response):], True
    return text, False
