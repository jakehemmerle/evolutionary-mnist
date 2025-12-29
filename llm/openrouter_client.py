from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenRouterError(RuntimeError):
    pass


def openrouter_chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout_seconds: int = 60,
) -> str:
    """
    Minimal OpenRouter chat-completions client.
    Returns assistant message content.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("Missing OPENROUTER_API_KEY (env var) and no api_key provided.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter:
            "X-Title": "mnist-evolution-tuner",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise OpenRouterError(f"OpenRouter HTTP error: {e.code} {e.reason}: {detail}") from e
    except urllib.error.URLError as e:
        raise OpenRouterError(f"OpenRouter connection error: {e}") from e

    data = json.loads(raw)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise OpenRouterError(f"Unexpected OpenRouter response shape: {data}") from e


