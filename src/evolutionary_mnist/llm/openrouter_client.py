from __future__ import annotations

import os
from typing import Any

import requests


class OpenRouterError(RuntimeError):
    pass


def openrouter_chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None = None,
    max_tokens: int = 800,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """
    OpenRouter chat-completions client with reasoning enabled.
    Returns the full assistant message dict (content, reasoning_details, etc.).
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("Missing OPENROUTER_API_KEY (env var) and no api_key provided.")

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "mnist-evolution-tuner",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "reasoning": {"enabled": True},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise OpenRouterError(f"OpenRouter request timed out after {timeout_seconds}s") from e
    except requests.exceptions.ConnectionError as e:
        raise OpenRouterError(f"OpenRouter connection error: {e}") from e
    except requests.exceptions.HTTPError as e:
        detail = response.text if response else str(e)
        raise OpenRouterError(f"OpenRouter HTTP error: {response.status_code}: {detail}") from e

    data = response.json()
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise OpenRouterError(f"Unexpected OpenRouter response shape: {data}") from e
