from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator

from config import HyperParams, TrainingHistory
from llm.openrouter_client import OpenRouterError, openrouter_chat_completion


def _save_llm_decision(
    output_dir: Path | None,
    generation: int,
    system_prompt: str,
    user_prompt: str,
    raw_response: str | None,
    validated_configs: list[dict] | None,
    error: str | None,
    attempts: int,
) -> None:
    """Save LLM decision details to llm_decisions.json for dashboard visualization."""
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_file = output_dir / "llm_decisions.json"

    # Load existing decisions or start fresh
    if decisions_file.exists():
        with open(decisions_file) as f:
            decisions = json.load(f)
    else:
        decisions = {"decisions": []}

    # Add this decision
    decision_entry = {
        "generation": generation + 1,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "validated_configs": validated_configs,
        "error": error,
        "attempts": attempts,
    }

    # Replace existing entry for this generation or append
    existing_idx = next(
        (i for i, d in enumerate(decisions["decisions"]) if d["generation"] == generation + 1),
        None
    )
    if existing_idx is not None:
        decisions["decisions"][existing_idx] = decision_entry
    else:
        decisions["decisions"].append(decision_entry)

    with open(decisions_file, "w") as f:
        json.dump(decisions, f, indent=2)


class LLMConfigListResponse(BaseModel):
    """Pydantic model for validating LLM-generated config list."""
    configs: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_configs(self) -> "LLMConfigListResponse":
        if not self.configs:
            raise ValueError("configs array cannot be empty")
        return self


def _build_system_prompt(cap: int, schema: dict[str, dict]) -> str:
    """Build system prompt for direct config output."""
    keys = list(schema.keys())
    return f"""You are a hyperparameter tuning assistant for neural network training.

Output a JSON array of training configurations. You can output anywhere from 1 to {cap} configs.

You MUST output valid JSON: an array of objects where each object has ALL these keys:
{keys}

Schema for each parameter:
{json.dumps(schema, indent=2)}

Rules:
- Propose diverse configs that explore promising regions of the search space
- Use previous generation results to guide your decisions
- DO NOT include duplicate configurations
- Each config must be unique"""


def _validate_configs(
    raw_configs: list[dict],
    schema: dict[str, dict],
    cap: int,
) -> list[dict]:
    """Validate configs against schema. Check cap and deduplicate."""
    if len(raw_configs) > cap:
        raise ValueError(f"LLM proposed {len(raw_configs)} configs, but cap is {cap}")

    # Deduplicate
    seen = set()
    unique = []
    for cfg in raw_configs:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)

    # Validate required keys exist
    for cfg in unique:
        for key in schema:
            if key not in cfg:
                raise ValueError(f"Config missing required key: {key}")

    return unique


def _strip_markdown_fences(content: str) -> str:
    """Strip markdown code fences from LLM response if present."""
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        # Remove closing fence
        if content.endswith("```"):
            content = content[:-3].rstrip()
    return content


def _parse_and_validate_response(
    content: str,
    schema: dict[str, dict],
    cap: int,
) -> list[dict]:
    """
    Parse LLM response and validate with Pydantic.
    Raises ValueError, ValidationError, or json.JSONDecodeError on failure.
    """
    content = _strip_markdown_fences(content)
    data = json.loads(content)

    # Handle both [{...}, {...}] and {"configs": [...]} formats
    if isinstance(data, list):
        raw_configs = data
    elif isinstance(data, dict) and "configs" in data:
        raw_configs = data["configs"]
    else:
        raise ValueError("Expected JSON array or object with 'configs' key")

    response = LLMConfigListResponse.model_validate({"configs": raw_configs})
    return _validate_configs(response.configs, schema, cap)


def _build_error_feedback(error: Exception, cap: int) -> str:
    """Build a user message explaining the error and asking for correction."""
    if isinstance(error, json.JSONDecodeError):
        return (
            "Your response was not valid JSON. "
            "Please respond with ONLY valid JSON: an array of config objects like "
            '[{"learning_rate": 0.01, "epochs": 5, ...}, ...]'
        )

    if isinstance(error, ValidationError):
        return (
            f"Validation error: {error}. "
            "Please fix your response. Ensure you output a non-empty array of config objects."
        )

    # ValueError - likely cap exceeded or missing keys
    error_msg = str(error)
    if "cap" in error_msg:
        return (
            f"ERROR: {error_msg}. "
            f"You MUST output at most {cap} configurations. "
            "Please provide a corrected response with fewer configs."
        )

    if "missing required key" in error_msg:
        return (
            f"ERROR: {error_msg}. "
            "Each config object must contain ALL required keys. "
            "Please provide a corrected response."
        )

    return f"Error: {error}. Please fix your response and try again."


def propose_next_generation_configs(
    *,
    schema: dict[str, dict],
    results_so_far: list[TrainingHistory],
    cap: int,
    seed: int,
    model: str,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 2,
    output_dir: Path | None = None,
    generation: int = 0,
) -> list[HyperParams]:
    """
    Ask an LLM to propose training configs for the next generation.
    Uses Pydantic for response validation. Retries with error feedback if validation fails.
    Retries indefinitely with exponential backoff until the LLM succeeds.

    If output_dir is provided, logs LLM decisions to llm_decisions.json.
    """
    system_prompt = _build_system_prompt(cap, schema)
    user_prompt = json.dumps({
        "max_configs": cap,
        "schema": schema,
        "previous_results": [
            {
                "config": asdict(h.params),
                "val_accuracy": h.val_accuracy,
                "val_loss": h.val_loss,
                "wall_time_seconds": h.wall_time_seconds,
            }
            for h in results_so_far
        ],
        "instruction": f"Propose up to {cap} unique training configs as a JSON array. Use previous results to inform your choices. No duplicates."
    })

    total_attempts = 0
    backoff_seconds = 1.0
    max_backoff = 60.0

    while True:
        # Reset messages for a fresh conversation each retry cycle
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        last_response: dict[str, Any] | None = None

        for attempt in range(max_retries + 1):
            total_attempts += 1
            try:
                message = openrouter_chat_completion(
                    model=model,
                    messages=messages,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=800,
                )
                last_response = message
                content = message.get("content") or ""

                validated = _parse_and_validate_response(content, schema, cap)

                # Log successful LLM decision
                _save_llm_decision(
                    output_dir=output_dir,
                    generation=generation,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    raw_response=content,
                    validated_configs=validated,
                    error=None,
                    attempts=total_attempts,
                )

                # Convert dicts to HyperParams
                return [HyperParams(**cfg) for cfg in validated]

            except OpenRouterError as e:
                # API errors: break inner loop and retry with backoff
                last_error = e
                print(f"[LLM advisor] API error: {e}")
                break

            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
                last_error = e
                if attempt < max_retries:
                    # Build error feedback message for retry
                    error_feedback = _build_error_feedback(e, cap)
                    print(f"[LLM advisor] Attempt {attempt + 1} failed: {e}. Retrying...")

                    # Append the assistant's failed response with reasoning_details preserved
                    messages.append({
                        "role": "assistant",
                        "content": message.get("content"),
                        "reasoning_details": message.get("reasoning_details"),
                    })
                    messages.append({"role": "user", "content": error_feedback})
                else:
                    print(f"[LLM advisor] All {max_retries + 1} attempts in this cycle failed: {e}")

        # All attempts in this cycle failed - wait and retry
        print(f"[LLM advisor] Retrying after {backoff_seconds:.1f}s backoff (last error: {last_error})")
        time.sleep(backoff_seconds)

        # Exponential backoff with cap
        backoff_seconds = min(backoff_seconds * 2, max_backoff)
