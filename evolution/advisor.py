from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator

from config import TrainingHistory
from evolution.mini_grid import propose_next_generation_option_lists
from evolution.pareto import pareto_front
from llm.openrouter_client import OpenRouterError, openrouter_chat_completion


def _save_llm_decision(
    output_dir: Path | None,
    generation: int,
    system_prompt: str,
    user_prompt: str,
    raw_response: str | None,
    validated_option_lists: dict[str, list] | None,
    fallback_used: bool,
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
        "validated_option_lists": validated_option_lists,
        "fallback_used": fallback_used,
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


class LLMOptionListsResponse(BaseModel):
    """Pydantic model for validating LLM-generated option_lists."""

    option_lists: dict[str, list[Any]]

    @model_validator(mode="after")
    def validate_option_lists(self) -> "LLMOptionListsResponse":
        # Ensure all lists are non-empty
        for key, values in self.option_lists.items():
            if not values:
                raise ValueError(f"option_lists['{key}'] cannot be empty")
        return self


def _compute_product_size(option_lists: dict[str, list]) -> int:
    """Compute the Cartesian product size of option_lists."""
    if not option_lists:
        return 0
    return reduce(lambda acc, v: acc * len(v), option_lists.values(), 1)


def _build_system_prompt(cap: int) -> str:
    """Build a system prompt that clearly explains the cap constraint."""
    # Calculate example scenarios based on cap
    examples = []
    if cap >= 4:
        examples.append(f"- 2 params × 2 options each = 2×2 = 4 configs ✓")
    if cap >= 8:
        examples.append(f"- 3 params × 2 options each = 2×2×2 = 8 configs ✓")
    else:
        examples.append(f"- 3 params × 2 options each = 2×2×2 = 8 configs ✗ (exceeds {cap})")
    if cap < 16:
        examples.append(f"- 4 params × 2 options each = 2×2×2×2 = 16 configs ✗ (exceeds {cap})")

    max_binary_params = int(math.floor(math.log2(cap))) if cap > 1 else 0

    return f"""You are a hyperparameter tuning assistant.

CRITICAL CONSTRAINT: The Cartesian product of all option lists must be AT MOST {cap} configurations.

Examples:
{chr(10).join(examples)}

With max_configs={cap}, you can safely vary up to {max_binary_params} parameters with 2 options each.

You MUST output valid JSON with this exact structure:
{{"option_lists": {{"param_name": [value1, value2], ...}}}}

Only use parameter names and values from the allowed_values provided."""


def _sanitize_and_validate(
    raw_option_lists: dict[str, list],
    base_search_space: dict[str, list],
    cap: int,
) -> dict[str, list]:
    """
    Sanitize option_lists: filter to allowed keys/values, fill missing keys with defaults.
    Raises ValueError if resulting product exceeds cap.
    """
    cleaned: dict[str, list] = {}

    for key, allowed in base_search_space.items():
        proposed = raw_option_lists.get(key)
        if not isinstance(proposed, list) or not proposed:
            # Use first allowed value as default
            cleaned[key] = [allowed[0]]
        else:
            # Keep only values that are in the allowed list
            kept = [v for v in proposed if v in allowed]
            cleaned[key] = kept if kept else [allowed[0]]

    product_size = _compute_product_size(cleaned)
    if product_size > cap:
        raise ValueError(
            f"LLM proposed {product_size} configs (product of option_lists), but cap is {cap}"
        )

    return cleaned


def _parse_and_validate_response(
    content: str,
    base_search_space: dict[str, list],
    cap: int,
) -> dict[str, list]:
    """
    Parse LLM response and validate with Pydantic.
    Raises ValueError, ValidationError, or json.JSONDecodeError on failure.
    """
    data = json.loads(content)
    # Handle both {"option_lists": {...}} and raw {...} formats
    if "option_lists" not in data and isinstance(data, dict):
        data = {"option_lists": data}

    response = LLMOptionListsResponse.model_validate(data)
    return _sanitize_and_validate(response.option_lists, base_search_space, cap)


def propose_next_generation_option_lists_with_llm(
    *,
    base_search_space: dict[str, list],
    results_so_far: list[TrainingHistory],
    cap: int,
    seed: int,
    model: str,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 2,
    output_dir: Path | None = None,
    generation: int = 0,
) -> dict[str, list]:
    """
    Ask an LLM to propose per-hparam option lists for the next generation mini-grid.
    Uses Pydantic for response validation. Retries with error feedback if validation fails.
    Falls back to heuristic proposal after max_retries.
    
    If output_dir is provided, logs LLM decisions to llm_decisions.json.
    """
    front = pareto_front(results_so_far)
    front_payload = [
        {
            "params": asdict(h.params),
            "val_accuracy": h.val_accuracy,
            "wall_time_seconds": h.wall_time_seconds,
            "param_count": h.param_count,
        }
        for h in front[:10]
    ]

    system_prompt = _build_system_prompt(cap)
    user_prompt = json.dumps(
        {
            "max_configs": cap,
            "allowed_values": base_search_space,
            "current_pareto_front": front_payload,
            "instruction": f"Propose option_lists for the next generation. "
            f"The Cartesian product must be <= {cap}.",
        }
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    last_response: str | None = None
    attempts = 0

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            content = openrouter_chat_completion(
                model=model,
                messages=messages,
                api_key=api_key,
                temperature=temperature,
                max_tokens=800,
            )
            last_response = content

            validated = _parse_and_validate_response(content, base_search_space, cap)
            
            # Log successful LLM decision
            _save_llm_decision(
                output_dir=output_dir,
                generation=generation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=content,
                validated_option_lists=validated,
                fallback_used=False,
                error=None,
                attempts=attempts,
            )
            
            return validated

        except OpenRouterError as e:
            # API errors are not retryable with feedback
            last_error = e
            print(f"[LLM advisor] API error: {e}")
            break

        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
            last_error = e
            if attempt < max_retries:
                # Build error feedback message for retry
                error_feedback = _build_error_feedback(e, cap)
                print(f"[LLM advisor] Attempt {attempt + 1} failed: {e}. Retrying...")

                # Append the assistant's failed response and error feedback
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": error_feedback})
            else:
                print(f"[LLM advisor] All {max_retries + 1} attempts failed: {e}")

    # Fall back to heuristic
    print(f"[LLM advisor] Falling back to heuristic after error: {last_error}")
    
    fallback_result = propose_next_generation_option_lists(
        base_search_space=base_search_space,
        results_so_far=results_so_far,
        cap=cap,
        seed=seed,
    )
    
    # Log fallback decision
    _save_llm_decision(
        output_dir=output_dir,
        generation=generation,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=last_response,
        validated_option_lists=fallback_result,
        fallback_used=True,
        error=str(last_error) if last_error else None,
        attempts=attempts,
    )
    
    return fallback_result


def _build_error_feedback(error: Exception, cap: int) -> str:
    """Build a user message explaining the error and asking for correction."""
    if isinstance(error, json.JSONDecodeError):
        return (
            "Your response was not valid JSON. "
            "Please respond with ONLY valid JSON in this format: "
            '{"option_lists": {"param_name": [value1, value2], ...}}'
        )

    if isinstance(error, ValidationError):
        return (
            f"Validation error: {error}. "
            "Please fix your response. Ensure option_lists is a dict with non-empty lists."
        )

    # ValueError - likely cap exceeded
    error_msg = str(error)
    if "configs" in error_msg and "cap" in error_msg:
        return (
            f"ERROR: {error_msg}. "
            f"You MUST reduce the number of options so the Cartesian product is <= {cap}. "
            f"Try using fewer parameters or fewer options per parameter. "
            "Please provide a corrected response."
        )

    return f"Error: {error}. Please fix your response and try again."


