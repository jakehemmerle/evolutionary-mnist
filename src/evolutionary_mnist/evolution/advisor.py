from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator

from evolutionary_mnist.config import HyperParams, TrainingHistory
from evolutionary_mnist.llm.openrouter_client import OpenRouterError, openrouter_chat_completion


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


def _build_system_prompt(cap: int, schema: dict[str, dict], current_generation: int, total_generations: int) -> str:
    """Build system prompt for direct config output."""
    keys = list(schema.keys())
    return f"""You are a hyperparameter tuning assistant for neural network training.

You are running generation {current_generation + 1} of {total_generations}.

Your response MUST follow this exact format:
1. First, write your reasoning and analysis (what you learned from all previous results, why you're choosing these hyperparameters)
2. Then, output your JSON array of training configurations inside a ```json code fence

You can output anywhere from 1 to {cap} configs.

The JSON must be valid: an array of objects where each object has ALL these keys:
{keys}

Schema for each parameter:
{json.dumps(schema, indent=2)}

Things to remember while reasoning:
- Be more aggressive with your hyperparameter variations. Explore. We are trying to find the best hyperparameters overall.
- Look at your best runs compared to your worst runs. What are the key differences? Explain from first principles why they were better and use that explination to guide your next generation. Bayesian inference!
- Think from first principles about which hyperparameters likely improved the accuracy relative to the other hyperparameters and training runs.
- DO NOT include duplicate configurations or configurations that have already been tried."""


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


def _extract_json_array(content: str) -> str:
    """Extract the first JSON array from LLM response, handling duplicates and extra text."""
    content = content.strip()

    # Try to find first code fence with JSON
    fence_match = re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', content)
    if fence_match:
        return fence_match.group(1)

    # Fall back: find first [...] structure using bracket-matching
    start = content.find('[')
    if start == -1:
        return content  # No array found, return as-is for downstream error

    depth = 0
    for i, char in enumerate(content[start:], start):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0:
                return content[start:i+1]

    return content  # Malformed, return as-is


def _log_response_diagnostics(content: str | None, error: Exception) -> str:
    """Generate diagnostic info for failed parse attempts."""
    if content is None:
        return "Response content is None"
    if content == "":
        return "Response content is empty string"
    
    length = len(content)
    # Show first and last 100 chars for context
    preview_len = 100
    if length <= preview_len * 2:
        preview = repr(content)
    else:
        preview = f"{repr(content[:preview_len])}...{repr(content[-preview_len:])}"
    
    # Check if extraction changed the content
    extracted = _extract_json_array(content)
    extract_info = ""
    if extracted != content.strip():
        extract_info = f" | After extraction: {len(extracted)} chars"
    
    return f"Length: {length} chars{extract_info} | Preview: {preview}"


def _append_debug_log(
    output_dir: Path | None,
    generation: int,
    attempt: int,
    content: str | None,
    error: Exception,
) -> None:
    """Append failed attempt details to llm_debug.log for post-mortem analysis."""
    if output_dir is None:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_file = output_dir / "llm_debug.log"
    
    timestamp = datetime.now().isoformat()
    diagnostics = _log_response_diagnostics(content, error)
    
    entry = f"""
{'=' * 80}
[{timestamp}] Generation {generation + 1}, Attempt {attempt}
Error: {type(error).__name__}: {error}
Diagnostics: {diagnostics}
--- Raw Response Start ---
{content}
--- Raw Response End ---
"""
    
    with open(debug_file, "a") as f:
        f.write(entry)


def _parse_and_validate_response(
    content: str,
    schema: dict[str, dict],
    cap: int,
) -> list[dict]:
    """
    Parse LLM response and validate with Pydantic.
    Raises ValueError, ValidationError, or json.JSONDecodeError on failure.
    """
    content = _extract_json_array(content)
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
    model: str,
    api_key: str | None = None,
    max_retries: int = 2,
    output_dir: Path | None = None,
    generation: int = 0,
    total_generations: int = 1,
) -> list[HyperParams]:
    """
    Ask an LLM to propose training configs for the next generation.
    Uses Pydantic for response validation. Retries with error feedback if validation fails.
    Retries indefinitely with exponential backoff until the LLM succeeds.

    If output_dir is provided, logs LLM decisions to llm_decisions.json.
    """
    system_prompt = _build_system_prompt(cap, schema, generation, total_generations)
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
                    max_tokens=8000,  # Increased: reasoning models need extra tokens for thinking
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
                raw_content = message.get("content") if message else None
                
                # Log diagnostics for debugging
                diagnostics = _log_response_diagnostics(raw_content, e)
                print(f"[LLM advisor] Attempt {total_attempts} failed: {type(e).__name__}: {e}")
                print(f"[LLM advisor] Response diagnostics: {diagnostics}")
                
                # Write detailed info to debug log file
                _append_debug_log(
                    output_dir=output_dir,
                    generation=generation,
                    attempt=total_attempts,
                    content=raw_content,
                    error=e,
                )
                
                if attempt < max_retries:
                    # Build error feedback message for retry
                    error_feedback = _build_error_feedback(e, cap)
                    print(f"[LLM advisor] Retrying with error feedback...")

                    # Append the assistant's failed response with reasoning_details preserved
                    messages.append({
                        "role": "assistant",
                        "content": raw_content,
                        "reasoning_details": message.get("reasoning_details") if message else None,
                    })
                    messages.append({"role": "user", "content": error_feedback})
                else:
                    print(f"[LLM advisor] All {max_retries + 1} attempts in this cycle failed")

        # All attempts in this cycle failed - wait and retry
        print(f"[LLM advisor] Retrying after {backoff_seconds:.1f}s backoff (last error: {last_error})")
        time.sleep(backoff_seconds)

        # Exponential backoff with cap
        backoff_seconds = min(backoff_seconds * 2, max_backoff)
