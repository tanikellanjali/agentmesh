from __future__ import annotations

import json
from typing import Any


def parse_model_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a model response as JSON, tolerating ```json fences."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
        candidate = candidate.rsplit("```", 1)[0]
        candidate = candidate.removeprefix("json").strip()

    if not candidate:
        return None, "model returned an empty response"

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"model response was not valid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, "model response was not a JSON object"
    return parsed, None


def missing_fields(output: dict[str, Any], output_contract: dict[str, Any]) -> list[str]:
    required = output_contract.get("required_fields", []) or []
    return [field for field in required if field not in output]
