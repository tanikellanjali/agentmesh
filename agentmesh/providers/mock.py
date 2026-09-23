from __future__ import annotations

import json

from agentmesh.providers.base import Completion, Usage


class MockProvider:
    """Deterministic provider used for tests, CI, and offline development.

    Returns a contract-shaped JSON payload without touching the network, so the
    full runtime can be exercised with no credentials.
    """

    name = "mock"

    def __init__(self, credentials: dict[str, str] | None = None) -> None:
        self.credentials = credentials or {}
        self.calls: list[dict[str, str]] = []

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> Completion:
        self.calls.append({"system": system, "prompt": prompt, "model": model})
        text = json.dumps(
            {
                "summary": f"Mock completion from {model}.",
                "findings": ["Mock provider: no model was called."],
                "confidence": 0.75,
            }
        )
        return Completion(
            text=text,
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4),
            stop_reason="end_turn",
        )
