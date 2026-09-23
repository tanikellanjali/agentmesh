from __future__ import annotations

import json
from collections.abc import Callable

from agentmesh.providers.base import Message, SingleShotMixin, Turn, Usage
from agentmesh.tools.base import ToolCall, ToolSpec


class MockProvider(SingleShotMixin):
    """Deterministic provider for tests, CI, and offline development.

    Calls each tool it is offered exactly once, then returns a contract-shaped
    JSON summary. That exercises the real tool loop without a network call, so
    offline runs prove the plumbing rather than skipping it.
    """

    name = "mock"

    def __init__(
        self,
        credentials: dict[str, str] | None = None,
        planner: Callable[[list[Message], list[ToolSpec] | None], Turn] | None = None,
    ) -> None:
        self.credentials = credentials or {}
        self.planner = planner
        self.turns: list[dict[str, object]] = []

    def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str = "deterministic",
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> Turn:
        self.turns.append({"system": system, "messages": list(messages), "model": model})

        if self.planner:
            return self.planner(messages, tools)

        used = {msg.tool_name for msg in messages if msg.role == "tool"}
        pending = [t for t in (tools or []) if t.name not in used]

        if pending:
            tool = pending[0]
            return Turn(
                text="",
                provider=self.name,
                model=model,
                tool_calls=[
                    ToolCall(
                        id=f"mock_{tool.name}",
                        name=tool.name,
                        arguments=_sample_arguments(tool, messages),
                    )
                ],
                usage=Usage(input_tokens=20, output_tokens=10),
                stop_reason="tool_use",
            )

        observed = [
            {"tool": msg.tool_name, "result": msg.content[:200]}
            for msg in messages
            if msg.role == "tool"
        ]
        text = json.dumps(
            {
                "summary": f"Mock completion from {model}.",
                "findings": [f"Called {len(observed)} tool(s)."] if observed else
                            ["Mock provider: no tools were offered."],
                "tool_results": observed,
                "confidence": 0.75,
            }
        )
        return Turn(
            text=text,
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=30, output_tokens=len(text) // 4),
            stop_reason="end_turn",
        )


def _sample_arguments(tool: ToolSpec, messages: list[Message]) -> dict[str, object]:
    """Fill a tool's required string fields with the user's request text."""
    request = next((m.content for m in messages if m.role == "user"), "")
    schema = tool.input_schema.get("properties", {})
    args: dict[str, object] = {}
    for field in tool.input_schema.get("required", []):
        kind = schema.get(field, {}).get("type", "string")
        args[field] = request if kind == "string" else None
    return args
