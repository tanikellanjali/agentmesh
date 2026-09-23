from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentmesh.providers.base import Message, Turn
from agentmesh.tools.base import ToolResult, ToolSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelCall

DEFAULT_MAX_ITERATIONS = 6


@dataclass
class LoopResult:
    text: str
    tool_results: list[ToolResult] = field(default_factory=list)
    turns: int = 0
    stopped_on_limit: bool = False

    @property
    def tools_used(self) -> list[str]:
        return [r.tool for r in self.tool_results]


def _render(result: ToolResult) -> str:
    import json

    if result.ok:
        return json.dumps(result.output, default=str)[:8000]
    return json.dumps({"error": result.error})


def run_tool_loop(
    call: "ModelCall",
    *,
    system: str,
    prompt: str,
    tools: list[ToolSpec],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    on_tool: Any = None,
    data_context: Any = None,
) -> LoopResult:
    """Drive a model until it stops asking for tools.

    The model may call tools repeatedly; each result is fed back so it can reason
    over real output. A tool that raises is reported to the model as an error
    rather than killing the run - the model can then try different arguments.
    """
    by_name = {t.name: t for t in tools}
    messages: list[Message] = [Message(role="user", content=prompt)]
    collected: list[ToolResult] = []

    for iteration in range(1, max_iterations + 1):
        turn: Turn = call.converse(system=system, messages=messages, tools=tools or None)

        if not turn.wants_tools:
            return LoopResult(text=turn.text, tool_results=collected, turns=iteration)

        messages.append(turn.as_message())
        for requested in turn.tool_calls:
            spec = by_name.get(requested.name)
            if spec is None:
                result = ToolResult(
                    tool=requested.name,
                    ok=False,
                    output=None,
                    error=f"tool '{requested.name}' is not available to this agent",
                )
            else:
                result = spec.call(requested.arguments, data_context)

            collected.append(result)
            if on_tool:
                on_tool(requested, result)

            messages.append(
                Message(
                    role="tool",
                    content=_render(result),
                    tool_call_id=requested.id,
                    tool_name=requested.name,
                )
            )

    # Out of iterations: ask once more, with tools withheld, for a final answer.
    messages.append(
        Message(
            role="user",
            content="Tool budget exhausted. Answer now using what you already have.",
        )
    )
    final = call.converse(system=system, messages=messages, tools=None)
    return LoopResult(
        text=final.text,
        tool_results=collected,
        turns=max_iterations + 1,
        stopped_on_limit=True,
    )
