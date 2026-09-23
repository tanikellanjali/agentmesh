from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ToolError(RuntimeError):
    """Raised when a tool cannot run. Caught and reported, never fatal."""


@dataclass(frozen=True)
class ToolSpec:
    """A callable an agent can invoke, described well enough for a model to use it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    tags: tuple[str, ...] = ()
    needs_context: bool = False

    def as_schema(self) -> dict[str, Any]:
        """Provider-neutral description; adapters reshape this per vendor."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def call(self, arguments: dict[str, Any], context: Any = None) -> "ToolResult":
        started = time.perf_counter()
        if self.needs_context and context is None:
            return ToolResult(
                tool=self.name, ok=False, output=None,
                error=f"{self.name} needs data access, but no data is bound to this agent",
            )
        try:
            output = self.fn(context, **arguments) if self.needs_context else self.fn(**arguments)
        except TypeError as exc:
            return ToolResult(
                tool=self.name, ok=False, output=None,
                error=f"invalid arguments for {self.name}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # a failing tool must not kill the run
            return ToolResult(
                tool=self.name, ok=False, output=None,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return ToolResult(
            tool=self.name, ok=True, output=output,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


@dataclass(frozen=True)
class ToolCall:
    """A model's request to run a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    output: Any
    error: str | None = None
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }
