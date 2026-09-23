from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from agentmesh.tools.base import ToolSpec


class ToolNotFound(KeyError):
    """Raised when a spec names a tool that is not registered."""


_TOOLS: dict[str, ToolSpec] = {}


def register_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    tags: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _TOOLS:
            raise ValueError(f"Tool already registered: {name}")
        # A tool whose first parameter is `ctx` is handed the agent's data
        # context; the model never sees or supplies it.
        params = list(inspect.signature(fn).parameters)
        _TOOLS[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            fn=fn,
            tags=tags,
            needs_context=bool(params) and params[0] == "ctx",
        )
        return fn

    return decorator


def get_tool(name: str) -> ToolSpec:
    try:
        return _TOOLS[name]
    except KeyError as exc:
        raise ToolNotFound(
            f"Unknown tool '{name}'. Registered tools: {', '.join(list_tools()) or '<none>'}"
        ) from exc


def list_tools() -> list[str]:
    return sorted(_TOOLS)


def catalog() -> list[ToolSpec]:
    """Every registered tool, for showing a model what it can reach for."""
    return [_TOOLS[name] for name in list_tools()]


def resolve_tools(names: list[str], strict: bool = False) -> tuple[list[ToolSpec], list[str]]:
    """Resolve declared tool names, reporting any that don't exist."""
    resolved: list[ToolSpec] = []
    missing: list[str] = []
    for name in names:
        try:
            resolved.append(get_tool(name))
        except ToolNotFound:
            if strict:
                raise
            missing.append(name)
    return resolved, missing
