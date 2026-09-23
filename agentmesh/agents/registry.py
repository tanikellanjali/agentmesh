from __future__ import annotations

from collections.abc import Callable

from agentmesh.agents.base_agent import BaseAgent


class ExecutorNotFound(KeyError):
    """Raised when an agent spec names an executor that is not registered."""


_EXECUTORS: dict[str, type[BaseAgent]] = {}


def register_executor(name: str) -> Callable[[type[BaseAgent]], type[BaseAgent]]:
    def decorator(executor: type[BaseAgent]) -> type[BaseAgent]:
        if name in _EXECUTORS:
            raise ValueError(f"Executor already registered: {name}")
        _EXECUTORS[name] = executor
        return executor

    return decorator


def get_executor(name: str) -> type[BaseAgent]:
    try:
        return _EXECUTORS[name]
    except KeyError as exc:
        known = ", ".join(list_executors()) or "<none>"
        raise ExecutorNotFound(
            f"Unknown executor '{name}'. Registered executors: {known}"
        ) from exc


def list_executors() -> list[str]:
    return sorted(_EXECUTORS)
