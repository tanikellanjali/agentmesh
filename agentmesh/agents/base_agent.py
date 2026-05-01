from __future__ import annotations

from typing import Any

from agentmesh.schemas.agent_spec import AgentSpec


class BaseAgent:
    """Base class for spec-backed agent executors."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _confidence(self) -> float:
        threshold = self.spec.validation.get("confidence_threshold", 0.7)
        return max(float(threshold), 0.75)

