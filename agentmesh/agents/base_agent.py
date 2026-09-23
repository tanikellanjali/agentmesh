from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentmesh.schemas.agent_spec import AgentSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelBroker


class BaseAgent:
    """Base class for spec-backed agent executors.

    Token usage, cost, and retry counts are recorded by the broker's
    ``RunRecorder``, keyed by agent id - executors do not track them.
    """

    def __init__(self, spec: AgentSpec, broker: "ModelBroker | None" = None) -> None:
        self.spec = spec
        self.broker = broker

    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _confidence_threshold(self) -> float:
        return float(self.spec.validation.get("confidence_threshold", 0.7))
