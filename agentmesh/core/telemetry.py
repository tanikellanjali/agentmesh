from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from agentmesh.providers.base import Usage

logger = logging.getLogger("agentmesh.llm")


class BudgetExceeded(RuntimeError):
    """Raised when a run hits the cost ceiling the caller set.

    AgentMesh sets no default ceiling. This is only raised when the caller
    supplies one.
    """


@dataclass(frozen=True)
class LLMCall:
    """One model call. A single agent may produce several of these."""

    call_id: str
    agent_id: str
    provider: str
    model_key: str
    model: str
    attempts: int
    outcome: str  # ok | error | refusal | unparsable
    duration_ms: float
    input_tokens: int
    output_tokens: int
    cost: float
    error: str | None = None
    retry_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunRecorder:
    """Collects every model call in a run and enforces an opt-in cost ceiling."""

    def __init__(self, run_id: str | None = None, max_cost: float | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.max_cost = max_cost
        self.calls: list[LLMCall] = []
        # Agents in one dependency level run concurrently.
        self._lock = threading.Lock()

    # -- aggregates -------------------------------------------------------
    @property
    def usage(self) -> Usage:
        return Usage(
            input_tokens=sum(call.input_tokens for call in self.calls),
            output_tokens=sum(call.output_tokens for call in self.calls),
        )

    @property
    def cost(self) -> float:
        return sum(call.cost for call in self.calls)

    @property
    def retries(self) -> int:
        """Total retry attempts across the run (attempts beyond the first)."""
        return sum(max(call.attempts - 1, 0) for call in self.calls)

    def for_agent(self, agent_id: str) -> list[LLMCall]:
        return [call for call in self.calls if call.agent_id == agent_id]

    def usage_for(self, agent_id: str) -> Usage:
        calls = self.for_agent(agent_id)
        return Usage(
            input_tokens=sum(call.input_tokens for call in calls),
            output_tokens=sum(call.output_tokens for call in calls),
        )

    def cost_for(self, agent_id: str) -> float:
        return sum(call.cost for call in self.for_agent(agent_id))

    def retries_for(self, agent_id: str) -> int:
        return sum(max(call.attempts - 1, 0) for call in self.for_agent(agent_id))

    # -- recording --------------------------------------------------------
    @property
    def over_budget(self) -> bool:
        return self.max_cost is not None and self.cost > self.max_cost

    def check_budget(self, agent_id: str) -> None:
        """Block a *new* call once the ceiling is reached.

        Calls already in flight are not cancelled, so a level running several
        agents concurrently can overshoot by up to one level's spend. Use
        ``max_workers=1`` when the ceiling must be strict.
        """
        if self.max_cost is None:
            return
        with self._lock:
            spent = sum(call.cost for call in self.calls)
        if spent >= self.max_cost:
            raise BudgetExceeded(
                f"Run {self.run_id} reached the ${self.max_cost:.4f} ceiling "
                f"(spent ${spent:.4f}) before agent '{agent_id}'."
            )

    def record(self, call: LLMCall) -> LLMCall:
        with self._lock:
            self.calls.append(call)
        logger.info(
            "llm_call agent=%s model=%s outcome=%s attempts=%s "
            "in=%s out=%s cost=$%.6f %.0fms",
            call.agent_id,
            call.model_key,
            call.outcome,
            call.attempts,
            call.input_tokens,
            call.output_tokens,
            call.cost,
            call.duration_ms,
            extra={"agentmesh": {"run_id": self.run_id, **call.as_dict()}},
        )
        return call

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "llm_calls": len(self.calls),
            "retries": self.retries,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "estimated_cost_usd": round(self.cost, 6),
            "max_cost_usd": self.max_cost,
            "calls": [call.as_dict() for call in self.calls],
        }


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "agentmesh", {}))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Attach a handler to the agentmesh logger. Safe to call more than once."""
    root = logging.getLogger("agentmesh")
    root.setLevel(level.upper())
    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler()
    handler.setFormatter(
        _JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.propagate = False
