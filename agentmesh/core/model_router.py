from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from agentmesh.core.telemetry import LLMCall, RunRecorder
from agentmesh.providers import Provider, ProviderUnavailable, get_provider
from agentmesh.providers.base import Completion, ProviderError, ProviderRefusal, Usage
from agentmesh.providers.retry import RetryOutcome, RetryPolicy, call_with_retry
from agentmesh.schemas.agent_spec import AgentSpec

DEFAULT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class ResolvedModel:
    key: str
    provider: str
    model: str
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = None

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens / 1000 * self.cost_per_1k_input_tokens
            + usage.output_tokens / 1000 * self.cost_per_1k_output_tokens
        )


def _resolve_key(key: str, models: dict[str, Any]) -> ResolvedModel:
    config = models.get(key, {})
    provider = config.get("provider") or (key.split("/", 1)[0] if "/" in key else key)
    # `model` lets a routing key stay stable while the provider's model id moves.
    model = config.get("model") or (key.split("/", 1)[1] if "/" in key else key)
    return ResolvedModel(
        key=key,
        provider=provider,
        model=model,
        cost_per_1k_input_tokens=float(config.get("cost_per_1k_input_tokens", 0.0)),
        cost_per_1k_output_tokens=float(config.get("cost_per_1k_output_tokens", 0.0)),
        max_tokens=int(config.get("max_tokens", DEFAULT_MAX_TOKENS)),
        effort=config.get("effort"),
    )


class ModelCall:
    """A provider + model bound to one agent.

    Every call made through this object is retried per policy, timed, costed,
    and recorded - so instrumentation does not depend on each executor
    remembering to do it.
    """

    def __init__(
        self,
        provider: Provider,
        model: ResolvedModel,
        agent_id: str,
        recorder: RunRecorder,
        retry_policy: RetryPolicy,
    ) -> None:
        self.provider = provider
        self.model = model
        self.agent_id = agent_id
        self.recorder = recorder
        self.retry_policy = retry_policy

    def complete(self, *, system: str, prompt: str) -> Completion:
        self.recorder.check_budget(self.agent_id)

        outcome = RetryOutcome()
        started = time.perf_counter()

        def operation() -> Completion:
            return self.provider.complete(
                system=system,
                prompt=prompt,
                model=self.model.model,
                max_tokens=self.model.max_tokens,
                effort=self.model.effort,
            )

        try:
            completion = call_with_retry(operation, self.retry_policy, outcome)
        except ProviderError as error:
            self._record(
                outcome=("refusal" if isinstance(error, ProviderRefusal) else "error"),
                attempts=outcome.attempts,
                usage=Usage(),
                duration_ms=(time.perf_counter() - started) * 1000,
                error=str(error),
                retry_errors=outcome.errors[:-1],
            )
            raise

        self._record(
            outcome="ok",
            attempts=outcome.attempts,
            usage=completion.usage,
            duration_ms=(time.perf_counter() - started) * 1000,
            retry_errors=outcome.errors,
        )
        return completion

    def _record(
        self,
        *,
        outcome: str,
        attempts: int,
        usage: Usage,
        duration_ms: float,
        error: str | None = None,
        retry_errors: list[str] | None = None,
    ) -> None:
        self.recorder.record(
            LLMCall(
                call_id=uuid.uuid4().hex[:12],
                agent_id=self.agent_id,
                provider=self.model.provider,
                model_key=self.model.key,
                model=self.model.model,
                attempts=attempts,
                outcome=outcome,
                duration_ms=round(duration_ms, 2),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost=self.model.cost(usage),
                error=error,
                retry_errors=retry_errors or [],
            )
        )


class ModelBroker:
    """Resolves an agent spec to a provider + model, and caches provider clients.

    Resolution order for each agent: an explicit override, then the spec's
    ``models.default``, then each entry in ``models.fallback``. A provider that
    cannot be constructed (missing SDK or credentials) is skipped in favour of
    the next candidate, so a declared fallback chain actually degrades. Nothing
    ever falls back to a fake model.
    """

    def __init__(
        self,
        models: dict[str, Any] | None = None,
        credentials: dict[str, str] | None = None,
        provider_override: str | None = None,
        model_override: str | None = None,
        recorder: RunRecorder | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.models = models or {}
        self.credentials = credentials or {}
        self.provider_override = provider_override
        self.model_override = model_override
        self.recorder = recorder or RunRecorder()
        self.retry_policy = retry_policy or RetryPolicy()
        self._clients: dict[str, Provider] = {}
        self._unavailable: dict[str, str] = {}

    def candidates(self, spec: AgentSpec) -> list[str]:
        if self.model_override:
            return [self.model_override]

        keys: list[str] = []
        default = spec.models.get("default")
        if default:
            keys.append(default)
        for key in spec.models.get("fallback", []) or []:
            if key not in keys:
                keys.append(key)
        return keys

    def provider_for(self, name: str) -> Provider:
        if name in self._clients:
            return self._clients[name]
        if name in self._unavailable:
            raise ProviderUnavailable(self._unavailable[name])

        try:
            client = get_provider(name, self.credentials)
        except ProviderUnavailable as exc:
            self._unavailable[name] = str(exc)
            raise

        self._clients[name] = client
        return client

    def resolve(self, spec: AgentSpec) -> ModelCall:
        attempts: list[str] = []

        for key in self.candidates(spec):
            resolved = _resolve_key(key, self.models)
            provider_name = self.provider_override or resolved.provider
            try:
                client = self.provider_for(provider_name)
            except ProviderUnavailable as exc:
                attempts.append(f"{key} -> {provider_name}: {exc}")
                continue
            return ModelCall(
                provider=client,
                model=resolved,
                agent_id=spec.id,
                recorder=self.recorder,
                retry_policy=self.retry_policy,
            )

        detail = "; ".join(attempts) or "the agent spec declares no models"
        raise ProviderUnavailable(
            f"No usable model for agent '{spec.id}' ({detail}). "
            "Run `agentmesh configure-models`, or use --provider mock to run offline."
        )
