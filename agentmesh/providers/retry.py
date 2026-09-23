from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from agentmesh.providers.base import ProviderError, ProviderTransientError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter for transient provider failures.

    Only `ProviderTransientError` (which `ProviderRateLimited` extends) is
    retried. Auth failures, refusals, and unavailable providers are permanent -
    retrying them wastes time and money.
    """

    max_attempts: int = 3
    initial_backoff: float = 0.5
    max_backoff: float = 8.0
    multiplier: float = 2.0
    jitter: bool = True

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, ProviderTransientError)

    def delay_for(self, attempt: int) -> float:
        """Delay in seconds before the attempt after `attempt` (1-indexed)."""
        raw = self.initial_backoff * (self.multiplier ** (attempt - 1))
        delay = min(raw, self.max_backoff)
        if self.jitter:
            delay *= 0.5 + random.random() / 2
        return delay


NO_RETRY = RetryPolicy(max_attempts=1)


@dataclass
class RetryOutcome:
    attempts: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def call_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy,
    outcome: RetryOutcome | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    outcome = outcome or RetryOutcome()
    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        outcome.attempts = attempt
        try:
            return operation()
        except ProviderError as error:
            last_error = error
            outcome.errors.append(str(error))
            if not policy.is_retryable(error) or attempt == policy.max_attempts:
                raise
            sleep(policy.delay_for(attempt))

    raise last_error  # pragma: no cover - loop always returns or raises
