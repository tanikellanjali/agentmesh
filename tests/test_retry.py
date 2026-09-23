import pytest

from agentmesh.agents.factory import build_agent
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.telemetry import RunRecorder
from agentmesh.providers.base import (
    Completion,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderTransientError,
    Usage,
)
from agentmesh.providers.retry import NO_RETRY, RetryOutcome, RetryPolicy, call_with_retry


class FlakyProvider:
    """Fails `failures` times, then succeeds."""

    name = "flaky"

    def __init__(self, failures=0, error=None, credentials=None):
        self.remaining = failures
        self.error = error or ProviderTransientError("upstream hiccup")
        self.attempts = 0

    def complete(self, *, system, prompt, model, max_tokens=4096, effort=None):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error
        return Completion(
            text='{"summary": "ok", "findings": [], "confidence": 0.9}',
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=10, output_tokens=5),
        )


def broker_with(provider, policy=None, recorder=None):
    broker = ModelBroker(
        provider_override=provider.name,
        retry_policy=policy or RetryPolicy(max_attempts=3, initial_backoff=0, jitter=False),
        recorder=recorder,
    )
    broker._clients[provider.name] = provider
    return broker


# -- policy ---------------------------------------------------------------


def test_only_transient_errors_are_retryable() -> None:
    policy = RetryPolicy()

    assert policy.is_retryable(ProviderTransientError("5xx"))
    assert policy.is_retryable(ProviderRateLimited("429"))
    assert not policy.is_retryable(ProviderAuthError("bad key"))
    assert not policy.is_retryable(ProviderRefusal("declined"))
    assert not policy.is_retryable(ProviderError("400"))


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(initial_backoff=1.0, multiplier=2.0, max_backoff=4.0, jitter=False)

    assert [policy.delay_for(n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 4.0]


def test_jitter_stays_within_half_the_delay() -> None:
    policy = RetryPolicy(initial_backoff=2.0, jitter=True)

    for _ in range(50):
        assert 1.0 <= policy.delay_for(1) <= 2.0


def test_call_with_retry_stops_once_it_succeeds() -> None:
    slept: list[float] = []
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderRateLimited("429")
        return "done"

    outcome = RetryOutcome()
    policy = RetryPolicy(max_attempts=5, initial_backoff=1.0, jitter=False)

    assert call_with_retry(operation, policy, outcome, sleep=slept.append) == "done"
    assert outcome.attempts == 3
    assert slept == [1.0, 2.0]


def test_permanent_errors_are_not_retried() -> None:
    slept: list[float] = []
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise ProviderAuthError("bad key")

    with pytest.raises(ProviderAuthError):
        call_with_retry(operation, RetryPolicy(max_attempts=5), sleep=slept.append)

    assert calls["n"] == 1
    assert slept == []


def test_retries_are_exhausted_then_the_error_surfaces() -> None:
    def operation():
        raise ProviderTransientError("still down")

    outcome = RetryOutcome()
    policy = RetryPolicy(max_attempts=3, initial_backoff=0, jitter=False)

    with pytest.raises(ProviderTransientError):
        call_with_retry(operation, policy, outcome, sleep=lambda _: None)

    assert outcome.attempts == 3
    assert len(outcome.errors) == 3


def test_no_retry_policy_makes_exactly_one_attempt() -> None:
    provider = FlakyProvider(failures=1)
    spec = synthesize_agent("sql_review")

    build_agent(spec, broker_with(provider, policy=NO_RETRY)).run("go", {})

    assert provider.attempts == 1


# -- integration through an agent -----------------------------------------


def test_agent_recovers_after_transient_failures(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = FlakyProvider(failures=2)
    recorder = RunRecorder()
    spec = synthesize_agent("sql_review")

    output = build_agent(spec, broker_with(provider, recorder=recorder)).run("go", {})

    assert output["summary"] == "ok"
    assert provider.attempts == 3


def test_retry_count_is_recorded_on_the_call(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = FlakyProvider(failures=2)
    recorder = RunRecorder()
    spec = synthesize_agent("sql_review")

    build_agent(spec, broker_with(provider, recorder=recorder)).run("go", {})

    call = recorder.calls[0]
    assert call.attempts == 3
    assert call.outcome == "ok"
    assert recorder.retries == 2
    assert recorder.retries_for("sql_review_agent") == 2


def test_auth_failure_is_recorded_without_retrying() -> None:
    provider = FlakyProvider(failures=99, error=ProviderAuthError("bad key"))
    recorder = RunRecorder()
    spec = synthesize_agent("sql_review")

    output = build_agent(spec, broker_with(provider, recorder=recorder)).run("go", {})

    assert provider.attempts == 1
    assert recorder.calls[0].attempts == 1
    assert recorder.calls[0].outcome == "error"
    assert "bad key" in output["error"]


def test_refusals_are_recorded_as_refusals() -> None:
    provider = FlakyProvider(failures=99, error=ProviderRefusal("declined"))
    recorder = RunRecorder()
    spec = synthesize_agent("sql_review")

    build_agent(spec, broker_with(provider, recorder=recorder)).run("go", {})

    assert recorder.calls[0].outcome == "refusal"
