"""Run persistence, API authentication, timeouts and large-result handling."""

import pytest

from agentmesh.api.auth import Authenticator, is_loopback, load_keys
from agentmesh.core.model_router import DEFAULT_TIMEOUT_S, _resolve_key
from agentmesh.core.run_store import list_runs, load_run
from agentmesh.core.runtime import run_project
from agentmesh.data.base import DataResultTooLarge, RowBatch
from agentmesh.data.context import DataContext, SourceRegistry
from agentmesh.data.sqlite_source import SqliteSource

# -- run persistence ------------------------------------------------------


@pytest.fixture()
def persisted(tmp_path):
    result = run_project("data_ops", "Review this SQL query", provider="mock",
                         persist=True, run_dir=tmp_path)
    return result, tmp_path


def test_a_run_writes_inspectable_artefacts(persisted) -> None:
    result, run_dir = persisted
    files = {p.name for p in (run_dir / result.recorder.run_id).iterdir()}

    assert files == {
        "run.json", "llm_calls.jsonl", "tool_runs.jsonl",
        "data_ops.jsonl", "agent_outputs.jsonl", "final_response.json",
    }


def test_a_persisted_run_reloads_with_its_detail(persisted) -> None:
    result, run_dir = persisted

    loaded = load_run(result.recorder.run_id, run_dir)

    assert loaded["selected_agents"] == result.run.selected_agents
    assert loaded["request"] == "Review this SQL query"
    assert len(loaded["llm_calls"]) == len(result.recorder.calls)
    assert loaded["graph"]["final_agent"] == result.run.final_agent_id


def test_runs_are_listed_newest_first(tmp_path) -> None:
    for message in ("first query", "second query"):
        run_project("data_ops", message, provider="mock", persist=True, run_dir=tmp_path)

    assert len(list_runs(tmp_path)) == 2


def test_nothing_is_written_unless_asked(tmp_path) -> None:
    run_project("data_ops", "Review this SQL query", provider="mock", run_dir=tmp_path)

    assert list_runs(tmp_path) == []


def test_loading_an_unknown_run_is_explicit(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run("nope", tmp_path)


# -- API authentication ---------------------------------------------------


def test_keys_parse_into_principals() -> None:
    keys = load_keys({"AGENTMESH_API_KEYS": "abc:anjali:acme, plain"})

    assert keys[0].principal.id == "anjali" and keys[0].principal.org == "acme"
    assert keys[1].principal.id.startswith("key:")


def test_a_valid_key_identifies_the_caller() -> None:
    auth = Authenticator({"AGENTMESH_API_KEYS": "s3cret:anjali:acme"})

    assert auth.authenticate("s3cret", "10.0.0.5").id == "anjali"


def test_an_invalid_key_is_refused() -> None:
    auth = Authenticator({"AGENTMESH_API_KEYS": "s3cret"})

    for presented in (None, "", "wrong"):
        with pytest.raises(PermissionError):
            auth.authenticate(presented, "127.0.0.1")


def test_without_keys_only_loopback_is_served() -> None:
    auth = Authenticator({})

    assert auth.authenticate(None, "127.0.0.1").id == "local"
    with pytest.raises(PermissionError, match="only loopback"):
        auth.authenticate(None, "203.0.113.9")


def test_loopback_detection() -> None:
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("10.0.0.1") and not is_loopback(None)


# -- timeouts -------------------------------------------------------------


def test_models_get_a_default_timeout() -> None:
    assert _resolve_key("anthropic/claude-opus-5", {}).timeout_s == DEFAULT_TIMEOUT_S


def test_a_timeout_can_be_set_per_model() -> None:
    routing = {"anthropic/fast": {"provider": "anthropic", "timeout_s": 15}}

    assert _resolve_key("anthropic/fast", routing).timeout_s == 15.0


# -- large results --------------------------------------------------------


@pytest.fixture()
def wide_table():
    source = SqliteSource("db")
    source.load_rows("events", ["id", "payload"], [(n, "x" * 200) for n in range(5_000)])
    registry = SourceRegistry.empty()
    registry.add_source(source)
    registry.bind("events", "db", "events")
    return DataContext(registry=registry)


def test_streaming_never_materialises_the_whole_result(wide_table) -> None:
    batches = list(wide_table.table("events").stream("SELECT * FROM {table}", batch_size=500))

    assert len(batches) == 10
    assert all(isinstance(b, RowBatch) and len(b) == 500 for b in batches)
    assert batches[-1].offset == 4_500


def test_streaming_respects_a_row_ceiling(wide_table) -> None:
    rows = sum(
        len(b)
        for b in wide_table.table("events").stream(
            "SELECT * FROM {table}", batch_size=300, max_rows=1_000
        )
    )

    assert rows == 1_000


def test_an_oversized_result_is_refused_by_bytes(wide_table) -> None:
    with pytest.raises(DataResultTooLarge, match="byte ceiling"):
        wide_table.table("events").query("SELECT * FROM {table}", max_bytes=1_000)


def test_aggregation_stays_small(wide_table) -> None:
    result = wide_table.table("events").query("SELECT COUNT(*) AS n FROM {table}")

    assert result.as_dicts() == [{"n": 5_000}]
    assert result.row_count == 1
