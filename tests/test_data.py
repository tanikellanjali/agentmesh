import pytest

from agentmesh.access.authorizer import AccessDenied, DenyAll
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.core.run_context import RunContext
from agentmesh.core.telemetry import RunRecorder
from agentmesh.data.base import DataBudgetExceeded, DataError, DataRequirement
from agentmesh.data.context import SourceRegistry
from agentmesh.data.sqlite_source import CsvSource, SqliteSource
from agentmesh.schemas.agent_spec import AgentSpec
from agentmesh.tools import get_tool

CSV = "supplier,units,unit_cost,sale_price\nDairyCo,80,1.10,2.20\nFreshFarms,120,0.40,0.95\n"


@pytest.fixture()
def csv_registry(tmp_path):
    (tmp_path / "purchases.csv").write_text(CSV)
    registry = SourceRegistry.empty()
    registry.add_source(CsvSource("files", root=tmp_path))
    registry.bind("transactions", "files", "purchases.csv")
    return registry


def spec_with_data(**overrides):
    base = dict(
        id="pl_agent", name="P&L Agent", category="finance",
        description="margins", capabilities=["profit_loss"],
        input_contract={"required": ["message"]},
        output_contract={"format": "json", "required_fields": ["summary"]},
        models={"default": "mock/deterministic"},
        data_requirements=[{
            "name": "transactions",
            "required_columns": ["supplier", "units"],
        }],
    )
    base.update(overrides)
    return AgentSpec(**base)


def context(registry, recorder=None, authorizer=None, max_data_cost=None, spec=None):
    return RunContext(
        principal=Principal(id="anjali"),
        authorizer=authorizer,
        sources=registry,
        recorder=recorder,
        max_data_cost=max_data_cost,
    ).data_context_for(spec or spec_with_data())


# -- sources --------------------------------------------------------------


def test_sql_aggregation_happens_in_the_source() -> None:
    src = SqliteSource("db")
    src.load_rows("t", ["region", "amount"], [("e", 100), ("w", 250), ("e", 75)])

    result = src.query("t", "SELECT region, SUM(amount) AS total FROM {table} GROUP BY region")

    assert sorted(result.as_dicts(), key=lambda r: r["region"]) == [
        {"region": "e", "total": 175}, {"region": "w", "total": 250}
    ]


def test_writes_are_refused_on_a_read_only_source() -> None:
    src = SqliteSource("db")
    src.load_rows("t", ["a"], [(1,)])

    for statement in ("DELETE FROM {table}", "DROP TABLE t", "UPDATE {table} SET a=2"):
        with pytest.raises(DataError, match="read-only"):
            src.query("t", statement)


def test_unsafe_table_references_are_rejected() -> None:
    with pytest.raises(DataError, match="unsafe table reference"):
        SqliteSource("db").describe("t; DROP TABLE users")


def test_results_are_capped_and_flagged() -> None:
    src = SqliteSource("db")
    src.load_rows("t", ["a"], [(n,) for n in range(50)])

    result = src.query("t", "SELECT * FROM {table}", max_rows=10)

    assert result.row_count == 10
    assert result.truncated is True


def test_csv_is_queryable_as_sql(csv_registry) -> None:
    source, ref = csv_registry.resolve("transactions")

    result = source.query(ref, "SELECT COUNT(*) AS n FROM {table}")

    assert result.as_dicts() == [{"n": 2}]


def test_missing_file_is_reported_clearly(tmp_path) -> None:
    source = CsvSource("files", root=tmp_path)

    with pytest.raises(DataError, match="file not found"):
        source.describe("nope.csv")


# -- bindings -------------------------------------------------------------


def test_an_unbound_requirement_explains_how_to_bind_it() -> None:
    ctx = context(SourceRegistry.empty())

    with pytest.raises(DataError, match="no data bound"):
        ctx.table("transactions").schema()


def test_unbound_requirements_are_listed(csv_registry) -> None:
    spec = spec_with_data(data_requirements=[
        {"name": "transactions"}, {"name": "inventory"}, {"name": "extras", "optional": True},
    ])
    ctx = context(csv_registry, spec=spec)

    assert ctx.unbound() == ["inventory"]
    assert ctx.available() == ["transactions"]


def test_required_columns_are_validated_against_real_data(csv_registry) -> None:
    ctx = context(csv_registry)
    assert ctx.table("transactions").validate() == []

    strict = context(csv_registry, spec=spec_with_data(data_requirements=[
        {"name": "transactions", "required_columns": ["supplier", "shelf_life"]}
    ]))
    assert strict.table("transactions").validate() == ["shelf_life"]


def test_binding_an_unknown_source_is_rejected() -> None:
    with pytest.raises(DataError, match="unknown source"):
        SourceRegistry.empty().bind("x", "nowhere", "t")


# -- policy + budget ------------------------------------------------------


def test_data_access_is_gated_by_the_authorizer(csv_registry) -> None:
    ctx = context(csv_registry, authorizer=DenyAll())

    with pytest.raises(AccessDenied):
        ctx.table("transactions").query("SELECT * FROM {table}")


def test_a_query_over_the_data_ceiling_is_refused_before_running(csv_registry) -> None:
    source, _ = csv_registry.resolve("transactions")
    source.cost_per_tb_scanned = 1_000_000_000  # make the tiny file "expensive"
    ctx = context(csv_registry, max_data_cost=0.0000001)

    with pytest.raises(DataBudgetExceeded, match="exceeds"):
        ctx.table("transactions").query("SELECT * FROM {table}")


def test_cost_is_estimated_before_execution(csv_registry) -> None:
    estimate = context(csv_registry).table("transactions").estimate("SELECT * FROM {table}")

    assert estimate.bytes_scanned > 0
    assert estimate.exact is False  # sqlite approximates; a warehouse dry-run would not


# -- telemetry ------------------------------------------------------------


def test_every_query_is_recorded(csv_registry) -> None:
    recorder = RunRecorder()
    context(csv_registry, recorder).table("transactions").query("SELECT * FROM {table}")

    op = recorder.data_ops[0]
    assert op.agent_id == "pl_agent"
    assert op.source == "files"
    assert op.ref == "purchases.csv"
    assert op.ok and op.rows == 2


def test_a_failed_query_is_recorded_too(csv_registry) -> None:
    recorder = RunRecorder()
    with pytest.raises(DataError):
        context(csv_registry, recorder).table("transactions").query("SELECT nope FROM {table}")

    assert recorder.data_ops[0].ok is False
    assert recorder.data_ops[0].error


# -- tools ----------------------------------------------------------------


def test_data_tools_require_a_context() -> None:
    result = get_tool("query_data").call({"requirement": "t", "sql": "SELECT 1"})

    assert result.ok is False
    assert "needs data access" in result.error


def test_query_data_returns_rows_and_cost(csv_registry) -> None:
    out = get_tool("query_data").call(
        {"requirement": "transactions",
         "sql": "SELECT supplier, SUM(units*(sale_price-unit_cost)) AS profit "
                "FROM {table} GROUP BY supplier"},
        context(csv_registry),
    ).output

    profits = {r["supplier"]: r["profit"] for r in out["rows"]}
    assert profits["DairyCo"] == pytest.approx(88.0)
    assert "cost_usd" in out


def test_describe_data_lists_bindings_without_a_requirement(csv_registry) -> None:
    out = get_tool("describe_data").call({}, context(csv_registry)).output

    assert out["bound"] == ["transactions"]
