"""Library-backed sources: DuckDB for files, SQLAlchemy for databases.

Each is skipped when its optional extra is absent.
"""

import pytest

from agentmesh.data.base import DataError

CSV = "item,units,price\napples,120,0.95\nmilk,80,2.20\nbread,45,2.50\n"


@pytest.fixture()
def csv_file(tmp_path):
    path = tmp_path / "stock.csv"
    path.write_text(CSV)
    return tmp_path


# -- duckdb ---------------------------------------------------------------


@pytest.fixture()
def duck(csv_file):
    pytest.importorskip("duckdb")
    from agentmesh.data.duckdb_source import DuckDbSource

    return DuckDbSource("files", root=csv_file)


def test_column_types_are_inferred_not_assumed_text(duck) -> None:
    from agentmesh.tools.builtin import _normalize_type

    types = duck.describe("stock.csv").types

    # The point is the inference, not the vendor's spelling of the type.
    assert _normalize_type(types["units"]) == "integer"
    assert _normalize_type(types["price"]) == "float"
    assert _normalize_type(types["item"]) == "string"


def test_numeric_comparison_is_numeric(duck) -> None:
    """The bug the hand-rolled loader had: '80' > '100' lexicographically."""
    rows = duck.query("stock.csv", "SELECT item FROM {table} WHERE units > 100").as_dicts()

    assert rows == [{"item": "apples"}]


def test_aggregation_is_pushed_into_the_engine(duck) -> None:
    result = duck.query(
        "stock.csv", "SELECT SUM(units * price) AS revenue FROM {table}"
    ).as_dicts()

    assert result[0]["revenue"] == pytest.approx(120 * 0.95 + 80 * 2.20 + 45 * 2.50)


def test_writes_are_refused(duck) -> None:
    for statement in ("DELETE FROM {table}", "DROP TABLE x", "COPY x TO 'y'"):
        with pytest.raises(DataError, match="read-only"):
            duck.query("stock.csv", statement)


def test_summarize_reports_nulls_and_cardinality(duck) -> None:
    summary = {row["column_name"]: row for row in duck.summarize("stock.csv")}

    assert summary["item"]["approx_unique"] == 3
    assert float(summary["units"]["null_percentage"]) == 0.0


def test_a_missing_relation_is_reported(duck) -> None:
    with pytest.raises(DataError):
        duck.describe("ghost.csv")


# -- sqlalchemy -----------------------------------------------------------


@pytest.fixture()
def alchemy(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    from agentmesh.data.sqlalchemy_source import SqlAlchemySource

    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = sa.create_engine(url)
    with engine.connect() as conn:
        conn.execute(sa.text("CREATE TABLE sales (region TEXT, amount INTEGER)"))
        conn.execute(sa.text("INSERT INTO sales VALUES ('east',100),('west',250),('east',75)"))
        conn.commit()
    return SqlAlchemySource("warehouse", url)


def test_schema_comes_from_the_database(alchemy) -> None:
    schema = alchemy.describe("sales")

    assert schema.columns == ["region", "amount"]
    assert schema.row_count == 3


def test_grouped_query_runs_in_the_database(alchemy) -> None:
    rows = alchemy.query(
        "sales", "SELECT region, SUM(amount) AS total FROM {table} GROUP BY region"
    ).as_dicts()

    assert sorted(rows, key=lambda r: r["region"]) == [
        {"region": "east", "total": 175},
        {"region": "west", "total": 250},
    ]


def test_parameters_are_bound_not_interpolated(alchemy) -> None:
    rows = alchemy.query(
        "sales", "SELECT region FROM {table} WHERE amount > :floor", {"floor": 200}
    ).as_dicts()

    assert rows == [{"region": "west"}]


def test_writes_are_refused_on_a_database(alchemy) -> None:
    with pytest.raises(DataError, match="read-only"):
        alchemy.query("sales", "DELETE FROM {table}")


def test_unsafe_references_are_rejected(alchemy) -> None:
    with pytest.raises(DataError, match="unsafe table reference"):
        alchemy.query("sales; DROP TABLE sales", "SELECT 1")
