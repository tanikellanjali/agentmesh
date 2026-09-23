"""SQLAlchemy-backed source: every SQL database, one adapter.

Postgres, MySQL, Oracle, SQL Server, Snowflake, BigQuery and SQLite all speak
SQLAlchemy through a driver. Writing a connector per vendor would be weeks of
work to reproduce what a mature library already does - including connection
pooling, type coercion and dialect differences.
"""

from __future__ import annotations

import re
import time
from typing import Any

from agentmesh.data.base import DataError, QueryEstimate, QueryResult, TableSchema

_WRITE = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT)\b", re.I)
_SAFE_REF = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*){0,2}$")


class SqlAlchemyUnavailable(RuntimeError):
    """The sqlalchemy extra is not installed."""


class SqlAlchemySource:
    """A SQL database reached by connection URL.

    Credentials belong in the URL's secret reference, never in an agent spec.
    """

    kind = "sql"

    def __init__(
        self,
        name: str,
        url: str,
        cost_per_tb_scanned: float = 0.0,
        read_only: bool = True,
        **engine_kwargs: Any,
    ) -> None:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:  # pragma: no cover - optional extra
            raise SqlAlchemyUnavailable(
                "sqlalchemy is not installed. Install with: pip install 'agentmesh[sql]'"
            ) from exc

        self.name = name
        self.read_only = read_only
        self.cost_per_tb_scanned = cost_per_tb_scanned
        self._text = text
        self._engine = create_engine(url, **engine_kwargs)

    def _guard(self, query: str) -> None:
        if self.read_only and _WRITE.search(query):
            raise DataError(f"source '{self.name}' is read-only; refusing a mutating statement")

    @staticmethod
    def _check_ref(ref: str) -> str:
        if not _SAFE_REF.match(ref):
            raise DataError(f"unsafe table reference: {ref!r}")
        return ref

    def _price(self, byte_count: int) -> float:
        return byte_count / 1_099_511_627_776 * self.cost_per_tb_scanned

    def _run(self, sql: str, params: dict[str, Any] | None = None):
        try:
            with self._engine.connect() as conn:
                result = conn.execute(self._text(sql), params or {})
                return list(result.keys()), result.fetchall()
        except Exception as exc:
            raise DataError(f"query failed on '{self.name}': {exc}") from exc

    def describe(self, ref: str) -> TableSchema:
        from sqlalchemy import inspect

        self._check_ref(ref)
        schema_name, _, table = ref.rpartition(".")
        inspector = inspect(self._engine)
        try:
            columns = inspector.get_columns(table, schema=schema_name or None)
        except Exception as exc:
            raise DataError(f"cannot describe {ref}: {exc}") from exc
        if not columns:
            raise DataError(f"table not found: {ref}")

        _, rows = self._run(f"SELECT COUNT(*) AS n FROM {ref}")
        return TableSchema(
            name=ref,
            columns=[c["name"] for c in columns],
            types={c["name"]: str(c["type"]) for c in columns},
            row_count=rows[0][0] if rows else None,
        )

    def estimate(self, ref: str, query: str, params: dict[str, Any] | None = None) -> QueryEstimate:
        self._guard(query)
        try:
            schema = self.describe(ref)
        except DataError:
            return QueryEstimate(detail="unknown table; no estimate available")
        approx = (schema.row_count or 0) * max(len(schema.columns), 1) * 32
        return QueryEstimate(
            bytes_scanned=approx,
            cost_usd=self._price(approx),
            exact=False,
            detail=f"approximated from {schema.row_count} rows",
        )

    def query(
        self,
        ref: str,
        query: str,
        params: dict[str, Any] | None = None,
        max_rows: int = 10_000,
    ) -> QueryResult:
        self._guard(query)
        rendered = query.replace("{table}", self._check_ref(ref))
        started = time.perf_counter()
        columns, rows = self._run(rendered, params)

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        scanned = sum(len(str(cell)) for row in rows for cell in row)
        return QueryResult(
            columns=columns,
            rows=[tuple(r) for r in rows],
            bytes_scanned=scanned,
            cost_usd=self._price(scanned),
            duration_ms=(time.perf_counter() - started) * 1000,
            truncated=truncated,
        )
