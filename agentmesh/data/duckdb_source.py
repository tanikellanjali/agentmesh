"""DuckDB-backed data source.

Replaces a hand-rolled CSV loader. DuckDB reads CSV, Parquet, JSON and globs
directly, infers column types correctly, and pushes aggregation down - all of
which we would otherwise have to write and get wrong. The hand-rolled loader
typed every column TEXT, which silently made `units > 100` a string comparison.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from agentmesh.data.base import (
    DEFAULT_BATCH_SIZE,
    DataError,
    QueryEstimate,
    QueryResult,
    RowBatch,
    TableSchema,
)

_WRITE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|INSTALL|LOAD)\b", re.I
)
_SAFE_TABLE = re.compile(r"^[A-Za-z_][\w.]*$")


class DuckDbUnavailable(RuntimeError):
    """The duckdb extra is not installed."""


class DuckDbSource:
    """Query files or a DuckDB database with real SQL and real types."""

    kind = "duckdb"

    def __init__(
        self,
        name: str,
        root: str | Path = ".",
        database: str = ":memory:",
        cost_per_tb_scanned: float = 0.0,
        read_only: bool = True,
    ) -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional extra
            raise DuckDbUnavailable(
                "duckdb is not installed. Install with: pip install 'agentmesh[duckdb]'"
            ) from exc

        self.name = name
        self.root = Path(root)
        self.read_only = read_only
        self.cost_per_tb_scanned = cost_per_tb_scanned
        self._duckdb = duckdb
        self._conn = duckdb.connect(database)

    # -- helpers ----------------------------------------------------------
    def _guard(self, query: str) -> None:
        if self.read_only and _WRITE.search(query):
            raise DataError(f"source '{self.name}' is read-only; refusing a mutating statement")

    def _relation(self, ref: str) -> str:
        """A file path becomes a scannable literal; anything else is a table."""
        path = self.root / ref
        if path.exists():
            return f"'{path.as_posix()}'"
        if "*" in ref or "?" in ref:  # glob across many files
            return f"'{(self.root / ref).as_posix()}'"
        if not _SAFE_TABLE.match(ref):
            raise DataError(f"unsafe table reference: {ref!r}")
        return ref

    def _price(self, byte_count: int) -> float:
        return byte_count / 1_099_511_627_776 * self.cost_per_tb_scanned

    def _execute(self, sql: str, params: dict[str, Any] | None = None):
        try:
            if params:
                return self._conn.execute(sql, params)
            return self._conn.execute(sql)
        except Exception as exc:
            raise DataError(f"query failed on '{self.name}': {exc}") from exc

    # -- api --------------------------------------------------------------
    def describe(self, ref: str) -> TableSchema:
        relation = self._relation(ref)
        cursor = self._execute(f"SELECT * FROM {relation} LIMIT 0")
        columns = [d[0] for d in cursor.description]
        types = {d[0]: str(d[1]) for d in cursor.description}
        count = self._execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
        return TableSchema(name=ref, columns=columns, types=types, row_count=count)

    def summarize(self, ref: str) -> list[dict[str, Any]]:
        """DuckDB's built-in profile: types, nulls, cardinality, quartiles."""
        cursor = self._execute(f"SUMMARIZE SELECT * FROM {self._relation(ref)}")
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    def estimate(self, ref: str, query: str, params: dict[str, Any] | None = None) -> QueryEstimate:
        self._guard(query)
        try:
            schema = self.describe(ref)
        except DataError:
            return QueryEstimate(detail="unknown relation; no estimate available")

        rows = schema.row_count or 0
        approx = rows * max(len(schema.columns), 1) * 32
        return QueryEstimate(
            bytes_scanned=approx,
            cost_usd=self._price(approx),
            exact=False,
            detail=f"approximated from {rows} rows x {len(schema.columns)} columns",
        )

    def query(
        self,
        ref: str,
        query: str,
        params: dict[str, Any] | None = None,
        max_rows: int = 10_000,
    ) -> QueryResult:
        self._guard(query)
        rendered = query.replace("{table}", self._relation(ref))
        started = time.perf_counter()
        cursor = self._execute(rendered, params)

        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        columns = [d[0] for d in cursor.description] if cursor.description else []
        scanned = sum(len(str(cell)) for row in rows for cell in row)

        return QueryResult(
            columns=columns,
            rows=[tuple(r) for r in rows],
            bytes_scanned=scanned,
            cost_usd=self._price(scanned),
            duration_ms=(time.perf_counter() - started) * 1000,
            truncated=truncated,
        )

    def iter_batches(
        self,
        ref: str,
        query: str,
        params: dict[str, Any] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_rows: int | None = None,
    ):
        """Stream results in chunks. DuckDB reads from disk, so a file larger
        than memory is fine as long as the caller does not hoard the batches."""
        self._guard(query)
        rendered = query.replace("{table}", self._relation(ref))
        cursor = self._execute(rendered, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        offset = 0
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                return
            if max_rows is not None and offset + len(rows) > max_rows:
                rows = rows[: max_rows - offset]
            if not rows:
                return
            yield RowBatch(columns=columns, rows=[tuple(r) for r in rows], offset=offset)
            offset += len(rows)
            if max_rows is not None and offset >= max_rows:
                return
