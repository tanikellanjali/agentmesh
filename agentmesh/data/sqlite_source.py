from __future__ import annotations

import csv
import io
import re
import sqlite3
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

_SAFE_REF = re.compile(r"^[A-Za-z_][\w]*$")
_WRITE = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA)\b", re.I)


class SqliteSource:
    """SQL query source backed by SQLite.

    Also the engine behind `CsvSource`, so tabular files and databases present
    one SQL surface. Queries are read-only: writes are rejected before they
    reach the database, because an agent with a live connection should never be
    able to mutate a tenant's data.
    """

    kind = "sql"

    def __init__(
        self,
        name: str,
        path: str | Path = ":memory:",
        cost_per_tb_scanned: float = 0.0,
        read_only: bool = True,
    ) -> None:
        self.name = name
        self.path = str(path)
        self.cost_per_tb_scanned = cost_per_tb_scanned
        self.read_only = read_only
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = None

    # -- helpers ----------------------------------------------------------
    def _guard(self, query: str) -> None:
        if self.read_only and _WRITE.search(query):
            raise DataError(
                f"source '{self.name}' is read-only; refusing a mutating statement"
            )

    @staticmethod
    def _check_ref(ref: str) -> str:
        if not _SAFE_REF.match(ref):
            raise DataError(f"unsafe table reference: {ref!r}")
        return ref

    def _price(self, byte_count: int) -> float:
        return byte_count / 1_099_511_627_776 * self.cost_per_tb_scanned

    # -- api --------------------------------------------------------------
    def describe(self, ref: str) -> TableSchema:
        self._check_ref(ref)
        try:
            info = self._conn.execute(f"PRAGMA table_info({ref})").fetchall()
        except sqlite3.Error as exc:
            raise DataError(f"cannot describe {ref}: {exc}") from exc
        if not info:
            raise DataError(f"table not found: {ref}")
        count = self._conn.execute(f"SELECT COUNT(*) FROM {ref}").fetchone()[0]
        return TableSchema(
            name=ref,
            columns=[row[1] for row in info],
            types={row[1]: (row[2] or "TEXT") for row in info},
            row_count=count,
        )

    def estimate(self, ref: str, query: str, params: dict[str, Any] | None = None) -> QueryEstimate:
        """Estimate before executing.

        SQLite has no dry-run, so this approximates from table size. A warehouse
        source (BigQuery) overrides this with a real dry-run and returns
        `exact=True`.
        """
        self._guard(query)
        try:
            schema = self.describe(ref)
        except DataError:
            return QueryEstimate(detail="unknown table; no estimate available")

        rows = schema.row_count or 0
        approx_bytes = rows * max(len(schema.columns), 1) * 32
        return QueryEstimate(
            bytes_scanned=approx_bytes,
            cost_usd=self._price(approx_bytes),
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
        rendered = query.replace("{table}", self._check_ref(ref))
        started = time.perf_counter()
        try:
            cursor = self._conn.execute(rendered, params or {})
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.Error as exc:
            raise DataError(f"query failed on '{self.name}': {exc}") from exc

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

    # -- loading ----------------------------------------------------------
    def load_rows(self, table: str, columns: list[str], rows: list[tuple[Any, ...]]) -> None:
        self._check_ref(table)
        cols = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        with self._conn:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            self._conn.execute(f"CREATE TABLE {table} ({cols})")
            self._conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)

    def iter_batches(
        self,
        ref: str,
        query: str,
        params: dict[str, Any] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_rows: int | None = None,
    ):
        """Stream results in chunks so a large table never lands in memory."""
        self._guard(query)
        rendered = query.replace("{table}", self._check_ref(ref))
        try:
            cursor = self._conn.execute(rendered, params or {})
        except sqlite3.Error as exc:
            raise DataError(f"query failed on '{self.name}': {exc}") from exc

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


class CsvSource(SqliteSource):
    """CSV and TSV files, queryable with SQL. Zero-dependency fallback.

    .. deprecated::
        Prefer :class:`agentmesh.data.duckdb_source.DuckDbSource`. This loader
        reads every column as TEXT, so ``units > 100`` is a *string* comparison
        and returns wrong rows. It is kept only so a bare install still has a
        file source; install the ``duckdb`` extra for correct types, Parquet,
        JSON and globs.
    """

    kind = "file"

    def __init__(self, name: str, root: str | Path = ".", delimiter: str = ",") -> None:
        super().__init__(name=name, path=":memory:", read_only=False)
        self.root = Path(root)
        self.delimiter = delimiter
        self._loaded: set[str] = set()

    def _table_for(self, ref: str) -> str:
        return re.sub(r"\W", "_", Path(ref).stem).strip("_") or "data"

    def _ensure(self, ref: str) -> str:
        table = self._table_for(ref)
        if table in self._loaded:
            return table

        path = self.root / ref
        if not path.exists():
            raise DataError(f"file not found: {path}")
        text = path.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(text), delimiter=self.delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise DataError(f"{ref} is empty") from exc

        rows = [tuple(r) + ("",) * (len(header) - len(r)) for r in reader]
        self.load_rows(table, header, [r[: len(header)] for r in rows])
        self._loaded.add(table)
        return table

    def describe(self, ref: str) -> TableSchema:
        return super().describe(self._ensure(ref))

    def estimate(self, ref: str, query: str, params: dict[str, Any] | None = None) -> QueryEstimate:
        return super().estimate(self._ensure(ref), query, params)

    def query(
        self, ref: str, query: str, params: dict[str, Any] | None = None, max_rows: int = 10_000
    ) -> QueryResult:
        return super().query(self._ensure(ref), query, params, max_rows)

    def iter_batches(
        self,
        ref: str,
        query: str,
        params: dict[str, Any] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_rows: int | None = None,
    ):
        yield from super().iter_batches(
            self._ensure(ref), query, params, batch_size, max_rows
        )
