from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class DataError(RuntimeError):
    """A data source could not be reached, bound, or queried."""


class DataBudgetExceeded(DataError):
    """A query's estimated cost exceeds the ceiling the caller set."""


@dataclass(frozen=True)
class DataRequirement:
    """What an agent needs, stated portably.

    This travels with a published agent. It names no server, no dataset and no
    credential - which is what lets one agent spec run against many tenants'
    data without any of them seeing each other's.
    """

    name: str
    kind: str = "table"                       # table | document
    required_columns: tuple[str, ...] = ()
    description: str = ""
    optional: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataRequirement":
        return cls(
            name=raw["name"],
            kind=raw.get("kind", "table"),
            required_columns=tuple(raw.get("required_columns", []) or []),
            description=raw.get("description", ""),
            optional=bool(raw.get("optional", False)),
        )


@dataclass(frozen=True)
class Binding:
    """Tenant-owned mapping from a requirement to a concrete location."""

    requirement: str
    source: str
    ref: str


@dataclass(frozen=True)
class QueryEstimate:
    """What a query will cost, known *before* running it."""

    bytes_scanned: int = 0
    cost_usd: float = 0.0
    exact: bool = False
    detail: str = ""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    bytes_scanned: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    truncated: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def summary(self) -> dict[str, Any]:
        """Compact form safe to hand a model - shape, not bulk."""
        return {
            "columns": self.columns,
            "row_count": self.row_count,
            "rows": self.as_dicts()[:50],
            "truncated": self.truncated or self.row_count > 50,
        }


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: list[str] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)
    row_count: int | None = None

    def missing(self, required: tuple[str, ...]) -> list[str]:
        present = {c.casefold() for c in self.columns}
        return [c for c in required if c.casefold() not in present]


@runtime_checkable
class DataSource(Protocol):
    """A place data lives. Query sources push work down; document sources parse."""

    name: str
    kind: str

    def describe(self, ref: str) -> TableSchema: ...

    def estimate(self, ref: str, query: str, params: dict[str, Any] | None = None) -> QueryEstimate: ...

    def query(
        self, ref: str, query: str, params: dict[str, Any] | None = None, max_rows: int = 10_000
    ) -> QueryResult: ...
