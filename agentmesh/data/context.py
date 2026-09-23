from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from agentmesh.access.authorizer import Action, AllowAll, Authorizer, Resource, ResourceKind
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.data.base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_RESULT_BYTES,
    Binding,
    DataBudgetExceeded,
    DataError,
    DataRequirement,
    DataResultTooLarge,
    DataSource,
    QueryEstimate,
    QueryResult,
    RowBatch,
    TableSchema,
)


@dataclass
class SourceRegistry:
    """Tenant-owned sources and the bindings that attach them to requirements."""

    sources: dict[str, DataSource]
    bindings: dict[str, Binding]

    @classmethod
    def empty(cls) -> SourceRegistry:
        return cls(sources={}, bindings={})

    def add_source(self, source: DataSource) -> None:
        self.sources[source.name] = source

    def bind(self, requirement: str, source: str, ref: str) -> None:
        if source not in self.sources:
            raise DataError(f"unknown source '{source}'")
        self.bindings[requirement] = Binding(requirement=requirement, source=source, ref=ref)

    def resolve(self, requirement: str) -> tuple[DataSource, str]:
        binding = self.bindings.get(requirement)
        if binding is None:
            raise DataError(
                f"no data bound to requirement '{requirement}'. "
                f"Bind it with: bindings.{requirement} = <source>/<ref>"
            )
        return self.sources[binding.source], binding.ref


class BoundTable:
    """One requirement, resolved to a real source and gated on every query."""

    def __init__(self, context: DataContext, requirement: DataRequirement) -> None:
        self._ctx = context
        self.requirement = requirement
        self.source, self.ref = context.registry.resolve(requirement.name)

    def schema(self) -> TableSchema:
        return self.source.describe(self.ref)

    def validate(self) -> list[str]:
        """Columns the agent declared it needs that the bound data lacks."""
        return self.schema().missing(self.requirement.required_columns)

    def estimate(self, sql: str, **params: Any) -> QueryEstimate:
        return self.source.estimate(self.ref, sql, params or None)

    def query(
        self,
        sql: str,
        max_rows: int = 10_000,
        max_bytes: int = DEFAULT_MAX_RESULT_BYTES,
        **params: Any,
    ) -> QueryResult:
        self._ctx.authorize(self.source)
        estimate = self.estimate(sql, **params)
        self._ctx.check_budget(estimate)

        try:
            result = self.source.query(self.ref, sql, params or None, max_rows=max_rows)
        except DataError as exc:
            self._ctx.record(self.source.name, self.ref, None, str(exc))
            raise

        self._ctx.record(self.source.name, self.ref, result, None)
        if result.bytes_scanned > max_bytes:
            raise DataResultTooLarge(
                f"result is {result.bytes_scanned:,} bytes, over the "
                f"{max_bytes:,} byte ceiling. Aggregate in SQL, or stream it."
            )
        return result

    def stream(
        self,
        sql: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_rows: int | None = None,
        **params: Any,
    ) -> Iterator[RowBatch]:
        """Iterate a large result in chunks.

        Gated exactly like `query`, but nothing larger than one batch is ever
        held in memory - so a table bigger than RAM can still be processed, as
        long as the caller reduces as it goes rather than collecting batches.
        """
        self._ctx.authorize(self.source)
        self._ctx.check_budget(self.estimate(sql, **params))

        rows = 0
        scanned = 0
        started = time.perf_counter()
        try:
            for batch in self.source.iter_batches(
                self.ref, sql, params or None, batch_size=batch_size, max_rows=max_rows
            ):
                rows += len(batch)
                scanned += sum(len(str(cell)) for row in batch.rows for cell in row)
                yield batch
        except DataError as exc:
            self._ctx.record(self.source.name, self.ref, None, str(exc))
            raise

        self._ctx.record(
            self.source.name,
            self.ref,
            QueryResult(
                columns=[],
                rows=[],
                bytes_scanned=scanned,
                cost_usd=0.0,
                duration_ms=(time.perf_counter() - started) * 1000,
            ),
            None,
        )


class DataContext:
    """What a tool receives to reach a tenant's data.

    Tools get a queryable handle, never a payload. Rows are aggregated at the
    source and only what a tool chooses to return is ever seen by a model -
    which is what keeps raw customer data out of the provider's hands and the
    token bill down.
    """

    def __init__(
        self,
        registry: SourceRegistry,
        requirements: list[DataRequirement] | None = None,
        principal: Principal = ANONYMOUS,
        authorizer: Authorizer | None = None,
        recorder: Any = None,
        agent_id: str = "",
        max_data_cost: float | None = None,
    ) -> None:
        self.registry = registry
        self.requirements = {r.name: r for r in (requirements or [])}
        self.principal = principal
        self.authorizer = authorizer or AllowAll()
        self.recorder = recorder
        self.agent_id = agent_id
        self.max_data_cost = max_data_cost
        self.spent = 0.0

    # -- gates ------------------------------------------------------------
    def authorize(self, source: DataSource) -> None:
        self.authorizer.can(
            self.principal,
            Action.USE,
            Resource(kind=ResourceKind.DATA_SOURCE, id=source.name),
        ).raise_if_denied()

    def check_budget(self, estimate: QueryEstimate) -> None:
        if self.max_data_cost is None:
            return
        if self.spent + estimate.cost_usd > self.max_data_cost:
            raise DataBudgetExceeded(
                f"query would scan {estimate.bytes_scanned:,} bytes "
                f"(~${estimate.cost_usd:.4f}); that exceeds the "
                f"${self.max_data_cost:.4f} data ceiling (spent ${self.spent:.4f})."
            )

    def record(self, source: str, ref: str, result: QueryResult | None, error: str | None) -> None:
        if result:
            self.spent += result.cost_usd
        if self.recorder is None:
            return
        from agentmesh.core.telemetry import DataOp

        self.recorder.record_data(
            DataOp(
                agent_id=self.agent_id,
                source=source,
                ref=ref,
                ok=error is None,
                rows=result.row_count if result else 0,
                bytes_scanned=result.bytes_scanned if result else 0,
                cost=result.cost_usd if result else 0.0,
                duration_ms=result.duration_ms if result else 0.0,
                error=error,
            )
        )

    # -- api --------------------------------------------------------------
    def table(self, name: str) -> BoundTable:
        requirement = self.requirements.get(name) or DataRequirement(name=name)
        return BoundTable(self, requirement)

    def available(self) -> list[str]:
        return sorted(self.registry.bindings)

    def unbound(self) -> list[str]:
        return sorted(
            name for name, req in self.requirements.items()
            if name not in self.registry.bindings and not req.optional
        )
