"""Tools that reach a tenant's bound data.

Each takes a leading `ctx`, which the runtime supplies. The model never sees a
connection string, a credential, or a raw table - only what a query returns.
"""

from __future__ import annotations

from typing import Any

from agentmesh.tools.registry import register_tool


@register_tool(
    name="describe_data",
    description=(
        "List the data bound to this agent and show a table's columns, types and "
        "row count. Call this first - never assume a column exists."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "requirement": {
                "type": "string",
                "description": "Name of the data requirement to describe. Omit to list all.",
            }
        },
        "additionalProperties": False,
    },
    tags=("data",),
)
def describe_data(ctx, requirement: str | None = None) -> dict[str, Any]:
    if requirement is None:
        return {"bound": ctx.available(), "unbound": ctx.unbound()}

    table = ctx.table(requirement)
    schema = table.schema()
    return {
        "requirement": requirement,
        "source": table.source.name,
        "columns": schema.columns,
        "types": schema.types,
        "row_count": schema.row_count,
        "missing_required_columns": table.validate(),
    }


@register_tool(
    name="query_data",
    description=(
        "Run a read-only SQL query against bound data and return the rows. Use "
        "{table} as the table name. Aggregate in SQL rather than pulling rows - "
        "it is faster, cheaper and the arithmetic is exact."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "requirement": {"type": "string", "description": "Which bound data to query."},
            "sql": {
                "type": "string",
                "description": "SQL SELECT using {table} as the table name.",
            },
            "max_rows": {"type": "integer", "description": "Row cap, default 500."},
        },
        "required": ["requirement", "sql"],
        "additionalProperties": False,
    },
    tags=("data", "sql"),
)
def query_data(ctx, requirement: str, sql: str, max_rows: int = 500) -> dict[str, Any]:
    result = ctx.table(requirement).query(sql, max_rows=max_rows)
    payload = result.summary()
    payload["bytes_scanned"] = result.bytes_scanned
    payload["cost_usd"] = round(result.cost_usd, 6)
    return payload


@register_tool(
    name="estimate_query_cost",
    description=(
        "Price a query BEFORE running it. Returns bytes scanned and estimated "
        "cost. Use this when a query might be expensive."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "requirement": {"type": "string"},
            "sql": {"type": "string"},
        },
        "required": ["requirement", "sql"],
        "additionalProperties": False,
    },
    tags=("data", "cost"),
)
def estimate_query_cost(ctx, requirement: str, sql: str) -> dict[str, Any]:
    estimate = ctx.table(requirement).estimate(sql)
    return {
        "bytes_scanned": estimate.bytes_scanned,
        "estimated_cost_usd": round(estimate.cost_usd, 6),
        "exact": estimate.exact,
        "detail": estimate.detail,
    }
