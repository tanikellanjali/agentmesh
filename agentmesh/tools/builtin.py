"""Working tools an agent can actually call.

Everything here does real analysis on real input - no placeholders. These are
deliberately dependency-free so a fresh clone can run them.
"""

from __future__ import annotations

import csv
import io
import re
import statistics
from pathlib import Path
from typing import Any

from agentmesh.tools.registry import register_tool

# ---------------------------------------------------------------- SQL

_SQL_RULES: list[tuple[str, str, str, str]] = [
    # (id, severity, regex, message)
    ("select_star", "medium", r"\bSELECT\s+\*",
     "SELECT * returns every column; name the columns you need."),
    ("leading_wildcard", "high", r"\bLIKE\s+'%",
     "LIKE with a leading wildcard cannot use an index and forces a scan."),
    ("function_on_column", "high", r"\bWHERE\b[^;]*?\b(?:UPPER|LOWER|TRIM|CAST|DATE)\s*\(\s*[\w.]+\s*\)\s*(?:=|<|>|LIKE)",
     "Wrapping a column in a function in WHERE prevents index use."),
    ("not_in_subquery", "medium", r"\bNOT\s+IN\s*\(\s*SELECT\b",
     "NOT IN with a subquery mishandles NULLs; prefer NOT EXISTS."),
    ("implicit_join", "high", r"\bFROM\s+[\w.]+(?:\s+(?:AS\s+)?\w+)?\s*,\s*[\w.]+",
     "Comma join is an implicit cross join; use explicit JOIN ... ON."),
    ("or_in_where", "low", r"\bWHERE\b[^;]*?\bOR\b",
     "OR in WHERE often defeats index use; consider UNION of two indexed queries."),
    ("distinct", "low", r"\bSELECT\s+DISTINCT\b",
     "DISTINCT often hides a join that duplicates rows; check the join grain."),
]


@register_tool(
    name="sql_analyze",
    description=(
        "Statically analyse a SQL query for correctness, safety and performance "
        "risks. Returns concrete findings with severity, plus the tables and "
        "columns referenced. Use this before commenting on any SQL."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The SQL query to analyse."},
            "dialect": {"type": "string", "description": "Optional SQL dialect."},
        },
        "required": ["sql"],
        "additionalProperties": False,
    },
    tags=("sql", "analysis"),
)
def sql_analyze(sql: str, dialect: str | None = None) -> dict[str, Any]:
    if not sql or not sql.strip():
        raise ValueError("sql must be a non-empty string")

    stripped = re.sub(r"--[^\n]*", " ", sql)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.S)
    flat = " ".join(stripped.split())
    upper = flat.upper()

    findings: list[dict[str, str]] = []
    for rule_id, severity, pattern, message in _SQL_RULES:
        if re.search(pattern, upper):
            findings.append({"rule": rule_id, "severity": severity, "message": message})

    # A write with no WHERE is the highest-value thing this catches.
    for verb in ("UPDATE", "DELETE"):
        if re.search(rf"\b{verb}\b", upper) and not re.search(r"\bWHERE\b", upper):
            findings.append({
                "rule": f"{verb.lower()}_without_where", "severity": "critical",
                "message": f"{verb} without WHERE affects every row in the table.",
            })

    if upper.startswith("SELECT") and not re.search(r"\b(LIMIT|TOP|FETCH FIRST)\b", upper):
        findings.append({
            "rule": "no_row_limit", "severity": "low",
            "message": "Unbounded SELECT; add LIMIT when exploring.",
        })

    tables = sorted({t for t in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.]*)", flat, re.I)})
    joins = len(re.findall(r"\bJOIN\b", upper))
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: order.get(f["severity"], 9))

    # Find the first real SQL verb rather than the first word, so surrounding
    # prose ("Review this query: SELECT ...") doesn't confuse the classification.
    verb = re.search(
        r"\b(SELECT|INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|WITH)\b", upper
    )

    return {
        "dialect": dialect or "generic",
        "statement_type": verb.group(1) if verb else "UNKNOWN",
        "tables": tables,
        "join_count": joins,
        "finding_count": len(findings),
        "risk_level": findings[0]["severity"] if findings else "none",
        "findings": findings,
    }


# ---------------------------------------------------------------- PII

def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        n = ord(ch) - 48
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


_PII_PATTERNS: dict[str, str] = {
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b",
    "ssn": r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
    "phone": r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",
    "ipv4": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "credit_card": r"\b(?:\d[ -]*?){13,19}\b",
    "api_key": r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b",
}


@register_tool(
    name="pii_scan",
    description=(
        "Scan text for personally identifiable information and secrets: emails, "
        "phone numbers, SSNs, credit cards (Luhn-validated), IP addresses and API "
        "keys. Returns counts, masked samples and character offsets. Use this "
        "instead of eyeballing text for sensitive data."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to scan."},
            "kinds": {
                "type": "array", "items": {"type": "string"},
                "description": "Optional subset: email, ssn, phone, ipv4, credit_card, api_key.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    tags=("pii", "compliance", "security"),
)
def pii_scan(text: str, kinds: list[str] | None = None) -> dict[str, Any]:
    if text is None:
        raise ValueError("text is required")

    wanted = kinds or list(_PII_PATTERNS)
    unknown = [k for k in wanted if k not in _PII_PATTERNS]
    if unknown:
        raise ValueError(f"unknown kinds: {', '.join(unknown)}")

    matches: list[dict[str, Any]] = []
    for kind in wanted:
        for m in re.finditer(_PII_PATTERNS[kind], text):
            value = m.group(0)
            if kind == "credit_card":
                digits = re.sub(r"\D", "", value)
                if not (13 <= len(digits) <= 19) or not _luhn(digits):
                    continue
            masked = value[:2] + "*" * max(len(value) - 4, 0) + value[-2:] if len(value) > 4 else "****"
            matches.append({"kind": kind, "masked": masked, "start": m.start(), "end": m.end()})

    counts: dict[str, int] = {}
    for m in matches:
        counts[m["kind"]] = counts.get(m["kind"], 0) + 1

    severity = "none"
    if any(k in counts for k in ("ssn", "credit_card", "api_key")):
        severity = "critical"
    elif counts:
        severity = "medium"

    return {
        "found": bool(matches),
        "severity": severity,
        "counts": counts,
        "total": len(matches),
        "matches": sorted(matches, key=lambda m: m["start"])[:50],
    }


@register_tool(
    name="redact_text",
    description="Replace every PII match found by pii_scan with a placeholder, returning safe text.",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "placeholder": {"type": "string", "description": "Default '[REDACTED]'."},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    tags=("pii", "compliance"),
)
def redact_text(text: str, placeholder: str = "[REDACTED]") -> dict[str, Any]:
    report = pii_scan(text)
    out = text
    for match in sorted(report["matches"], key=lambda m: m["start"], reverse=True):
        out = out[: match["start"]] + placeholder + out[match["end"] :]
    return {"redacted": out, "replacements": report["total"], "counts": report["counts"]}


# ---------------------------------------------------------------- data

def _infer(values: list[str]) -> str:
    real = [v for v in values if v not in ("", None)]
    if not real:
        return "empty"
    def is_int(v):
        try:
            int(v); return True
        except ValueError:
            return False
    def is_float(v):
        try:
            float(v); return True
        except ValueError:
            return False
    if all(is_int(v) for v in real):
        return "integer"
    if all(is_float(v) for v in real):
        return "float"
    if all(v.lower() in ("true", "false", "0", "1", "yes", "no") for v in real):
        return "boolean"
    if all(re.match(r"^\d{4}-\d{2}-\d{2}", v) for v in real):
        return "date"
    return "string"


_TYPE_ALIASES = {
    "integer": ("BIGINT", "INTEGER", "HUGEINT", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER"),
    "float": ("DOUBLE", "FLOAT", "REAL", "DECIMAL"),
    "boolean": ("BOOLEAN",),
    "date": ("DATE", "TIMESTAMP", "TIME", "DATETIME"),
}


def _normalize_type(raw: str) -> str:
    """One type vocabulary regardless of which engine profiled the data.

    A tool's contract must not change because an optional extra is installed.
    """
    upper = str(raw).upper()
    for canonical, aliases in _TYPE_ALIASES.items():
        if any(upper.startswith(alias) for alias in aliases):
            return canonical
    return "string"


def _as_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _profile_with_duckdb(csv_text: str, delimiter: str) -> dict[str, Any] | None:
    """Use DuckDB when it is installed: correct type inference, real quartiles.

    Hand-rolled CSV typing gets this wrong - it reads every column as text, so
    `units > 100` becomes a string comparison. Returns None if duckdb is absent.
    """
    try:
        import duckdb
    except ImportError:
        return None

    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
        handle.write(csv_text)
        path = handle.name

    try:
        conn = duckdb.connect()
        relation = f"read_csv_auto('{path}', delim='{delimiter}')"
        cursor = conn.execute(f"SUMMARIZE SELECT * FROM {relation}")
        names = [d[0] for d in cursor.description]
        summary = [dict(zip(names, row)) for row in cursor.fetchall()]

        total = conn.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
        distinct = conn.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {relation})").fetchone()[0]
    except Exception:
        return None
    finally:
        Path(path).unlink(missing_ok=True)

    columns: list[dict[str, Any]] = []
    issues: list[str] = []
    for row in summary:
        null_pct = float(row.get("null_percentage") or 0)
        column = {
            "name": row["column_name"],
            "type": _normalize_type(row["column_type"]),
            "sql_type": str(row["column_type"]),
            "null_pct": round(null_pct, 1),
            "null_count": round(null_pct / 100 * total),
            "distinct": row.get("approx_unique"),
            "min": row.get("min"),
            "max": row.get("max"),
        }
        if column["type"] in ("integer", "float"):
            column["min"] = _as_float(row.get("min"))
            column["max"] = _as_float(row.get("max"))
            column["mean"] = _as_float(row.get("avg"))
            column["stddev"] = _as_float(row.get("std"))
            column["quartiles"] = [_as_float(row.get(q)) for q in ("q25", "q50", "q75")]
        if null_pct:
            issues.append(f"{column['name']}: {column['null_count']} missing value(s) ({column['null_pct']}%)")
        if column["distinct"] == 1:
            issues.append(f"{column['name']}: constant column")
        columns.append(column)

    if total - distinct:
        issues.append(f"{total - distinct} duplicate row(s)")

    return {
        "engine": "duckdb",
        "row_count": total,
        "column_count": len(columns),
        "duplicate_rows": total - distinct,
        "columns": columns,
        "issues": issues,
    }


@register_tool(
    name="profile_csv",
    description=(
        "Profile CSV data: per-column type, null counts, distinct values, "
        "min/max/mean and quartiles, plus duplicate and constant-column checks. "
        "Use this to ground any claim about a dataset's shape or quality."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "csv_text": {"type": "string", "description": "Raw CSV including a header row."},
            "delimiter": {"type": "string", "description": "Default ','."},
        },
        "required": ["csv_text"],
        "additionalProperties": False,
    },
    tags=("data", "quality"),
)
def profile_csv(csv_text: str, delimiter: str = ",") -> dict[str, Any]:
    if not csv_text or not csv_text.strip():
        raise ValueError("csv_text must be a non-empty string")

    profiled = _profile_with_duckdb(csv_text, delimiter)
    if profiled is not None:
        return profiled
    return _profile_with_stdlib(csv_text, delimiter)


def _profile_with_stdlib(csv_text: str, delimiter: str) -> dict[str, Any]:
    """Zero-dependency fallback for installs without the duckdb extra."""
    rows = list(csv.DictReader(io.StringIO(csv_text), delimiter=delimiter))
    if not rows:
        return {"engine": "stdlib", "row_count": 0, "columns": [], "issues": ["no data rows"]}

    columns: list[dict[str, Any]] = []
    issues: list[str] = []
    for name in rows[0].keys():
        raw = [(r.get(name) or "").strip() for r in rows]
        blanks = sum(1 for v in raw if v == "")
        kind = _infer(raw)
        col: dict[str, Any] = {
            "name": name,
            "type": kind,
            "null_count": blanks,
            "null_pct": round(blanks / len(raw) * 100, 1),
            "distinct": len({v for v in raw if v != ""}),
        }
        if kind in ("integer", "float"):
            nums = [float(v) for v in raw if v != ""]
            if nums:
                col |= {"min": min(nums), "max": max(nums),
                        "mean": round(statistics.fmean(nums), 4)}
        if blanks:
            issues.append(f"{name}: {blanks} missing value(s) ({col['null_pct']}%)")
        if col["distinct"] == 1 and blanks < len(raw):
            issues.append(f"{name}: constant column")
        columns.append(col)

    keyed = [tuple((r.get(c) or "") for c in rows[0]) for r in rows]
    dupes = len(keyed) - len(set(keyed))
    if dupes:
        issues.append(f"{dupes} duplicate row(s)")

    return {
        "engine": "stdlib",
        "row_count": len(rows),
        "column_count": len(columns),
        "duplicate_rows": dupes,
        "columns": columns,
        "issues": issues,
    }


@register_tool(
    name="text_stats",
    description="Count characters, words, lines and estimated tokens in text. Use for size checks.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    tags=("text",),
)
def text_stats(text: str) -> dict[str, Any]:
    words = text.split()
    return {
        "characters": len(text),
        "words": len(words),
        "lines": len(text.splitlines()),
        "estimated_tokens": max(1, len(text) // 4) if text else 0,
    }
