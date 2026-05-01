from __future__ import annotations

from dataclasses import dataclass


CAPABILITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "request_intake": (
        "build",
        "help",
        "request",
        "triage",
        "user wants",
    ),
    "constraint_extraction": (
        "constraint",
        "constraints",
        "requirement",
        "requirements",
        "must",
        "should",
    ),
    "domain_research": (
        "research",
        "lookup",
        "investigate",
        "understand",
    ),
    "option_generation": (
        "option",
        "options",
        "choices",
        "alternatives",
    ),
    "plan_generation": (
        "plan",
        "workflow",
        "steps",
        "process",
    ),
    "decision_support": (
        "decide",
        "decision",
        "recommend",
        "recommendation",
        "triage",
    ),
    "data_ingestion": (
        "csv",
        "dataset",
        "file",
        "ingest",
        "load",
        "source",
        "table",
        "upload",
    ),
    "schema_profiling": (
        "column",
        "field",
        "profile",
        "schema",
        "table",
    ),
    "data_normalization": (
        "canonical",
        "cleanup",
        "clean up",
        "normalize",
        "standardize",
        "transform",
    ),
    "schema_mapping": (
        "map fields",
        "mapping",
        "rename",
        "schema mapping",
    ),
    "data_quality_check": (
        "data quality",
        "duplicate",
        "missing",
        "null",
        "outlier",
        "quality",
        "validate",
    ),
    "anomaly_detection": (
        "anomaly",
        "outlier",
        "unexpected",
    ),
    "pii_detection": (
        "email",
        "pii",
        "phone",
        "redact",
        "sensitive",
        "ssn",
    ),
    "compliance_review": (
        "compliance",
        "gdpr",
        "hipaa",
        "policy",
        "regulatory",
    ),
    "sql_analysis": (
        "query",
        "select ",
        "sql",
        "where ",
        "join ",
    ),
    "query_review": (
        "optimize",
        "performance",
        "query",
        "review",
        "sql",
    ),
    "report_generation": (
        "report",
        "summary",
        "summarize",
        "write up",
    ),
    "response_composition": (
        "compose",
        "final response",
        "recommendation",
        "recommendations",
    ),
}


@dataclass(frozen=True)
class NeedResolution:
    required_capabilities: list[str]
    reasoning: str


def resolve_needs(message: str) -> NeedResolution:
    normalized = message.casefold()
    required_capabilities = [
        capability
        for capability, keywords in CAPABILITY_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]

    if not required_capabilities:
        required_capabilities = ["report_generation"]
        reasoning = "No specific data operation keywords were detected; defaulted to report generation."
    else:
        reasoning = (
            "Detected request terms for "
            + ", ".join(required_capabilities)
            + "."
        )

    return NeedResolution(
        required_capabilities=required_capabilities,
        reasoning=reasoning,
    )
