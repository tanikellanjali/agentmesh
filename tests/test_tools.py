import pytest

from agentmesh.tools import ToolNotFound, get_tool, list_tools, resolve_tools


def call(name, **kwargs):
    return get_tool(name).call(kwargs)


# -- registry -------------------------------------------------------------


def test_builtin_tools_are_registered() -> None:
    for name in ("sql_analyze", "pii_scan", "redact_text", "profile_csv", "text_stats"):
        assert name in list_tools()


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ToolNotFound):
        get_tool("no_such_tool")


def test_resolve_tools_reports_missing_without_raising() -> None:
    tools, missing = resolve_tools(["sql_analyze", "imaginary_tool"])

    assert [t.name for t in tools] == ["sql_analyze"]
    assert missing == ["imaginary_tool"]


def test_tools_describe_themselves_for_a_model() -> None:
    schema = get_tool("sql_analyze").as_schema()

    assert schema["name"] == "sql_analyze"
    assert "sql" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["sql"]
    assert len(schema["description"]) > 40


def test_a_raising_tool_returns_an_error_result_not_an_exception() -> None:
    result = call("sql_analyze", sql="")

    assert result.ok is False
    assert "non-empty" in result.error


def test_bad_arguments_are_reported_clearly() -> None:
    result = call("sql_analyze", wrong_arg="x")

    assert result.ok is False
    assert "invalid arguments" in result.error


# -- sql_analyze ----------------------------------------------------------


def test_write_without_where_is_critical() -> None:
    out = call("sql_analyze", sql="DELETE FROM orders").output

    assert out["risk_level"] == "critical"
    assert out["findings"][0]["rule"] == "delete_without_where"


def test_a_scoped_write_is_not_flagged_as_critical() -> None:
    out = call("sql_analyze", sql="DELETE FROM orders WHERE id = 1").output

    assert "delete_without_where" not in [f["rule"] for f in out["findings"]]


def test_performance_smells_are_detected() -> None:
    sql = "SELECT * FROM users u, orders o WHERE UPPER(u.email) = 'A' AND name LIKE '%x'"
    rules = {f["rule"] for f in call("sql_analyze", sql=sql).output["findings"]}

    assert {"select_star", "implicit_join", "function_on_column", "leading_wildcard"} <= rules


def test_tables_and_joins_are_extracted() -> None:
    out = call(
        "sql_analyze", sql="SELECT a.x FROM alpha a JOIN beta b ON a.id=b.id LIMIT 10"
    ).output

    assert out["tables"] == ["alpha", "beta"]
    assert out["join_count"] == 1
    assert out["statement_type"] == "SELECT"


def test_comments_do_not_trigger_findings() -> None:
    out = call("sql_analyze", sql="SELECT id FROM t LIMIT 1 -- SELECT * FROM everything").output

    assert "select_star" not in [f["rule"] for f in out["findings"]]


def test_surrounding_prose_does_not_confuse_the_parser() -> None:
    out = call("sql_analyze", sql="Please review: UPDATE t SET a=1 WHERE id=2").output

    assert out["statement_type"] == "UPDATE"


# -- pii_scan -------------------------------------------------------------


def test_pii_kinds_are_detected() -> None:
    text = "Reach bob@corp.com or 555-123-4567, SSN 123-45-6789, host 10.0.0.8"
    out = call("pii_scan", text=text).output

    assert out["severity"] == "critical"  # SSN present
    assert set(out["counts"]) >= {"email", "phone", "ssn", "ipv4"}


def test_credit_cards_are_luhn_validated() -> None:
    out = call("pii_scan", text="good 4111111111111111 bad 4111111111111112").output

    assert out["counts"].get("credit_card") == 1


def test_clean_text_reports_nothing() -> None:
    out = call("pii_scan", text="The quarterly revenue rose by twelve percent.").output

    assert out["found"] is False
    assert out["severity"] == "none"


def test_matches_are_masked_not_echoed() -> None:
    out = call("pii_scan", text="bob@corp.com").output

    assert out["matches"][0]["masked"] != "bob@corp.com"
    assert "*" in out["matches"][0]["masked"]


def test_unknown_kind_is_rejected() -> None:
    result = call("pii_scan", text="x", kinds=["nope"])

    assert result.ok is False
    assert "unknown kinds" in result.error


def test_redaction_removes_every_match() -> None:
    out = call("redact_text", text="mail bob@corp.com now").output

    assert "bob@corp.com" not in out["redacted"]
    assert out["replacements"] == 1


# -- profile_csv ----------------------------------------------------------


def test_csv_profiling_infers_types_and_finds_gaps() -> None:
    csv_text = "id,score,name\n1,10,ann\n2,,bob\n3,30,\n"
    out = call("profile_csv", csv_text=csv_text).output

    by_name = {c["name"]: c for c in out["columns"]}
    assert out["row_count"] == 3
    assert by_name["id"]["type"] == "integer"
    assert by_name["score"]["null_count"] == 1
    assert by_name["name"]["null_count"] == 1
    assert any("missing value" in issue for issue in out["issues"])


def test_numeric_columns_get_statistics() -> None:
    out = call("profile_csv", csv_text="v\n1\n2\n3\n").output
    col = out["columns"][0]

    assert (col["min"], col["max"], col["mean"]) == (1.0, 3.0, 2.0)


def test_duplicate_rows_are_counted() -> None:
    out = call("profile_csv", csv_text="a\nx\nx\ny\n").output

    assert out["duplicate_rows"] == 1


def test_constant_columns_are_flagged() -> None:
    out = call("profile_csv", csv_text="a,b\n1,z\n2,z\n").output

    assert any("constant column" in issue for issue in out["issues"])


def test_empty_csv_is_rejected() -> None:
    assert call("profile_csv", csv_text="   ").ok is False


# -- text_stats -----------------------------------------------------------


def test_text_stats_counts() -> None:
    out = call("text_stats", text="one two\nthree").output

    assert out["words"] == 3
    assert out["lines"] == 2
