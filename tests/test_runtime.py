from agentmesh.core.runtime import run_project


def test_runtime_runs_agents_internally_for_data_ops_request() -> None:
    result = run_project(
        "data_ops",
        "Review this SQL query for performance, missing values, and summarize the risks.",
        provider="mock",
    )

    assert result.project.spec.id == "data_ops"
    assert result.run.selected_agents == [
        "data_quality_agent",
        "sql_analyzer_agent",
        "report_agent",
    ]
    assert result.run.executions[-1].output["agent_id"] == "report_agent"


def test_runtime_result_serializes_for_api_clients() -> None:
    result = run_project(
        "data_ops",
        "Review this SQL query for performance, missing values, and summarize the risks.",
        provider="mock",
    )

    payload = result.as_dict()

    assert payload["project"]["id"] == "data_ops"
    assert payload["selected_agents"] == [
        "data_quality_agent",
        "sql_analyzer_agent",
        "report_agent",
    ]
    assert payload["agent_outputs"][0]["agent_id"] == "data_quality_agent"
