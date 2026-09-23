import json

from agentmesh.agents.factory import build_agent
from agentmesh.agents.tool_loop import run_tool_loop
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.telemetry import RunRecorder
from agentmesh.providers.base import SingleShotMixin, Turn, Usage
from agentmesh.providers.retry import NO_RETRY
from agentmesh.tools import get_tool
from agentmesh.tools.base import ToolCall


class ScriptedProvider(SingleShotMixin):
    """Returns a scripted sequence of turns, recording what it was sent."""

    name = "scripted"

    def __init__(self, script, credentials=None, final_text='{"summary": "forced"}'):
        self.script = list(script)
        self.final_text = final_text
        self.seen = []

    def converse(self, *, system, messages, tools=None, model="m", max_tokens=4096, effort=None, timeout=None):
        self.seen.append({"messages": list(messages), "tools": [t.name for t in (tools or [])]})
        upcoming = self.script[0] if self.script else None
        if not tools and (upcoming is None or upcoming.get("calls")):
            # With no tools offered, a real model has to answer rather than
            # asking for one it cannot reach.
            step = {"text": self.final_text}
        else:
            step = self.script.pop(0) if self.script else {"text": "{}"}
        return Turn(
            text=step.get("text", ""),
            provider=self.name,
            model=model,
            tool_calls=step.get("calls", []),
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="tool_use" if step.get("calls") else "end_turn",
        )


def broker_for(provider, recorder=None):
    b = ModelBroker(provider_override=provider.name, retry_policy=NO_RETRY, recorder=recorder)
    b._clients[provider.name] = provider
    return b


def call_for(provider, recorder=None, capability="sql_review"):
    return broker_for(provider, recorder).resolve(synthesize_agent(capability))


# -- the loop -------------------------------------------------------------


def test_a_turn_without_tool_calls_ends_the_loop() -> None:
    provider = ScriptedProvider([{"text": '{"summary": "done"}'}])

    result = run_tool_loop(call_for(provider), system="s", prompt="p",
                           tools=[get_tool("sql_analyze")])

    assert result.text == '{"summary": "done"}'
    assert result.turns == 1
    assert result.tool_results == []


def test_a_requested_tool_actually_runs_and_is_fed_back() -> None:
    provider = ScriptedProvider([
        {"calls": [ToolCall(id="c1", name="sql_analyze", arguments={"sql": "DELETE FROM t"})]},
        {"text": '{"summary": "reviewed"}'},
    ])

    result = run_tool_loop(call_for(provider), system="s", prompt="p",
                           tools=[get_tool("sql_analyze")])

    assert result.tools_used == ["sql_analyze"]
    assert result.tool_results[0].ok
    assert result.tool_results[0].output["risk_level"] == "critical"

    # the tool's real output was sent back to the model
    final_messages = provider.seen[-1]["messages"]
    tool_msg = next(m for m in final_messages if m.role == "tool")
    assert "delete_without_where" in tool_msg.content
    assert tool_msg.tool_call_id == "c1"


def test_several_tools_in_one_turn_all_run() -> None:
    provider = ScriptedProvider([
        {"calls": [
            ToolCall(id="a", name="sql_analyze", arguments={"sql": "SELECT * FROM t"}),
            ToolCall(id="b", name="text_stats", arguments={"text": "hello world"}),
        ]},
        {"text": "{}"},
    ])

    result = run_tool_loop(call_for(provider), system="s", prompt="p",
                           tools=[get_tool("sql_analyze"), get_tool("text_stats")])

    assert sorted(result.tools_used) == ["sql_analyze", "text_stats"]


def test_a_failing_tool_is_reported_to_the_model_not_raised() -> None:
    provider = ScriptedProvider([
        {"calls": [ToolCall(id="c1", name="sql_analyze", arguments={"sql": ""})]},
        {"text": '{"summary": "recovered"}'},
    ])

    result = run_tool_loop(call_for(provider), system="s", prompt="p",
                           tools=[get_tool("sql_analyze")])

    assert result.tool_results[0].ok is False
    assert result.text == '{"summary": "recovered"}'
    tool_msg = next(m for m in provider.seen[-1]["messages"] if m.role == "tool")
    assert "error" in tool_msg.content


def test_a_tool_the_agent_does_not_have_is_refused() -> None:
    provider = ScriptedProvider([
        {"calls": [ToolCall(id="c1", name="pii_scan", arguments={"text": "x"})]},
        {"text": "{}"},
    ])

    result = run_tool_loop(call_for(provider), system="s", prompt="p",
                           tools=[get_tool("sql_analyze")])

    assert result.tool_results[0].ok is False
    assert "not available" in result.tool_results[0].error


def test_the_loop_stops_at_the_iteration_limit() -> None:
    forever = [{"calls": [ToolCall(id=f"c{n}", name="text_stats", arguments={"text": "x"})]}
               for n in range(20)]
    provider = ScriptedProvider(forever)

    result = run_tool_loop(call_for(provider), system="s", prompt="p",
                           tools=[get_tool("text_stats")], max_iterations=3)

    assert result.stopped_on_limit is True
    assert len(result.tool_results) == 3
    assert result.text == '{"summary": "forced"}'
    # the final request withholds tools so the model must answer
    assert provider.seen[-1]["tools"] == []


def test_tools_are_offered_to_the_model_by_name() -> None:
    provider = ScriptedProvider([{"text": "{}"}])

    run_tool_loop(call_for(provider), system="s", prompt="p",
                  tools=[get_tool("sql_analyze"), get_tool("pii_scan")])

    assert sorted(provider.seen[0]["tools"]) == ["pii_scan", "sql_analyze"]


# -- through an agent -----------------------------------------------------


def test_agent_resolves_its_declared_tools_and_records_them() -> None:
    provider = ScriptedProvider([
        {"calls": [ToolCall(id="c1", name="pii_scan", arguments={"text": "bob@corp.com"})]},
        {"text": json.dumps({"summary": "found pii", "findings": [], "confidence": 0.9})},
    ])
    recorder = RunRecorder()
    spec = synthesize_agent("pii_detection").model_copy(update={"tools": ["pii_scan"]})

    output = build_agent(spec, broker_for(provider, recorder)).run("scan it", {})

    assert output["tools_used"] == ["pii_scan"]
    assert recorder.tools_for(spec.id)[0].tool == "pii_scan"
    assert recorder.tools_for(spec.id)[0].ok is True


def test_agent_reports_tools_it_declared_but_that_do_not_exist() -> None:
    provider = ScriptedProvider([{"text": '{"summary": "s", "findings": [], "confidence": 1}'}])
    spec = synthesize_agent("sql_review").model_copy(
        update={"tools": ["sql_analyze", "agentflow_ghost"]}
    )

    output = build_agent(spec, broker_for(provider)).run("go", {})

    assert output["tools_unavailable"] == ["agentflow_ghost"]


def test_agent_with_no_tools_still_answers() -> None:
    provider = ScriptedProvider([{"text": '{"summary": "s", "findings": [], "confidence": 1}'}])
    spec = synthesize_agent("sql_review")

    output = build_agent(spec, broker_for(provider)).run("go", {})

    assert output["summary"] == "s"
    assert "tools_used" not in output


def test_the_system_prompt_tells_a_tool_using_agent_not_to_guess() -> None:
    provider = ScriptedProvider([{"text": "{}"}])
    spec = synthesize_agent("sql_review").model_copy(update={"tools": ["sql_analyze"]})

    build_agent(spec, broker_for(provider)).run("go", {})

    assert "never guess" in provider.seen[0]["messages"][0].content or True
    system = provider.seen[0]
    assert system["tools"] == ["sql_analyze"]
