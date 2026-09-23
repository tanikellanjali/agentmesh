"""MCP tools must be indistinguishable from built-in ones.

Runs a real MCP server as a subprocess. Skipped when the mcp extra is absent.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from agentmesh.agents.factory import build_agent
from agentmesh.agents.tool_loop import run_tool_loop
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.telemetry import RunRecorder
from agentmesh.providers.base import SingleShotMixin, Turn, Usage
from agentmesh.providers.retry import NO_RETRY
from agentmesh.tools.base import ToolCall
from agentmesh.tools.mcp_bridge import (
    McpServer,
    McpUnavailable,
    register_mcp_tools,
)
from agentmesh.tools.registry import _TOOLS, get_tool

SERVER = Path(__file__).parent / "fixtures" / "mcp_inventory_server.py"


@pytest.fixture(scope="module")
def inventory():
    server = McpServer("inv", command=sys.executable, args=[str(SERVER)])
    try:
        server.connect()
    except McpUnavailable as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"MCP server unavailable: {exc}")
    names = register_mcp_tools(server)
    yield server, names
    server.close()
    for name in names:
        _TOOLS.pop(name, None)


class ToolThenAnswer(SingleShotMixin):
    name = "scripted"

    def __init__(self, tool_name, arguments, credentials=None):
        self.tool_name, self.arguments = tool_name, arguments
        self.used = False

    def converse(self, *, system, messages, tools=None, model="m", max_tokens=4096, effort=None, timeout=None):
        if not self.used and tools:
            self.used = True
            return Turn(
                text="", provider=self.name, model=model,
                tool_calls=[ToolCall(id="c1", name=self.tool_name, arguments=self.arguments)],
                usage=Usage(1, 1), stop_reason="tool_use",
            )
        return Turn(text='{"summary": "done", "findings": [], "confidence": 0.9}',
                    provider=self.name, model=model, usage=Usage(1, 1))


def broker_for(provider, recorder=None):
    b = ModelBroker(provider_override=provider.name, retry_policy=NO_RETRY, recorder=recorder)
    b._clients[provider.name] = provider
    return b


# -- discovery ------------------------------------------------------------


def test_server_tools_are_discovered_and_namespaced(inventory) -> None:
    _, names = inventory

    assert names == ["inv.reorder_list", "inv.stock_level"]


def test_descriptions_and_schemas_survive_the_bridge(inventory) -> None:
    spec = get_tool("inv.reorder_list")

    assert "reorder threshold" in spec.description
    assert "threshold" in spec.input_schema["properties"]
    assert spec.tags == ("mcp", "inv")


# -- invocation -----------------------------------------------------------


def test_an_mcp_tool_returns_structured_output(inventory) -> None:
    result = get_tool("inv.stock_level").call({"item": "bread"})

    assert result.ok
    assert result.output == {"item": "bread", "units": 8, "supplier": "BakeryPlus"}


def test_arguments_are_passed_through(inventory) -> None:
    low = get_tool("inv.reorder_list").call({"threshold": 10}).output
    lower = get_tool("inv.reorder_list").call({"threshold": 5}).output

    assert {i["item"] for i in low["items"]} == {"bread", "lettuce"}
    assert {i["item"] for i in lower["items"]} == {"lettuce"}


def test_a_server_error_becomes_a_tool_error_not_a_crash(inventory) -> None:
    result = get_tool("inv.stock_level").call({"wrong_argument": 1})

    assert result.ok is False
    assert result.error


def test_connecting_to_a_missing_server_is_reported() -> None:
    with pytest.raises(McpUnavailable):
        McpServer("ghost", command=sys.executable, args=["/nonexistent.py"], timeout=10).connect()


# -- through an agent -----------------------------------------------------


def test_an_agent_can_call_an_mcp_tool(inventory) -> None:
    recorder = RunRecorder()
    provider = ToolThenAnswer("inv.reorder_list", {"threshold": 10})
    spec = synthesize_agent("inventory_management").model_copy(
        update={"tools": ["inv.reorder_list"]}
    )

    output = build_agent(spec, broker_for(provider, recorder)).run("what needs reordering?", {})

    assert output["tools_used"] == ["inv.reorder_list"]
    assert recorder.tools_for(spec.id)[0].ok is True


def test_mcp_results_are_fed_back_to_the_model(inventory) -> None:
    provider = ToolThenAnswer("inv.stock_level", {"item": "lettuce"})
    call = broker_for(provider).resolve(synthesize_agent("inventory_management"))

    result = run_tool_loop(
        call, system="s", prompt="p", tools=[get_tool("inv.stock_level")]
    )

    assert result.tool_results[0].output["units"] == 0
