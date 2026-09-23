"""Bridge MCP servers into AgentMesh tools.

MCP is the emerging standard for connecting models to tools and data, and it
already has servers for Postgres, BigQuery, Slack, GitHub and hundreds more,
maintained by the people who own those systems. Rather than reimplement that
long tail, AgentMesh speaks the protocol: any MCP tool becomes an ordinary
`ToolSpec`, indistinguishable to an agent from a built-in one.

The MCP client is async and AgentMesh tools are sync, so each server owns a
dedicated event loop on a background thread and the session is held open for
the life of the connection.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Future
from typing import Any

from agentmesh.tools.base import ToolSpec
from agentmesh.tools.registry import _TOOLS


class McpUnavailable(RuntimeError):
    """The mcp SDK is not installed, or a server could not be reached."""


class McpServer:
    """A connected MCP server whose tools can be registered with AgentMesh."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self.timeout = timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._ready = threading.Event()
        self._shutdown: asyncio.Event | None = None
        self._error: BaseException | None = None

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> McpServer:
        try:
            from mcp import ClientSession, StdioServerParameters  # noqa: F401
            from mcp.client.stdio import stdio_client  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional extra
            raise McpUnavailable(
                "The mcp SDK is not installed. Install with: pip install 'agentmesh[mcp]'"
            ) from exc

        self._thread = threading.Thread(target=self._serve, daemon=True, name=f"mcp-{self.name}")
        self._thread.start()
        if not self._ready.wait(self.timeout):
            raise McpUnavailable(f"MCP server '{self.name}' did not start within {self.timeout}s")
        if self._error:
            raise McpUnavailable(f"MCP server '{self.name}' failed to start: {self._error}")
        return self

    def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def main() -> None:
            self._shutdown = asyncio.Event()
            params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env, cwd=self.cwd
            )
            try:
                async with (
                    stdio_client(params) as (read, write),
                    ClientSession(read, write) as session,
                ):
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
            except BaseException as exc:  # surfaced to connect()
                self._error = exc
                self._ready.set()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    def close(self) -> None:
        if self._loop and self._shutdown and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._shutdown.set)
        if self._thread:
            self._thread.join(timeout=self.timeout)
        self._session = None

    def __enter__(self) -> McpServer:
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- calls ------------------------------------------------------------
    def _run(self, coro) -> Any:
        if self._session is None or self._loop is None:
            raise McpUnavailable(f"MCP server '{self.name}' is not connected")
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.timeout)

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._run(self._session.list_tools())
        return [
            {
                "name": tool.name,
                "description": tool.description or f"{tool.name} (from {self.name})",
                "input_schema": _field(
                    tool, "input_schema", "inputSchema",
                    default={"type": "object", "properties": {}},
                ),
            }
            for tool in result.tools
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = self._run(self._session.call_tool(name, arguments))
        if _field(result, "is_error", "isError", default=False):
            raise RuntimeError(_render(result))
        return _render(result)


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field across MCP SDK versions (v1 camelCase, v2 snake_case)."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _render(result: Any) -> Any:
    """Prefer structured content; fall back to concatenated text blocks."""
    structured = _field(result, "structured_content", "structuredContent")
    if structured:
        return structured

    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    joined = "\n".join(parts)
    try:
        return json.loads(joined)
    except (json.JSONDecodeError, ValueError):
        return joined


def register_mcp_tools(
    server: McpServer, namespace: str | None = None, overwrite: bool = True
) -> list[str]:
    """Register every tool the server exposes as an AgentMesh tool.

    Names are namespaced (`postgres.query`) so two servers can expose a tool of
    the same name without colliding.
    """
    prefix = namespace or server.name
    registered: list[str] = []

    for tool in server.list_tools():
        qualified = f"{prefix}.{tool['name']}"
        if qualified in _TOOLS and not overwrite:
            continue

        def make_fn(tool_name: str):
            def call(**arguments: Any) -> Any:
                return server.call_tool(tool_name, arguments)

            return call

        _TOOLS[qualified] = ToolSpec(
            name=qualified,
            description=tool["description"],
            input_schema=tool["input_schema"],
            fn=make_fn(tool["name"]),
            tags=("mcp", prefix),
        )
        registered.append(qualified)

    return sorted(registered)
