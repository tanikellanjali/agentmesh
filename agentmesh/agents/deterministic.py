"""An executor that runs tools directly, with no model call at all.

Most of what a mesh does is computation, not judgement. A profit-and-loss
calculator, a stock-level check, a data profile - these have exact answers, and
paying a model to produce them is slower, costlier and less correct than running
the code. Reserve model calls for the parts that genuinely need reasoning:
understanding the request at the front, and composing the answer at the end.

A deterministic agent declares its tools and how to fill their arguments::

    executor: deterministic
    tools: [profile_csv]
    deterministic:
      arguments:
        profile_csv:
          csv_text: $message
      summary: "Profiled the supplied dataset."
"""

from __future__ import annotations

from typing import Any

from agentmesh.agents.base_agent import BaseAgent
from agentmesh.agents.registry import register_executor
from agentmesh.core.contracts import missing_fields
from agentmesh.tools.registry import resolve_tools

MESSAGE_TOKEN = "$message"
CONTEXT_PREFIX = "$context."


def resolve_argument(value: Any, message: str, context: dict[str, Any]) -> Any:
    """Substitute `$message` and `$context.<agent>.<field>` references."""
    if isinstance(value, list):
        return [resolve_argument(item, message, context) for item in value]
    if isinstance(value, dict):
        return {k: resolve_argument(v, message, context) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    if value == MESSAGE_TOKEN:
        return message
    if value.startswith(CONTEXT_PREFIX):
        cursor: Any = context
        for part in value[len(CONTEXT_PREFIX):].split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            elif isinstance(cursor, list) and part.isdigit() and int(part) < len(cursor):
                cursor = cursor[int(part)]
            else:
                return None
        return cursor
    return value


@register_executor("deterministic")
class DeterministicAgent(BaseAgent):
    """Runs the agent's tools in order and returns their results. No LLM."""

    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        config = self.spec.model_extra.get("deterministic", {}) if self.spec.model_extra else {}
        config = config or getattr(self.spec, "deterministic", {}) or {}
        arguments = config.get("arguments", {}) or {}

        tools, unknown = resolve_tools(self.spec.tools)
        data_context = (
            self.run_context.data_context_for(self.spec) if self.run_context else None
        )

        results: dict[str, Any] = {}
        findings: list[str] = []
        failed: list[str] = []

        for tool in tools:
            resolved = resolve_argument(arguments.get(tool.name, {}), message, context)
            outcome = tool.call(resolved if isinstance(resolved, dict) else {}, data_context)
            self._record_tool(outcome)

            if outcome.ok:
                results[tool.name] = outcome.output
                findings.extend(self._describe(tool.name, outcome.output))
            else:
                failed.append(tool.name)
                findings.append(f"{tool.name} failed: {outcome.error}")

        output: dict[str, Any] = {
            "agent_id": self.spec.id,
            "executor": "deterministic",
            "model": None,
            "summary": config.get("summary")
            or f"{self.spec.name} ran {len(results)} tool(s) with no model call.",
            "findings": findings,
            "results": results,
            "confidence": 0.0 if failed else 1.0,
        }
        if failed:
            output["tools_failed"] = failed
        if unknown:
            output["tools_unavailable"] = unknown
        if tools:
            output["tools_used"] = [t.name for t in tools if t.name not in failed]

        gaps = missing_fields(output, self.spec.output_contract)
        if gaps:
            output["contract_violations"] = gaps

        context[self.spec.id] = output
        return output

    @staticmethod
    def _describe(tool_name: str, output: Any) -> list[str]:
        """Surface a tool's own headline findings so downstream agents see them."""
        if not isinstance(output, dict):
            return []
        if isinstance(output.get("issues"), list):
            return [str(issue) for issue in output["issues"]]
        if isinstance(output.get("findings"), list):
            return [
                f["message"] if isinstance(f, dict) and "message" in f else str(f)
                for f in output["findings"]
            ]
        return []

    def _record_tool(self, outcome: Any) -> None:
        recorder = getattr(self.broker, "recorder", None)
        if recorder is None:
            return
        from agentmesh.core.telemetry import ToolInvocation

        recorder.record_tool(
            ToolInvocation(
                agent_id=self.spec.id,
                tool=outcome.tool,
                ok=outcome.ok,
                duration_ms=outcome.duration_ms,
                error=outcome.error,
            )
        )
