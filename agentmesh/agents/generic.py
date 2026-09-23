from __future__ import annotations

import json
from typing import Any

from agentmesh.agents.base_agent import BaseAgent
from agentmesh.agents.registry import register_executor
from agentmesh.agents.tool_loop import run_tool_loop
from agentmesh.core.contracts import missing_fields, parse_model_json
from agentmesh.providers.base import ProviderError
from agentmesh.tools.registry import resolve_tools


@register_executor("generic")
class GenericAgent(BaseAgent):
    """Spec-driven, model-backed, tool-using executor.

    The agent's own spec is the prompt: its description sets the role, its
    capabilities scope the work, its ``tools`` list is what it can actually do,
    and its ``output_contract`` defines the JSON it must return. Synthesized
    agents use this without any bespoke code.
    """

    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        if self.broker is None:
            raise RuntimeError(
                f"Agent '{self.spec.id}' has no model broker; the runtime must supply one."
            )

        call = self.broker.resolve(self.spec)
        tools, unknown_tools = resolve_tools(self.spec.tools)

        try:
            loop = run_tool_loop(
                call,
                system=self.system_prompt(tools),
                prompt=self.user_prompt(message, context),
                tools=tools,
                max_iterations=int(self.spec.validation.get("max_tool_iterations", 6)),
                on_tool=self._record_tool,
                data_context=self.run_context.data_context_for(self.spec) if self.run_context else None,
            )
        except ProviderError as exc:
            output = self._error_output(str(exc), call.model.key)
            context[self.spec.id] = output
            return output

        parsed, parse_error = parse_model_json(loop.text)
        if parsed is None:
            output = self._unparsed_output(loop.text, parse_error, call.model.key)
        else:
            output = dict(parsed)
            gaps = missing_fields(output, self.spec.output_contract)
            if gaps:
                output["contract_violations"] = gaps

        output["agent_id"] = self.spec.id
        output["model"] = call.model.key
        if loop.tool_results:
            output["tools_used"] = loop.tools_used
            failed = [r.tool for r in loop.tool_results if not r.ok]
            if failed:
                output["tools_failed"] = failed
        if unknown_tools:
            output["tools_unavailable"] = unknown_tools
        if self.run_context:
            unbound = self.run_context.data_context_for(self.spec).unbound()
            if unbound:
                output["data_unbound"] = unbound
        if loop.stopped_on_limit:
            output["truncated"] = "tool iteration limit reached"

        context[self.spec.id] = output
        return output

    # -- prompting --------------------------------------------------------
    def system_prompt(self, tools: list[Any] | None = None) -> str:
        required = self.spec.output_contract.get("required_fields", []) or []
        lines = [
            f"You are {self.spec.name}, an agent in a multi-agent mesh.",
            f"Purpose: {self.spec.description.strip()}",
            f"Capabilities you are responsible for: {', '.join(self.spec.capabilities)}.",
        ]
        if tools:
            lines += [
                "",
                "You have tools. Call them to get real data - never guess at a result "
                "a tool could give you, and never invent findings you have not verified.",
            ]
        lines += [
            "",
            "When you have finished, respond with a single JSON object and nothing "
            "else - no prose, no code fences.",
        ]
        if required:
            lines.append(f"The object must contain these keys: {', '.join(required)}.")
        lines.append(
            "Stay within your capabilities. If the request falls outside them, say so "
            "in your summary rather than guessing."
        )
        return "\n".join(lines)

    def user_prompt(self, message: str, context: dict[str, Any]) -> str:
        sections = [f"User request:\n{message}"]
        upstream = {
            agent_id: output for agent_id, output in context.items() if agent_id != self.spec.id
        }
        if upstream:
            sections.append(
                "Output from agents that ran before you:\n"
                + json.dumps(upstream, indent=2, default=str)
            )
        return "\n\n".join(sections)

    # -- bookkeeping ------------------------------------------------------
    def _record_tool(self, requested: Any, result: Any) -> None:
        recorder = getattr(self.broker, "recorder", None)
        if recorder is None:
            return
        from agentmesh.core.telemetry import ToolInvocation

        recorder.record_tool(
            ToolInvocation(
                agent_id=self.spec.id,
                tool=result.tool,
                ok=result.ok,
                duration_ms=result.duration_ms,
                error=result.error,
            )
        )

    def _error_output(self, error: str, model_key: str) -> dict[str, Any]:
        return {
            "agent_id": self.spec.id,
            "model": model_key,
            "summary": f"{self.spec.name} could not complete: {error}",
            "findings": [],
            "confidence": 0.0,
            "error": error,
        }

    def _unparsed_output(self, text: str, error: str | None, model_key: str) -> dict[str, Any]:
        return {
            "summary": text.strip()[:500],
            "findings": [],
            "confidence": 0.0,
            "contract_violations": [error or "unparsable response"],
            "raw_response": text,
        }
