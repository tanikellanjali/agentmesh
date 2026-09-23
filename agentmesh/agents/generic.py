from __future__ import annotations

import json
from typing import Any

from agentmesh.agents.base_agent import BaseAgent
from agentmesh.agents.registry import register_executor
from agentmesh.core.contracts import missing_fields, parse_model_json
from agentmesh.providers.base import ProviderError


@register_executor("generic")
class GenericAgent(BaseAgent):
    """Spec-driven, model-backed executor.

    The agent's own spec is the prompt: its description sets the role, its
    capabilities scope the work, and its ``output_contract`` defines the JSON
    the model must return. Synthesized agents use this without any bespoke code.
    """

    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        if self.broker is None:
            raise RuntimeError(
                f"Agent '{self.spec.id}' has no model broker; the runtime must supply one."
            )

        call = self.broker.resolve(self.spec)

        try:
            completion = call.complete(
                system=self.system_prompt(),
                prompt=self.user_prompt(message, context),
            )
        except ProviderError as exc:
            output = self._error_output(str(exc), call.model.key)
            context[self.spec.id] = output
            return output

        parsed, parse_error = parse_model_json(completion.text)
        if parsed is None:
            output = self._unparsed_output(completion.text, parse_error, call.model.key)
            context[self.spec.id] = output
            return output

        output = dict(parsed)
        output["agent_id"] = self.spec.id
        output["model"] = call.model.key

        gaps = missing_fields(output, self.spec.output_contract)
        if gaps:
            output["contract_violations"] = gaps

        context[self.spec.id] = output
        return output

    def system_prompt(self) -> str:
        required = self.spec.output_contract.get("required_fields", []) or []
        lines = [
            f"You are {self.spec.name}, an agent in a multi-agent mesh.",
            f"Purpose: {self.spec.description.strip()}",
            f"Capabilities you are responsible for: {', '.join(self.spec.capabilities)}.",
            "",
            "Respond with a single JSON object and nothing else - no prose, no code fences.",
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
            "agent_id": self.spec.id,
            "model": model_key,
            "summary": text.strip()[:500],
            "findings": [],
            "confidence": 0.0,
            "contract_violations": [error or "unparsable response"],
            "raw_response": text,
        }
