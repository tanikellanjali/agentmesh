from __future__ import annotations

from agentmesh.schemas.agent_spec import AgentSpec

GENERIC_EXECUTOR = "generic"
SYNTHESIZED_CATEGORY = "synthesized"
DEFAULT_MODEL_KEY = "anthropic/claude-opus-5"
FALLBACK_MODEL_KEYS = ("openai/gpt-4.1-mini",)


def agent_id_for(capability: str) -> str:
    return f"{capability}_agent"


def _title(capability: str) -> str:
    return " ".join(part.capitalize() for part in capability.split("_"))


def synthesize_agent(capability: str, category: str = SYNTHESIZED_CATEGORY) -> AgentSpec:
    """Build an ephemeral agent spec for a capability no registered agent covers.

    Deterministic today. The signature is the seam for a model-backed
    synthesizer that drafts richer specs from the user's request.
    """
    return AgentSpec(
        id=agent_id_for(capability),
        name=f"{_title(capability)} Agent",
        category=category,
        description=(
            f"Ephemeral agent synthesized to satisfy the '{capability}' capability."
        ),
        capabilities=[capability],
        input_contract={"required": ["message"], "optional": ["context"]},
        output_contract={
            "format": "json",
            "required_fields": ["summary", "findings", "confidence"],
        },
        models={"default": DEFAULT_MODEL_KEY, "fallback": list(FALLBACK_MODEL_KEYS)},
        executor=GENERIC_EXECUTOR,
        troubleshooting={"enabled": True, "agent": "troubleshooting_agent"},
        validation={"strict_json": True, "confidence_threshold": 0.7},
    )
