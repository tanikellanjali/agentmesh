"""Deciding which agents should handle a request.

This replaces substring keyword matching. Given a request, the catalog of
agents the caller may see, and the tools available, a resolver produces a plan:
which existing agents to **reuse**, which to **adapt** into a new version, and
what to **create** from scratch.

Two rules shape the design.

*Denial blocks.* An agent the caller can see but may not use is reported as
blocked and never routed around - otherwise every denial would silently become
a synthesis bypass. An agent they cannot even *see* is absent from the catalog,
so creating an equivalent is legitimate rather than an escape hatch.

*Adaptation versions.* Tweaking an existing agent produces a new version of that
same agent, preserving one identity and one history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentmesh.access.catalog import Catalog, CatalogEntry, CatalogResult
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.contracts import parse_model_json
from agentmesh.providers.base import Message, ProviderError
from agentmesh.schemas.agent_spec import AgentSpec
from agentmesh.tools.registry import catalog as tool_catalog

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelCall

MAX_CATALOG_AGENTS = 60


@dataclass(frozen=True)
class Adaptation:
    """An existing agent forked into a new version of itself."""

    base: CatalogEntry
    spec: AgentSpec
    reason: str = ""

    @property
    def id(self) -> str:
        return self.spec.id


@dataclass(frozen=True)
class AgentPlan:
    capabilities: list[str] = field(default_factory=list)
    reuse: list[CatalogEntry] = field(default_factory=list)
    adapt: list[Adaptation] = field(default_factory=list)
    create: list[AgentSpec] = field(default_factory=list)
    blocked: list[CatalogEntry] = field(default_factory=list)
    reasoning: str = ""
    source: str = "model"

    @property
    def agents(self) -> list[AgentSpec]:
        """Every spec that should actually run, in a stable order."""
        specs = [entry.spec for entry in self.reuse]
        specs += [adaptation.spec for adaptation in self.adapt]
        specs += list(self.create)
        return sorted(specs, key=lambda spec: spec.id)

    @property
    def blocked_ids(self) -> list[str]:
        return [entry.id for entry in self.blocked]

    def as_dict(self) -> dict[str, Any]:
        return {
            "capabilities": self.capabilities,
            "reuse": [e.id for e in self.reuse],
            "adapt": [{"id": a.id, "version": a.spec.version, "reason": a.reason}
                      for a in self.adapt],
            "create": [s.id for s in self.create],
            "blocked": self.blocked_ids,
            "reasoning": self.reasoning,
            "source": self.source,
        }


@runtime_checkable
class Resolver(Protocol):
    name: str

    def resolve(
        self,
        request: str,
        catalog: Catalog,
        principal: Principal = ANONYMOUS,
    ) -> AgentPlan: ...


def _describe_agents(result: CatalogResult) -> list[dict[str, Any]]:
    entries = result.usable[:MAX_CATALOG_AGENTS]
    return [
        {
            "id": e.spec.id,
            "version": e.spec.version,
            "name": e.spec.name,
            "description": e.spec.description.strip(),
            "capabilities": e.spec.capabilities,
            "tools": e.spec.tools,
        }
        for e in entries
    ]


def _describe_tools() -> list[dict[str, str]]:
    return [{"name": t.name, "description": t.description} for t in tool_catalog()]


SYSTEM_PROMPT = """You plan multi-agent meshes.

Given a user request, a catalog of existing agents, and the tools available,
decide how to satisfy the request. Prefer reusing an existing agent over
adapting one, and adapting one over creating something new - duplicated agents
are the problem this system exists to prevent.

Respond with a single JSON object and nothing else:

{
  "capabilities": ["short_snake_case_capability", ...],
  "reuse":   [{"id": "<existing agent id>", "why": "..."}],
  "adapt":   [{"id": "<existing agent id>", "why": "...",
               "add_capabilities": [...], "add_tools": [...]}],
  "create":  [{"id": "<new_snake_case_id>", "name": "...", "description": "...",
               "capabilities": [...], "tools": [...], "depends_on": [...],
               "deterministic": true|false}],
  "reasoning": "one or two sentences"
}

Rules:
- Name capabilities from the request itself. There is no fixed vocabulary.
- Only use tool names that appear in the tool catalog.
- Only reference agent ids that appear in the agent catalog.
- `depends_on` may reference any agent id in this plan.
- Set "deterministic": true when the work is pure computation over tools and
  needs no language model. Prefer it - it is faster, cheaper and exact.
- Include exactly one agent whose capabilities contain "response_composition";
  it writes the final answer."""


class ModelResolver:
    """Asks a model to plan the mesh. No keyword table, no fixed vocabulary."""

    name = "model"

    def __init__(self, call: ModelCall, fallback: Resolver | None = None) -> None:
        self.call = call
        self.fallback = fallback

    def resolve(
        self,
        request: str,
        catalog: Catalog,
        principal: Principal = ANONYMOUS,
    ) -> AgentPlan:
        found = catalog.search(principal)
        prompt = json.dumps(
            {
                "request": request,
                "agent_catalog": _describe_agents(found),
                "tool_catalog": _describe_tools(),
            },
            indent=2,
        )

        try:
            turn = self.call.converse(
                system=SYSTEM_PROMPT,
                messages=[Message(role="user", content=prompt)],
                tools=None,
            )
            plan, error = parse_model_json(turn.text)
        except ProviderError as exc:
            plan, error = None, str(exc)

        if plan is None:
            if self.fallback is None:
                raise ResolutionFailed(f"could not plan the mesh: {error}")
            fell_back = self.fallback.resolve(request, catalog, principal)
            return AgentPlan(
                capabilities=fell_back.capabilities,
                reuse=fell_back.reuse,
                adapt=fell_back.adapt,
                create=fell_back.create,
                blocked=fell_back.blocked,
                reasoning=f"Model planning unavailable ({error}); {fell_back.reasoning}",
                source=f"fallback:{self.fallback.name}",
            )

        return build_plan(plan, found, catalog, principal)


class ResolutionFailed(RuntimeError):
    """No plan could be produced."""


def build_plan(
    raw: dict[str, Any],
    found: CatalogResult,
    catalog: Catalog,
    principal: Principal,
) -> AgentPlan:
    """Turn a model's proposed plan into specs, enforcing what it may not do."""
    usable = {entry.id: entry for entry in found.usable}
    known_tools = {t.name for t in tool_catalog()}

    reuse: list[CatalogEntry] = []
    adapt: list[Adaptation] = []
    create: list[AgentSpec] = []
    notes: list[str] = []

    for item in raw.get("reuse", []) or []:
        entry = usable.get(_id_of(item))
        if entry:
            reuse.append(entry)
        else:
            notes.append(f"dropped unknown agent '{_id_of(item)}' from reuse")

    for item in raw.get("adapt", []) or []:
        agent_id = _id_of(item)
        entry = usable.get(agent_id)
        if entry is None:
            notes.append(f"dropped unknown agent '{agent_id}' from adapt")
            continue
        adapt.append(_adapt(entry, item, catalog, principal, known_tools))

    taken = {e.id for e in reuse} | {a.id for a in adapt}
    for item in raw.get("create", []) or []:
        spec = _create(item, known_tools, principal)
        if spec is None:
            continue
        if spec.id in taken:
            notes.append(f"skipped duplicate agent '{spec.id}'")
            continue
        taken.add(spec.id)
        create.append(spec)

    reasoning = str(raw.get("reasoning", "")).strip()
    if notes:
        reasoning = f"{reasoning} ({'; '.join(notes)})".strip()

    return AgentPlan(
        capabilities=[str(c) for c in (raw.get("capabilities") or [])],
        reuse=reuse,
        adapt=adapt,
        create=create,
        blocked=found.blocked,
        reasoning=reasoning,
    )


def _id_of(item: Any) -> str:
    return str(item.get("id") if isinstance(item, dict) else item)


def _adapt(
    entry: CatalogEntry,
    item: dict[str, Any],
    catalog: Catalog,
    principal: Principal,
    known_tools: set[str],
) -> Adaptation:
    """Fork an agent into a new version of itself, never a new identity."""
    base = entry.spec
    capabilities = list(
        dict.fromkeys(base.capabilities + [str(c) for c in (item.get("add_capabilities") or [])])
    )
    tools = list(
        dict.fromkeys(
            base.tools + [t for t in (item.get("add_tools") or []) if t in known_tools]
        )
    )
    version = (
        catalog.next_version(base.id) if hasattr(catalog, "next_version") else base.version + 1
    )
    spec = base.model_copy(
        update={
            "version": version,
            "capabilities": capabilities,
            "tools": tools,
            "derived_from": f"{base.id}@{base.version}",
            "owner": base.owner or (None if principal.is_anonymous else principal.id),
        }
    )
    return Adaptation(base=entry, spec=spec, reason=str(item.get("why", "")).strip())


def _create(item: Any, known_tools: set[str], principal: Principal) -> AgentSpec | None:
    if not isinstance(item, dict):
        return None
    capabilities = [str(c) for c in (item.get("capabilities") or [])]
    if not capabilities:
        return None

    spec = synthesize_agent(capabilities[0])
    tools = [t for t in (item.get("tools") or []) if t in known_tools]
    deterministic = bool(item.get("deterministic")) and bool(tools)

    return spec.model_copy(
        update={
            "id": str(item.get("id") or spec.id),
            "name": str(item.get("name") or spec.name),
            "description": str(item.get("description") or spec.description),
            "capabilities": capabilities,
            "tools": tools,
            "depends_on": [str(d) for d in (item.get("depends_on") or [])],
            "executor": "deterministic" if deterministic else "generic",
            "owner": None if principal.is_anonymous else principal.id,
        }
    )


class CatalogResolver:
    """Deterministic fallback: match the request against the catalog directly.

    Used when no model is reachable. It scores each agent by how many of its
    capability and description words appear in the request, which is weaker than
    a model but is honest about being a fallback rather than pretending to plan.
    """

    name = "catalog"

    # A length filter would drop exactly the terms that carry the meaning -
    # sql, pii, api, csv - so filter by stopword instead.
    STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
        "my", "of", "on", "or", "please", "that", "the", "their", "then",
        "there", "these", "this", "to", "was", "what", "when", "where",
        "which", "who", "will", "with", "you", "your",
    })

    def __init__(self, minimum_score: int = 1) -> None:
        self.minimum_score = minimum_score

    def resolve(
        self,
        request: str,
        catalog: Catalog,
        principal: Principal = ANONYMOUS,
    ) -> AgentPlan:
        found = catalog.search(principal)
        words = {
            word
            for word in (w.strip(".,!?;:'\"").casefold() for w in request.split())
            if word and word not in self.STOPWORDS
        }

        scored: list[tuple[int, CatalogEntry]] = []
        for entry in found.usable:
            terms = set()
            for capability in entry.spec.capabilities:
                terms |= set(capability.split("_"))
            terms |= {
                w.strip(".,'\"").casefold()
                for w in entry.spec.description.split()
                if w.strip(".,'\"").casefold() not in self.STOPWORDS
            }
            score = len(words & terms)
            if score >= self.minimum_score:
                scored.append((score, entry))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        reuse = [entry for _, entry in scored]
        capabilities = sorted({c for entry in reuse for c in entry.spec.capabilities})

        return AgentPlan(
            capabilities=capabilities,
            reuse=reuse,
            blocked=found.blocked,
            reasoning=(
                f"Matched {len(reuse)} agent(s) by overlap with the request. "
                "No model was used to plan."
            ),
            source=self.name,
        )
