import json

import pytest

from agentmesh.access.authorizer import Action, Decision
from agentmesh.access.catalog import LocalCatalog
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.resolver import CatalogResolver, ModelResolver, ResolutionFailed
from agentmesh.core.telemetry import RunRecorder
from agentmesh.providers.base import ProviderError, SingleShotMixin, Turn, Usage
from agentmesh.providers.retry import NO_RETRY


class Planner(SingleShotMixin):
    name = "planner"

    def __init__(self, plan=None, error=None, credentials=None):
        self.plan, self.error = plan, error
        self.prompts = []

    def converse(self, *, system, messages, tools=None, model="m",
                 max_tokens=4096, effort=None, timeout=None):
        self.prompts.append({"system": system, "user": messages[0].content})
        if self.error:
            raise self.error
        text = self.plan if isinstance(self.plan, str) else json.dumps(self.plan or {})
        return Turn(text=text, provider=self.name, model=model, usage=Usage(10, 10))


def call_for(provider):
    broker = ModelBroker(provider_override=provider.name, retry_policy=NO_RETRY,
                         recorder=RunRecorder())
    broker._clients[provider.name] = provider
    return broker.resolve(synthesize_agent("planning"))


def agent(agent_id, capability, **overrides):
    base = {"id": agent_id, **overrides}
    return synthesize_agent(capability).model_copy(update=base)


def catalog(specs, authorizer=None):
    return LocalCatalog(entries=specs, authorizer=authorizer)


class DenyUse:
    def __init__(self, blocked_id):
        self.blocked_id = blocked_id

    def can(self, principal, action, resource):
        denied = resource.id == self.blocked_id and action == Action.USE
        return Decision(allowed=not denied, principal=principal.id, action=action,
                        resource=str(resource), reason="restricted" if denied else "ok")


# -- capability naming is free, not from a table --------------------------


def test_capabilities_come_from_the_request_not_a_fixed_vocabulary() -> None:
    provider = Planner({"capabilities": ["translation", "legal_risk_review"],
                        "create": [{"id": "translation_agent", "name": "T",
                                    "description": "d", "capabilities": ["translation"]}],
                        "reasoning": "needs translation"})

    plan = ModelResolver(call_for(provider)).resolve(
        "Translate this contract into French and check legal risk", catalog([])
    )

    assert plan.capabilities == ["translation", "legal_risk_review"]
    assert [s.id for s in plan.create] == ["translation_agent"]


def test_the_catalog_and_tools_are_shown_to_the_planner() -> None:
    provider = Planner({"capabilities": [], "reasoning": ""})
    existing = agent("sql_analyzer_agent", "sql_analysis")

    ModelResolver(call_for(provider)).resolve("review sql", catalog([existing]))

    sent = provider.prompts[0]["user"]
    assert "sql_analyzer_agent" in sent
    assert "sql_analyze" in sent          # the tool catalog
    assert "response_composition" in provider.prompts[0]["system"]


# -- reuse / adapt / create ----------------------------------------------


def test_an_existing_agent_is_reused() -> None:
    provider = Planner({"reuse": [{"id": "sql_analyzer_agent", "why": "covers it"}]})
    plan = ModelResolver(call_for(provider)).resolve(
        "review sql", catalog([agent("sql_analyzer_agent", "sql_analysis")])
    )

    assert [e.id for e in plan.reuse] == ["sql_analyzer_agent"]
    assert plan.create == []


def test_adapting_produces_a_new_version_of_the_same_agent() -> None:
    provider = Planner({"adapt": [{"id": "report_agent", "why": "also legal",
                                   "add_capabilities": ["legal_summary"],
                                   "add_tools": ["text_stats"]}]})
    base = agent("report_agent", "report_generation", version=2)

    plan = ModelResolver(call_for(provider)).resolve("summarise legal risk", catalog([base]))
    adapted = plan.adapt[0].spec

    assert adapted.id == "report_agent"          # same identity
    assert adapted.version == 3                  # new version
    assert adapted.derived_from == "report_agent@2"
    assert "legal_summary" in adapted.capabilities
    assert "text_stats" in adapted.tools


def test_a_created_agent_can_be_deterministic() -> None:
    provider = Planner({"create": [{"id": "pii_agent", "name": "P", "description": "d",
                                    "capabilities": ["pii_detection"],
                                    "tools": ["pii_scan"], "deterministic": True}]})

    plan = ModelResolver(call_for(provider)).resolve("find PII", catalog([]))

    assert plan.create[0].executor == "deterministic"


def test_deterministic_is_ignored_without_tools() -> None:
    provider = Planner({"create": [{"id": "x_agent", "name": "X", "description": "d",
                                    "capabilities": ["thinking"], "deterministic": True}]})

    plan = ModelResolver(call_for(provider)).resolve("think", catalog([]))

    assert plan.create[0].executor == "generic"


# -- the plan is validated, not trusted -----------------------------------


def test_unknown_agent_ids_are_dropped() -> None:
    provider = Planner({"reuse": [{"id": "ghost_agent"}],
                        "adapt": [{"id": "phantom_agent"}]})

    plan = ModelResolver(call_for(provider)).resolve("go", catalog([]))

    assert plan.reuse == [] and plan.adapt == []
    assert "ghost_agent" in plan.reasoning


def test_unknown_tools_are_stripped_from_created_agents() -> None:
    provider = Planner({"create": [{"id": "a_agent", "name": "A", "description": "d",
                                    "capabilities": ["x"],
                                    "tools": ["sql_analyze", "imaginary_tool"]}]})

    plan = ModelResolver(call_for(provider)).resolve("go", catalog([]))

    assert plan.create[0].tools == ["sql_analyze"]


def test_duplicate_agents_are_not_created_twice() -> None:
    provider = Planner({"reuse": [{"id": "sql_analyzer_agent"}],
                        "create": [{"id": "sql_analyzer_agent", "name": "dup",
                                    "description": "d", "capabilities": ["sql_analysis"]}]})

    plan = ModelResolver(call_for(provider)).resolve(
        "go", catalog([agent("sql_analyzer_agent", "sql_analysis")])
    )

    assert [s.id for s in plan.agents] == ["sql_analyzer_agent"]


# -- denial blocks, and is never routed around ----------------------------


def test_a_denied_agent_is_reported_blocked_not_replaced() -> None:
    provider = Planner({"reuse": [{"id": "finance_agent"}], "reasoning": "needs finance"})
    specs = [agent("finance_agent", "financial_analysis")]

    plan = ModelResolver(call_for(provider)).resolve(
        "budget my store", catalog(specs, DenyUse("finance_agent")), Principal(id="anjali")
    )

    assert plan.blocked_ids == ["finance_agent"]
    assert plan.reuse == []          # it was not usable
    assert plan.create == []         # and no replacement was invented
    assert plan.agents == []


def test_a_denied_agent_is_never_offered_to_the_planner() -> None:
    provider = Planner({"reasoning": ""})
    specs = [agent("finance_agent", "financial_analysis")]

    ModelResolver(call_for(provider)).resolve(
        "budget", catalog(specs, DenyUse("finance_agent")), Principal(id="anjali")
    )

    # visible in the catalog it may not use -> excluded from the planning prompt
    assert "finance_agent" not in provider.prompts[0]["user"]


# -- fallback -------------------------------------------------------------


def test_an_unusable_model_response_falls_back_and_says_so() -> None:
    provider = Planner("not json at all")

    plan = ModelResolver(call_for(provider), fallback=CatalogResolver()).resolve(
        "review the sql query", catalog([agent("sql_analyzer_agent", "sql_analysis")])
    )

    assert plan.source == "fallback:catalog"
    assert "Model planning unavailable" in plan.reasoning


def test_a_provider_failure_falls_back() -> None:
    provider = Planner(error=ProviderError("down"))

    plan = ModelResolver(call_for(provider), fallback=CatalogResolver()).resolve(
        "review the sql query", catalog([agent("sql_analyzer_agent", "sql_analysis")])
    )

    assert plan.source == "fallback:catalog"


def test_without_a_fallback_resolution_failure_is_explicit() -> None:
    provider = Planner("not json")

    with pytest.raises(ResolutionFailed):
        ModelResolver(call_for(provider)).resolve("go", catalog([]))


def test_catalog_resolver_matches_on_overlap() -> None:
    specs = [agent("sql_analyzer_agent", "sql_analysis"),
             agent("pii_agent", "pii_detection")]

    plan = CatalogResolver().resolve("please analyse the sql", catalog(specs), ANONYMOUS)

    assert [e.id for e in plan.reuse] == ["sql_analyzer_agent"]
    assert plan.source == "catalog"
