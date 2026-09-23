import pytest

from agentmesh.access.authorizer import (
    AccessDenied,
    Action,
    AllowAll,
    DenyAll,
    Resource,
    ResourceKind,
)
from agentmesh.access.catalog import LocalCatalog
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.core.agent_synthesizer import synthesize_agent


class ViewOnly:
    """Can see everything, may use nothing - the case that matters."""

    def can(self, principal, action, resource):
        from agentmesh.access.authorizer import Decision

        return Decision(
            allowed=action == Action.VIEW,
            principal=principal.id,
            action=action,
            resource=str(resource),
            reason="view-only policy",
        )


class HideFinance:
    """Cannot even see finance agents - they must appear not to exist."""

    def can(self, principal, action, resource):
        from agentmesh.access.authorizer import Decision

        hidden = "finance" in resource.id
        return Decision(
            allowed=not hidden,
            principal=principal.id,
            action=action,
            resource=str(resource),
            reason="finance is hidden" if hidden else "ok",
        )


def agent(capability, **overrides):
    return synthesize_agent(capability).model_copy(update=overrides)


def catalog(specs, authorizer=None):
    return LocalCatalog(entries=specs, authorizer=authorizer)


# -- principal ------------------------------------------------------------


def test_default_principal_is_anonymous() -> None:
    assert ANONYMOUS.is_anonymous
    assert Principal(id="anjali").is_anonymous is False


# -- decisions ------------------------------------------------------------


def test_allow_all_is_honest_about_being_unenforced() -> None:
    decision = AllowAll().can(ANONYMOUS, Action.USE, Resource(ResourceKind.AGENT, "x"))

    assert decision.allowed
    assert "AllowAll" in decision.reason


def test_a_denial_raises_with_its_reason_attached() -> None:
    decision = DenyAll().can(ANONYMOUS, Action.USE, Resource(ResourceKind.AGENT, "x"))

    with pytest.raises(AccessDenied) as exc:
        decision.raise_if_denied()

    assert exc.value.decision.allowed is False
    assert exc.value.decision.action == Action.USE


# -- the catalog split that prevents the bypass ---------------------------


def test_usable_agents_are_returned() -> None:
    result = catalog([agent("sql_analysis")]).search(ANONYMOUS)

    assert [e.id for e in result.usable] == ["sql_analysis_agent"]
    assert result.blocked == []


def test_viewable_but_unusable_agents_are_blocked_not_hidden() -> None:
    result = catalog([agent("sql_analysis")], ViewOnly()).search(ANONYMOUS)

    assert result.usable == []
    assert result.blocked_ids == ["sql_analysis_agent"]


def test_unviewable_agents_are_absent_entirely() -> None:
    result = catalog(
        [agent("finance_reporting"), agent("sql_analysis")], HideFinance()
    ).search(ANONYMOUS)

    assert [e.id for e in result.usable] == ["sql_analysis_agent"]
    # invisible, not blocked: the principal is never told it exists
    assert result.blocked == []


def test_search_filters_by_capability() -> None:
    result = catalog([agent("sql_analysis"), agent("pii_detection")]).search(
        ANONYMOUS, capabilities=["pii_detection"]
    )

    assert [e.id for e in result.usable] == ["pii_detection_agent"]


def test_disabled_agents_are_never_returned() -> None:
    result = catalog([agent("sql_analysis", enabled=False)]).search(ANONYMOUS)

    assert result.usable == [] and result.blocked == []


# -- versioning -----------------------------------------------------------


def test_latest_version_wins_and_history_is_kept() -> None:
    c = catalog([])
    c.add(agent("sql_analysis", version=1))
    c.add(agent("sql_analysis", version=2, derived_from="sql_analysis_agent@1"))

    assert c.get("sql_analysis_agent").version == 2
    assert c.get("sql_analysis_agent", version=1).version == 1
    assert [e.version for e in c.versions("sql_analysis_agent")] == [1, 2]


def test_next_version_increments() -> None:
    c = catalog([agent("sql_analysis", version=3)])

    assert c.next_version("sql_analysis_agent") == 4
    assert c.next_version("brand_new_agent") == 1


def test_an_adapted_agent_records_what_it_came_from() -> None:
    adapted = agent("sql_analysis", version=2, derived_from="sql_analysis_agent@1")

    assert adapted.id == "sql_analysis_agent"   # same identity, new version
    assert adapted.derived_from == "sql_analysis_agent@1"


def test_specs_carry_ownership_and_visibility() -> None:
    spec = agent("sql_analysis", owner="anjali", visibility="public")

    assert spec.owner == "anjali"
    assert spec.visibility == "public"
