from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from agentmesh.access.principal import Principal


class Action(StrEnum):
    """What a principal may do to a resource."""

    VIEW = "view"          # know it exists, read its definition
    USE = "use"            # invoke it in a run
    FORK = "fork"          # copy it into your own namespace
    VERSION = "version"    # create a new version of it
    EDIT = "edit"          # change it in place
    PUBLISH = "publish"    # change its visibility
    DELETE = "delete"


class ResourceKind(StrEnum):
    AGENT = "agent"
    TOOL = "tool"
    MODEL = "model"
    DATA_SOURCE = "data_source"
    PROJECT = "project"


@dataclass(frozen=True)
class Resource:
    kind: ResourceKind
    id: str
    owner: str | None = None
    visibility: str = "private"   # private | team | org | public

    def __str__(self) -> str:
        return f"{self.kind}:{self.id}"


class AccessDenied(PermissionError):
    """Raised when a principal may not perform an action.

    Carrying the decision means the caller can explain *why* rather than
    failing opaquely - which is what lets the resolver offer to create a
    replacement instead of silently routing around a denial.
    """

    def __init__(self, decision: "Decision") -> None:
        super().__init__(decision.reason)
        self.decision = decision


@dataclass(frozen=True)
class Decision:
    allowed: bool
    principal: str
    action: Action
    resource: str
    reason: str = ""
    constraints: dict[str, Any] | None = None

    def raise_if_denied(self) -> "Decision":
        if not self.allowed:
            raise AccessDenied(self)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "principal": self.principal,
            "action": str(self.action),
            "resource": self.resource,
            "reason": self.reason,
        }


@runtime_checkable
class Authorizer(Protocol):
    """Policy decision point. The library asks; the host application decides."""

    def can(self, principal: Principal, action: Action, resource: Resource) -> Decision: ...


class AllowAll:
    """Default for the open-source library used standalone.

    A library cannot enforce anything - the user can edit the YAML, the code and
    the policy. Real enforcement requires a host that owns the catalog and runs
    the mesh server-side. This default is honest about that.
    """

    def can(self, principal: Principal, action: Action, resource: Resource) -> Decision:
        return Decision(
            allowed=True,
            principal=principal.id,
            action=action,
            resource=str(resource),
            reason="no policy configured (AllowAll)",
        )


class DenyAll:
    """Useful for tests and for proving a call site is actually gated."""

    def can(self, principal: Principal, action: Action, resource: Resource) -> Decision:
        return Decision(
            allowed=False,
            principal=principal.id,
            action=action,
            resource=str(resource),
            reason="denied by policy (DenyAll)",
        )
