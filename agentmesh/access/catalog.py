from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentmesh.access.authorizer import Action, AllowAll, Authorizer, Resource, ResourceKind
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.schemas.agent_spec import AgentSpec


@dataclass(frozen=True)
class CatalogEntry:
    spec: AgentSpec
    source: str = "local"

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def version(self) -> int:
        return self.spec.version

    def as_resource(self) -> Resource:
        return Resource(
            kind=ResourceKind.AGENT,
            id=self.spec.id,
            owner=self.spec.owner,
            visibility=self.spec.visibility,
        )


@dataclass(frozen=True)
class CatalogResult:
    """What a principal may see, split by what they may do with it.

    `blocked` is the important half: agents the principal can see but not use.
    Dropping them would make "denied" indistinguishable from "does not exist",
    and the resolver would silently synthesize a replacement - turning every
    denial into a bypass.
    """

    usable: list[CatalogEntry] = field(default_factory=list)
    blocked: list[CatalogEntry] = field(default_factory=list)

    @property
    def blocked_ids(self) -> list[str]:
        return [e.id for e in self.blocked]


@runtime_checkable
class Catalog(Protocol):
    """Where agents come from. Local YAML here; a database in the host app."""

    def search(
        self,
        principal: Principal,
        capabilities: list[str] | None = None,
        text: str | None = None,
    ) -> CatalogResult: ...

    def get(self, agent_id: str, version: int | None = None) -> CatalogEntry | None: ...

    def versions(self, agent_id: str) -> list[CatalogEntry]: ...


class LocalCatalog:
    """Agents loaded from spec packs on disk, filtered by the authorizer."""

    def __init__(
        self,
        entries: list[AgentSpec] | None = None,
        authorizer: Authorizer | None = None,
    ) -> None:
        self.authorizer = authorizer or AllowAll()
        self._by_id: dict[str, list[CatalogEntry]] = {}
        for spec in entries or []:
            self.add(spec)

    # -- population -------------------------------------------------------
    def add(self, spec: AgentSpec, source: str = "local") -> CatalogEntry:
        entry = CatalogEntry(spec=spec, source=source)
        self._by_id.setdefault(spec.id, []).append(entry)
        self._by_id[spec.id].sort(key=lambda e: e.version)
        return entry

    @classmethod
    def from_spec_packs(
        cls, spec_packs_dir: Path, authorizer: Authorizer | None = None
    ) -> LocalCatalog:
        from agentmesh.core.project_loader import list_projects, load_project

        catalog = cls(authorizer=authorizer)
        if not Path(spec_packs_dir).exists():
            return catalog
        for project in list_projects(Path(spec_packs_dir)):
            loaded = load_project(project.id, Path(spec_packs_dir))
            for spec in loaded.agents:
                catalog.add(spec, source=project.id)
        return catalog

    # -- reads ------------------------------------------------------------
    def get(self, agent_id: str, version: int | None = None) -> CatalogEntry | None:
        entries = self._by_id.get(agent_id, [])
        if not entries:
            return None
        if version is None:
            return entries[-1]  # latest
        return next((e for e in entries if e.version == version), None)

    def versions(self, agent_id: str) -> list[CatalogEntry]:
        return list(self._by_id.get(agent_id, []))

    def latest(self) -> list[CatalogEntry]:
        return [entries[-1] for entries in self._by_id.values() if entries]

    def search(
        self,
        principal: Principal = ANONYMOUS,
        capabilities: list[str] | None = None,
        text: str | None = None,
    ) -> CatalogResult:
        wanted = set(capabilities or [])
        needle = (text or "").casefold()

        usable: list[CatalogEntry] = []
        blocked: list[CatalogEntry] = []

        for entry in sorted(self.latest(), key=lambda e: e.id):
            spec = entry.spec
            if not spec.enabled:
                continue
            if wanted and not wanted & set(spec.capabilities):
                continue
            if needle and needle not in f"{spec.id} {spec.name} {spec.description}".casefold():
                continue

            resource = entry.as_resource()
            # Not viewable means invisible: the principal is never told it exists.
            if not self.authorizer.can(principal, Action.VIEW, resource).allowed:
                continue
            if self.authorizer.can(principal, Action.USE, resource).allowed:
                usable.append(entry)
            else:
                blocked.append(entry)

        return CatalogResult(usable=usable, blocked=blocked)

    def next_version(self, agent_id: str) -> int:
        entries = self._by_id.get(agent_id, [])
        return (entries[-1].version + 1) if entries else 1
