from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentmesh.access.authorizer import AllowAll, Authorizer
from agentmesh.access.principal import ANONYMOUS, Principal
from agentmesh.data.base import DataRequirement
from agentmesh.data.context import DataContext, SourceRegistry

if TYPE_CHECKING:
    from agentmesh.core.telemetry import RunRecorder
    from agentmesh.schemas.agent_spec import AgentSpec


@dataclass
class RunContext:
    """Everything a run carries that is not about models.

    Identity, policy and data travel together because every one of them is a
    property of *who asked*, not of the mesh itself.
    """

    principal: Principal = ANONYMOUS
    authorizer: Authorizer = field(default_factory=AllowAll)
    sources: SourceRegistry = field(default_factory=SourceRegistry.empty)
    recorder: RunRecorder | None = None
    max_data_cost: float | None = None

    def requirements_for(self, spec: AgentSpec) -> list[DataRequirement]:
        return [DataRequirement.from_dict(raw) for raw in spec.data_requirements]

    def data_context_for(self, spec: AgentSpec) -> DataContext:
        return DataContext(
            registry=self.sources,
            requirements=self.requirements_for(spec),
            principal=self.principal,
            authorizer=self.authorizer,
            recorder=self.recorder,
            agent_id=spec.id,
            max_data_cost=self.max_data_cost,
        )
