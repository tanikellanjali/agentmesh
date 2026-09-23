from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Principal:
    """Who is making a request.

    The library never authenticates anyone - it only carries identity so the
    host application can decide and so every record can be attributed.
    """

    id: str = "anonymous"
    display_name: str = ""
    org: str | None = None
    teams: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anonymous(self) -> bool:
        return self.id == "anonymous"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org": self.org,
            "teams": list(self.teams),
            "roles": list(self.roles),
        }


ANONYMOUS = Principal()
