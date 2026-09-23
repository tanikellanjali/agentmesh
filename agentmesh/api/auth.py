"""API authentication.

Secure by default without making local development painful:

* If ``AGENTMESH_API_KEYS`` is set, every request must present a valid key and
  the key determines the caller's :class:`Principal`.
* If it is not set, only loopback callers are allowed. A remote request is
  refused rather than silently served, so an unconfigured server cannot be
  exposed by accident.

Keys are compared in constant time, and never logged or echoed.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from dataclasses import dataclass

from agentmesh.access.principal import Principal

logger = logging.getLogger("agentmesh.api")

KEYS_ENV = "AGENTMESH_API_KEYS"
LOOPBACK_ENV = "AGENTMESH_ALLOW_LOOPBACK"


@dataclass(frozen=True)
class ApiKey:
    secret: str
    principal: Principal


def _parse(entry: str) -> ApiKey | None:
    """``secret`` or ``secret:principal_id`` or ``secret:principal_id:org``."""
    parts = [p.strip() for p in entry.split(":")]
    if not parts or not parts[0]:
        return None
    secret = parts[0]
    principal_id = parts[1] if len(parts) > 1 and parts[1] else f"key:{secret[:6]}"
    org = parts[2] if len(parts) > 2 and parts[2] else None
    return ApiKey(secret=secret, principal=Principal(id=principal_id, org=org))


def load_keys(env: dict[str, str] | None = None) -> list[ApiKey]:
    raw = (env or os.environ).get(KEYS_ENV, "")
    return [key for key in (_parse(part) for part in raw.split(",") if part.strip()) if key]


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host == "testclient":  # starlette's in-process test transport
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost", "::1"}


class Authenticator:
    """Resolves a request's credentials to a principal, or rejects it."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.keys = load_keys(env)
        source = env or os.environ
        self.allow_loopback = source.get(LOOPBACK_ENV, "1") not in ("0", "false", "False")
        if not self.keys:
            logger.warning(
                "%s is not set: only loopback requests will be served. "
                "Set it to a comma-separated list of keys before exposing this API.",
                KEYS_ENV,
            )

    @property
    def configured(self) -> bool:
        return bool(self.keys)

    def match(self, presented: str | None) -> ApiKey | None:
        if not presented:
            return None
        for key in self.keys:
            if hmac.compare_digest(key.secret, presented):
                return key
        return None

    def authenticate(self, presented: str | None, client_host: str | None) -> Principal:
        """Return the caller's principal, or raise ``PermissionError``."""
        if self.keys:
            key = self.match(presented)
            if key is None:
                raise PermissionError("invalid or missing API key")
            return key.principal

        if self.allow_loopback and is_loopback(client_host):
            return Principal(id="local", display_name="loopback")

        raise PermissionError(
            f"{KEYS_ENV} is not configured, so only loopback requests are served. "
            "Set it before exposing this API."
        )
