"""Who is calling the dashboard API, for which tenant, and what they may do.

The ledger's requirement for Part 5 was that the API over the graphs is *"itself an
outward-facing surface and gets the same treatment as a connector: authenticated,
tenant-scoped, and every approval it records must name a real approver."* Each of
those three is a property of this module:

*   **Authenticated.** Every ``/api`` route requires ``Authorization: Bearer
    <token>``. Tokens are looked up by their SHA-256 digest, so the registry never
    holds a usable secret after construction and a lookup does not compare secrets
    character by character. Tokens shorter than :data:`MIN_TOKEN_LENGTH` are
    refused at configuration time — a guessable token is an unauthenticated API.
*   **Tenant-scoped.** An :class:`Identity` names exactly one tenant, and the API
    resolves the workspace *from the identity*, never from the request. There is no
    tenant parameter to tamper with.
*   **A real approver.** The approver recorded on a decision is
    :attr:`Identity.principal`. The decision body cannot carry an approver — the
    request model forbids extra fields — so a client cannot approve as someone else,
    and a viewer, who may read everything, cannot decide anything.

Bearer tokens in a header rather than a cookie, deliberately: a cookie is sent by
the browser on cross-site requests, which makes every state-changing route a CSRF
target. A header is only ever sent by code the page itself runs.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sentinel.core.errors import SentinelError

__all__ = [
    "MIN_TOKEN_LENGTH",
    "AuthError",
    "Identity",
    "Role",
    "TokenRegistry",
]

MIN_TOKEN_LENGTH: Final[int] = 24

#: A principal is an email-style login or a directory name. No whitespace, so what
#: the audit log records as the approver is exactly one unambiguous string.
_PRINCIPAL: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._%+\-@]{0,127}$")
_TENANT: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}$")


class AuthError(SentinelError):
    """Missing, unknown or malformed credentials, or a bad registry configuration."""


class Role(StrEnum):
    #: Reads everything for its tenant, and answers the approval gate.
    ANALYST = "analyst"
    #: Reads everything for its tenant. Cannot decide, launch, replay or recover.
    VIEWER = "viewer"


@dataclass(frozen=True, slots=True)
class Identity:
    principal: str
    tenant_id: str
    role: Role

    def __post_init__(self) -> None:
        if not _PRINCIPAL.match(self.principal):
            raise AuthError(f"principal {self.principal!r} is not a valid login name")
        if not _TENANT.match(self.tenant_id):
            raise AuthError(f"tenant {self.tenant_id!r} is not a valid tenant id")

    @property
    def can_act(self) -> bool:
        return self.role is Role.ANALYST


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenRegistry:
    """Bearer token -> :class:`Identity`, holding only token digests."""

    def __init__(self, entries: Mapping[str, Identity]) -> None:
        self._by_digest: dict[str, Identity] = {}
        for token, identity in entries.items():
            if len(token) < MIN_TOKEN_LENGTH or token != token.strip():
                raise AuthError(
                    f"token for {identity.principal} is shorter than {MIN_TOKEN_LENGTH} "
                    "characters or padded; generate one with secrets.token_urlsafe(32)"
                )
            digest = _digest(token)
            if digest in self._by_digest:
                raise AuthError("two identities share one token")
            self._by_digest[digest] = identity

    def __len__(self) -> int:
        return len(self._by_digest)

    @property
    def tenants(self) -> frozenset[str]:
        return frozenset(identity.tenant_id for identity in self._by_digest.values())

    def authenticate(self, header: str | None) -> Identity:
        """Resolve an ``Authorization`` header value, or raise :class:`AuthError`."""
        if not header:
            raise AuthError("missing bearer token")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token or token != token.strip():
            raise AuthError("expected 'Authorization: Bearer <token>'")
        identity = self._by_digest.get(_digest(token))
        if identity is None:
            raise AuthError("unknown token")
        return identity

    @classmethod
    def parse(cls, spec: str) -> TokenRegistry:
        """Parse ``token:principal:tenant:role`` entries separated by commas.

        This is the ``SENTINEL_DASHBOARD_TOKENS`` format. Whitespace around entries
        is ignored; an entry with the wrong number of fields is an error rather than
        a skipped line, because a silently dropped analyst is a lock-out discovered
        during an incident.
        """
        entries: dict[str, Identity] = {}
        for raw in spec.split(","):
            entry = raw.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) != 4:
                raise AuthError(
                    "each token entry must be token:principal:tenant:role; got "
                    f"{len(parts)} field(s)"
                )
            token, principal, tenant, role = parts
            try:
                parsed_role = Role(role)
            except ValueError as exc:
                raise AuthError(f"unknown role {role!r}") from exc
            if token in entries:
                raise AuthError("two identities share one token")
            entries[token] = Identity(principal, tenant, parsed_role)
        if not entries:
            raise AuthError("no dashboard tokens configured")
        return cls(entries)

    @classmethod
    def generate(
        cls, identities: Iterable[Identity]
    ) -> tuple[TokenRegistry, dict[str, str]]:
        """Fresh random tokens for ``identities``. Returns ``(registry, principal->token)``.

        For a local demo: the tokens are printed once at start-up and never stored.
        """
        issued: dict[str, str] = {}
        entries: dict[str, Identity] = {}
        for identity in identities:
            token = secrets.token_urlsafe(32)
            entries[token] = identity
            issued[f"{identity.principal}@{identity.tenant_id}"] = token
        return cls(entries), issued
