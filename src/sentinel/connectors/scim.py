"""The identity connector: disable an account over SCIM 2.0 (RFC 7643 / RFC 7644).

SCIM rather than one vendor's API because it is the provisioning protocol Okta,
Microsoft Entra ID, OneLogin, JumpCloud and Google Workspace all serve, so one
connector covers the directories a mid-market customer actually runs. The whole
capability is two calls:

*   ``GET /Users?filter=userName eq "<name>"`` — exactly one result, or refuse.
*   ``PATCH /Users/{id}`` with a ``PatchOp`` that replaces ``active`` with ``false``.

``active=false`` is a state assignment, so the call is idempotent and is sent with an
idempotency key, which makes the ``PATCH`` safe to retry. An account that is already
inactive succeeds without a write, and the outcome says so.

The filter value is the one place untrusted text meets a query language. SCIM filter
syntax uses ``"``, ``(`` and ``)``; the target has already been through
:func:`~sentinel.connectors.targets.canonical_account`, which admits none of them, and
the value is escaped anyway, because the day the account regex is widened is not the
day anyone will remember this line.

Least privilege: ``users:read`` and ``users:write``, and the ``PATCH`` body can only
ever name the ``active`` attribute — the connector has no code path that sets a
password, a group membership or a role.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Final

from sentinel.connectors.base import (
    Capability,
    ConnectorError,
    Credential,
    ExecutionOutcome,
    LeastPrivilegeError,
    check_scopes,
    require_executable,
)
from sentinel.connectors.http import (
    SEGMENT,
    CallRecord,
    EgressPolicy,
    RetryPolicy,
    Route,
    ScopedHttpClient,
    Transport,
    UrllibTransport,
    quote_segment,
)
from sentinel.connectors.targets import canonical_account
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import ActionRequest, ActionType

__all__ = ["PATCH_OP_SCHEMA", "REQUIRED_SCOPES", "ScimIdentityConnector", "scim_filter_value"]

PATCH_OP_SCHEMA: Final[str] = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
REQUIRED_SCOPES: Final[frozenset[str]] = frozenset({"users:read", "users:write"})
_SCIM_JSON: Final[str] = "application/scim+json"


def scim_filter_value(value: str) -> str:
    """Quote a value for a SCIM filter string literal (RFC 7644 Section 3.4.2.2)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class ScimIdentityConnector:
    """Disables directory accounts. Cannot enable, create, delete or modify anything else."""

    name = "scim"

    def __init__(
        self,
        *,
        base_url: str,
        credential: Credential,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        allow_insecure_loopback: bool = False,
        capabilities: Iterable[Capability] = (Capability.ACCOUNT_DISABLE,),
    ) -> None:
        caps = frozenset(capabilities)
        if caps != {Capability.ACCOUNT_DISABLE}:
            raise LeastPrivilegeError(
                "scim: this connector holds account.disable and nothing else"
            )
        check_scopes(connector=self.name, declared=credential.scopes, required=REQUIRED_SCOPES)
        self._capabilities = caps
        self.http = ScopedHttpClient(
            connector=self.name,
            policy=EgressPolicy(
                base_url=base_url,
                routes=(
                    Route("users.search", "GET", r"/Users",
                          query={"filter", "attributes", "count"}),
                    Route("users.patch", "PATCH", rf"/Users/{SEGMENT}"),
                ),
                allow_insecure_loopback=allow_insecure_loopback,
            ),
            transport=transport or UrllibTransport(),
            auth=lambda: {"Authorization": f"Bearer {credential.secret.reveal()}"},
            retry=retry or RetryPolicy(),
            base_headers={"Accept": _SCIM_JSON},
            **({"sleep": sleep} if sleep is not None else {}),
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def observe(self, callback: Callable[[CallRecord], None] | None) -> None:
        self.http.on_call = callback

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        require_executable(action, connector=self.name)
        if action.action_type is not ActionType.DISABLE_ACCOUNT:
            raise GuardrailViolation(
                f"scim: cannot execute {action.action_type.value}; it disables accounts"
            )
        account = canonical_account(action.target)
        found = self.http.request(
            "GET",
            "/Users",
            query={
                "filter": f"userName eq {scim_filter_value(account)}",
                "attributes": "id,userName,active",
                "count": "2",
            },
        ).json() or {}
        resources = found.get("Resources") or []
        total = int(found.get("totalResults", len(resources)))
        if total == 0 or not resources:
            raise ConnectorError(f"scim: no account named {account}")
        if total > 1 or len(resources) > 1:
            raise GuardrailViolation(
                f"scim: {total} accounts match {account}; disabling one of them is not "
                "what was approved"
            )
        user = resources[0]
        if str(user.get("userName", "")).lower() != account.lower():
            raise ConnectorError(
                f"scim: the directory answered a filter for {account} with a different "
                "account; refusing to act on it"
            )
        user_id = str(user.get("id") or "")
        if not user_id:
            raise ConnectorError(f"scim: account {account} has no id")
        if user.get("active") is False:
            return ExecutionOutcome(
                succeeded=True,
                detail=f"scim: account {account} was already disabled; nothing written",
                reference=f"scim:Users/{user_id}",
            )
        response = self.http.request(
            "PATCH",
            f"/Users/{quote_segment(user_id)}",
            json_body={
                "schemas": [PATCH_OP_SCHEMA],
                "Operations": [{"op": "replace", "path": "active", "value": False}],
            },
            content_type=_SCIM_JSON,
            idempotency_key=action.action_id,
            expect=(200, 204),
        )
        if response.status == 200:
            body = response.json() or {}
            if body.get("active") is not False:
                raise ConnectorError(
                    f"scim: the directory accepted the change but reports {account} as "
                    "still active"
                )
        return ExecutionOutcome(
            succeeded=True,
            detail=f"scim: disabled account {account}",
            reference=f"scim:Users/{user_id}",
        )
