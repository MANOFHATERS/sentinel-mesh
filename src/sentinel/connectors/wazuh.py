"""The EDR / firewall connector: Wazuh active response (PRD Section 4.2, weeks 1-2).

PRD Section 4.2 names *"a free-tier Wazuh or Elastic SIEM instance"* as the first
real integration, so this connector speaks the Wazuh 4.x server API:

*   ``POST /security/user/authenticate`` with HTTP Basic credentials returns a JWT
    (15 minutes by default). The token is cached and refreshed once on a ``401``.
*   ``GET /agents?ip=...`` or ``?name=...`` resolves the host an action names to an
    agent id. Exactly one active agent must match: zero is "we do not manage that
    host", and more than one means isolating *a* machine with that address, which is
    not what the analyst approved.
*   ``PUT /active-response?agents_list=...`` runs an active-response command on the
    agent(s). ``!firewall-drop`` is Wazuh's built-in, reading the address from
    ``alert.data.srcip``; host isolation has no built-in, so it defaults to a custom
    command named ``sentinel-isolate`` that the customer installs (the command map is
    configurable, and a missing command is a configuration error rather than a
    silent no-op).

Which agent enforces a block
----------------------------
``isolate_host`` runs on the host itself. ``block_ip`` runs wherever the firewall
is, which the action does not say — an alert names the attacker's address, not the
customer's perimeter. The operator therefore configures ``firewall_agents``, and an
``IP_BLOCK`` capability without them is refused at construction rather than at the
first incident.

Least privilege in Wazuh terms
------------------------------
Wazuh RBAC grants ``agent:read`` and ``active-response:command`` as separate
policies; those are the scopes this connector requires, and a declared
``*:*`` / ``administrator`` role is refused. The manager (agent ``000``) is never an
isolation target: it is the thing that would have to un-isolate everything else.
"""

from __future__ import annotations

import base64
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final

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
    CallRecord,
    EgressPolicy,
    HttpError,
    RetryPolicy,
    Route,
    ScopedHttpClient,
    Transport,
    UrllibTransport,
)
from sentinel.connectors.targets import canonical_host, canonical_ip
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import ActionRequest, ActionType

__all__ = ["DEFAULT_COMMANDS", "SCOPES_FOR_CAPABILITY", "WazuhConnector"]

SCOPES_FOR_CAPABILITY: Final[dict[Capability, frozenset[str]]] = {
    Capability.HOST_ISOLATE: frozenset({"agent:read", "active-response:command"}),
    Capability.IP_BLOCK: frozenset({"active-response:command"}),
}

DEFAULT_COMMANDS: Final[dict[Capability, str]] = {
    Capability.HOST_ISOLATE: "sentinel-isolate",
    Capability.IP_BLOCK: "!firewall-drop",
}

_AGENT_ID: Final[re.Pattern[str]] = re.compile(r"^\d{3,}$")
_COMMAND: Final[re.Pattern[str]] = re.compile(r"^!?[A-Za-z0-9_.-]{1,64}$")
_MANAGER_ID: Final[str] = "000"
#: Refresh this many seconds before the JWT's nominal expiry.
_TOKEN_SLACK_SECONDS: Final[float] = 60.0


class WazuhConnector:
    """Isolates hosts and blocks addresses through Wazuh active response."""

    name = "wazuh"

    def __init__(
        self,
        *,
        base_url: str,
        credential: Credential,
        capabilities: Iterable[Capability] = (Capability.HOST_ISOLATE, Capability.IP_BLOCK),
        firewall_agents: tuple[str, ...] = (),
        commands: Mapping[Capability, str] | None = None,
        token_lifetime_seconds: float = 900.0,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        allow_insecure_loopback: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        caps = frozenset(capabilities)
        unsupported = caps - SCOPES_FOR_CAPABILITY.keys()
        if unsupported or not caps:
            raise LeastPrivilegeError(
                f"wazuh: cannot hold {sorted(c.value for c in unsupported) or 'nothing'}; "
                "this connector isolates hosts and blocks addresses"
            )
        if credential.username is None:
            raise ValueError("wazuh: the API credential needs a username")
        check_scopes(
            connector=self.name,
            declared=credential.scopes,
            required=frozenset().union(*(SCOPES_FOR_CAPABILITY[c] for c in caps)),
        )
        if Capability.IP_BLOCK in caps:
            if not firewall_agents:
                raise ValueError(
                    "wazuh: ip.block needs firewall_agents — the agents that enforce the "
                    "perimeter. An alert names the attacker, not where to block them"
                )
            for agent in firewall_agents:
                if not _AGENT_ID.match(agent):
                    raise ValueError(f"wazuh: invalid agent id {agent!r}")
        resolved = dict(DEFAULT_COMMANDS)
        resolved.update(commands or {})
        for capability in caps:
            if not _COMMAND.match(resolved[capability]):
                raise ValueError(f"wazuh: invalid active-response command {resolved[capability]!r}")

        self._capabilities = caps
        self._credential = credential
        self.firewall_agents = tuple(firewall_agents)
        self.commands = {c: resolved[c] for c in caps}
        self._token: str | None = None
        self._token_expires = 0.0
        self._token_lifetime = token_lifetime_seconds
        self._monotonic = monotonic
        self._lock = threading.Lock()

        routes = [Route("auth", "POST", r"/security/user/authenticate")]
        if Capability.HOST_ISOLATE in caps:
            routes.append(
                Route("agents.get", "GET", r"/agents", query={"ip", "name", "select", "limit"})
            )
        routes.append(Route("active_response.run", "PUT", r"/active-response",
                            query={"agents_list"}))
        self.http = ScopedHttpClient(
            connector=self.name,
            policy=EgressPolicy(
                base_url=base_url,
                routes=tuple(routes),
                allow_insecure_loopback=allow_insecure_loopback,
            ),
            transport=transport or UrllibTransport(),
            auth=self._bearer,
            retry=retry or RetryPolicy(),
            **({"sleep": sleep} if sleep is not None else {}),
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def observe(self, callback: Callable[[CallRecord], None] | None) -> None:
        self.http.on_call = callback

    # --- auth ------------------------------------------------------------------ #

    def _bearer(self) -> dict[str, str]:
        with self._lock:
            if self._token is None or self._monotonic() >= self._token_expires:
                self._authenticate()
            return {"Authorization": f"Bearer {self._token}"}

    def _authenticate(self) -> None:
        pair = f"{self._credential.username}:{self._credential.secret.reveal()}"
        basic = base64.b64encode(pair.encode("utf-8")).decode("ascii")
        response = self.http.request(
            "POST",
            "/security/user/authenticate",
            headers={"Authorization": f"Basic {basic}"},
            use_auth=False,
            # A login creates nothing; repeating it after a 503 is harmless.
            retryable=True,
        )
        token = ((response.json() or {}).get("data") or {}).get("token")
        if not isinstance(token, str) or not token:
            raise ConnectorError("wazuh: authentication returned no token")
        self._token = token
        self._token_expires = self._monotonic() + max(
            1.0, self._token_lifetime - _TOKEN_SLACK_SECONDS
        )

    def _invalidate(self) -> None:
        with self._lock:
            self._token = None

    def _call(self, method: str, path: str, **kwargs: Any):
        """One API call, re-authenticating once if the token was revoked or expired."""
        try:
            return self.http.request(method, path, **kwargs)
        except HttpError as exc:
            if exc.status != 401:
                raise
            self._invalidate()
            return self.http.request(method, path, **kwargs)

    # --- actions ------------------------------------------------------------------ #

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        require_executable(action, connector=self.name)
        if action.action_type is ActionType.ISOLATE_HOST:
            self._require(Capability.HOST_ISOLATE)
            host = canonical_host(action.target)
            agent = self._resolve_agent(host)
            return self._run(
                Capability.HOST_ISOLATE,
                agents=(agent,),
                alert_data={"sentinel_action_id": action.action_id, "host": host},
                subject=f"host {host} (agent {agent})",
            )
        if action.action_type is ActionType.BLOCK_IP:
            self._require(Capability.IP_BLOCK)
            address = canonical_ip(action.target)
            return self._run(
                Capability.IP_BLOCK,
                agents=self.firewall_agents,
                alert_data={"srcip": address, "sentinel_action_id": action.action_id},
                subject=f"address {address}",
            )
        raise GuardrailViolation(
            f"wazuh: cannot execute {action.action_type.value}; it isolates hosts and "
            "blocks addresses"
        )

    def _require(self, capability: Capability) -> None:
        if capability not in self._capabilities:
            raise GuardrailViolation(f"wazuh: this connector was not granted {capability.value}")

    def _resolve_agent(self, host: str) -> str:
        key = "ip" if _looks_like_ip(host) else "name"
        body = self._call(
            "GET",
            "/agents",
            query={key: host, "select": "id,name,ip,status", "limit": "2"},
        ).json() or {}
        items = (body.get("data") or {}).get("affected_items") or []
        if not items:
            raise ConnectorError(f"wazuh: no agent manages {host}; nothing to isolate")
        if len(items) > 1:
            raise GuardrailViolation(
                f"wazuh: {len(items)} agents match {host}; isolating one of them is not "
                "what was approved"
            )
        agent = items[0]
        agent_id = str(agent.get("id", ""))
        if not _AGENT_ID.match(agent_id):
            raise ConnectorError(f"wazuh: malformed agent id for {host}")
        if agent_id == _MANAGER_ID:
            raise GuardrailViolation(
                f"wazuh: {host} is the Wazuh manager; isolating it would cut off the "
                "system that un-isolates everything else"
            )
        if agent.get("status") != "active":
            raise ConnectorError(
                f"wazuh: agent {agent_id} for {host} is {agent.get('status')}; an "
                "active-response command would be queued, not executed"
            )
        return agent_id

    def _run(
        self,
        capability: Capability,
        *,
        agents: tuple[str, ...],
        alert_data: dict[str, str],
        subject: str,
    ) -> ExecutionOutcome:
        command = self.commands[capability]
        response = self._call(
            "PUT",
            "/active-response",
            query={"agents_list": ",".join(agents)},
            json_body={"command": command, "arguments": [], "alert": {"data": alert_data}},
        )
        body = response.json() or {}
        data = body.get("data") or {}
        failed = data.get("failed_items") or []
        affected = data.get("affected_items") or []
        if failed or len(affected) < len(agents):
            ids = sorted(str(item) for item in affected)
            return ExecutionOutcome(
                succeeded=False,
                detail=(
                    f"wazuh: {command} reached {len(affected)} of {len(agents)} agent(s) "
                    f"for {subject}; affected {ids}, {len(failed)} failed"
                ),
            )
        return ExecutionOutcome(
            succeeded=True,
            detail=f"wazuh {command} on {subject} via agent(s) {', '.join(agents)}",
            reference=f"wazuh:active-response:{command}:{','.join(agents)}",
        )


def _looks_like_ip(host: str) -> bool:
    try:
        canonical_ip(host)
    except GuardrailViolation:
        return False
    return True
