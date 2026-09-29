"""Building the real connector layer from the environment.

Every setting is an environment variable, because that is where a container
orchestrator puts secrets and where they stay out of the repository. Nothing is
enabled by default: a connector exists only if its variables are set, and a
half-configured connector (a URL with no token) raises rather than being skipped,
because a SOC that silently lost its firewall connector would find out during an
incident.

==================================  ==========================================
``SENTINEL_TENANT``                 the tenant this process acts for (required)
``SENTINEL_GITHUB_REPO``            ``owner/repo``
``SENTINEL_GITHUB_TOKEN``           fine-grained token
``SENTINEL_GITHUB_SCOPES``          comma list, e.g. ``contents:write,pull_requests:write``
``SENTINEL_GITHUB_API``             default ``https://api.github.com`` (GHES: your host)
``SENTINEL_WAZUH_URL``              ``https://wazuh.example:55000``
``SENTINEL_WAZUH_USER``             API user
``SENTINEL_WAZUH_PASSWORD``         API password
``SENTINEL_WAZUH_SCOPES``           comma list, e.g. ``agent:read,active-response:command``
``SENTINEL_WAZUH_FIREWALL_AGENTS``  comma list of agent ids that enforce blocks
``SENTINEL_SCIM_URL``               ``https://directory.example/scim/v2``
``SENTINEL_SCIM_TOKEN``             bearer token
``SENTINEL_SCIM_SCOPES``            ``users:read,users:write``
``SENTINEL_SLACK_WEBHOOK``          incoming-webhook URL (a secret)
``SENTINEL_PROTECTED_NETWORKS``     comma list of CIDRs never to act against
``SENTINEL_PROTECTED_HOSTS``        comma list of names/accounts never to act against
``SENTINEL_JOURNAL``                path of the SQLite execution journal
``SENTINEL_MAX_ACTIONS_PER_HOUR``   blast-radius ceiling (default 25)
==================================  ==========================================
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import timedelta

from sentinel.connectors.base import Capability, Credential, Secret
from sentinel.connectors.github import GITHUB_API, GitHubConnector
from sentinel.connectors.journal import BlastRadiusLimiter, MemoryJournal, SqliteJournal
from sentinel.connectors.notify import LocalEnrichmentConnector, SlackWebhookConnector
from sentinel.connectors.router import ConnectorRouter
from sentinel.connectors.scim import ScimIdentityConnector
from sentinel.connectors.targets import TargetPolicy
from sentinel.connectors.wazuh import WazuhConnector

__all__ = ["ConfigurationError", "router_from_env"]


class ConfigurationError(ValueError):
    """The environment describes a connector incompletely or inconsistently."""


def _list(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _group(env: Mapping[str, str], name: str, keys: tuple[str, ...]) -> dict[str, str] | None:
    """All of ``keys`` or none of them. Some-but-not-all is a configuration error."""
    present = {key: env[key] for key in keys if env.get(key, "").strip()}
    if not present:
        return None
    missing = [key for key in keys if key not in present]
    if missing:
        raise ConfigurationError(
            f"{name} is partly configured; set {', '.join(missing)} or unset the rest"
        )
    return present


def router_from_env(env: Mapping[str, str] | None = None, audit=None) -> ConnectorRouter:
    """Build a :class:`ConnectorRouter` from environment variables."""
    env = dict(os.environ if env is None else env)
    tenant = env.get("SENTINEL_TENANT", "").strip()
    if not tenant:
        raise ConfigurationError("SENTINEL_TENANT is required: a router serves one tenant")

    connectors: list[object] = [LocalEnrichmentConnector()]
    draft_opener = None

    github = _group(env, "GitHub", ("SENTINEL_GITHUB_REPO", "SENTINEL_GITHUB_TOKEN",
                                    "SENTINEL_GITHUB_SCOPES"))
    if github:
        owner, _, repo = github["SENTINEL_GITHUB_REPO"].partition("/")
        scopes = frozenset(_list(github["SENTINEL_GITHUB_SCOPES"]))
        caps = [Capability.PR_OPEN_DRAFT]
        if "issues:write" in scopes:
            caps.append(Capability.ISSUE_OPEN)
        connector = GitHubConnector(
            owner=owner,
            repo=repo,
            credential=Credential(Secret(github["SENTINEL_GITHUB_TOKEN"]), scopes),
            capabilities=caps,
            base_url=env.get("SENTINEL_GITHUB_API", GITHUB_API),
        )
        draft_opener = connector
        if Capability.ISSUE_OPEN in caps:
            connectors.append(connector)

    wazuh = _group(env, "Wazuh", ("SENTINEL_WAZUH_URL", "SENTINEL_WAZUH_USER",
                                  "SENTINEL_WAZUH_PASSWORD", "SENTINEL_WAZUH_SCOPES"))
    if wazuh:
        firewall = _list(env.get("SENTINEL_WAZUH_FIREWALL_AGENTS"))
        caps = [Capability.HOST_ISOLATE] + ([Capability.IP_BLOCK] if firewall else [])
        scopes = frozenset(_list(wazuh["SENTINEL_WAZUH_SCOPES"]))
        connectors.append(
            WazuhConnector(
                base_url=wazuh["SENTINEL_WAZUH_URL"],
                credential=Credential(Secret(wazuh["SENTINEL_WAZUH_PASSWORD"]), scopes,
                                      username=wazuh["SENTINEL_WAZUH_USER"]),
                capabilities=caps,
                firewall_agents=firewall,
            )
        )

    scim = _group(env, "SCIM", ("SENTINEL_SCIM_URL", "SENTINEL_SCIM_TOKEN",
                                "SENTINEL_SCIM_SCOPES"))
    if scim:
        connectors.append(
            ScimIdentityConnector(
                base_url=scim["SENTINEL_SCIM_URL"],
                credential=Credential(Secret(scim["SENTINEL_SCIM_TOKEN"]),
                                      frozenset(_list(scim["SENTINEL_SCIM_SCOPES"]))),
            )
        )

    if env.get("SENTINEL_SLACK_WEBHOOK", "").strip():
        connectors.append(SlackWebhookConnector(webhook_url=Secret(env["SENTINEL_SLACK_WEBHOOK"])))

    journal_path = env.get("SENTINEL_JOURNAL", "").strip()
    try:
        ceiling = int(env.get("SENTINEL_MAX_ACTIONS_PER_HOUR", "25"))
    except ValueError as exc:
        raise ConfigurationError("SENTINEL_MAX_ACTIONS_PER_HOUR must be an integer") from exc

    return ConnectorRouter(
        tenant_id=tenant,
        connectors=connectors,  # type: ignore[arg-type]
        draft_opener=draft_opener,
        targets=TargetPolicy(
            protected_networks=_list(env.get("SENTINEL_PROTECTED_NETWORKS")),
            protected_hosts=frozenset(_list(env.get("SENTINEL_PROTECTED_HOSTS"))),
        ),
        journal=SqliteJournal(journal_path) if journal_path else MemoryJournal(),
        limiter=BlastRadiusLimiter(max_actions=ceiling, window=timedelta(hours=1)),
        audit=audit,
    )
