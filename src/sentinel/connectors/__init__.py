"""Layer 5: the connector layer (PRD Figure 2, Section 5.7). Part 4.

Replaces Part 3's ``SimulatedConnector`` and ``DraftPullRequestConnector`` with real
connectors behind a least-privilege interface, without changing a node in any of the
three graphs:

==================================  ===============================================
:mod:`~sentinel.connectors.base`     capabilities, credentials, the F-08 check
:mod:`~sentinel.connectors.http`     the egress allowlist; the only network path
:mod:`~sentinel.connectors.targets`  what a target may be, and what it may never be
:mod:`~sentinel.connectors.journal`  exactly-once on resume; the blast-radius ceiling
:mod:`~sentinel.connectors.github`   draft pull requests and issues — never a merge
:mod:`~sentinel.connectors.wazuh`    host isolation and IP blocks (EDR / firewall)
:mod:`~sentinel.connectors.scim`     account disablement over SCIM 2.0
:mod:`~sentinel.connectors.notify`   Slack, signed webhooks, local enrichment
:mod:`~sentinel.connectors.router`   one object the graphs call; the last guardrail
:mod:`~sentinel.connectors.sandbox`  live local emulators of every API above
:mod:`~sentinel.connectors.config`   building the layer from environment variables
==================================  ===============================================

Nothing here imports the agent layer. A connector that needed an orchestrator to run
would be a connector nobody could test on its own.
"""

from sentinel.connectors.base import (
    CAPABILITY_FOR_ACTION,
    ActionConnector,
    Capability,
    ConnectorError,
    Credential,
    EgressDenied,
    ExecutionOutcome,
    LeastPrivilegeError,
    Secret,
    TargetRejected,
    require_executable,
)
from sentinel.connectors.http import EgressPolicy, Route, ScopedHttpClient, UrllibTransport
from sentinel.connectors.journal import (
    BlastRadiusExceeded,
    BlastRadiusLimiter,
    MemoryJournal,
    SqliteJournal,
)
from sentinel.connectors.targets import TargetPolicy

__all__ = [
    "CAPABILITY_FOR_ACTION",
    "ActionConnector",
    "BlastRadiusExceeded",
    "BlastRadiusLimiter",
    "Capability",
    "ConnectorError",
    "Credential",
    "EgressDenied",
    "EgressPolicy",
    "ExecutionOutcome",
    "LeastPrivilegeError",
    "MemoryJournal",
    "Route",
    "ScopedHttpClient",
    "Secret",
    "SqliteJournal",
    "TargetPolicy",
    "TargetRejected",
    "UrllibTransport",
    "require_executable",
]
