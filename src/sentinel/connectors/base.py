"""The connector contract: what a Layer 5 connector is allowed to be (PRD Section 5.7).

PRD Section 5.7: *"every connector (EDR, firewall, Git) is scoped to the minimum API
permissions needed for its specific action set, never a broad admin credential."*
That sentence is three separate requirements, and each is enforced in a different
place because each fails differently:

1.  **Capabilities.** A connector declares the :class:`Capability` set it holds, and
    the router refuses to hand it an action outside that set. This is the
    *application-level* scope — what the connector's code will attempt.
2.  **Credential scopes.** A :class:`Credential` carries the scopes the operator
    says the token grants, and a connector refuses at construction if those exceed
    what its enabled capabilities need, or omit something they require. This is the
    *identity-level* scope — what the remote system would let the token do.
3.  **Egress routes.** Every HTTP call passes through
    :class:`~sentinel.connectors.http.EgressPolicy`, an allowlist of
    ``(method, path)`` pairs per connector. This is the *network-level* scope — what
    can physically leave the process — and it is the one that holds when the first
    two are wrong. A GitHub token with ``contents:write`` *can* merge a pull request;
    the Git connector cannot, because ``PUT .../pulls/{n}/merge`` is not a route it
    has.

And one requirement from F-08 that every connector repeats rather than trusts:
:func:`require_executable`. The graph already refuses to reach an execution node for
an unapproved action, and the schema already refuses to mark one executed. A
connector that trusts its caller executes whatever a bug upstream hands it, so the
last object before the wire checks too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from sentinel.core.errors import GuardrailViolation, SentinelError
from sentinel.core.schemas import ActionRequest, ActionType, ApprovalStatus

__all__ = [
    "CAPABILITY_FOR_ACTION",
    "ActionConnector",
    "Capability",
    "ConnectorError",
    "Credential",
    "EgressDenied",
    "ExecutesActions",
    "ExecutionOutcome",
    "LeastPrivilegeError",
    "Secret",
    "TargetRejected",
    "check_scopes",
    "require_executable",
]


class ConnectorError(SentinelError):
    """A connector could not complete a call it was permitted to make."""


class EgressDenied(GuardrailViolation):
    """A request fell outside the connector's egress allowlist. Never retried."""


class LeastPrivilegeError(GuardrailViolation):
    """A credential grants more than the connector needs, or less than it requires."""


class TargetRejected(GuardrailViolation):
    """An action's target is malformed, protected, or the wrong kind of thing."""


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """What a connector reported back.

    ``reference`` is the remote system's handle for what happened — a pull request
    URL, an active-response id — so an analyst can go and look. ``replayed`` marks an
    outcome returned from the execution journal rather than from a second call to the
    remote system, which is what a resumed run should see for an action that already
    ran.
    """

    succeeded: bool
    detail: str
    reference: str | None = None
    replayed: bool = False


class Capability(StrEnum):
    """What a connector can do to the outside world. One per side effect."""

    HOST_ISOLATE = "host.isolate"
    IP_BLOCK = "ip.block"
    ACCOUNT_DISABLE = "account.disable"
    PROCESS_KILL = "process.kill"
    FILE_QUARANTINE = "file.quarantine"
    PR_OPEN_DRAFT = "pr.open_draft"
    ISSUE_OPEN = "issue.open"
    NOTIFY = "notify.send"
    ENRICH = "enrich.local"


#: The capability an action type needs. ``OPEN_PATCH_PR`` arrives two ways and needs
#: a different capability for each: from the Code-Scan Agent it carries a validated
#: diff and becomes a draft PR (``open_draft``); from the Supply-Chain Agent it
#: carries only a node id — the sprint graph is synthetic, so there is no manifest to
#: edit — and becomes a tracking issue (``execute``). Mapping it here to the draft
#: capability and routing the no-diff case explicitly keeps "an action type means
#: one side effect" true everywhere except the one place that says why.
CAPABILITY_FOR_ACTION: Final[dict[ActionType, Capability]] = {
    ActionType.ISOLATE_HOST: Capability.HOST_ISOLATE,
    ActionType.BLOCK_IP: Capability.IP_BLOCK,
    ActionType.DISABLE_ACCOUNT: Capability.ACCOUNT_DISABLE,
    ActionType.KILL_PROCESS: Capability.PROCESS_KILL,
    ActionType.QUARANTINE_FILE: Capability.FILE_QUARANTINE,
    ActionType.OPEN_PATCH_PR: Capability.PR_OPEN_DRAFT,
    ActionType.NOTIFY_ANALYST: Capability.NOTIFY,
    ActionType.ENRICH_ONLY: Capability.ENRICH,
}


class Secret:
    """A string that does not print.

    ``repr``, ``str`` and f-strings all render ``Secret(***)``, so a token that
    reaches an exception message, a log line or an audit payload arrives redacted.
    The value is only available through :meth:`reveal`, which is greppable — every
    call site that handles the raw credential is one search away.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("a secret must be a non-empty string")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__

    def __format__(self, _spec: str) -> str:
        return "Secret(***)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Secret):
            return NotImplemented
        import hmac

        return hmac.compare_digest(self._value, other._value)

    def __hash__(self) -> int:
        return hash(("Secret", len(self._value)))

    def __reduce__(self):  # pragma: no cover - exercised by the pickle test
        raise TypeError("a Secret cannot be pickled; pass it explicitly instead")


@dataclass(frozen=True, slots=True)
class Credential:
    """A secret plus the scopes the operator says it grants.

    The scopes are a *declaration*, and :func:`check_scopes` holds the connector to it
    in both directions. Where the remote system reports the token's real scopes
    (GitHub's ``X-OAuth-Scopes`` on classic tokens), the connector also checks the
    report against its forbidden set before its first write — a declaration can be
    wrong, and a report is evidence.
    """

    secret: Secret
    scopes: frozenset[str] = field(default_factory=frozenset)
    #: For credentials that are a username/password pair (Wazuh's API user).
    username: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scopes", frozenset(s.strip() for s in self.scopes))
        if any(not s for s in self.scopes):
            raise ValueError("credential scopes must be non-empty strings")

    def __repr__(self) -> str:
        return f"Credential(scopes={sorted(self.scopes)!r}, secret=Secret(***))"


def check_scopes(
    *,
    connector: str,
    declared: Iterable[str],
    required: Iterable[str],
    permitted_extra: Iterable[str] = (),
) -> None:
    """Refuse a credential that is broader or narrower than the connector needs.

    ``required`` must all be present — a connector missing a scope fails on the
    first real incident, at 3 a.m., after the approval, which is the worst time to
    find out. Anything declared beyond ``required | permitted_extra`` is refused: the
    PRD's rule is *minimum* permissions, and a token that can do more than its
    connector needs is the broad admin credential Section 5.7 forbids, however it is
    named.
    """
    declared_set = frozenset(declared)
    required_set = frozenset(required)
    missing = required_set - declared_set
    if missing:
        raise LeastPrivilegeError(
            f"{connector}: credential is missing required scope(s) {sorted(missing)}"
        )
    excess = declared_set - required_set - frozenset(permitted_extra)
    if excess:
        raise LeastPrivilegeError(
            f"{connector}: credential grants {sorted(excess)} beyond what its enabled "
            f"capabilities need ({sorted(required_set)}); issue a narrower token "
            "(PRD Section 5.7: never a broad admin credential)"
        )


def require_executable(action: ActionRequest, *, connector: str) -> None:
    """The connector-side half of F-08. Raises unless ``action`` may run now."""
    if action.approval_status in (
        ApprovalStatus.REJECTED,
        ApprovalStatus.EXPIRED,
        ApprovalStatus.EXECUTED,
        ApprovalStatus.FAILED,
    ):
        raise GuardrailViolation(
            f"{connector} refused {action.action_type.value} on {action.target}: the "
            f"action is already {action.approval_status.value}"
        )
    if action.requires_human_approval and action.approval_status is not (
        ApprovalStatus.APPROVED
    ):
        raise GuardrailViolation(
            f"{connector} refused {action.action_type.value} on {action.target}: human "
            f"approval required, status is {action.approval_status.value}"
        )


class ExecutesActions(Protocol):
    """What an execution node needs: ``SimulatedConnector`` and the router both fit."""

    def execute(self, action: ActionRequest) -> ExecutionOutcome: ...


@runtime_checkable
class ActionConnector(Protocol):
    """A Layer 5 connector. ``execute`` must call :func:`require_executable` first."""

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> frozenset[Capability]: ...

    def execute(self, action: ActionRequest) -> ExecutionOutcome: ...
