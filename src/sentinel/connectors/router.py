"""The router: one object the graphs call, and the last guardrail before the wire.

The three graphs were written against a one-method interface —
``execute(action) -> ExecutionOutcome``, plus ``open_draft(action, draft)`` for the
code-scan graph — so that Part 4 could replace the connectors without touching a
node. :class:`ConnectorRouter` *is* that interface. It holds the real connectors and
applies, in order, every check that belongs to "may this reach the outside world",
none of which a single connector can apply for itself:

1.  **Tenant isolation.** A router serves one tenant. An MSSP runs one per customer,
    and an action for ``globex`` handed to ``acme``'s router is refused — one
    customer's approval must never drive another customer's EDR.
2.  **The F-08 check**, again (:func:`~sentinel.connectors.base.require_executable`).
3.  **The journal.** An action already executed returns its recorded outcome,
    ``replayed=True``, with no second call (see :mod:`sentinel.connectors.journal`).
4.  **Capability dispatch, fail closed.** Exactly one connector per capability, fixed
    at construction. An action whose capability no connector holds is refused, not
    dropped: "we have no firewall connector" must surface as a failed action an
    analyst sees, never as a silent success.
5.  **Target policy.** Format and the customer's protected infrastructure.
6.  **Blast radius.** A ceiling on destructive executions per tenant per window.

And it writes the record: a ``connector_called`` audit row per HTTP attempt (route
name, status, digest — see :class:`~sentinel.connectors.http.CallRecord`) and a
``guardrail_blocked`` row per refusal. The execution node still writes
``action_executed`` itself, so :func:`~sentinel.agents.contain.verify_no_ungated_execution`
reads the same rows it always has and needs no change — which the ledger said would
be the test of whether this layer had been given authority it should not have.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from sentinel.audit.log import HashChainedAuditLog
from sentinel.connectors.base import (
    CAPABILITY_FOR_ACTION,
    ActionConnector,
    Capability,
    ExecutionOutcome,
    require_executable,
)
from sentinel.connectors.http import CallRecord
from sentinel.connectors.journal import BlastRadiusLimiter, ExecutionJournal, MemoryJournal
from sentinel.connectors.targets import TargetPolicy
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import ActionRequest, ActionType, AuditEventType
from sentinel.scan.patch import PullRequestDraft

__all__ = ["ConnectorRouter", "DraftPullRequestOpener", "RoutedExecution"]


class DraftPullRequestOpener(Protocol):
    name: str

    @property
    def capabilities(self) -> frozenset[Capability]: ...

    def open_draft(self, action: ActionRequest, draft: PullRequestDraft) -> ExecutionOutcome: ...


@dataclass(frozen=True, slots=True)
class RoutedExecution:
    """One call the router let through, for tests and the dashboard."""

    action: ActionRequest
    connector: str
    outcome: ExecutionOutcome


_CURRENT_ACTION: contextvars.ContextVar[ActionRequest | None] = contextvars.ContextVar(
    "sentinel_connector_action", default=None
)


class ConnectorRouter:
    """Dispatches approved actions to least-privilege connectors for one tenant."""

    name = "router"

    def __init__(
        self,
        *,
        tenant_id: str,
        connectors: Iterable[ActionConnector] = (),
        draft_opener: DraftPullRequestOpener | None = None,
        targets: TargetPolicy | None = None,
        journal: ExecutionJournal | None = None,
        limiter: BlastRadiusLimiter | None = None,
        audit: HashChainedAuditLog | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("a router serves exactly one tenant; name it")
        self.tenant_id = tenant_id
        self.targets = targets or TargetPolicy()
        self.journal: ExecutionJournal = journal or MemoryJournal()
        self.limiter = limiter
        self.audit = audit
        self._by_capability: dict[Capability, ActionConnector] = {}
        self._draft_opener = draft_opener
        self._lock = threading.Lock()
        self.executions: list[RoutedExecution] = []
        self.refusals: list[tuple[ActionRequest, str]] = []

        for connector in connectors:
            for capability in connector.capabilities:
                if capability is Capability.PR_OPEN_DRAFT:
                    continue  # reached through open_draft, not execute
                holder = self._by_capability.get(capability)
                if holder is not None:
                    raise ValueError(
                        f"capability {capability.value} is held by both {holder.name} and "
                        f"{connector.name}; exactly one connector may hold each, or which "
                        "system acted becomes a question the audit log cannot answer"
                    )
                self._by_capability[capability] = connector
            self._observe(connector)
        if draft_opener is not None:
            if Capability.PR_OPEN_DRAFT not in draft_opener.capabilities:
                raise ValueError(f"{draft_opener.name} does not hold pr.open_draft")
            self._observe(draft_opener)

    # --- introspection ------------------------------------------------------------- #

    @property
    def capabilities(self) -> frozenset[Capability]:
        held = set(self._by_capability)
        if self._draft_opener is not None:
            held.add(Capability.PR_OPEN_DRAFT)
        return frozenset(held)

    @property
    def executed(self) -> list[ActionRequest]:
        """Actions that reached a connector and succeeded. Mirrors ``SimulatedConnector``."""
        return [item.action for item in self.executions if item.outcome.succeeded]

    def connector_for(self, action_type: ActionType) -> str | None:
        capability = _capability_for_execute(action_type)
        holder = self._by_capability.get(capability)
        return None if holder is None else holder.name

    # --- the graph-facing interface ---------------------------------------------- #

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        capability = _capability_for_execute(action.action_type)
        connector = self._by_capability.get(capability)

        def call() -> ExecutionOutcome:
            if connector is None:
                raise GuardrailViolation(
                    f"no connector holds {capability.value} for tenant {self.tenant_id}; "
                    f"{action.action_type.value} is refused rather than silently dropped"
                )
            return connector.execute(action)

        return self._dispatch(action, connector.name if connector else "none", call)

    def open_draft(self, action: ActionRequest, draft: PullRequestDraft) -> ExecutionOutcome:
        opener = self._draft_opener

        def call() -> ExecutionOutcome:
            if opener is None:
                raise GuardrailViolation(
                    f"no connector holds pr.open_draft for tenant {self.tenant_id}"
                )
            return opener.open_draft(action, draft)

        return self._dispatch(action, opener.name if opener else "none", call)

    # --- internals --------------------------------------------------------------- #

    def _dispatch(self, action: ActionRequest, connector_name: str, call) -> ExecutionOutcome:
        try:
            if action.tenant_id != self.tenant_id:
                raise GuardrailViolation(
                    f"router for tenant {self.tenant_id} refused an action for tenant "
                    f"{action.tenant_id}"
                )
            require_executable(action, connector=self.name)
            prior = self.journal.lookup(action.action_id)
            if prior is not None and prior.completed and prior.outcome is not None:
                outcome = ExecutionOutcome(
                    succeeded=prior.outcome.succeeded,
                    detail=f"{prior.outcome.detail} (replayed from the execution journal)",
                    reference=prior.outcome.reference,
                    replayed=True,
                )
                with self._lock:
                    self.executions.append(RoutedExecution(action, prior.connector, outcome))
                return outcome
            self.targets.validate(action.action_type, action.target)
        except GuardrailViolation as exc:
            self._refused(action, connector_name, exc)
            raise

        reserved = False
        if self.limiter is not None and action.action_type.is_destructive:
            try:
                self.limiter.acquire(self.tenant_id)
            except GuardrailViolation as exc:
                self._refused(action, connector_name, exc)
                raise
            reserved = True

        self.journal.start(action.action_id, connector=connector_name)
        token = _CURRENT_ACTION.set(action)
        try:
            outcome = call()
        except GuardrailViolation as exc:
            # A refusal happens before the wire (egress, capability, target, scope),
            # so the blast-radius slot was never used.
            if reserved and self.limiter is not None:
                self.limiter.release(self.tenant_id)
            self._refused(action, connector_name, exc)
            raise
        finally:
            _CURRENT_ACTION.reset(token)

        if outcome.succeeded:
            self.journal.complete(action.action_id, outcome)
        with self._lock:
            self.executions.append(RoutedExecution(action, connector_name, outcome))
        return outcome

    def _observe(self, connector: object) -> None:
        observe = getattr(connector, "observe", None)
        if callable(observe):
            observe(self._record_call)

    def _record_call(self, record: CallRecord) -> None:
        action = _CURRENT_ACTION.get()
        if self.audit is None or action is None:
            return
        self.audit.append(
            AuditEventType.CONNECTOR_CALLED,
            actor=f"connector:{record.connector}",
            tenant_id=action.tenant_id,
            subject_id=action.action_id,
            payload={
                **record.as_payload(),
                "action_type": action.action_type.value,
                "approval_status": action.approval_status.value,
                "requires_human_approval": action.requires_human_approval,
            },
        )

    def _refused(self, action: ActionRequest, connector: str, exc: Exception) -> None:
        reason = f"{type(exc).__name__}: {exc}"[:500]
        with self._lock:
            self.refusals.append((action, reason))
        if self.audit is None:
            return
        self.audit.append(
            AuditEventType.GUARDRAIL_BLOCKED,
            actor=f"connector:{connector}",
            tenant_id=action.tenant_id or self.tenant_id,
            subject_id=action.action_id,
            payload={
                "action_type": action.action_type.value,
                "guardrail": type(exc).__name__,
                "reason": reason,
            },
        )


def _capability_for_execute(action_type: ActionType) -> Capability:
    """The capability ``execute`` needs. A diff-less ``OPEN_PATCH_PR`` is an issue."""
    if action_type is ActionType.OPEN_PATCH_PR:
        return Capability.ISSUE_OPEN
    return CAPABILITY_FOR_ACTION[action_type]
