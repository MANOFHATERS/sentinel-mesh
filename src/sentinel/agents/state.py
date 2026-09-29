"""The checkpointed state that flows through the orchestration graph (PRD F-04).

PRD Section 5.4: *"The orchestrator node inspects the shared, checkpointed state
after each step and routes to the next agent."* F-04's acceptance criterion is
*"any node can pause for human input and resume with full context intact"*, which
makes this module the thing that criterion is actually about: if the state is not
complete, not serializable and not verifiable, resume cannot be faithful no matter
how good the runtime is.

Three decisions here are load-bearing.

**The state is a** :class:`~sentinel.core.schemas.Contract`, not a ``TypedDict``.
Every orchestration framework in this space models state as a mutable dict, and
that is the wrong shape for a system whose audit story is *"the checkpoint you
resumed from is the checkpoint I wrote"*. Being a ``Contract`` gives the state
immutability (a node cannot corrupt its input), ``extra="forbid"`` (a typo'd key
is an error rather than a silently ignored write), and
:meth:`~sentinel.core.schemas.Contract.canonical_hash` — which is what lets
:mod:`sentinel.agents.checkpoint` detect a checkpoint that was edited between
write and resume.

**Triage lives on the alert, not beside it.** ``IncidentState.triage`` is a
property reading ``alert.triage``. The obvious alternative — a separate
``triage`` field — gives two places a verdict can live and therefore one place
they can disagree, and the disagreement would be invisible: both fields
serialize, both hash, and nothing in the type system says which one the approval
gate should believe. :meth:`~sentinel.core.schemas.Alert.with_triage` already
exists and already validates, so the state reuses it.

**The step history is part of the state, not a side log.** A resumed run must be
able to answer "what has already happened" without consulting the audit log,
because the audit log is a separate durability domain: it can be keyed
differently, rotated, or (in the MSSP deployment the PRD targets) live in a
different tenant's storage. The history is therefore carried in the checkpoint,
and :attr:`Checkpoint.audit_head` binds it to the audit chain rather than
depending on it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from sentinel.core.schemas import (
    ActionRequest,
    Alert,
    Contract,
    Evidence,
    InvestigationReport,
    RiskTier,
    TriageResult,
    UtcDatetime,
)

__all__ = [
    "END",
    "HumanDecision",
    "IncidentState",
    "IncidentStatus",
    "Interrupt",
    "StepOutcome",
    "StepRecord",
]


END = "__end__"
"""The terminal cursor value. Named as LangGraph names it, for the same reason."""


class IncidentStatus(StrEnum):
    """Where a run is. Distinct from ``ApprovalStatus``, which is per action."""

    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    DISMISSED = "dismissed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """True when no further node will run without new input."""
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES: frozenset[IncidentStatus] = frozenset(
    {IncidentStatus.COMPLETED, IncidentStatus.DISMISSED, IncidentStatus.FAILED}
)


class StepOutcome(StrEnum):
    """How one node execution ended."""

    OK = "ok"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class StepRecord(Contract):
    """One node execution, as the state remembers it.

    ``duration_ms`` is stored rather than derived from the timestamps because the
    PRD's MTTD/MTTC metrics (Section 9.1) are measured off these records, and a
    duration recomputed from two wall-clock instants silently absorbs any clock
    adjustment that happened between them.
    """

    node: str = Field(min_length=1, max_length=64)
    started_at: UtcDatetime
    ended_at: UtcDatetime
    outcome: StepOutcome
    duration_ms: float = Field(ge=0.0, le=86_400_000.0)
    detail: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _ordering(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError(f"step {self.node!r} ended before it started")
        return self


class Interrupt(Contract):
    """What the graph is waiting for, recorded so a UI can render the question.

    The PRD's only interrupt is the Human Approval Gate, so ``subject_id`` is an
    ``action_id`` in every path that exists today. It is typed as a plain
    identifier rather than as ``action_id`` because the field's meaning is "the
    thing this question is about", and a second interrupt kind (the PRD's Phase 2
    Threat-Hunting agent would want one) should not require a schema migration to
    reuse the mechanism.
    """

    node: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=1000)
    subject_id: str = Field(min_length=1, max_length=64)
    requested_at: UtcDatetime


class HumanDecision(Contract):
    """The analyst's answer to an :class:`Interrupt`. The resume value.

    Typed, rather than the ``Any`` that a general-purpose resume channel would
    carry, because this value is the sole input that can turn a *proposed*
    destructive action into an *executed* one. F-08 says zero actions execute
    without a logged approval; an untyped resume payload is the hole through
    which a truthy dict becomes an approval.
    """

    approver: str = Field(min_length=1, max_length=128)
    approved: bool
    decided_at: UtcDatetime
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _approver_is_identified(self) -> Self:
        if not self.approver.strip():
            raise ValueError("an approval must name a human; F-08 requires an approver")
        return self


class IncidentState(Contract):
    """Everything one incident's run knows. Immutable; nodes return a new one."""

    incident_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=64)
    alert: Alert
    report: InvestigationReport | None = None
    actions: tuple[ActionRequest, ...] = Field(default=(), max_length=64)
    evidence: tuple[Evidence, ...] = Field(default=(), max_length=256)
    trust_tier: RiskTier = RiskTier.RECOMMEND
    status: IncidentStatus = IncidentStatus.RUNNING
    cursor: str = Field(default=END, min_length=1, max_length=64)
    history: tuple[StepRecord, ...] = Field(default=(), max_length=256)
    interrupt: Interrupt | None = None
    error: str | None = Field(default=None, max_length=2000)
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.alert.tenant_id != self.tenant_id:
            raise ValueError(
                f"state tenant {self.tenant_id!r} does not match alert tenant "
                f"{self.alert.tenant_id!r}; a cross-tenant state is a data leak, not a bug"
            )
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes created_at")
        if (self.status is IncidentStatus.AWAITING_APPROVAL) != (self.interrupt is not None):
            raise ValueError(
                "awaiting_approval and interrupt must agree: a run that is waiting must "
                "say what it waits for, and a run that is not waiting must not carry a "
                "stale question"
            )
        if self.status is IncidentStatus.FAILED and not self.error:
            raise ValueError("a failed run must record why")
        if self.status.is_terminal and self.cursor != END:
            raise ValueError(
                f"status {self.status.value} is terminal but cursor points at "
                f"{self.cursor!r}; a terminal state with a live cursor would resume"
            )
        seen: set[str] = set()
        for action in self.actions:
            if action.action_id in seen:
                raise ValueError(f"duplicate action_id {action.action_id!r} in state")
            seen.add(action.action_id)
            if action.alert_id != self.alert.alert_id:
                raise ValueError(
                    f"action {action.action_id!r} belongs to alert {action.alert_id!r}, "
                    f"not {self.alert.alert_id!r}"
                )
        return self

    # --- derived views -------------------------------------------------------- #

    @property
    def triage(self) -> TriageResult | None:
        """The triage verdict. Single-sourced from the alert (see module docstring)."""
        return self.alert.triage

    @property
    def is_waiting(self) -> bool:
        return self.status is IncidentStatus.AWAITING_APPROVAL

    @property
    def pending_action(self) -> ActionRequest | None:
        """The action the current interrupt is about, if any."""
        if self.interrupt is None:
            return None
        return self.action(self.interrupt.subject_id)

    def action(self, action_id: str) -> ActionRequest | None:
        for candidate in self.actions:
            if candidate.action_id == action_id:
                return candidate
        return None

    def steps_for(self, node: str) -> tuple[StepRecord, ...]:
        return tuple(record for record in self.history if record.node == node)

    @property
    def visited(self) -> tuple[str, ...]:
        """Node names in execution order, including repeats."""
        return tuple(record.node for record in self.history)

    # --- transitions ---------------------------------------------------------- #

    def with_alert(self, alert: Alert, *, at: datetime) -> IncidentState:
        return self.updated(alert=alert, updated_at=at)

    def with_triage(self, triage: TriageResult, *, at: datetime) -> IncidentState:
        """Attach a verdict by routing through :meth:`Alert.with_triage`."""
        return self.updated(alert=self.alert.with_triage(triage), updated_at=at)

    def with_report(self, report: InvestigationReport, *, at: datetime) -> IncidentState:
        if self.alert.alert_id not in report.alert_ids:
            raise ValueError(
                f"report {report.report_id!r} does not cover alert "
                f"{self.alert.alert_id!r}; attaching it would make the narrative "
                "describe a different incident than the one it is filed under"
            )
        return self.updated(
            report=report,
            evidence=_merge_evidence(self.evidence, report.evidence),
            updated_at=at,
        )

    def with_action(self, action: ActionRequest, *, at: datetime) -> IncidentState:
        """Add or replace an action by ``action_id``, keeping insertion order."""
        replaced = False
        actions: list[ActionRequest] = []
        for existing in self.actions:
            if existing.action_id == action.action_id:
                actions.append(action)
                replaced = True
            else:
                actions.append(existing)
        if not replaced:
            actions.append(action)
        return self.updated(
            actions=tuple(actions),
            evidence=_merge_evidence(self.evidence, action.evidence),
            updated_at=at,
        )

    def with_evidence(self, evidence: tuple[Evidence, ...], *, at: datetime) -> IncidentState:
        return self.updated(evidence=_merge_evidence(self.evidence, evidence), updated_at=at)

    def with_step(self, record: StepRecord) -> IncidentState:
        return self.updated(history=(*self.history, record), updated_at=record.ended_at)

    def waiting_on(self, interrupt: Interrupt) -> IncidentState:
        return self.updated(
            status=IncidentStatus.AWAITING_APPROVAL,
            interrupt=interrupt,
            updated_at=interrupt.requested_at,
        )

    def resumed(self, *, cursor: str, at: datetime) -> IncidentState:
        """Clear the interrupt and point at the node that will now run."""
        if self.interrupt is None:
            raise ValueError("resumed() on a state that is not waiting")
        return self.updated(
            status=IncidentStatus.RUNNING,
            interrupt=None,
            cursor=cursor,
            updated_at=at,
        )

    def routed_to(self, cursor: str, *, at: datetime) -> IncidentState:
        return self.updated(cursor=cursor, updated_at=at)

    def finished(self, status: IncidentStatus, *, at: datetime, error: str | None = None) -> (
        IncidentState
    ):
        if not status.is_terminal:
            raise ValueError(f"{status.value} is not a terminal status")
        return self.updated(
            status=status,
            cursor=END,
            interrupt=None,
            error=error,
            updated_at=at,
        )


def _merge_evidence(
    existing: tuple[Evidence, ...], incoming: tuple[Evidence, ...]
) -> tuple[Evidence, ...]:
    """Union by ``ref``, first writer wins, order preserved.

    First-writer-wins rather than last, because the first citation of a ref is the
    one whose ``relevance`` was computed against the query that actually retrieved
    it. A later, incidental re-citation at a different relevance would otherwise
    rewrite the evidence an earlier claim was graded on.
    """
    seen = {item.ref for item in existing}
    merged = list(existing)
    for item in incoming:
        if item.ref not in seen:
            seen.add(item.ref)
            merged.append(item)
    return tuple(merged)
