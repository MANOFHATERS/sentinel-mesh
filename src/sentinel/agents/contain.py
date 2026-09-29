"""The Containment Agent and the Human Approval Gate (PRD F-08, Section 5.5.4).

F-08's acceptance criterion is a zero: *"Zero actions executed without a logged
approval event."* A zero is a claim about every path, which is why the defence is
in four independent places rather than in one careful function:

1.  **The schema.** :meth:`~sentinel.core.schemas.ActionRequest.propose` derives
    ``requires_human_approval`` from the action type and the trust tier; the
    caller never sets it. :meth:`~sentinel.core.schemas.ActionRequest.mark_executed`
    refuses a request that needs approval and has not been approved. Part 1 built
    this before any agent existed, which is the point — it covers code that had
    not been written yet.
2.  **The action mask.** :class:`~sentinel.rl.actions.ActionMask` removes
    ``auto_contain`` from the policy's arm set below ``auto_with_notify``, so at
    the tiers a customer starts on the policy cannot *want* to contain
    autonomously (PRD Section 5.7's tiered autonomy).
3.  **The graph.** The execution node sits behind ``interrupt_before``, so the
    run stops and checkpoints before it, not after.
4.  **The audit log.** Every proposal, request, grant, denial and execution is a
    chained row, so F-08 is *checkable after the fact* rather than only
    preventable. :func:`verify_no_ungated_execution` is the check, and it reads
    the log rather than the code.

Defence in depth here is not belt and braces. Each layer fails differently: the
schema cannot see the trust tier a customer configured, the mask cannot see what
a caller does after selection, the graph cannot see an action executed outside
it, and the log cannot prevent anything at all. The intersection is the
guarantee.

Why the bandit does not choose the action type
----------------------------------------------
The policy (Section 5.5.4) chooses among four *decisions* — contain, escalate,
monitor, dismiss — not among ``isolate_host`` / ``block_ip`` /
``disable_account``. That separation is deliberate. The reward signal the policy
learns from is analyst feedback on whether intervening was right, which says
nothing about whether isolating the host was better than blocking the address.
Letting the bandit pick the concrete action would have it optimising a choice it
receives no signal about, and the resulting preferences would be an artefact of
the exploration schedule. The *type* comes from the investigation's
recommendation, which is derived from the technique and is explainable; the
*decision to act at all* comes from the policy, which is what it has evidence
for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

import numpy as np

from sentinel.audit.log import HashChainedAuditLog
from sentinel.connectors.base import ExecutionOutcome
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import GuardrailViolation, SentinelError
from sentinel.core.schemas import (
    ActionRequest,
    ActionType,
    AgentName,
    Alert,
    ApprovalStatus,
    AuditEventType,
    Evidence,
    InvestigationReport,
    RiskTier,
    Severity,
)
from sentinel.rl.actions import ActionMask, ResponseAction
from sentinel.rl.bandit import LinearThompsonBandit
from sentinel.rl.simulate import build_context

__all__ = [
    "ContainmentAgent",
    "ContainmentError",
    "ExecutionOutcome",
    "Proposal",
    "SimulatedConnector",
    "approval_queue",
    "executable_action",
    "pending_action",
    "verify_no_ungated_execution",
]


class ContainmentError(SentinelError):
    """The Containment Agent could not produce a proposal it is allowed to make."""


#: Where an action's target comes from, per action type. An ``isolate_host``
#: aimed at an IP address and a ``block_ip`` aimed at a hostname are both
#: syntactically fine and operationally wrong, and the connector layer (Part 4)
#: would be the first thing to notice — after the approval had been granted.
_TARGET_FIELD: Final[dict[ActionType, str]] = {
    ActionType.ISOLATE_HOST: "asset_id",
    ActionType.BLOCK_IP: "src_ip",
    ActionType.DISABLE_ACCOUNT: "asset_id",
    ActionType.KILL_PROCESS: "asset_id",
    ActionType.QUARANTINE_FILE: "asset_id",
    ActionType.OPEN_PATCH_PR: "asset_id",
    ActionType.NOTIFY_ANALYST: "asset_id",
    ActionType.ENRICH_ONLY: "asset_id",
}


@dataclass(frozen=True, slots=True)
class Proposal:
    """What the agent decided, and the policy reasoning behind it."""

    action: ActionRequest | None
    response: ResponseAction
    tier: RiskTier
    exploratory: bool
    policy_confidence: float
    #: What the learned policy itself chose, before any floor was applied. ``None``
    #: when no policy is fitted and the response simply follows triage.
    policy_choice: ResponseAction | None = None
    #: True when the triage floor overrode a less attentive policy choice.
    floored: bool = False
    #: The policy's posterior-mean reward estimate per allowed option.
    expected: dict[str, float] = field(default_factory=dict)

    def policy_view(self) -> dict[str, object]:
        """The policy's reasoning as plain data, for the audit row and the dashboard."""
        return {
            "fitted": self.policy_choice is not None,
            "choice": None if self.policy_choice is None else self.policy_choice.value,
            "response": self.response.value,
            "confidence": round(float(self.policy_confidence), 6),
            "exploratory": self.exploratory,
            "floored": self.floored,
            "expected": {k: round(float(v), 6) for k, v in self.expected.items()},
        }

    @property
    def intervenes(self) -> bool:
        return self.response.intervenes

    @property
    def needs_human(self) -> bool:
        return self.action is not None and self.action.requires_human_approval


class SimulatedConnector:
    """A mocked EDR/firewall, per PRD Section 4.1's explicit scope.

    Part 4 built the real layer in :mod:`sentinel.connectors` —
    :class:`~sentinel.connectors.router.ConnectorRouter` in front of Wazuh, SCIM,
    GitHub and Slack connectors — and this stays as the hermetic default, because
    the one-command evaluation must run with no network and F-08 still needs
    *something* to execute in order to prove that nothing executes without approval.

    It records what it was asked to do, so a test can assert the negative: that
    a rejected action never reached it.
    """

    def __init__(self) -> None:
        self.executed: list[ActionRequest] = []

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        if action.requires_human_approval and action.approval_status is not (
            ApprovalStatus.APPROVED
        ):
            # The connector refuses too. A connector that trusts its caller is a
            # connector that executes whatever a bug upstream hands it.
            raise GuardrailViolation(
                f"connector refused {action.action_type.value} on {action.target}: "
                f"human approval required, status is {action.approval_status.value}"
            )
        self.executed.append(action)
        return ExecutionOutcome(
            succeeded=True,
            detail=f"simulated {action.action_type.value} on {action.target}",
        )


@dataclass(slots=True)
class ContainmentAgent:
    """Proposes a response action. Never executes one."""

    policy: LinearThompsonBandit | None = None
    clock: Clock = field(default_factory=SystemClock)
    #: The customer's current trust tier for this action class (Section 5.7).
    #: ``recommend`` is where every action type starts, by the PRD's own rule.
    default_tier: RiskTier = RiskTier.RECOMMEND
    #: Part 5. When true, the policy may raise the level of attention above what
    #: triage decided but never lower it: an alert triage escalated is escalated
    #: whatever the policy prefers. The same monotone-caution rule the reasoning
    #: engine is held to (see :mod:`sentinel.agents.engine`), applied to the learned
    #: policy. Off by default so the offline F-09 replay measures the policy alone.
    triage_floor: bool = False
    version: str = "containment-1"

    def propose(
        self,
        alert: Alert,
        *,
        report: InvestigationReport | None = None,
        tier: RiskTier | None = None,
        now: datetime | None = None,
        audit_hash_prev: str | None = None,
    ) -> Proposal:
        """Choose a response and, if it implies one, build an ``ActionRequest``."""
        triage = alert.triage
        if triage is None:
            raise ContainmentError(
                f"alert {alert.alert_id} has no triage verdict; containment runs "
                "downstream of triage and needs its score to build a policy context"
            )
        effective_tier = tier if tier is not None else self.default_tier
        created_at = now or self.clock.now()
        mask = ActionMask.for_tier(effective_tier)

        response, exploratory, confidence, choice, floored, expected = self._select(
            alert, triage, effective_tier
        )
        # Asserted rather than trusted: this is the invariant the mask exists for,
        # and the assertion costs nothing on a path that runs once per incident.
        mask.require(response)

        action_type = self._action_type(
            response, report, triage.severity, effective_tier
        )
        if action_type is None:
            return Proposal(
                action=None,
                response=response,
                tier=effective_tier,
                exploratory=exploratory,
                policy_confidence=confidence,
                policy_choice=choice,
                floored=floored,
                expected=expected,
            )

        target = self._target(alert, action_type)
        action = ActionRequest.propose(
            alert_id=alert.alert_id,
            tenant_id=alert.tenant_id,
            proposed_by=AgentName.CONTAINMENT,
            action_type=action_type,
            target=target,
            rationale=self._rationale(response, action_type, triage, confidence),
            risk_tier=effective_tier,
            created_at=created_at,
            evidence=self._evidence(report),
            **({"audit_hash_prev": audit_hash_prev} if audit_hash_prev else {}),
        )
        return Proposal(
            action=action,
            response=response,
            tier=effective_tier,
            exploratory=exploratory,
            policy_confidence=confidence,
            policy_choice=choice,
            floored=floored,
            expected=expected,
        )

    # --- internals ------------------------------------------------------------ #

    def _select(
        self, alert: Alert, triage, tier: RiskTier
    ) -> tuple[ResponseAction, bool, float, ResponseAction | None, bool, dict[str, float]]:
        """``(response, exploratory, confidence, policy_choice, floored, expected)``."""
        score = float(triage.anomaly_score if triage.anomaly_score is not None else 0.5)
        if self.policy is None:
            # No policy fitted. Fall back to the triage decision, which is the
            # conservative reading: without a learned policy the system does what
            # the detectors said, and never more.
            fallback = _RESPONSE_FOR_DECISION[triage.decision]
            mask = ActionMask.for_tier(tier)
            if not mask.permits(fallback):
                fallback = ResponseAction.ESCALATE
            return fallback, False, triage.confidence, None, False, {}
        context = build_context(alert, anomaly_score=min(1.0, max(0.0, score)))
        decision = self.policy.select(np.asarray(context, dtype=float), tier=tier)
        expected = {a.value: float(v) for a, v in decision.expected_values.items()}
        response = decision.action
        floored = False
        if self.triage_floor:
            floor = _RESPONSE_FOR_DECISION[triage.decision]
            if _ATTENTION[response] < _ATTENTION[floor] and ActionMask.for_tier(
                tier
            ).permits(floor):
                response = floor
                floored = True
        return (
            response,
            decision.was_exploratory,
            decision.confidence,
            decision.action,
            floored,
            expected,
        )

    def _action_type(
        self,
        response: ResponseAction,
        report: InvestigationReport | None,
        severity: Severity,
        tier: RiskTier,
    ) -> ActionType | None:
        """Turn a policy decision into a concrete action, given the trust tier.

        The subtlety here is a genuine contradiction between two parts of the
        PRD, and resolving it the obvious way makes F-08 unreachable.

        Section 5.5.4 gives the policy the action ``auto_contain``, and
        :mod:`sentinel.rl.actions` correctly sets its minimum trust tier to
        ``auto_with_notify`` — "contain without asking" is exactly what the top
        two tiers mean. Section 5.7 then says every action type starts at the
        *lowest* tier and is promoted only after a run of correct
        recommendations. Put together, a customer on the default ``recommend``
        tier has ``auto_contain`` masked out of the policy's arm set, so the
        Containment Agent never proposes a destructive action, so the Human
        Approval Gate never fires — and F-08's acceptance criterion passes
        vacuously, on a system that has never gated anything.

        Measured before this was fixed: 200 incidents through the graph at the
        default tier produced 133 executions, 0 approval requests.

        The resolution is that "contain" names two different things. The
        *policy's* ``auto_contain`` means "act without a human", and it is right
        that ``recommend`` forbids it. *Recommending* a containment action for a
        human to approve is what the ``recommend`` tier is for, and that is an
        escalation, not an autonomous action. So an escalation at ``recommend``
        or above proposes the action the investigation recommends, gated;
        ``observe`` proposes nothing but a notification, which is what
        observe-only means.
        """
        if response is ResponseAction.DISMISS:
            return None
        if response is ResponseAction.MONITOR:
            return ActionType.ENRICH_ONLY

        if response is ResponseAction.AUTO_CONTAIN:
            # The tier already permits unattended execution, or the mask would
            # not have offered this arm. propose() still derives the flag.
            return self._from_report(report, severity)

        # ESCALATE. At observe-only the agent may tell a human and nothing else.
        if tier.rank < RiskTier.RECOMMEND.rank:
            return ActionType.NOTIFY_ANALYST
        recommended = self._from_report(report, severity, require_recommendation=True)
        return recommended if recommended is not None else ActionType.NOTIFY_ANALYST

    def _from_report(
        self,
        report: InvestigationReport | None,
        severity: Severity,
        *,
        require_recommendation: bool = False,
    ) -> ActionType | None:
        """The action the investigation recommends, if it recommends a real one."""
        if report is not None:
            for candidate in report.recommended_actions:
                if candidate.is_destructive:
                    return candidate
        if require_recommendation:
            return None
        # No investigation to draw on. Isolating the host is the broadest
        # containment available and therefore the wrong default; blocking the
        # source address is reversible and proportionate.
        return (
            ActionType.ISOLATE_HOST
            if severity >= Severity.CRITICAL
            else ActionType.BLOCK_IP
        )

    def _target(self, alert: Alert, action_type: ActionType) -> str:
        field_name = _TARGET_FIELD[action_type]
        value = getattr(alert, field_name, None)
        if value is None:
            value = alert.asset_id
        text = str(value).strip()
        if not text:
            raise ContainmentError(
                f"cannot propose {action_type.value}: alert {alert.alert_id} carries no "
                f"{field_name}, and an action with an empty target is an action aimed "
                "at everything"
            )
        return text[:256]

    def _rationale(
        self, response: ResponseAction, action_type: ActionType, triage, confidence: float
    ) -> str:
        return (
            f"Policy chose {response.value} (confidence {confidence:.2f}) for a "
            f"{triage.severity.value} alert triaged {triage.decision.value}"
            + (f" and mapped to {triage.technique_id}" if triage.technique_id else "")
            + f". Proposed action: {action_type.value}."
        )[:4000]

    def _evidence(self, report: InvestigationReport | None) -> tuple[Evidence, ...]:
        """Carry the investigation's citations onto the action request.

        The analyst approving this sees the evidence on the object they are
        approving, not in a separate view they have to go and find. An approval
        UI that makes the reviewer navigate away to see why is an approval UI
        that collects clicks rather than judgements.
        """
        if report is None:
            return ()
        return report.evidence[:16]


_RESPONSE_FOR_DECISION: Final[dict[object, ResponseAction]] = {}


def _build_decision_map() -> None:
    from sentinel.core.schemas import TriageDecision

    _RESPONSE_FOR_DECISION.update(
        {
            TriageDecision.ESCALATE: ResponseAction.ESCALATE,
            TriageDecision.MONITOR: ResponseAction.MONITOR,
            TriageDecision.AUTO_DISMISS: ResponseAction.DISMISS,
        }
    )


_build_decision_map()

#: How much human attention each response gets. Containing and escalating both put
#: the alert in front of a human or act on it; monitoring records it; dismissing
#: drops it. The triage floor compares on this, never on reward.
_ATTENTION: Final[dict[ResponseAction, int]] = {
    ResponseAction.AUTO_CONTAIN: 2,
    ResponseAction.ESCALATE: 2,
    ResponseAction.MONITOR: 1,
    ResponseAction.DISMISS: 0,
}


# --------------------------------------------------------------------------- #
# F-08, checked from the log rather than from the code
# --------------------------------------------------------------------------- #


def verify_no_ungated_execution(
    log: HashChainedAuditLog, *, tenant_id: str | None = None
) -> tuple[str, ...]:
    """Return the action ids that executed without a logged approval.

    F-08's acceptance criterion read back out of the audit trail, which is where
    a compliance reviewer would read it. This deliberately does not consult the
    :class:`~sentinel.core.schemas.ActionRequest` objects: they enforce the rule,
    so asking them whether the rule held is circular. The log is an independent
    record, and an empty return value means the record agrees.

    An execution is ungated when an ``action_executed`` row exists for an action
    whose payload says approval was required, and no ``approval_granted`` row for
    the same action precedes it. Ordering matters: an approval logged *after* the
    execution is not an approval, it is a cover story.
    """
    granted: dict[str, int] = {}
    ungated: list[str] = []
    for record in log.iter_records():
        if tenant_id is not None and record.tenant_id != tenant_id:
            continue
        if record.event_type is AuditEventType.APPROVAL_GRANTED:
            granted.setdefault(record.subject_id, record.seq)
        elif record.event_type is AuditEventType.ACTION_EXECUTED:
            required = bool(record.payload.get("requires_human_approval", True))
            if not required:
                continue
            approval_seq = granted.get(record.subject_id)
            if approval_seq is None or approval_seq > record.seq:
                ungated.append(record.subject_id)
    return tuple(ungated)


def pending_action(actions: Sequence[ActionRequest]) -> ActionRequest | None:
    """The newest action still awaiting a decision, or ``None``.

    Newest first because a run proposes at most one action per pass, so the last
    pending one is the one the current routing decision is about.
    """
    for action in reversed(actions):
        if action.approval_status is ApprovalStatus.PENDING:
            return action
    return None


def executable_action(actions: Sequence[ActionRequest]) -> ActionRequest | None:
    """The newest action that may be executed now, or ``None``.

    Two cases qualify, and missing the second one is a real bug rather than a
    conservative omission:

    1.  An action a human **approved**.
    2.  An action still ``PENDING`` whose ``requires_human_approval`` is false —
        which is what an ``auto_with_notify`` or ``autonomous`` trust tier produces.
        Such an action never becomes ``APPROVED``, because nobody approves it.

    This lives here, beside :func:`verify_no_ungated_execution`, because it is the
    positive form of the same predicate and there must be exactly one of it. It was
    previously written out longhand in each graph, and the third copy — the
    code-scan graph's — omitted case 2, so an autonomous-tier scan failed its run
    with *"the PR node was reached with no approved action"* while the routing that
    sent it there was correct. Three copies of a security predicate means the rule
    holds in three implementations, and only one of them had a test.

    Note what this does **not** do: it never relaxes
    :meth:`~sentinel.core.schemas.ActionRequest.mark_executed`, which still refuses
    to execute a human-gated action that has not been approved. This function
    chooses a candidate; the schema decides whether it may run.
    """
    for action in reversed(actions):
        if action.approval_status is ApprovalStatus.APPROVED:
            return action
        if (
            action.approval_status is ApprovalStatus.PENDING
            and not action.requires_human_approval
        ):
            return action
    return None


def approval_queue(actions: Sequence[ActionRequest]) -> tuple[ActionRequest, ...]:
    """The actions waiting on a human, newest first. The dashboard's data source."""
    pending = [
        action
        for action in actions
        if action.requires_human_approval
        and action.approval_status is ApprovalStatus.PENDING
    ]
    return tuple(sorted(pending, key=lambda a: a.created_at, reverse=True))
