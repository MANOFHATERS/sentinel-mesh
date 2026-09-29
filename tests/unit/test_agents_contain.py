"""The Containment Agent, its guardrails, and the F-08 log check.

The interesting tests here are the negative ones. ``ActionRequest`` already
refuses to be constructed ungated, so this file does not re-test the schema; it
tests the agent's *choices* — which action type, at which tier, with which
target — and the audit-log reader that checks F-08 without consulting the
objects that enforce it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from sentinel.agents.contain import (
    ContainmentAgent,
    ContainmentError,
    SimulatedConnector,
    approval_queue,
    verify_no_ungated_execution,
)
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import (
    ActionRequest,
    ActionType,
    AgentName,
    Alert,
    ApprovalStatus,
    AuditEventType,
    InvestigationReport,
    RiskTier,
    Severity,
    TriageDecision,
    TriageResult,
)
from sentinel.rl.actions import PolicyError, ResponseAction
from sentinel.rl.bandit import LinearThompsonBandit
from sentinel.rl.simulate import N_FEATURES, build_context

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def _triaged(
    alert: Alert,
    *,
    decision: TriageDecision = TriageDecision.ESCALATE,
    severity: Severity = Severity.HIGH,
    score: float = 0.95,
    technique: str | None = "T1110",
) -> Alert:
    return alert.with_triage(
        TriageResult(
            severity=severity,
            confidence=0.9,
            decision=decision,
            technique_id=technique,
            supporting_fields=("dst_port",) if technique else (),
            rationale="fixture",
            anomaly_score=score,
            model_version="triage-test",
            latency_ms=1.0,
            decided_at=NOW,
        )
    )


def _report(alert: Alert, actions: tuple[ActionType, ...]) -> InvestigationReport:
    return InvestigationReport(
        report_id="r1",
        alert_ids=(alert.alert_id,),
        tenant_id=alert.tenant_id,
        summary="fixture",
        severity=Severity.HIGH,
        confidence=0.9,
        recommended_actions=actions,
        model_version="investigation-test",
        created_at=NOW,
    )


# --------------------------------------------------------------------------- #


class TestTierBehaviour:
    def test_recommend_tier_proposes_a_gated_destructive_action(self, alert: Alert) -> None:
        """The default tier every customer starts on must be able to reach the gate.

        The regression this pins: with ``auto_contain`` masked out at
        ``recommend`` and ``escalate`` mapped only to ``notify_analyst``, the
        Human Approval Gate was unreachable in the default configuration and
        F-08 passed on a system that had never gated anything. Measured then:
        200 incidents, 133 executions, 0 approval requests.
        """
        triaged = _triaged(alert)
        proposal = ContainmentAgent().propose(
            triaged,
            report=_report(triaged, (ActionType.BLOCK_IP,)),
            tier=RiskTier.RECOMMEND,
            now=NOW,
        )
        assert proposal.action is not None
        assert proposal.action.action_type is ActionType.BLOCK_IP
        assert proposal.action.requires_human_approval
        assert proposal.needs_human

    def test_observe_tier_only_notifies(self, alert: Alert) -> None:
        triaged = _triaged(alert)
        proposal = ContainmentAgent().propose(
            triaged,
            report=_report(triaged, (ActionType.ISOLATE_HOST,)),
            tier=RiskTier.OBSERVE,
            now=NOW,
        )
        assert proposal.action is not None
        assert proposal.action.action_type is ActionType.NOTIFY_ANALYST
        assert not proposal.action.action_type.is_destructive
        assert not proposal.action.requires_human_approval

    def test_a_monitor_verdict_proposes_enrichment_only(self, alert: Alert) -> None:
        triaged = _triaged(alert, decision=TriageDecision.MONITOR)
        proposal = ContainmentAgent().propose(triaged, tier=RiskTier.RECOMMEND, now=NOW)
        assert proposal.action is not None
        assert proposal.action.action_type is ActionType.ENRICH_ONLY

    def test_a_dismissal_proposes_nothing(self, alert: Alert) -> None:
        triaged = _triaged(alert, decision=TriageDecision.AUTO_DISMISS, severity=Severity.LOW)
        proposal = ContainmentAgent().propose(triaged, tier=RiskTier.RECOMMEND, now=NOW)
        assert proposal.action is None
        assert proposal.response is ResponseAction.DISMISS
        assert not proposal.needs_human

    def test_an_unattended_tier_does_not_require_approval(self, alert: Alert) -> None:
        """What the top tiers mean: propose() derives the flag, and it flips."""
        triaged = _triaged(alert)
        proposal = ContainmentAgent().propose(
            triaged,
            report=_report(triaged, (ActionType.BLOCK_IP,)),
            tier=RiskTier.AUTONOMOUS,
            now=NOW,
        )
        assert proposal.action is not None
        assert not proposal.action.requires_human_approval

    def test_the_agent_never_proposes_outside_its_mask(self, alert: Alert) -> None:
        from sentinel.rl.actions import ActionMask

        triaged = _triaged(alert)
        for tier in RiskTier:
            proposal = ContainmentAgent().propose(triaged, tier=tier, now=NOW)
            assert ActionMask.for_tier(tier).permits(proposal.response)


class TestActionSelection:
    def test_the_investigation_recommendation_is_honoured(self, alert: Alert) -> None:
        triaged = _triaged(alert)
        proposal = ContainmentAgent().propose(
            triaged,
            report=_report(triaged, (ActionType.ISOLATE_HOST, ActionType.BLOCK_IP)),
            tier=RiskTier.RECOMMEND,
            now=NOW,
        )
        assert proposal.action is not None
        assert proposal.action.action_type is ActionType.ISOLATE_HOST

    def test_without_an_investigation_an_escalation_only_notifies(
        self, alert: Alert
    ) -> None:
        """A destructive action needs a reason, and the reason is the report."""
        proposal = ContainmentAgent().propose(
            _triaged(alert), tier=RiskTier.RECOMMEND, now=NOW
        )
        assert proposal.action is not None
        assert proposal.action.action_type is ActionType.NOTIFY_ANALYST

    def test_a_non_destructive_recommendation_does_not_become_the_action(
        self, alert: Alert
    ) -> None:
        triaged = _triaged(alert)
        proposal = ContainmentAgent().propose(
            triaged,
            report=_report(triaged, (ActionType.ENRICH_ONLY,)),
            tier=RiskTier.RECOMMEND,
            now=NOW,
        )
        assert proposal.action is not None
        assert proposal.action.action_type is ActionType.NOTIFY_ANALYST

    def test_block_ip_targets_the_source_address(self, alert: Alert) -> None:
        """An isolate_host aimed at an IP is syntactically fine and operationally wrong."""
        triaged = _triaged(alert)
        proposal = ContainmentAgent().propose(
            triaged,
            report=_report(triaged, (ActionType.BLOCK_IP,)),
            tier=RiskTier.RECOMMEND,
            now=NOW,
        )
        assert proposal.action is not None
        assert proposal.action.target == alert.src_ip

    def test_isolate_host_targets_the_asset(self, alert: Alert) -> None:
        triaged = _triaged(alert)
        proposal = ContainmentAgent().propose(
            triaged,
            report=_report(triaged, (ActionType.ISOLATE_HOST,)),
            tier=RiskTier.RECOMMEND,
            now=NOW,
        )
        assert proposal.action is not None
        assert proposal.action.target == alert.asset_id

    def test_the_investigations_evidence_travels_with_the_action(
        self, alert: Alert
    ) -> None:
        """The approver sees why on the object they are approving."""
        from sentinel.core.schemas import Evidence, EvidenceKind

        triaged = _triaged(alert)
        report = _report(triaged, (ActionType.BLOCK_IP,)).updated(
            evidence=(
                Evidence(kind=EvidenceKind.KB_CHUNK, ref="kb://x#1", excerpt="why"),
            ),
            claims=(("because", ("kb://x#1",)),),
        )
        proposal = ContainmentAgent().propose(
            triaged, report=report, tier=RiskTier.RECOMMEND, now=NOW
        )
        assert proposal.action is not None
        assert [e.ref for e in proposal.action.evidence] == ["kb://x#1"]

    def test_an_untriaged_alert_is_refused(self, alert: Alert) -> None:
        with pytest.raises(ContainmentError, match="no triage verdict"):
            ContainmentAgent().propose(alert, now=NOW)

    def test_an_empty_target_is_refused(self, minimal_alert_kwargs) -> None:
        """An action with no target is an action aimed at everything."""
        bare = _triaged(Alert(**minimal_alert_kwargs).updated(asset_id="x"))
        agent = ContainmentAgent()
        with pytest.raises(ContainmentError, match="empty target"):
            agent._target(bare.updated(asset_id=" "), ActionType.ISOLATE_HOST)


class TestPolicyIntegration:
    def test_a_fitted_policy_drives_the_decision(self, alert: Alert) -> None:
        policy = LinearThompsonBandit(n_features=N_FEATURES, seed=3)
        triaged = _triaged(alert)
        proposal = ContainmentAgent(policy=policy).propose(
            triaged, tier=RiskTier.AUTONOMOUS, now=NOW
        )
        assert proposal.response in ResponseAction
        assert 0.0 <= proposal.policy_confidence <= 1.0

    def test_the_policy_context_is_the_shared_definition(self, alert: Alert) -> None:
        """One construction of the observation vector, or the posteriors mean nothing."""
        triaged = _triaged(alert, score=0.7)
        context = build_context(triaged, anomaly_score=0.7)
        assert context.shape == (N_FEATURES,)
        assert np.all(np.isfinite(context))

    def test_without_a_policy_the_triage_verdict_stands(self, alert: Alert) -> None:
        """No learned policy means do what the detectors said, never more."""
        for decision, expected in (
            (TriageDecision.ESCALATE, ResponseAction.ESCALATE),
            (TriageDecision.MONITOR, ResponseAction.MONITOR),
            (TriageDecision.AUTO_DISMISS, ResponseAction.DISMISS),
        ):
            triaged = _triaged(alert, decision=decision, severity=Severity.MEDIUM)
            proposal = ContainmentAgent().propose(
                triaged, tier=RiskTier.AUTONOMOUS, now=NOW
            )
            assert proposal.response is expected

    def test_a_fallback_the_tier_forbids_becomes_an_escalation(self, alert: Alert) -> None:
        """``dismiss`` needs ``recommend``; at ``observe`` it must not be taken."""
        triaged = _triaged(alert, decision=TriageDecision.AUTO_DISMISS, severity=Severity.LOW)
        proposal = ContainmentAgent().propose(triaged, tier=RiskTier.OBSERVE, now=NOW)
        assert proposal.response is ResponseAction.ESCALATE

    def test_the_mask_is_asserted_after_selection(self, alert: Alert) -> None:
        """The invariant the mask exists for, checked at the point of use."""

        class RogueBandit:
            def select(self, context, *, tier, greedy=False):
                from sentinel.rl.bandit import Decision

                return Decision(
                    action=ResponseAction.AUTO_CONTAIN,
                    tier=tier,
                    sampled_values={ResponseAction.AUTO_CONTAIN: 1.0},
                    expected_values={ResponseAction.AUTO_CONTAIN: 1.0},
                    uncertainties={ResponseAction.AUTO_CONTAIN: 0.0},
                )

        agent = ContainmentAgent(policy=RogueBandit())  # type: ignore[arg-type]
        with pytest.raises(PolicyError, match="requires tier"):
            agent.propose(_triaged(alert), tier=RiskTier.RECOMMEND, now=NOW)


class TestSimulatedConnector:
    def test_it_executes_an_approved_action(self, alert: Alert) -> None:
        action = ActionRequest.propose(
            alert_id=alert.alert_id,
            tenant_id=alert.tenant_id,
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.BLOCK_IP,
            target="10.0.0.1",
            rationale="test",
            risk_tier=RiskTier.RECOMMEND,
            created_at=NOW,
        ).approve(approver="analyst", at=NOW)
        connector = SimulatedConnector()
        assert connector.execute(action).succeeded
        assert connector.executed == [action]

    def test_it_refuses_an_unapproved_action(self, alert: Alert) -> None:
        action = ActionRequest.propose(
            alert_id=alert.alert_id,
            tenant_id=alert.tenant_id,
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.BLOCK_IP,
            target="10.0.0.1",
            rationale="test",
            risk_tier=RiskTier.RECOMMEND,
            created_at=NOW,
        )
        connector = SimulatedConnector()
        with pytest.raises(GuardrailViolation):
            connector.execute(action)
        assert connector.executed == []

    def test_it_executes_an_ungated_action_at_an_unattended_tier(
        self, alert: Alert
    ) -> None:
        action = ActionRequest.propose(
            alert_id=alert.alert_id,
            tenant_id=alert.tenant_id,
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.BLOCK_IP,
            target="10.0.0.1",
            rationale="test",
            risk_tier=RiskTier.AUTONOMOUS,
            created_at=NOW,
        )
        assert SimulatedConnector().execute(action).succeeded


class TestF08LogCheck:
    """``verify_no_ungated_execution`` reads the log, not the objects."""

    def test_a_clean_log_reports_nothing(self, audit_log) -> None:
        audit_log.append(
            AuditEventType.APPROVAL_GRANTED,
            actor="orchestrator",
            tenant_id="acme",
            subject_id="action-1",
            payload={"approver": "dana"},
        )
        audit_log.append(
            AuditEventType.ACTION_EXECUTED,
            actor="containment_agent",
            tenant_id="acme",
            subject_id="action-1",
            payload={"requires_human_approval": True},
        )
        assert verify_no_ungated_execution(audit_log) == ()

    def test_an_execution_with_no_approval_is_reported(self, audit_log) -> None:
        audit_log.append(
            AuditEventType.ACTION_EXECUTED,
            actor="containment_agent",
            tenant_id="acme",
            subject_id="action-9",
            payload={"requires_human_approval": True},
        )
        assert verify_no_ungated_execution(audit_log) == ("action-9",)

    def test_an_approval_logged_afterwards_is_not_an_approval(self, audit_log) -> None:
        """A cover story, not an approval. Ordering is the whole point of a chain."""
        audit_log.append(
            AuditEventType.ACTION_EXECUTED,
            actor="containment_agent",
            tenant_id="acme",
            subject_id="action-7",
            payload={"requires_human_approval": True},
        )
        audit_log.append(
            AuditEventType.APPROVAL_GRANTED,
            actor="orchestrator",
            tenant_id="acme",
            subject_id="action-7",
            payload={"approver": "dana"},
        )
        assert verify_no_ungated_execution(audit_log) == ("action-7",)

    def test_an_ungated_tier_execution_is_not_a_violation(self, audit_log) -> None:
        audit_log.append(
            AuditEventType.ACTION_EXECUTED,
            actor="containment_agent",
            tenant_id="acme",
            subject_id="action-3",
            payload={"requires_human_approval": False},
        )
        assert verify_no_ungated_execution(audit_log) == ()

    def test_a_missing_flag_is_treated_as_gated(self, audit_log) -> None:
        """Absence of evidence is not evidence of an ungated tier."""
        audit_log.append(
            AuditEventType.ACTION_EXECUTED,
            actor="containment_agent",
            tenant_id="acme",
            subject_id="action-4",
            payload={},
        )
        assert verify_no_ungated_execution(audit_log) == ("action-4",)

    def test_another_tenants_rows_are_excluded(self, audit_log) -> None:
        audit_log.append(
            AuditEventType.ACTION_EXECUTED,
            actor="containment_agent",
            tenant_id="other",
            subject_id="action-5",
            payload={"requires_human_approval": True},
        )
        assert verify_no_ungated_execution(audit_log, tenant_id="acme") == ()
        assert verify_no_ungated_execution(audit_log, tenant_id="other") == ("action-5",)


class TestApprovalQueue:
    def test_lists_only_pending_gated_actions(self, alert: Alert) -> None:
        def build(action_type: ActionType, tier: RiskTier) -> ActionRequest:
            return ActionRequest.propose(
                alert_id=alert.alert_id,
                tenant_id=alert.tenant_id,
                proposed_by=AgentName.CONTAINMENT,
                action_type=action_type,
                target="t",
                rationale="r",
                risk_tier=tier,
                created_at=NOW,
            )

        gated = build(ActionType.BLOCK_IP, RiskTier.RECOMMEND)
        ungated = build(ActionType.NOTIFY_ANALYST, RiskTier.RECOMMEND)
        decided = build(ActionType.ISOLATE_HOST, RiskTier.RECOMMEND).approve(
            approver="dana", at=NOW
        )
        queue = approval_queue([gated, ungated, decided])
        assert [a.action_id for a in queue] == [gated.action_id]

    def test_is_newest_first(self, alert: Alert) -> None:
        from datetime import timedelta

        def build(offset: int) -> ActionRequest:
            return ActionRequest.propose(
                alert_id=alert.alert_id,
                tenant_id=alert.tenant_id,
                proposed_by=AgentName.CONTAINMENT,
                action_type=ActionType.BLOCK_IP,
                target="t",
                rationale="r",
                risk_tier=RiskTier.RECOMMEND,
                created_at=NOW + timedelta(seconds=offset),
            )

        old, new = build(0), build(60)
        assert approval_queue([old, new])[0].action_id == new.action_id

    def test_an_executed_action_leaves_the_queue(self, alert: Alert) -> None:
        action = ActionRequest.propose(
            alert_id=alert.alert_id,
            tenant_id=alert.tenant_id,
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.BLOCK_IP,
            target="t",
            rationale="r",
            risk_tier=RiskTier.RECOMMEND,
            created_at=NOW,
        )
        executed = action.approve(approver="dana", at=NOW).mark_executed(at=NOW)
        assert executed.approval_status is ApprovalStatus.EXECUTED
        assert approval_queue([executed]) == ()
