"""Part 5.1: the response policy in the live flow, and the model lab.

Three properties matter here and each has a test:

1. The live Containment Agent consults the trained policy, and records what it chose.
2. The **triage floor**: the policy may raise the level of attention above triage,
   never lower it — an escalated alert cannot be dismissed by a learned policy.
3. The Models page reports what training actually did, including a result that
   makes a model look bad (diffusion augmentation does not help on this corpus).
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pytest

from sentinel.agents.contain import ContainmentAgent
from sentinel.core.schemas import (
    ActionType,
    Alert,
    InvestigationReport,
    RiskTier,
    Severity,
    TriageDecision,
    TriageResult,
)
from sentinel.dashboard.lab import (
    DiffusionStudy,
    ServingPolicy,
    train_response_policy,
)
from sentinel.rl.actions import ResponseAction
from sentinel.rl.bandit import Decision
from sentinel.rl.simulate import N_FEATURES

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)


def _triaged(alert: Alert, decision: TriageDecision) -> Alert:
    return alert.with_triage(
        TriageResult(
            severity=Severity.HIGH,
            confidence=0.9,
            decision=decision,
            technique_id="T1110",
            supporting_fields=("dst_port",),
            rationale="fixture",
            anomaly_score=0.97,
            model_version="triage-test",
            latency_ms=1.0,
            decided_at=NOW,
        )
    )


def _report(alert: Alert) -> InvestigationReport:
    return InvestigationReport(
        report_id="r1",
        alert_ids=(alert.alert_id,),
        tenant_id=alert.tenant_id,
        summary="fixture",
        severity=Severity.HIGH,
        confidence=0.9,
        recommended_actions=(ActionType.BLOCK_IP,),
        model_version="investigation-test",
        created_at=NOW,
    )


class FixedPolicy:
    """A policy that always prefers one response — the adversarial case for the floor."""

    def __init__(self, action: ResponseAction) -> None:
        self.action = action

    def select(self, context, *, tier, greedy=False) -> Decision:
        from sentinel.rl.actions import ActionMask

        allowed = ActionMask.for_tier(tier).allowed
        values = {a: (1.0 if a is self.action else 0.0) for a in allowed}
        return Decision(action=self.action, tier=tier, sampled_values=values,
                        expected_values=values, uncertainties=dict.fromkeys(allowed, 0.1))


class TestTriageFloor:
    def test_without_the_floor_a_policy_can_dismiss_an_escalation(self, alert):
        triaged = _triaged(alert, TriageDecision.ESCALATE)
        proposal = ContainmentAgent(policy=FixedPolicy(ResponseAction.DISMISS)).propose(
            triaged, report=_report(triaged), tier=RiskTier.RECOMMEND, now=NOW
        )
        assert proposal.response is ResponseAction.DISMISS
        assert proposal.action is None

    def test_with_the_floor_an_escalation_stays_escalated(self, alert):
        triaged = _triaged(alert, TriageDecision.ESCALATE)
        agent = ContainmentAgent(policy=FixedPolicy(ResponseAction.DISMISS), triage_floor=True)
        proposal = agent.propose(triaged, report=_report(triaged), tier=RiskTier.RECOMMEND,
                                 now=NOW)
        assert proposal.policy_choice is ResponseAction.DISMISS, "what the policy wanted"
        assert proposal.response is ResponseAction.ESCALATE, "what the system did"
        assert proposal.floored
        assert proposal.action is not None and proposal.action.requires_human_approval
        view = proposal.policy_view()
        assert view["choice"] == "dismiss" and view["response"] == "escalate"
        assert view["floored"] is True

    def test_the_floor_never_lowers_a_more_attentive_choice(self, alert):
        # Triage only monitored; the policy escalates. That is allowed and kept.
        triaged = _triaged(alert, TriageDecision.MONITOR)
        agent = ContainmentAgent(policy=FixedPolicy(ResponseAction.ESCALATE), triage_floor=True)
        proposal = agent.propose(triaged, report=_report(triaged), tier=RiskTier.RECOMMEND,
                                 now=NOW)
        assert proposal.response is ResponseAction.ESCALATE
        assert not proposal.floored
        assert proposal.action is not None and proposal.action.action_type is ActionType.BLOCK_IP

    def test_monitor_is_floored_from_dismiss(self, alert):
        triaged = _triaged(alert, TriageDecision.MONITOR)
        agent = ContainmentAgent(policy=FixedPolicy(ResponseAction.DISMISS), triage_floor=True)
        proposal = agent.propose(triaged, report=_report(triaged), tier=RiskTier.RECOMMEND,
                                 now=NOW)
        assert proposal.response is ResponseAction.MONITOR and proposal.floored

    def test_no_policy_is_reported_as_unfitted(self, alert):
        triaged = _triaged(alert, TriageDecision.ESCALATE)
        proposal = ContainmentAgent().propose(triaged, report=_report(triaged),
                                              tier=RiskTier.RECOMMEND, now=NOW)
        assert proposal.policy_view()["fitted"] is False
        assert proposal.policy_choice is None


class TestResponsePolicyTraining:
    @pytest.fixture(scope="class")
    def trained(self):
        return train_response_policy(seed=20260928, episodes=400)

    def test_it_learns_better_than_a_policy_that_learns_nothing(self, trained):
        _bandit, training = trained
        assert training.policy_total_regret < 0.5 * training.baseline_total_regret
        curve = training.policy_cumulative_regret
        assert len(curve) == 400 and all(b >= a for a, b in pairwise(curve))
        assert sum(training.action_counts.values()) == 400

    def test_the_curve_is_downsampled_for_the_page_but_keeps_the_end(self, trained):
        _bandit, training = trained
        view = training.as_dict()
        assert len(view["curve"]["policy"]) <= 200
        assert view["curve"]["policy"][-1] == pytest.approx(training.policy_total_regret)
        assert view["curve"]["episode"][-1] == 400
        assert 0 < view["regret_ratio"] < 1

    def test_serving_is_greedy_and_deterministic(self, trained):
        bandit, _training = trained
        serving = ServingPolicy(bandit)
        context = np.linspace(0.1, 0.9, N_FEATURES)
        context[0] = 1.0
        first = serving.select(context, tier=RiskTier.RECOMMEND)
        again = serving.select(context, tier=RiskTier.RECOMMEND, greedy=False)
        assert first.action is again.action
        assert not first.was_exploratory
        assert serving.decisions == 2

    def test_training_is_reproducible(self):
        _, one = train_response_policy(seed=7, episodes=120)
        _, two = train_response_policy(seed=7, episodes=120)
        assert one.policy_cumulative_regret == two.policy_cumulative_regret


class TestDiffusionStudy:
    def test_a_small_study_runs_and_reports_both_directions(self):
        study = DiffusionStudy(seed=20260928, n_alerts=3000, levels=(None,), epochs=40)
        assert study.view()["status"] == "pending"
        study.start()
        study.join(timeout=600)
        view = study.view()
        assert view["status"] == "done", view["error"]
        assert view["levels_done"] == 1
        row = view["results"][0]
        assert set(row["before"]) == set(view["rare_families"])
        for value in (*row["before"].values(), *row["after"].values()):
            assert 0.0 <= value <= 1.0
        assert row["synthetic_rows"] > 0
        assert view["loss_curve"] and view["loss_curve"][-1] < view["loss_curve"][0]

    def test_a_failure_is_reported_not_raised(self):
        study = DiffusionStudy(seed=1, n_alerts=-5, levels=(None,), epochs=2)
        study.run()
        view = study.view()
        assert view["status"] == "failed" and view["error"]

    def test_start_is_idempotent(self):
        study = DiffusionStudy(seed=1, n_alerts=-5, levels=(None,), epochs=2)
        assert study.start() is study.start()
        study.join(timeout=60)
