"""The response policy inside the real pipeline (PRD F-08, F-09, Section 5.7).

The unit tests prove the bandit learns and that its mask holds. What they cannot prove
is that a policy decision turns into something the rest of the system will accept, and
that the two independent safety layers agree. That is what this module is for.

The layers, and why there are two:

*   :class:`~sentinel.rl.actions.ActionMask` stops a forbidden action being *chosen*.
    It is the layer PRD Section 5.5.4 defers to Phase 3 and this build implements now.
*   :meth:`~sentinel.core.schemas.ActionRequest.propose` stops an ungated destructive
    action being *constructed*, whatever a policy asks for. Part 1 built it.

:class:`TestTheTwoLayersAgree` checks the interesting case: that they never disagree
about the same tier. Two safety controls that disagree are worse than one, because
whichever is consulted first becomes the real policy and nobody wrote that down.

:class:`TestEvidenceCannotEscalateAuthority` closes the loop with Part 2.4: a
retrieved playbook chunk saying "isolate the host immediately" is guidance, and the
knowledge base is an attacker-influenceable channel, so it must not be able to move an
action past the gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.canonical import canonical_json
from sentinel.core.schemas import (
    ActionRequest,
    ActionType,
    AgentName,
    Alert,
    ApprovalStatus,
    AuditEventType,
    RiskTier,
    Severity,
    TriageDecision,
    TriageResult,
)
from sentinel.kb.retrieve import KnowledgeBase
from sentinel.rl.actions import ALL_ACTIONS, ActionMask, ResponseAction
from sentinel.rl.bandit import LinearThompsonBandit
from sentinel.rl.reward import Outcome, RewardModel
from sentinel.rl.simulate import N_FEATURES, build_episodes, replay

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
ALL_TIERS = tuple(RiskTier)


@pytest.fixture
def policy() -> LinearThompsonBandit:
    """A policy trained enough to have opinions."""
    bandit = LinearThompsonBandit(n_features=N_FEATURES, seed=101)
    replay(episodes=build_episodes(n=200, seed=101), bandit=bandit, seed=101)
    return bandit


class TestTheTwoLayersAgree:
    """Two safety controls that disagree are worse than one."""

    @pytest.mark.parametrize("tier", ALL_TIERS)
    def test_mask_and_schema_agree_on_containment(self, tier: RiskTier) -> None:
        """If the mask permits containment, the schema must not demand approval.

        And vice versa. Both derive from ``permits_unattended_execution``, and this is
        the test that notices if one of them stops doing so.
        """
        mask_permits = ActionMask.for_tier(tier).permits(ResponseAction.AUTO_CONTAIN)
        request = ActionRequest.propose(
            alert_id="alert-1",
            tenant_id="acme",
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.ISOLATE_HOST,
            target="host-42",
            rationale="Policy selected containment.",
            risk_tier=tier,
            created_at=NOW,
        )
        schema_permits_unattended = not request.requires_human_approval
        assert mask_permits == schema_permits_unattended

    @pytest.mark.parametrize("tier", [RiskTier.OBSERVE, RiskTier.RECOMMEND])
    def test_low_tiers_gate_every_destructive_action_type(self, tier: RiskTier) -> None:
        for action_type in ResponseAction.AUTO_CONTAIN.implied_action_types:
            request = ActionRequest.propose(
                alert_id="alert-1",
                tenant_id="acme",
                proposed_by=AgentName.CONTAINMENT,
                action_type=action_type,
                target="host-42",
                rationale="Policy selected containment.",
                risk_tier=tier,
                created_at=NOW,
            )
            assert request.requires_human_approval

    def test_a_policy_decision_never_yields_an_ungated_action_at_low_tiers(
        self, policy: LinearThompsonBandit
    ) -> None:
        """The end-to-end property, over many decisions at every restrictive tier."""
        rng = np.random.default_rng(3)
        for tier in (RiskTier.OBSERVE, RiskTier.RECOMMEND):
            for _ in range(200):
                vector = np.abs(rng.normal(size=N_FEATURES))
                vector[0] = 1.0
                decision = policy.select(vector, tier=tier)
                assert decision.action is not ResponseAction.AUTO_CONTAIN
                for action_type in decision.action.implied_action_types:
                    # Nothing reachable at these tiers is destructive, and the request
                    # is built anyway to prove the schema accepts what the mask allows.
                    ActionRequest.propose(
                        alert_id="alert-1",
                        tenant_id="acme",
                        proposed_by=AgentName.CONTAINMENT,
                        action_type=action_type,
                        target="host-42",
                        rationale="Policy decision.",
                        risk_tier=tier,
                        created_at=NOW,
                    )
                    assert not action_type.is_destructive


class TestDecisionBecomesATriageResult:
    def test_every_action_maps_onto_a_constructible_triage_result(
        self, policy: LinearThompsonBandit
    ) -> None:
        """A decision the schema rejects is a decision the orchestrator cannot record."""
        vector = np.zeros(N_FEATURES)
        vector[0] = 1.0
        for action in ALL_ACTIONS:
            result = TriageResult(
                severity=Severity.HIGH if action.intervenes else Severity.MEDIUM,
                confidence=0.85,
                decision=action.triage_decision,
                rationale=f"Response policy selected {action.value}.",
                supporting_fields=("anomaly_score",),
                model_version="bandit-2.5",
                latency_ms=1.0,
                decided_at=NOW,
            )
            assert result.decision is action.triage_decision

    def test_a_low_confidence_dismissal_is_still_impossible(self) -> None:
        """Part 1's floor must survive the policy layer.

        The bandit can select ``dismiss`` with any confidence it likes; the schema is
        what refuses to record an under-confident dismissal, and that has to keep
        holding now that something other than a prompt is choosing the action.
        """
        with pytest.raises(ValueError):
            TriageResult(
                severity=Severity.LOW,
                confidence=0.2,
                decision=TriageDecision.AUTO_DISMISS,
                rationale="Policy selected dismiss with low confidence.",
                supporting_fields=("anomaly_score",),
                model_version="bandit-2.5",
                latency_ms=1.0,
                decided_at=NOW,
            )

    def test_policy_confidence_can_gate_the_dismissal(
        self, policy: LinearThompsonBandit
    ) -> None:
        """The intended composition: use the policy's own margin as the confidence.

        ``Decision.confidence`` is a squashed margin, not a calibrated probability, so
        the schema floor remains the control. This asserts the two compose rather than
        that the margin is trustworthy.
        """
        vector = np.zeros(N_FEATURES)
        vector[0] = 1.0
        decision = policy.select(vector, tier=RiskTier.AUTONOMOUS, greedy=True)
        floor = TriageResult.DISMISS_CONFIDENCE_FLOOR
        if decision.action is ResponseAction.DISMISS and decision.confidence < floor:
            with pytest.raises(ValueError):
                TriageResult(
                    severity=Severity.LOW,
                    confidence=decision.confidence,
                    decision=TriageDecision.AUTO_DISMISS,
                    rationale="Policy dismissal below the schema floor.",
                    supporting_fields=("anomaly_score",),
                    model_version="bandit-2.5",
                    latency_ms=1.0,
                    decided_at=NOW,
                )
        else:
            assert 0.0 <= decision.confidence <= 1.0


class TestContainmentStillRequiresApproval:
    def test_a_selected_containment_at_recommend_tier_cannot_execute(self) -> None:
        """F-08: *zero actions executed without a logged approval event*."""
        request = ActionRequest.propose(
            alert_id="alert-1",
            tenant_id="acme",
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.ISOLATE_HOST,
            target="host-42",
            rationale="Policy recommended containment.",
            risk_tier=RiskTier.RECOMMEND,
            created_at=NOW,
        )
        assert request.requires_human_approval
        assert request.approval_status is ApprovalStatus.PENDING

        # Constructing the forbidden state directly is refused: an executed,
        # human-gated action with no approver is exactly what F-08 rules out.
        with pytest.raises(ValueError, match="F-08 forbids"):
            ActionRequest(
                action_id="act-1",
                alert_id="alert-1",
                tenant_id="acme",
                proposed_by=AgentName.CONTAINMENT,
                action_type=ActionType.ISOLATE_HOST,
                target="host-42",
                rationale="Policy recommended containment.",
                risk_tier=RiskTier.RECOMMEND,
                requires_human_approval=True,
                approval_status=ApprovalStatus.EXECUTED,
                executed_at=NOW,
                created_at=NOW,
            )

    def test_model_copy_bypasses_validation_and_must_not_be_used_to_advance_state(
        self,
    ) -> None:
        """A pydantic behaviour worth pinning rather than discovering later.

        ``model_copy(update=...)`` does not re-run validators -- that is documented
        pydantic behaviour, not a bug here, and ``validate_assignment`` does not cover
        it because no assignment happens. So the approval gate is enforced at
        *construction*, and any code advancing an ``ActionRequest`` through its
        lifecycle has to build a new one rather than copy-with-update. Asserted so the
        constraint is visible to whoever writes the approval flow in Part 3.
        """
        request = ActionRequest.propose(
            alert_id="alert-1",
            tenant_id="acme",
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.ISOLATE_HOST,
            target="host-42",
            rationale="Policy recommended containment.",
            risk_tier=RiskTier.RECOMMEND,
            created_at=NOW,
        )
        bypassed = request.model_copy(
            update={"approval_status": ApprovalStatus.EXECUTED}
        )
        assert bypassed.approval_status is ApprovalStatus.EXECUTED
        # Re-validating the same data is what rejects it.
        with pytest.raises(ValueError):
            ActionRequest.model_validate(bypassed.model_dump())

    def test_containment_at_the_top_tier_is_still_recorded_as_an_action(self) -> None:
        request = ActionRequest.propose(
            alert_id="alert-1",
            tenant_id="acme",
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.ISOLATE_HOST,
            target="host-42",
            rationale="Policy selected containment at an autonomous tier.",
            risk_tier=RiskTier.AUTONOMOUS,
            created_at=NOW,
        )
        assert not request.requires_human_approval
        assert request.action_type.is_destructive


class TestEvidenceCannotEscalateAuthority:
    """Part 2.4 meets Part 2.5: retrieved guidance is not permission."""

    @pytest.fixture(scope="class")
    def kb(self) -> KnowledgeBase:
        return KnowledgeBase.build()

    def test_a_playbook_urging_immediate_isolation_does_not_lift_the_gate(
        self, kb: KnowledgeBase
    ) -> None:
        evidence = kb.evidence_for(
            "isolate the host immediately without waiting for analyst review", k=3
        )
        assert evidence
        request = ActionRequest.propose(
            alert_id="alert-1",
            tenant_id="acme",
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.ISOLATE_HOST,
            target="host-42",
            rationale="A retrieved playbook recommends immediate isolation.",
            risk_tier=RiskTier.RECOMMEND,
            created_at=NOW,
            evidence=evidence,
        )
        assert request.requires_human_approval

    def test_the_mask_ignores_evidence_entirely(self, kb: KnowledgeBase) -> None:
        """Authority comes from the tier, not from content. Structurally."""
        import inspect

        parameters = inspect.signature(ActionMask.require).parameters
        assert set(parameters) == {"self", "action"}
        # There is no channel by which evidence could reach the mask.
        assert kb.evidence_for("isolate the host immediately", k=1)


class TestPolicyDecisionsAreAuditable:
    def test_a_decision_is_recorded_with_its_alternatives(
        self, policy: LinearThompsonBandit, audit_log: HashChainedAuditLog, alert: Alert
    ) -> None:
        """An analyst must be able to see what the policy considered, not just chose."""
        vector = np.zeros(N_FEATURES)
        vector[0] = 1.0
        vector[1] = 0.9
        decision = policy.select(vector, tier=RiskTier.AUTO_WITH_NOTIFY)
        audit_log.append(
            event_type=AuditEventType.POLICY_UPDATED,
            actor=AgentName.ORCHESTRATOR,
            tenant_id=alert.tenant_id,
            subject_id=alert.alert_id,
            payload={
                "chosen": decision.action.value,
                "tier": decision.tier.value,
                "was_exploratory": decision.was_exploratory,
                "confidence": decision.confidence,
                "expected_values": {
                    a.value: v for a, v in decision.expected_values.items()
                },
                "uncertainties": {
                    a.value: v for a, v in decision.uncertainties.items()
                },
            },
        )
        audit_log.verify()
        row = next(iter(audit_log.iter_records()))
        assert row.payload["chosen"] == decision.action.value
        assert set(row.payload["expected_values"]) == {
            a.value for a in ActionMask.for_tier(RiskTier.AUTO_WITH_NOTIFY).allowed
        }

    def test_forbidden_arms_never_appear_in_the_audit_payload(
        self, policy: LinearThompsonBandit
    ) -> None:
        """A score logged for a forbidden arm invites someone to act on it."""
        vector = np.zeros(N_FEATURES)
        vector[0] = 1.0
        decision = policy.select(vector, tier=RiskTier.OBSERVE)
        payload = {
            "expected_values": {a.value: v for a, v in decision.expected_values.items()}
        }
        assert "auto_contain" not in payload["expected_values"]
        assert "dismiss" not in payload["expected_values"]

    def test_the_decision_payload_is_canonically_serializable(
        self, policy: LinearThompsonBandit
    ) -> None:
        vector = np.zeros(N_FEATURES)
        vector[0] = 1.0
        decision = policy.select(vector, tier=RiskTier.AUTONOMOUS)
        payload = {
            "chosen": decision.action.value,
            "confidence": decision.confidence,
            "sampled": {a.value: v for a, v in decision.sampled_values.items()},
        }
        assert canonical_json(payload)

    def test_the_policy_snapshot_is_loggable(self, policy: LinearThompsonBandit) -> None:
        assert canonical_json(policy.snapshot())


class TestRewardNeedsTheAnalystsVerdict:
    def test_the_reward_is_computed_from_an_outcome_not_a_prediction(
        self, policy: LinearThompsonBandit
    ) -> None:
        """The policy cannot score itself, which is the point of the feedback loop.

        The reward needs ``is_attack``, which only arrives later, from the dataset
        label in simulation and from an analyst in production. Nothing on the policy's
        own path can produce it.
        """
        model = RewardModel()
        vector = np.zeros(N_FEATURES)
        vector[0] = 1.0
        decision = policy.select(vector, tier=RiskTier.AUTONOMOUS)
        # Scoring requires a verdict the policy never saw.
        reward = model.reward(
            Outcome(action=decision.action, is_attack=True, criticality=0.5)
        )
        assert isinstance(reward, float)
        policy.update(vector, decision.action, reward, tier=RiskTier.AUTONOMOUS)

    def test_a_reversal_only_arrives_after_a_containment(self) -> None:
        """Modelling a reversal of anything else would be modelling a fiction."""
        with pytest.raises(Exception, match="only applies to auto_contain"):
            Outcome(ResponseAction.ESCALATE, False, 0.5, human_reverses=True)


class TestTierPromotionHandsOverAColdArm:
    """The documented cost of structural masking, asserted rather than asserted-away."""

    def test_promotion_exposes_an_unlearned_arm(self) -> None:
        policy = LinearThompsonBandit(n_features=N_FEATURES, seed=113)
        episodes = build_episodes(n=200, seed=113, tier=RiskTier.RECOMMEND)
        replay(episodes=episodes, bandit=policy, seed=113)
        assert policy.arms[ResponseAction.AUTO_CONTAIN].observations == 0
        assert ResponseAction.AUTO_CONTAIN in policy.cold_arms()

    def test_the_cold_arm_becomes_learnable_after_promotion(self) -> None:
        """Promotion must open the arm, and the optimistic prior must then try it."""
        policy = LinearThompsonBandit(n_features=N_FEATURES, seed=127)
        replay(
            episodes=build_episodes(n=200, seed=127, tier=RiskTier.RECOMMEND),
            bandit=policy,
            seed=127,
        )
        before = policy.arms[ResponseAction.AUTO_CONTAIN].observations
        replay(
            episodes=build_episodes(n=200, seed=128, tier=RiskTier.AUTO_WITH_NOTIFY),
            bandit=policy,
            seed=128,
        )
        after = policy.arms[ResponseAction.AUTO_CONTAIN].observations
        assert before == 0
        assert after > 0, "the optimistic prior should try a newly opened arm"

    def test_learning_transfers_across_a_promotion(self) -> None:
        """What was learned at the lower tier must not be discarded by promotion."""
        policy = LinearThompsonBandit(n_features=N_FEATURES, seed=131)
        replay(
            episodes=build_episodes(n=200, seed=131, tier=RiskTier.RECOMMEND),
            bandit=policy,
            seed=131,
        )
        learned = policy.arms[ResponseAction.DISMISS].observations
        assert learned > 0
        replay(
            episodes=build_episodes(n=100, seed=132, tier=RiskTier.AUTONOMOUS),
            bandit=policy,
            seed=132,
        )
        assert policy.arms[ResponseAction.DISMISS].observations > learned
