"""The shaped reward (:mod:`sentinel.rl.reward`).

The reward is the only place the bandit learns anything about the security domain, so
a subtly wrong ordering here produces a policy that is confidently wrong and shows no
symptom anywhere else — it will optimise whatever it was given, perfectly.

:class:`TestOrderingHoldsAcrossTheParameterSpace` is the load-bearing class. Asserting
that the *default* constants satisfy the ordering proves almost nothing: the defaults
were written to satisfy it. What matters is that the ordering survives being tuned,
which is why the invariants are checked over a grid and why
:func:`~sentinel.rl.reward.assert_ordering_invariants` lives in the module rather than
in this file.

:class:`TestRewardNeverSeesGroundTruthOnThePolicyPath` closes the loop the other way:
``Outcome.is_attack`` exists so the oracle is computable, and the one thing that must
never happen is the policy reading it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sentinel.core.schemas import RiskTier
from sentinel.rl.actions import ALL_ACTIONS, ActionMask, ResponseAction
from sentinel.rl.reward import (
    CRITICALITY_RANGE,
    Outcome,
    RewardError,
    RewardModel,
    assert_ordering_invariants,
    best_action,
)

MODEL = RewardModel()
ALL_TIERS = tuple(RiskTier)


class TestOutcomeValidation:
    def test_valid_outcome(self) -> None:
        assert Outcome(ResponseAction.DISMISS, False, 0.5).criticality == 0.5

    @pytest.mark.parametrize("criticality", [-0.01, 1.01, 2.0, -5.0])
    def test_criticality_outside_range_rejected(self, criticality: float) -> None:
        with pytest.raises(RewardError, match="outside"):
            Outcome(ResponseAction.DISMISS, False, criticality)

    @pytest.mark.parametrize("criticality", list(CRITICALITY_RANGE))
    def test_range_endpoints_accepted(self, criticality: float) -> None:
        assert Outcome(ResponseAction.MONITOR, True, criticality)

    @pytest.mark.parametrize(
        "action",
        [ResponseAction.ESCALATE, ResponseAction.MONITOR, ResponseAction.DISMISS],
    )
    def test_reversal_only_applies_to_containment(
        self, action: ResponseAction
    ) -> None:
        """There is nothing to reverse about an escalation or a dismissal."""
        with pytest.raises(RewardError, match="only applies to auto_contain"):
            Outcome(action, False, 0.5, human_reverses=True)

    def test_outcome_is_immutable(self) -> None:
        outcome = Outcome(ResponseAction.DISMISS, False, 0.5)
        with pytest.raises(AttributeError):
            outcome.is_attack = True  # type: ignore[misc]


class TestPrdRequirement:
    """PRD Section 5.5.4's one explicit ordering constraint."""

    def test_reversed_false_containment_costs_more_than_over_caution(self) -> None:
        reversed_contain = MODEL.reward(
            Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5, human_reverses=True)
        )
        over_cautious = MODEL.reward(Outcome(ResponseAction.ESCALATE, False, 0.5))
        assert reversed_contain < over_cautious

    def test_a_reversal_costs_more_than_no_reversal(self) -> None:
        assert MODEL.reward(
            Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5, human_reverses=True)
        ) < MODEL.reward(Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5))

    def test_correct_containment_is_rewarded(self) -> None:
        assert MODEL.reward(Outcome(ResponseAction.AUTO_CONTAIN, True, 0.5)) > 0.0


class TestDomainOrdering:
    """The orderings the PRD sentence omits and a security product requires."""

    def test_missing_a_real_attack_is_the_worst_outcome(self) -> None:
        missed = MODEL.reward(Outcome(ResponseAction.DISMISS, True, 0.5))
        others = [
            MODEL.reward(Outcome(action, attack, 0.5, reverses))
            for action, attack, reverses in itertools.product(
                ALL_ACTIONS, (True, False), (False, True)
            )
            if not (reverses and action is not ResponseAction.AUTO_CONTAIN)
            and not (action is ResponseAction.DISMISS and attack)
        ]
        assert missed < min(others)

    def test_on_a_real_attack_reward_falls_with_inaction(self) -> None:
        rewards = [
            MODEL.reward(Outcome(action, True, 0.5))
            for action in (
                ResponseAction.AUTO_CONTAIN,
                ResponseAction.ESCALATE,
                ResponseAction.MONITOR,
                ResponseAction.DISMISS,
            )
        ]
        assert rewards == sorted(rewards, reverse=True)
        assert len(set(rewards)) == 4

    def test_on_a_benign_alert_reward_falls_with_escalation(self) -> None:
        rewards = [
            MODEL.reward(Outcome(action, False, 0.5))
            for action in (
                ResponseAction.DISMISS,
                ResponseAction.MONITOR,
                ResponseAction.ESCALATE,
                ResponseAction.AUTO_CONTAIN,
            )
        ]
        assert rewards == sorted(rewards, reverse=True)

    def test_dismissing_a_benign_alert_is_rewarded(self) -> None:
        """The alert-volume reduction F-12 measures is the product's core value."""
        assert MODEL.reward(Outcome(ResponseAction.DISMISS, False, 0.5)) > 0.0

    def test_the_gain_from_containing_over_escalating_is_small(self) -> None:
        """Why the learned policy is risk-averse, and correctly so.

        The upside of containing rather than escalating a true attack is small, while
        the downside of containing a benign one is large. That asymmetry is the reason
        the measured policy escalates far more often than it contains, and pinning it
        here means the behaviour is explained by the reward rather than treated as a
        learning failure.
        """
        gain = MODEL.reward(Outcome(ResponseAction.AUTO_CONTAIN, True, 0.5)) - MODEL.reward(
            Outcome(ResponseAction.ESCALATE, True, 0.5)
        )
        loss = MODEL.reward(Outcome(ResponseAction.ESCALATE, False, 0.5)) - MODEL.reward(
            Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5, human_reverses=True)
        )
        assert gain > 0.0
        assert loss / gain > 5.0


class TestCriticality:
    def test_criticality_scales_penalties(self) -> None:
        for action, attack in (
            (ResponseAction.DISMISS, True),
            (ResponseAction.AUTO_CONTAIN, False),
            (ResponseAction.MONITOR, True),
            (ResponseAction.ESCALATE, False),
        ):
            low = MODEL.reward(Outcome(action, attack, 0.0))
            high = MODEL.reward(Outcome(action, attack, 1.0))
            assert high < low

    def test_criticality_does_not_inflate_rewards(self) -> None:
        """Otherwise the policy acquires an appetite for critical assets."""
        for action, attack in (
            (ResponseAction.AUTO_CONTAIN, True),
            (ResponseAction.DISMISS, False),
            (ResponseAction.MONITOR, False),
            (ResponseAction.ESCALATE, True),
        ):
            assert MODEL.reward(Outcome(action, attack, 1.0)) == MODEL.reward(
                Outcome(action, attack, 0.0)
            )

    def test_penalty_scaling_is_monotone(self) -> None:
        values = [
            MODEL.reward(Outcome(ResponseAction.DISMISS, True, c))
            for c in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert values == sorted(values, reverse=True)

    def test_zero_criticality_weight_disables_scaling(self) -> None:
        flat = RewardModel(criticality_weight=0.0)
        assert flat.reward(Outcome(ResponseAction.DISMISS, True, 1.0)) == flat.reward(
            Outcome(ResponseAction.DISMISS, True, 0.0)
        )


class TestOrderingHoldsAcrossTheParameterSpace:
    """The defaults satisfying the ordering proves nothing; surviving tuning does."""

    def test_default_model_passes(self) -> None:
        assert_ordering_invariants(RewardModel())

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 4.0])
    def test_ordering_survives_uniform_penalty_rescaling(self, scale: float) -> None:
        """Every penalty scaled together. Tests the shape, not the magnitudes."""
        base = RewardModel()
        assert_ordering_invariants(
            RewardModel(
                contain_false=base.contain_false * scale,
                reversal_penalty=base.reversal_penalty * scale,
                escalate_false=base.escalate_false * scale,
                monitor_true=base.monitor_true * scale,
                dismiss_true=base.dismiss_true * scale,
            )
        )

    @pytest.mark.parametrize("contain_true", [0.7, 1.0, 1.5, 3.0])
    def test_ordering_survives_varying_the_containment_reward(
        self, contain_true: float
    ) -> None:
        """Values above ``escalate_true``; below it the model is self-inconsistent."""
        assert_ordering_invariants(RewardModel(contain_true=contain_true))

    @pytest.mark.parametrize("weight", [0.0, 0.25, 1.0, 2.0, 5.0])
    def test_ordering_survives_varying_criticality_weight(self, weight: float) -> None:
        assert_ordering_invariants(RewardModel(criticality_weight=weight))

    @pytest.mark.parametrize("dismiss_true", [-3.0, -6.0, -12.0])
    @pytest.mark.parametrize("reversal_penalty", [-0.5, -1.2])
    def test_ordering_survives_deepening_the_two_worst_penalties(
        self, dismiss_true: float, reversal_penalty: float
    ) -> None:
        """Both combinations keep a missed attack the worst outcome, as required."""
        assert_ordering_invariants(
            RewardModel(dismiss_true=dismiss_true, reversal_penalty=reversal_penalty)
        )

    def test_a_containment_reward_below_escalation_is_rejected(self) -> None:
        """Self-inconsistent: containing a real attack would be worse than escalating."""
        with pytest.raises(RewardError, match="decrease monotonically"):
            assert_ordering_invariants(RewardModel(contain_true=0.3))

    def test_a_reversal_penalty_deeper_than_a_missed_attack_is_rejected(self) -> None:
        """Isolating a benign host must never be treated as worse than being breached."""
        with pytest.raises(RewardError, match="worst outcome"):
            assert_ordering_invariants(
                RewardModel(dismiss_true=-2.0, reversal_penalty=-3.0)
            )

    def test_a_model_that_underweights_missed_attacks_is_rejected(self) -> None:
        """The invariant must actually bite, not merely be satisfiable."""
        broken = RewardModel(dismiss_true=-0.1)
        with pytest.raises(RewardError, match="worst outcome"):
            assert_ordering_invariants(broken)

    def test_a_model_that_rewards_false_containment_is_rejected(self) -> None:
        broken = RewardModel(contain_false=1.0, reversal_penalty=-0.01)
        with pytest.raises(RewardError):
            assert_ordering_invariants(broken)

    def test_a_model_where_reversal_is_free_is_rejected(self) -> None:
        broken = RewardModel(reversal_penalty=0.0)
        with pytest.raises(RewardError, match="reversal must cost"):
            assert_ordering_invariants(broken)

    def test_a_model_that_inflates_rewards_by_criticality_is_rejected(self) -> None:
        """Guards the sign convention: scaling must apply to losses only."""

        class Inflating(RewardModel):
            def reward(self, outcome: Outcome) -> float:  # type: ignore[override]
                base = RewardModel.reward(self, outcome)
                return base * (1.0 + outcome.criticality)

        with pytest.raises(RewardError, match="must not inflate"):
            assert_ordering_invariants(Inflating())

    def test_failure_message_lists_every_violation(self) -> None:
        broken = RewardModel(dismiss_true=-0.1, reversal_penalty=0.0)
        with pytest.raises(RewardError) as caught:
            assert_ordering_invariants(broken)
        assert str(caught.value).count("- ") >= 2


class TestRewardSpread:
    def test_is_positive_and_finite(self) -> None:
        assert MODEL.reward_spread > 0.0
        assert np.isfinite(MODEL.reward_spread)

    def test_matches_the_standard_deviation_of_the_outcome_grid(self) -> None:
        values = [
            MODEL.reward(Outcome(action, attack, criticality, reverses))
            for action, attack, criticality, reverses in itertools.product(
                ALL_ACTIONS, (True, False), (0.0, 0.5, 1.0), (False, True)
            )
            if not (reverses and action is not ResponseAction.AUTO_CONTAIN)
        ]
        assert MODEL.reward_spread == pytest.approx(float(np.std(values)))

    def test_scales_with_the_reward_constants(self) -> None:
        """This is the decoupling the bandit's ``reward_scale`` relies on."""
        wide = RewardModel(dismiss_true=-12.0)
        assert wide.reward_spread > MODEL.reward_spread

    def test_never_returns_zero(self) -> None:
        """A zero scale would divide by zero inside the bandit's update."""
        flat = RewardModel(
            contain_true=0.0, contain_false=0.0, reversal_penalty=0.0,
            escalate_true=0.0, escalate_false=0.0, monitor_true=0.0,
            monitor_false=0.0, dismiss_true=0.0, dismiss_false=0.0,
        )
        assert flat.reward_spread == 1.0


class TestOracle:
    def test_oracle_contains_a_real_attack_when_allowed(self) -> None:
        assert (
            best_action(
                MODEL,
                is_attack=True,
                criticality=0.5,
                mask=ActionMask.for_tier(RiskTier.AUTONOMOUS),
            )
            is ResponseAction.AUTO_CONTAIN
        )

    def test_oracle_dismisses_a_benign_alert_when_allowed(self) -> None:
        assert (
            best_action(
                MODEL,
                is_attack=False,
                criticality=0.5,
                mask=ActionMask.for_tier(RiskTier.AUTONOMOUS),
            )
            is ResponseAction.DISMISS
        )

    def test_oracle_respects_the_mask(self) -> None:
        """An oracle outside the mask would make obedience look like regret."""
        for tier in ALL_TIERS:
            mask = ActionMask.for_tier(tier)
            for is_attack in (True, False):
                chosen = best_action(
                    MODEL, is_attack=is_attack, criticality=0.5, mask=mask
                )
                assert mask.permits(chosen)

    def test_oracle_escalates_an_attack_at_observe_tier(self) -> None:
        """The best available action when containment is forbidden."""
        assert (
            best_action(
                MODEL,
                is_attack=True,
                criticality=0.5,
                mask=ActionMask.for_tier(RiskTier.OBSERVE),
            )
            is ResponseAction.ESCALATE
        )

    def test_oracle_monitors_a_benign_alert_at_observe_tier(self) -> None:
        """Dismissal is forbidden there, so monitoring is the best remaining call."""
        assert (
            best_action(
                MODEL,
                is_attack=False,
                criticality=0.5,
                mask=ActionMask.for_tier(RiskTier.OBSERVE),
            )
            is ResponseAction.MONITOR
        )

    def test_best_reward_equals_the_best_actions_reward(self) -> None:
        for tier, is_attack, criticality in itertools.product(
            ALL_TIERS, (True, False), (0.0, 0.5, 1.0)
        ):
            mask = ActionMask.for_tier(tier)
            chosen = best_action(
                MODEL, is_attack=is_attack, criticality=criticality, mask=mask
            )
            assert MODEL.best_reward(
                is_attack=is_attack, criticality=criticality, mask=mask
            ) == pytest.approx(
                MODEL.expected_reward(
                    action=chosen, is_attack=is_attack, criticality=criticality
                )
            )

    def test_best_reward_dominates_every_allowed_action(self) -> None:
        for tier, is_attack in itertools.product(ALL_TIERS, (True, False)):
            mask = ActionMask.for_tier(tier)
            best = MODEL.best_reward(is_attack=is_attack, criticality=0.5, mask=mask)
            for action in mask.allowed:
                assert (
                    MODEL.expected_reward(
                        action=action, is_attack=is_attack, criticality=0.5
                    )
                    <= best + 1e-12
                )

    def test_higher_tiers_never_have_a_worse_oracle(self) -> None:
        """More permitted actions cannot lower the maximum over them."""
        ordered = sorted(ALL_TIERS, key=lambda t: t.rank)
        for is_attack in (True, False):
            rewards = [
                MODEL.best_reward(
                    is_attack=is_attack,
                    criticality=0.5,
                    mask=ActionMask.for_tier(tier),
                )
                for tier in ordered
            ]
            assert rewards == sorted(rewards)


class TestExpectedReward:
    def test_reversal_probability_interpolates(self) -> None:
        kept = MODEL.reward(Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5))
        reversed_ = MODEL.reward(
            Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5, human_reverses=True)
        )
        half = MODEL.expected_reward(
            action=ResponseAction.AUTO_CONTAIN,
            is_attack=False,
            criticality=0.5,
            reversal_probability=0.5,
        )
        assert half == pytest.approx(0.5 * (kept + reversed_))

    @pytest.mark.parametrize("probability", [0.0, 1.0])
    def test_endpoints_match_the_deterministic_reward(self, probability: float) -> None:
        expected = MODEL.expected_reward(
            action=ResponseAction.AUTO_CONTAIN,
            is_attack=False,
            criticality=0.5,
            reversal_probability=probability,
        )
        assert expected == pytest.approx(
            MODEL.reward(
                Outcome(
                    ResponseAction.AUTO_CONTAIN,
                    False,
                    0.5,
                    human_reverses=bool(probability),
                )
            )
        )

    def test_reversal_probability_is_irrelevant_to_other_actions(self) -> None:
        for action in (
            ResponseAction.ESCALATE,
            ResponseAction.MONITOR,
            ResponseAction.DISMISS,
        ):
            assert MODEL.expected_reward(
                action=action, is_attack=True, criticality=0.5, reversal_probability=0.0
            ) == MODEL.expected_reward(
                action=action, is_attack=True, criticality=0.5, reversal_probability=1.0
            )

    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_invalid_probability_rejected(self, probability: float) -> None:
        with pytest.raises(RewardError, match="reversal_probability"):
            MODEL.expected_reward(
                action=ResponseAction.AUTO_CONTAIN,
                is_attack=False,
                criticality=0.5,
                reversal_probability=probability,
            )


class TestRewardNeverSeesGroundTruthOnThePolicyPath:
    def test_the_bandit_update_signature_takes_only_a_scalar(self) -> None:
        """The structural guarantee: there is no channel for the label to leak.

        ``Outcome.is_attack`` exists so the oracle is computable. The policy must
        receive a number, and this asserts the API gives it no other option.
        """
        import inspect

        from sentinel.rl.bandit import LinearThompsonBandit

        parameters = inspect.signature(LinearThompsonBandit.update).parameters
        assert set(parameters) == {"self", "context", "action", "reward", "tier"}
        assert parameters["reward"].annotation == "float"

    def test_select_takes_no_outcome_argument(self) -> None:
        import inspect

        from sentinel.rl.bandit import LinearThompsonBandit

        parameters = inspect.signature(LinearThompsonBandit.select).parameters
        assert set(parameters) == {"self", "context", "tier", "greedy"}
