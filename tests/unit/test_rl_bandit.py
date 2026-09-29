"""Linear contextual Thompson sampling (:mod:`sentinel.rl.bandit`).

Three things are checked here that the F-09 regret replay cannot check for itself,
because a bandit with any of them broken still produces a plausible-looking regret
curve:

*   :class:`TestShermanMorrisonMatchesDirectSolve` — the incremental inverse *is* the
    matrix it claims to be. A drifting rank-1 update degrades the posterior gradually,
    which shows up as slightly worse learning and never as an error.
*   :class:`TestPosteriorActuallyLearns` — the arms converge on the coefficients of a
    known linear reward function. Without this, "the regret curve went down" could be
    the environment rather than the policy.
*   :class:`TestMaskIsStructural` — a forbidden arm is not scored, not sampled, and
    not updated, at any tier, for any context. This is the safety property, so it is
    asserted over the full cross product rather than on a representative case.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sentinel.core.schemas import RiskTier
from sentinel.rl.actions import ALL_ACTIONS, ResponseAction
from sentinel.rl.bandit import (
    DEFAULT_PRIOR_PRECISION,
    DEFAULT_PRIOR_REWARD,
    ArmPosterior,
    BanditError,
    LinearThompsonBandit,
)

ALL_TIERS = tuple(RiskTier)
D = 5


def context(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


@pytest.fixture
def bandit() -> LinearThompsonBandit:
    return LinearThompsonBandit(n_features=D, seed=7)


class TestArmPosteriorConstruction:
    def test_starts_at_the_prior(self) -> None:
        arm = ArmPosterior(n_features=D, prior_precision=2.0)
        np.testing.assert_allclose(arm.precision, 2.0 * np.eye(D))
        np.testing.assert_allclose(arm.inverse, np.eye(D) / 2.0)
        assert arm.observations == 0

    def test_optimistic_prior_sits_on_the_bias_coefficient(self) -> None:
        """The prior must enter through the same channel as the data."""
        arm = ArmPosterior(n_features=D, prior_precision=4.0, prior_reward=1.0)
        assert arm.mean[0] == pytest.approx(1.0)
        np.testing.assert_allclose(arm.mean[1:], 0.0)

    def test_zero_prior_reward_gives_a_zero_mean(self) -> None:
        arm = ArmPosterior(n_features=D, prior_reward=0.0)
        np.testing.assert_allclose(arm.mean, 0.0)

    def test_optimistic_prior_predicts_optimistically_for_a_bias_context(self) -> None:
        arm = ArmPosterior(n_features=D, prior_precision=4.0, prior_reward=1.0)
        assert arm.predict(context(1.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx(1.0)

    @pytest.mark.parametrize("n_features", [0, -1])
    def test_invalid_width_rejected(self, n_features: int) -> None:
        with pytest.raises(BanditError, match="n_features"):
            ArmPosterior(n_features=n_features)

    @pytest.mark.parametrize("precision", [0.0, -1.0])
    def test_non_positive_ridge_rejected(self, precision: float) -> None:
        """A zero ridge leaves ``A`` singular until enough data arrives."""
        with pytest.raises(BanditError, match="prior_precision"):
            ArmPosterior(n_features=D, prior_precision=precision)


class TestArmPosteriorUpdate:
    def test_absorbs_an_observation(self) -> None:
        arm = ArmPosterior(n_features=D)
        arm.update(context(1, 0, 0, 0, 0), 1.0)
        assert arm.observations == 1
        assert arm.reward_sum == pytest.approx(1.0)

    def test_precision_accumulates_the_outer_product(self) -> None:
        arm = ArmPosterior(n_features=D, prior_precision=1.0)
        vector = context(1, 2, 0, 0, 0)
        arm.update(vector, 0.5)
        np.testing.assert_allclose(arm.precision, np.eye(D) + np.outer(vector, vector))

    def test_moment_accumulates_reward_times_context(self) -> None:
        arm = ArmPosterior(n_features=D, prior_precision=1.0, prior_reward=0.0)
        vector = context(1, 2, 0, 0, 0)
        arm.update(vector, 3.0)
        np.testing.assert_allclose(arm.moment, 3.0 * vector)

    def test_uncertainty_shrinks_where_data_arrives(self) -> None:
        arm = ArmPosterior(n_features=D)
        vector = context(1, 1, 0, 0, 0)
        before = arm.uncertainty(vector)
        for _ in range(20):
            arm.update(vector, 1.0)
        assert arm.uncertainty(vector) < before

    def test_uncertainty_stays_high_where_no_data_arrives(self) -> None:
        """The property Thompson sampling relies on to keep exploring."""
        arm = ArmPosterior(n_features=D)
        seen = context(1, 1, 0, 0, 0)
        unseen = context(0, 0, 0, 0, 1)
        before = arm.uncertainty(unseen)
        for _ in range(50):
            arm.update(seen, 1.0)
        assert arm.uncertainty(unseen) == pytest.approx(before, rel=0.05)

    def test_uncertainty_is_never_nan(self) -> None:
        """A marginally negative quadratic form must not become a NaN."""
        arm = ArmPosterior(n_features=D)
        rng = np.random.default_rng(3)
        for _ in range(300):
            arm.update(rng.normal(size=D), float(rng.normal()))
        for _ in range(50):
            value = arm.uncertainty(rng.normal(size=D))
            assert np.isfinite(value)
            assert value >= 0.0

    def test_mean_reward_reports_the_average(self) -> None:
        arm = ArmPosterior(n_features=D)
        arm.update(context(1, 0, 0, 0, 0), 1.0)
        arm.update(context(1, 0, 0, 0, 0), 3.0)
        assert arm.mean_reward == pytest.approx(2.0)
        assert ArmPosterior(n_features=D).mean_reward == 0.0

    def test_wrong_shaped_context_rejected(self) -> None:
        arm = ArmPosterior(n_features=D)
        with pytest.raises(BanditError, match="shape"):
            arm.update(context(1.0, 2.0), 1.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_context_rejected(self, bad: float) -> None:
        arm = ArmPosterior(n_features=D)
        with pytest.raises(BanditError, match="non-finite"):
            arm.update(context(1.0, bad, 0.0, 0.0, 0.0), 1.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_reward_rejected(self, bad: float) -> None:
        arm = ArmPosterior(n_features=D)
        with pytest.raises(BanditError, match="reward must be finite"):
            arm.update(context(1, 0, 0, 0, 0), bad)


class TestShermanMorrisonMatchesDirectSolve:
    """The incremental inverse must be the matrix it claims to be."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_after_many_random_updates(self, seed: int) -> None:
        arm = ArmPosterior(n_features=D, prior_precision=DEFAULT_PRIOR_PRECISION)
        rng = np.random.default_rng(seed)
        for _ in range(500):
            arm.update(rng.normal(size=D), float(rng.normal()))
        np.testing.assert_allclose(arm.inverse, arm.direct_inverse(), atol=1e-9)

    def test_matches_with_correlated_contexts(self) -> None:
        """Nearly collinear contexts are where a rank-1 update is worst behaved."""
        arm = ArmPosterior(n_features=D)
        rng = np.random.default_rng(11)
        base = rng.normal(size=D)
        for _ in range(400):
            arm.update(base + 1e-4 * rng.normal(size=D), float(rng.normal()))
        np.testing.assert_allclose(arm.inverse, arm.direct_inverse(), atol=1e-7)

    def test_matches_with_large_magnitude_contexts(self) -> None:
        arm = ArmPosterior(n_features=D)
        rng = np.random.default_rng(13)
        for _ in range(200):
            arm.update(1e3 * rng.normal(size=D), float(rng.normal()))
        np.testing.assert_allclose(
            arm.inverse, arm.direct_inverse(), atol=1e-12, rtol=1e-6
        )

    def test_inverse_stays_symmetric(self) -> None:
        """An asymmetric 'covariance' makes the Cholesky either fail or lie."""
        arm = ArmPosterior(n_features=D)
        rng = np.random.default_rng(17)
        for _ in range(300):
            arm.update(rng.normal(size=D), float(rng.normal()))
            np.testing.assert_array_equal(arm.inverse, arm.inverse.T)

    def test_inverse_stays_positive_definite(self) -> None:
        arm = ArmPosterior(n_features=D)
        rng = np.random.default_rng(19)
        for _ in range(300):
            arm.update(rng.normal(size=D), float(rng.normal()))
        assert np.all(np.linalg.eigvalsh(arm.inverse) > 0.0)

    def test_bandit_reports_zero_drift_after_a_long_run(
        self, bandit: LinearThompsonBandit
    ) -> None:
        rng = np.random.default_rng(23)
        for _ in range(400):
            vector = rng.normal(size=D)
            action = ALL_ACTIONS[int(rng.integers(len(ALL_ACTIONS)))]
            bandit.update(vector, action, float(rng.normal()), tier=RiskTier.AUTONOMOUS)
        assert bandit.max_inverse_drift() < 1e-9


class TestPosteriorActuallyLearns:
    """Recover the coefficients of a known linear reward function."""

    def test_recovers_a_known_linear_reward(self) -> None:
        truth = np.asarray([0.5, -1.0, 2.0, 0.0, 0.25])
        arm = ArmPosterior(n_features=D, prior_precision=1e-6, prior_reward=0.0)
        rng = np.random.default_rng(29)
        for _ in range(4000):
            vector = rng.normal(size=D)
            arm.update(vector, float(vector @ truth))
        np.testing.assert_allclose(arm.mean, truth, atol=1e-6)

    def test_recovers_under_observation_noise(self) -> None:
        truth = np.asarray([1.0, -0.5, 0.75, 0.0, 0.0])
        arm = ArmPosterior(n_features=D, prior_precision=1e-3, prior_reward=0.0)
        rng = np.random.default_rng(31)
        for _ in range(8000):
            vector = rng.normal(size=D)
            arm.update(vector, float(vector @ truth + rng.normal(0.0, 0.3)))
        np.testing.assert_allclose(arm.mean, truth, atol=0.05)

    def test_prediction_improves_with_data(self) -> None:
        truth = np.asarray([0.2, 1.5, -0.8, 0.0, 0.1])
        arm = ArmPosterior(n_features=D, prior_precision=1.0, prior_reward=0.0)
        rng = np.random.default_rng(37)
        probe = rng.normal(size=D)
        first_error = abs(arm.predict(probe) - float(probe @ truth))
        for _ in range(2000):
            vector = rng.normal(size=D)
            arm.update(vector, float(vector @ truth))
        assert abs(arm.predict(probe) - float(probe @ truth)) < first_error * 0.05

    def test_the_optimistic_prior_is_eventually_overwritten(self) -> None:
        """Optimism must be a starting point, not a permanent bias."""
        arm = ArmPosterior(n_features=D, prior_precision=4.0, prior_reward=1.0)
        bias = context(1, 0, 0, 0, 0)
        assert arm.predict(bias) > 0.5
        for _ in range(500):
            arm.update(bias, -2.0)
        assert arm.predict(bias) < -1.9


class TestSelection:
    def test_returns_a_permitted_action(self, bandit: LinearThompsonBandit) -> None:
        for tier in ALL_TIERS:
            decision = bandit.select(context(1, 0.5, 0, 0, 0), tier=tier)
            assert decision.action.minimum_tier.rank <= tier.rank

    def test_scores_only_permitted_arms(self, bandit: LinearThompsonBandit) -> None:
        decision = bandit.select(context(1, 0.5, 0, 0, 0), tier=RiskTier.OBSERVE)
        assert set(decision.sampled_values) == {
            ResponseAction.MONITOR,
            ResponseAction.ESCALATE,
        }
        assert set(decision.expected_values) == set(decision.sampled_values)
        assert set(decision.uncertainties) == set(decision.sampled_values)

    def test_greedy_mode_suppresses_sampling(self, bandit: LinearThompsonBandit) -> None:
        vector = context(1, 0.5, 0.25, 0, 0)
        decision = bandit.select(vector, tier=RiskTier.AUTONOMOUS, greedy=True)
        assert decision.sampled_values == decision.expected_values
        assert not decision.was_exploratory

    def test_greedy_mode_is_deterministic(self, bandit: LinearThompsonBandit) -> None:
        vector = context(1, 0.5, 0.25, 0, 0)
        first = bandit.select(vector, tier=RiskTier.AUTONOMOUS, greedy=True)
        assert first.action is bandit.select(
            vector, tier=RiskTier.AUTONOMOUS, greedy=True
        ).action

    def test_zero_posterior_scale_is_greedy(self) -> None:
        policy = LinearThompsonBandit(n_features=D, posterior_scale=0.0, seed=3)
        decision = policy.select(context(1, 0.5, 0, 0, 0), tier=RiskTier.AUTONOMOUS)
        assert decision.sampled_values == decision.expected_values

    def test_sampling_does_explore(self) -> None:
        """A sampler that never overrides the mean is not sampling."""
        policy = LinearThompsonBandit(
            n_features=D, posterior_scale=1.0, seed=5, prior_reward=0.0
        )
        vector = context(1, 0.5, 0, 0, 0)
        # Give one arm a clear advantage, then check the others still get tried.
        for _ in range(30):
            policy.update(vector, ResponseAction.DISMISS, 1.0, tier=RiskTier.AUTONOMOUS)
        chosen = {
            policy.select(vector, tier=RiskTier.AUTONOMOUS).action for _ in range(200)
        }
        assert len(chosen) > 1

    def test_ties_prefer_the_less_interventionist_action(self) -> None:
        """With no information, the policy must not default to containment."""
        policy = LinearThompsonBandit(n_features=D, posterior_scale=0.0, seed=1)
        decision = policy.select(context(1, 0, 0, 0, 0), tier=RiskTier.AUTONOMOUS)
        assert decision.action is ResponseAction.DISMISS

    def test_learned_preference_is_followed(self) -> None:
        policy = LinearThompsonBandit(
            n_features=D, posterior_scale=0.0, seed=1, prior_reward=0.0
        )
        vector = context(1, 1.0, 0, 0, 0)
        for _ in range(50):
            policy.update(vector, ResponseAction.ESCALATE, 5.0, tier=RiskTier.AUTONOMOUS)
            policy.update(vector, ResponseAction.DISMISS, -5.0, tier=RiskTier.AUTONOMOUS)
        assert (
            policy.select(vector, tier=RiskTier.AUTONOMOUS).action
            is ResponseAction.ESCALATE
        )

    def test_same_seed_gives_the_same_sequence(self) -> None:
        vector = context(1, 0.5, 0.2, 0, 0)
        a = LinearThompsonBandit(n_features=D, seed=42)
        b = LinearThompsonBandit(n_features=D, seed=42)
        assert [a.select(vector, tier=RiskTier.AUTONOMOUS).action for _ in range(50)] == [
            b.select(vector, tier=RiskTier.AUTONOMOUS).action for _ in range(50)
        ]

    def test_different_seeds_differ(self) -> None:
        vector = context(1, 0.5, 0.2, 0, 0)
        a = LinearThompsonBandit(n_features=D, seed=1, posterior_scale=1.0)
        b = LinearThompsonBandit(n_features=D, seed=2, posterior_scale=1.0)
        assert [a.select(vector, tier=RiskTier.AUTONOMOUS).action for _ in range(50)] != [
            b.select(vector, tier=RiskTier.AUTONOMOUS).action for _ in range(50)
        ]

    def test_never_raises_on_extreme_contexts(self, bandit: LinearThompsonBandit) -> None:
        for vector in (
            np.zeros(D),
            np.ones(D) * 1e6,
            np.ones(D) * -1e6,
            np.asarray([1.0, 0.0, 0.0, 0.0, 1e-12]),
        ):
            assert bandit.select(vector, tier=RiskTier.AUTONOMOUS).action in ALL_ACTIONS


class TestMaskIsStructural:
    """A forbidden arm is not scored, not sampled, and not updated. Ever."""

    def test_no_tier_ever_yields_a_forbidden_action(self) -> None:
        rng = np.random.default_rng(43)
        policy = LinearThompsonBandit(n_features=D, seed=43, posterior_scale=1.0)
        for tier in ALL_TIERS:
            for _ in range(300):
                vector = rng.normal(size=D)
                decision = policy.select(vector, tier=tier)
                assert decision.action.minimum_tier.rank <= tier.rank

    def test_forbidden_arms_are_absent_from_the_decision_not_zeroed(self) -> None:
        """Absent, so nothing downstream can read a score for a forbidden arm."""
        policy = LinearThompsonBandit(n_features=D, seed=47)
        decision = policy.select(context(1, 1, 1, 1, 1), tier=RiskTier.RECOMMEND)
        assert ResponseAction.AUTO_CONTAIN not in decision.sampled_values
        assert ResponseAction.AUTO_CONTAIN not in decision.expected_values

    def test_a_strongly_rewarded_forbidden_arm_is_still_never_chosen(self) -> None:
        """The exact failure a reward penalty cannot prevent.

        The arm is trained to look overwhelmingly good, then requested at a tier that
        forbids it. A soft penalty would be outvoted by a large enough modelled
        benefit; a structural mask cannot be.
        """
        policy = LinearThompsonBandit(
            n_features=D, seed=53, posterior_scale=0.0, prior_reward=0.0
        )
        vector = context(1, 1, 0, 0, 0)
        for _ in range(500):
            policy.update(
                vector, ResponseAction.AUTO_CONTAIN, 1000.0, tier=RiskTier.AUTONOMOUS
            )
        assert (
            policy.select(vector, tier=RiskTier.AUTONOMOUS).action
            is ResponseAction.AUTO_CONTAIN
        )
        for tier in (RiskTier.OBSERVE, RiskTier.RECOMMEND):
            assert (
                policy.select(vector, tier=tier).action
                is not ResponseAction.AUTO_CONTAIN
            )

    def test_update_refuses_an_action_the_tier_forbids(self) -> None:
        """Updating a masked arm means it was played, which is the violation itself."""
        policy = LinearThompsonBandit(n_features=D, seed=59)
        with pytest.raises(Exception, match="requires tier"):
            policy.update(
                context(1, 0, 0, 0, 0),
                ResponseAction.AUTO_CONTAIN,
                1.0,
                tier=RiskTier.OBSERVE,
            )

    def test_update_without_a_tier_is_permitted_for_offline_fitting(self) -> None:
        policy = LinearThompsonBandit(n_features=D, seed=61)
        policy.update(context(1, 0, 0, 0, 0), ResponseAction.AUTO_CONTAIN, 1.0)
        assert policy.arms[ResponseAction.AUTO_CONTAIN].observations == 1

    def test_masked_arms_stay_cold_and_are_reported(self) -> None:
        """The documented cost of structural masking, made visible rather than hidden."""
        policy = LinearThompsonBandit(n_features=D, seed=67, posterior_scale=0.5)
        rng = np.random.default_rng(67)
        for _ in range(200):
            vector = rng.normal(size=D)
            decision = policy.select(vector, tier=RiskTier.OBSERVE)
            policy.update(vector, decision.action, 0.5, tier=RiskTier.OBSERVE)
        assert policy.arms[ResponseAction.AUTO_CONTAIN].observations == 0
        assert policy.arms[ResponseAction.DISMISS].observations == 0
        cold = policy.cold_arms()
        assert ResponseAction.AUTO_CONTAIN in cold
        assert ResponseAction.DISMISS in cold
        assert ResponseAction.MONITOR not in cold


class TestDecision:
    def test_greedy_action_uses_the_posterior_mean(self) -> None:
        policy = LinearThompsonBandit(
            n_features=D, seed=101, posterior_scale=0.0, prior_reward=0.0
        )
        vector = context(1, 0.5, 0, 0, 0)
        for _ in range(20):
            policy.update(vector, ResponseAction.ESCALATE, 2.0, tier=RiskTier.AUTONOMOUS)
        decision = policy.select(vector, tier=RiskTier.AUTONOMOUS)
        best_mean = max(
            decision.expected_values, key=lambda a: decision.expected_values[a]
        )
        assert decision.greedy_action is best_mean is ResponseAction.ESCALATE

    def test_greedy_action_uses_the_same_cautious_tie_break_as_select(self) -> None:
        """A greedy action computed under a different tie rule reports phantom exploration."""
        policy = LinearThompsonBandit(n_features=D, seed=103, posterior_scale=0.0)
        decision = policy.select(context(1, 0, 0, 0, 0), tier=RiskTier.AUTONOMOUS)
        # All four arms tie at the optimistic prior.
        assert len(set(decision.expected_values.values())) == 1
        assert decision.greedy_action is ResponseAction.DISMISS
        assert decision.action is ResponseAction.DISMISS
        assert not decision.was_exploratory

    def test_exploratory_flag_detects_an_override(self) -> None:
        policy = LinearThompsonBandit(
            n_features=D, seed=71, posterior_scale=2.0, prior_reward=0.0
        )
        vector = context(1, 1, 0, 0, 0)
        for _ in range(40):
            policy.update(vector, ResponseAction.DISMISS, 3.0, tier=RiskTier.AUTONOMOUS)
        decisions = [
            policy.select(vector, tier=RiskTier.AUTONOMOUS) for _ in range(200)
        ]
        assert any(d.was_exploratory for d in decisions)
        assert any(not d.was_exploratory for d in decisions)

    def test_confidence_is_bounded(self, bandit: LinearThompsonBandit) -> None:
        for tier in ALL_TIERS:
            value = bandit.select(context(1, 0.5, 0, 0, 0), tier=tier).confidence
            assert 0.0 <= value <= 1.0

    def test_confidence_rises_with_a_clearer_margin(self) -> None:
        policy = LinearThompsonBandit(
            n_features=D, seed=73, posterior_scale=0.0, prior_reward=0.0
        )
        vector = context(1, 1, 0, 0, 0)
        before = policy.select(vector, tier=RiskTier.AUTONOMOUS).confidence
        for _ in range(100):
            policy.update(vector, ResponseAction.ESCALATE, 5.0, tier=RiskTier.AUTONOMOUS)
        assert policy.select(vector, tier=RiskTier.AUTONOMOUS).confidence > before

    def test_decision_is_immutable(self, bandit: LinearThompsonBandit) -> None:
        decision = bandit.select(context(1, 0, 0, 0, 0), tier=RiskTier.AUTONOMOUS)
        with pytest.raises(AttributeError):
            decision.action = ResponseAction.DISMISS  # type: ignore[misc]

    def test_tier_is_recorded(self, bandit: LinearThompsonBandit) -> None:
        for tier in ALL_TIERS:
            assert bandit.select(context(1, 0, 0, 0, 0), tier=tier).tier is tier


class TestConfiguration:
    @pytest.mark.parametrize("n_features", [0, -3])
    def test_invalid_width_rejected(self, n_features: int) -> None:
        with pytest.raises(BanditError, match="n_features"):
            LinearThompsonBandit(n_features=n_features)

    def test_negative_posterior_scale_rejected(self) -> None:
        with pytest.raises(BanditError, match="posterior_scale"):
            LinearThompsonBandit(n_features=D, posterior_scale=-1.0)

    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_non_positive_reward_scale_rejected(self, scale: float) -> None:
        """A zero scale would divide by zero inside every update."""
        with pytest.raises(BanditError, match="reward_scale"):
            LinearThompsonBandit(n_features=D, reward_scale=scale)

    def test_reward_scale_divides_the_update(self) -> None:
        """This is what makes ``posterior_scale`` mean sigmas rather than raw units."""
        raw = LinearThompsonBandit(n_features=D, reward_scale=1.0, prior_reward=0.0)
        scaled = LinearThompsonBandit(n_features=D, reward_scale=4.0, prior_reward=0.0)
        vector = context(1, 0, 0, 0, 0)
        raw.update(vector, ResponseAction.DISMISS, 4.0)
        scaled.update(vector, ResponseAction.DISMISS, 16.0)
        assert raw.arms[ResponseAction.DISMISS].predict(vector) == pytest.approx(
            scaled.arms[ResponseAction.DISMISS].predict(vector)
        )

    def test_every_arm_gets_a_posterior(self) -> None:
        policy = LinearThompsonBandit(n_features=D)
        assert set(policy.arms) == set(ALL_ACTIONS)

    def test_missing_arm_rejected(self) -> None:
        with pytest.raises(BanditError, match="no posterior for arms"):
            LinearThompsonBandit(
                n_features=D,
                arms={ResponseAction.MONITOR: ArmPosterior(n_features=D)},
            )

    def test_unknown_action_rejected_on_update(self, bandit: LinearThompsonBandit) -> None:
        with pytest.raises(BanditError, match="unknown action"):
            bandit.update(context(1, 0, 0, 0, 0), "not_an_action", 1.0)  # type: ignore[arg-type]

    def test_wrong_width_context_rejected(self, bandit: LinearThompsonBandit) -> None:
        with pytest.raises(BanditError, match="features"):
            bandit.select(np.zeros(D + 1), tier=RiskTier.AUTONOMOUS)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_context_rejected(
        self, bandit: LinearThompsonBandit, bad: float
    ) -> None:
        vector = np.zeros(D)
        vector[2] = bad
        with pytest.raises(BanditError, match="non-finite"):
            bandit.select(vector, tier=RiskTier.AUTONOMOUS)

    def test_default_prior_reward_is_optimistic(self) -> None:
        """The measured fix for bimodal convergence; not a neutral default."""
        assert DEFAULT_PRIOR_REWARD > 0.0

    def test_context_accepts_a_list(self, bandit: LinearThompsonBandit) -> None:
        assert bandit.select([1.0, 0.0, 0.0, 0.0, 0.0], tier=RiskTier.AUTONOMOUS)  # type: ignore[arg-type]


class TestIntrospection:
    def test_total_observations_counts_every_arm(self, bandit: LinearThompsonBandit) -> None:
        bandit.update(context(1, 0, 0, 0, 0), ResponseAction.MONITOR, 0.1)
        bandit.update(context(1, 0, 0, 0, 0), ResponseAction.DISMISS, 0.2)
        assert bandit.total_observations == 2

    def test_cold_arms_reports_everything_before_any_data(
        self, bandit: LinearThompsonBandit
    ) -> None:
        assert set(bandit.cold_arms()) == set(ALL_ACTIONS)

    def test_cold_arms_respects_the_minimum(self, bandit: LinearThompsonBandit) -> None:
        for _ in range(3):
            bandit.update(context(1, 0, 0, 0, 0), ResponseAction.MONITOR, 0.1)
        assert ResponseAction.MONITOR not in bandit.cold_arms(minimum=3)
        assert ResponseAction.MONITOR in bandit.cold_arms(minimum=4)

    def test_snapshot_is_serializable_and_complete(self, bandit: LinearThompsonBandit) -> None:
        import json

        bandit.update(context(1, 0, 0, 0, 0), ResponseAction.ESCALATE, 0.5)
        snapshot = bandit.snapshot()
        assert json.dumps(snapshot)
        assert set(snapshot["arms"]) == {a.value for a in ALL_ACTIONS}  # type: ignore[arg-type]
        assert snapshot["total_observations"] == 1
        for key in ("prior_precision", "posterior_scale", "reward_scale", "prior_reward"):
            assert key in snapshot

    def test_jitter_is_not_needed_in_normal_operation(self) -> None:
        """The ladder is insurance. If it fires routinely, something else is wrong."""
        policy = LinearThompsonBandit(n_features=D, seed=79, posterior_scale=0.5)
        rng = np.random.default_rng(79)
        for _ in range(500):
            vector = rng.normal(size=D)
            decision = policy.select(vector, tier=RiskTier.AUTONOMOUS)
            policy.update(vector, decision.action, float(rng.normal()), tier=RiskTier.AUTONOMOUS)
        assert policy.jitter_events == 0

    def test_drift_is_zero_for_a_fresh_policy(self, bandit: LinearThompsonBandit) -> None:
        assert bandit.max_inverse_drift() < 1e-12


class TestLongRunStability:
    """A bandit that fails after five thousand decisions has not been tested."""

    @pytest.mark.slow
    def test_survives_a_long_adversarial_run(self) -> None:
        policy = LinearThompsonBandit(n_features=D, seed=83, posterior_scale=0.5)
        rng = np.random.default_rng(83)
        for step in range(5000):
            # Alternate benign contexts with degenerate and near-collinear ones, which
            # is where an incremental inverse actually breaks.
            if step % 3 == 0:
                vector = np.zeros(D)
                vector[0] = 1.0
            elif step % 3 == 1:
                vector = np.full(D, 1e-8)
                vector[0] = 1.0
            else:
                vector = rng.normal(size=D)
            tier = ALL_TIERS[step % len(ALL_TIERS)]
            decision = policy.select(vector, tier=tier)
            assert decision.action.minimum_tier.rank <= tier.rank
            policy.update(vector, decision.action, float(rng.normal()), tier=tier)
        assert policy.max_inverse_drift() < 1e-6
        assert all(
            np.all(np.isfinite(arm.inverse)) for arm in policy.arms.values()
        )
        for arm in policy.arms.values():
            if arm.observations:
                assert np.all(np.linalg.eigvalsh(arm.inverse) > -1e-12)

    @pytest.mark.slow
    def test_all_tiers_interleaved_never_violate(self) -> None:
        policy = LinearThompsonBandit(n_features=D, seed=89, posterior_scale=1.0)
        rng = np.random.default_rng(89)
        for tier, _ in itertools.product(ALL_TIERS, range(500)):
            vector = rng.normal(size=D)
            decision = policy.select(vector, tier=tier)
            assert decision.action.minimum_tier.rank <= tier.rank
            policy.update(vector, decision.action, float(rng.normal()), tier=tier)
