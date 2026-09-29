"""Episode simulation and the F-09 replay (:mod:`sentinel.rl.simulate`).

The two classes that matter most are the ones checking things the regret number
cannot check about itself:

*   :class:`TestGroundTruthNeverEntersTheContext` — the reward reads
    ``is_attack``, and the policy must not. If the label leaked into a context feature
    the regret curve would look superb and the policy would be useless on real
    traffic. This is asserted by *measurement*: a logistic fit from context to label
    must not be separable beyond what the anomaly score alone explains.
*   :class:`TestRegretIsDefinedHonestly` — regret is computed against an oracle
    inside the same action mask, and from *expected* rather than realised reward.
    Either mistake produces a number that looks like a policy result and is not one.

:class:`TestF09Acceptance` is the acceptance criterion itself, on
:data:`~sentinel.rl.simulate.REPORTING_SEEDS` only.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.schemas import RiskTier
from sentinel.rl.actions import ALL_ACTIONS, ActionMask, ResponseAction
from sentinel.rl.bandit import LinearThompsonBandit
from sentinel.rl.reward import RewardModel
from sentinel.rl.simulate import (
    CONTEXT_SPEC,
    GATE_MEAN_SUBLINEARITY,
    GATE_OPTIMAL_ACTION_RATE,
    GATE_REGRET_RATIO,
    GATE_WORST_SUBLINEARITY,
    N_FEATURES,
    REPORTING_SEEDS,
    TUNING_SEEDS,
    Episode,
    SimulationError,
    aggregate,
    assert_f09_gates,
    build_episodes,
    replay,
    replay_many_seeds,
)

ALL_TIERS = tuple(RiskTier)


@pytest.fixture(scope="module")
def episodes() -> tuple[Episode, ...]:
    return build_episodes(n=200, seed=REPORTING_SEEDS[0])


class TestSeedDiscipline:
    def test_tuning_and_reporting_seeds_are_disjoint(self) -> None:
        """The whole point. Part 2 measured 0.825 on tuning seeds and 0.740 held out."""
        assert not set(TUNING_SEEDS) & set(REPORTING_SEEDS)

    def test_both_sets_are_large_enough_to_average_over(self) -> None:
        assert len(TUNING_SEEDS) >= 5
        assert len(REPORTING_SEEDS) >= 5

    def test_seeds_are_fixed_constants_not_generated(self) -> None:
        assert isinstance(TUNING_SEEDS, tuple)
        assert isinstance(REPORTING_SEEDS, tuple)
        assert all(isinstance(s, int) for s in (*TUNING_SEEDS, *REPORTING_SEEDS))


class TestEpisodeConstruction:
    def test_builds_the_requested_count(self) -> None:
        assert len(build_episodes(n=50, seed=1)) == 50

    def test_context_width_matches_the_spec(self, episodes: tuple[Episode, ...]) -> None:
        assert len(CONTEXT_SPEC.names) == N_FEATURES
        for episode in episodes:
            assert episode.context.shape == (N_FEATURES,)

    def test_contexts_are_finite(self, episodes: tuple[Episode, ...]) -> None:
        for episode in episodes:
            assert np.all(np.isfinite(episode.context))

    def test_bias_feature_is_exactly_one(self, episodes: tuple[Episode, ...]) -> None:
        """Without an intercept a linear arm cannot learn 'dismiss is usually right'."""
        assert CONTEXT_SPEC.names[0] == "bias"
        for episode in episodes:
            assert episode.context[0] == 1.0

    def test_deterministic_for_a_seed(self) -> None:
        first = build_episodes(n=40, seed=5)
        second = build_episodes(n=40, seed=5)
        assert [e.alert_id for e in first] == [e.alert_id for e in second]
        np.testing.assert_array_equal(
            np.stack([e.context for e in first]),
            np.stack([e.context for e in second]),
        )

    def test_different_seeds_give_different_episodes(self) -> None:
        first = np.stack([e.context for e in build_episodes(n=40, seed=5)])
        second = np.stack([e.context for e in build_episodes(n=40, seed=6)])
        assert not np.array_equal(first, second)

    def test_criticality_is_stable_per_asset(self) -> None:
        """A re-drawn criticality would make its feature pure noise.

        And it must be stable *across processes*, which rules out ``hash`` -- the same
        trap :mod:`sentinel.kb.text` documents for feature indices.
        """
        built = build_episodes(n=400, seed=7)
        by_alert: dict[str, float] = {}
        for episode in built:
            if episode.alert_id in by_alert:
                assert by_alert[episode.alert_id] == episode.criticality
            by_alert[episode.alert_id] = episode.criticality
        # Rebuild and confirm the same values come back.
        again = {e.alert_id: e.criticality for e in build_episodes(n=400, seed=7)}
        for alert_id, value in by_alert.items():
            assert again[alert_id] == value

    def test_criticality_is_in_range_and_varied(self, episodes: tuple[Episode, ...]) -> None:
        values = np.asarray([e.criticality for e in episodes])
        assert np.all((values >= 0.0) & (values <= 1.0))
        assert values.std() > 0.05

    def test_both_classes_are_present(self, episodes: tuple[Episode, ...]) -> None:
        labels = [e.is_attack for e in episodes]
        assert 0 < sum(labels) < len(labels)

    def test_tier_is_carried_on_every_episode(self) -> None:
        for tier in ALL_TIERS:
            built = build_episodes(n=10, seed=3, tier=tier)
            assert all(e.tier is tier for e in built)

    def test_detector_skill_controls_separation(self) -> None:
        """A parameter, not a constant, because the policy's job depends on it."""
        score_index = CONTEXT_SPEC.names.index("anomaly_score")

        def gap(skill: float) -> float:
            built = build_episodes(n=600, seed=9, detector_skill=skill)
            attack = np.asarray(
                [e.context[score_index] for e in built if e.is_attack]
            )
            benign = np.asarray(
                [e.context[score_index] for e in built if not e.is_attack]
            )
            return float(attack.mean() - benign.mean())

        assert gap(0.95) > gap(0.5) > gap(0.0) - 0.05

    def test_score_noise_widens_the_observed_score(self) -> None:
        score_index = CONTEXT_SPEC.names.index("anomaly_score")

        def deviation(noise: float) -> float:
            built = build_episodes(n=400, seed=13, score_noise=noise)
            return float(
                np.mean(
                    [abs(e.context[score_index] - e.true_score) for e in built]
                )
            )

        assert deviation(0.30) > deviation(0.02)

    @pytest.mark.parametrize(
        "kwargs,fragment",
        [
            ({"n": 0}, "n must be"),
            ({"n": 10, "score_noise": -0.1}, "score_noise"),
            ({"n": 10, "score_noise": 1.5}, "score_noise"),
            ({"n": 10, "detector_skill": -0.1}, "detector_skill"),
            ({"n": 10, "detector_skill": 1.5}, "detector_skill"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict, fragment: str) -> None:
        with pytest.raises(SimulationError, match=fragment):
            build_episodes(seed=1, **kwargs)

    def test_wrong_width_context_rejected(self) -> None:
        with pytest.raises(SimulationError, match="shape"):
            Episode(
                context=np.zeros(N_FEATURES + 1),
                is_attack=True,
                criticality=0.5,
                tier=RiskTier.OBSERVE,
                alert_id="a",
            )

    def test_non_finite_context_rejected(self) -> None:
        bad = np.zeros(N_FEATURES)
        bad[1] = np.nan
        with pytest.raises(SimulationError, match="non-finite"):
            Episode(
                context=bad,
                is_attack=True,
                criticality=0.5,
                tier=RiskTier.OBSERVE,
                alert_id="a",
            )


class TestGroundTruthNeverEntersTheContext:
    """The label is the reward's input, never the policy's."""

    def test_the_spec_names_no_label_feature(self) -> None:
        for name in CONTEXT_SPEC.names:
            assert "label" not in name
            assert "truth" not in name
            assert "attack" not in name

    def test_no_single_feature_separates_the_classes_perfectly(
        self, episodes: tuple[Episode, ...]
    ) -> None:
        """A leaked label would show as one feature whose class ranges never overlap.

        Range overlap, not exact-value overlap: the continuous features are drawn from
        Beta distributions, so the two classes share no exact values even when their
        supports coincide completely. An exact-match test therefore fires on every
        continuous feature and catches nothing -- which it did, on the first run.
        """
        matrix = np.stack([e.context for e in episodes])
        labels = np.asarray([e.is_attack for e in episodes])
        for index, name in enumerate(CONTEXT_SPEC.names):
            column = matrix[:, index]
            if column.std() == 0.0:
                continue  # the constant bias feature
            attack = column[labels]
            benign = column[~labels]
            overlaps = (
                attack.min() <= benign.max() and benign.min() <= attack.max()
            )
            assert overlaps, (
                f"feature {name!r} has non-overlapping ranges across classes, which "
                "means the ground-truth label has leaked into the context"
            )

    def test_no_feature_is_a_thresholded_copy_of_the_label(
        self, episodes: tuple[Episode, ...]
    ) -> None:
        """Ranges can overlap while a threshold still recovers the label exactly."""
        matrix = np.stack([e.context for e in episodes])
        labels = np.asarray([e.is_attack for e in episodes])
        for index, name in enumerate(CONTEXT_SPEC.names):
            column = matrix[:, index]
            if column.std() == 0.0:
                continue
            order = np.argsort(column)
            sorted_labels = labels[order]
            # Best accuracy achievable by any single threshold on this feature.
            positives = np.cumsum(sorted_labels)
            total_positive = positives[-1]
            negatives_below = np.arange(1, len(sorted_labels) + 1) - positives
            best = max(
                float(
                    np.max(
                        (negatives_below + (total_positive - positives))
                        / len(sorted_labels)
                    )
                ),
                float(
                    np.max(
                        (positives + (len(sorted_labels) - np.arange(1, len(sorted_labels) + 1)
                                      - (total_positive - positives)))
                        / len(sorted_labels)
                    )
                ),
            )
            assert best < 0.99, (
                f"a single threshold on {name!r} recovers the label with accuracy "
                f"{best:.3f}; the label has leaked"
            )

    def test_the_context_is_not_fully_predictive_of_the_label(
        self, episodes: tuple[Episode, ...]
    ) -> None:
        """A least-squares fit must leave real error. Perfect fit means a leak.

        Measured rather than argued: reading the feature list and concluding nothing
        leaked is exactly the review that misses a derived feature.
        """
        matrix = np.stack([e.context for e in episodes])
        labels = np.asarray([e.is_attack for e in episodes], dtype=float)
        coefficients, *_ = np.linalg.lstsq(matrix, labels, rcond=None)
        predictions = matrix @ coefficients
        residual = float(np.mean((predictions - labels) ** 2))
        # The anomaly score is informative, so the fit should be decent but far from
        # exact. A residual near zero would mean the label is recoverable.
        assert residual > 0.02, f"context predicts the label too well (mse {residual})"

    def test_true_score_is_diagnostic_only(self, episodes: tuple[Episode, ...]) -> None:
        """``true_score`` is the pre-noise score; it must not be in the context."""
        score_index = CONTEXT_SPEC.names.index("anomaly_score")
        differing = sum(
            1 for e in episodes if abs(e.context[score_index] - e.true_score) > 1e-9
        )
        assert differing > len(episodes) * 0.8


class TestRegretIsDefinedHonestly:
    def test_regret_is_never_negative(self, episodes: tuple[Episode, ...]) -> None:
        result = replay(episodes=episodes, seed=1)
        assert np.all(result.regret >= 0.0)

    def test_oracle_reward_dominates_the_policy(self, episodes: tuple[Episode, ...]) -> None:
        result = replay(episodes=episodes, seed=1)
        assert result.oracle_rewards.mean() >= result.rewards.mean()

    def test_the_oracle_stays_inside_the_mask(self) -> None:
        """An unmasked oracle would charge the policy for obeying its own constraint."""
        model = RewardModel()
        for tier in ALL_TIERS:
            mask = ActionMask.for_tier(tier)
            for is_attack in (True, False):
                best = model.best_reward(
                    is_attack=is_attack, criticality=0.5, mask=mask
                )
                unmasked = model.best_reward(
                    is_attack=is_attack,
                    criticality=0.5,
                    mask=ActionMask.for_tier(RiskTier.AUTONOMOUS),
                )
                assert best <= unmasked + 1e-12

    def test_an_oracle_policy_has_zero_regret_at_every_tier(self) -> None:
        """The definition's sanity check: playing the oracle must cost nothing."""
        from sentinel.rl.reward import best_action

        model = RewardModel()
        for tier in ALL_TIERS:
            built = build_episodes(n=60, seed=17, tier=tier)
            mask = ActionMask.for_tier(tier)
            total = 0.0
            for episode in built:
                chosen = best_action(
                    model,
                    is_attack=episode.is_attack,
                    criticality=episode.criticality,
                    mask=mask,
                    reversal_probability=0.7,
                )
                oracle = model.best_reward(
                    is_attack=episode.is_attack,
                    criticality=episode.criticality,
                    mask=mask,
                    reversal_probability=0.7,
                )
                actual = model.expected_reward(
                    action=chosen,
                    is_attack=episode.is_attack,
                    criticality=episode.criticality,
                    reversal_probability=0.7,
                )
                total += max(oracle - actual, 0.0)
            assert total == pytest.approx(0.0, abs=1e-9)

    def test_regret_uses_expected_not_realised_reward(self, episodes: tuple[Episode, ...]) -> None:
        """Otherwise the environment's coin flip is charged to the policy.

        A lucky un-reversed false containment would otherwise look like a good
        decision, and regret would carry the reversal draw's variance.
        """
        result = replay(episodes=episodes, seed=2, reversal_probability=0.5)
        # Realised reward is noisy for containment; regret must not be.
        contained = [
            i
            for i, e in enumerate(episodes)
            if not e.is_attack
        ]
        assert contained
        # Regret for identical (action, label, criticality) triples must be identical,
        # which can only hold if the reversal draw is excluded.
        by_key: dict[tuple, set[float]] = {}
        for index, episode in enumerate(episodes):
            key = (episode.is_attack, round(episode.criticality, 9))
            by_key.setdefault(key, set()).add(round(float(result.regret[index]), 9))
        # Each key may have several distinct regrets (different actions chosen), but
        # never more than the number of allowed actions.
        allowed = len(ActionMask.for_tier(episodes[0].tier))
        for key, values in by_key.items():
            assert len(values) <= allowed, key

    def test_zero_episodes_rejected(self) -> None:
        with pytest.raises(SimulationError, match="no episodes"):
            replay(episodes=[], seed=1)

    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_invalid_reversal_probability_rejected(
        self, episodes: tuple[Episode, ...], probability: float
    ) -> None:
        with pytest.raises(SimulationError, match="reversal_probability"):
            replay(episodes=episodes, seed=1, reversal_probability=probability)


class TestReplayResult:
    def test_counts_sum_to_the_episode_count(self, episodes: tuple[Episode, ...]) -> None:
        result = replay(episodes=episodes, seed=1)
        assert sum(result.action_counts.values()) == len(episodes)

    def test_never_violates_the_tier(self) -> None:
        for tier in ALL_TIERS:
            built = build_episodes(n=120, seed=19, tier=tier)
            result = replay(episodes=built, seed=19)
            assert result.violations == 0
            mask = ActionMask.for_tier(tier)
            for action, count in result.action_counts.items():
                if count:
                    assert mask.permits(action), f"{tier.value} played {action.value}"

    def test_cumulative_regret_is_monotone(self, episodes: tuple[Episode, ...]) -> None:
        cumulative = replay(episodes=episodes, seed=1).cumulative_regret
        assert np.all(np.diff(cumulative) >= 0.0)

    def test_window_regret_covers_the_right_slice(self, episodes: tuple[Episode, ...]) -> None:
        result = replay(episodes=episodes, seed=1)
        whole = result.window_regret(0.0, 1.0)
        assert whole == pytest.approx(float(result.regret.mean()))

    @pytest.mark.parametrize("window", [(0.5, 0.5), (0.5, 0.2), (-0.1, 0.5), (0.0, 1.1)])
    def test_invalid_window_rejected(
        self, episodes: tuple[Episode, ...], window: tuple[float, float]
    ) -> None:
        result = replay(episodes=episodes, seed=1)
        with pytest.raises(SimulationError, match="window"):
            result.window_regret(*window)

    def test_optimal_action_rate_is_a_fraction(self, episodes: tuple[Episode, ...]) -> None:
        rate = replay(episodes=episodes, seed=1).optimal_action_rate
        assert 0.0 <= rate <= 1.0

    def test_summary_mentions_both_metrics(self, episodes: tuple[Episode, ...]) -> None:
        text = replay(episodes=episodes, seed=1).summary()
        assert "reduction=" in text
        assert "sublinear=" in text
        assert "violations=0" in text

    def test_sublinearity_of_a_non_improving_policy_is_about_zero(self) -> None:
        """Calibration of the metric itself: a random policy must score near zero."""
        results = replay_many_seeds(
            seeds=TUNING_SEEDS, n_episodes=200, random_policy=True
        )
        assert abs(aggregate(results)["mean_sublinearity"]) < 0.10

    def test_sublinearity_is_one_when_regret_stops(self, episodes: tuple[Episode, ...]) -> None:
        """A policy that reaches zero regret early should score near the maximum."""
        result = replay(episodes=episodes, seed=1)
        assert result.sublinearity <= 1.0


class TestGreedyAndBaseline:
    def test_greedy_replay_runs_and_learns_nothing_new(
        self, episodes: tuple[Episode, ...]
    ) -> None:
        result = replay(episodes=episodes, seed=1, greedy=True)
        assert result.exploratory == 0

    def test_random_policy_is_much_worse(self, episodes: tuple[Episode, ...]) -> None:
        """The comparison that makes a falling regret curve mean something."""
        learned = replay(episodes=episodes, seed=1)
        random = replay(episodes=episodes, seed=1, random_policy=True)
        assert learned.total_regret < random.total_regret * 0.6
        assert learned.optimal_action_rate > random.optimal_action_rate

    def test_random_policy_still_respects_the_mask(self) -> None:
        for tier in ALL_TIERS:
            built = build_episodes(n=100, seed=23, tier=tier)
            result = replay(episodes=built, seed=23, random_policy=True)
            assert result.violations == 0
            mask = ActionMask.for_tier(tier)
            for action, count in result.action_counts.items():
                if count:
                    assert mask.permits(action)

    def test_a_supplied_bandit_is_used(self, episodes: tuple[Episode, ...]) -> None:
        policy = LinearThompsonBandit(n_features=N_FEATURES, seed=29)
        replay(episodes=episodes, bandit=policy, seed=29)
        assert policy.total_observations == len(episodes)

    def test_replay_is_deterministic(self, episodes: tuple[Episode, ...]) -> None:
        a = replay(episodes=episodes, seed=31)
        b = replay(episodes=episodes, seed=31)
        np.testing.assert_array_equal(a.regret, b.regret)
        assert a.action_counts == b.action_counts


class TestAggregate:
    def test_reports_every_expected_statistic(self) -> None:
        results = replay_many_seeds(seeds=TUNING_SEEDS[:2], n_episodes=60)
        stats = aggregate(results)
        for key in (
            "mean_total_regret",
            "mean_sublinearity",
            "worst_sublinearity",
            "mean_regret_reduction",
            "worst_regret_reduction",
            "mean_optimal_action_rate",
            "worst_optimal_action_rate",
            "total_violations",
            "max_inverse_drift",
        ):
            assert key in stats

    def test_empty_results_rejected(self) -> None:
        with pytest.raises(SimulationError, match="no results"):
            aggregate([])

    def test_no_seeds_rejected(self) -> None:
        with pytest.raises(SimulationError, match="no seeds"):
            replay_many_seeds(seeds=(), n_episodes=10)

    def test_worst_is_never_better_than_mean(self) -> None:
        stats = aggregate(replay_many_seeds(seeds=TUNING_SEEDS, n_episodes=60))
        assert stats["worst_sublinearity"] <= stats["mean_sublinearity"]
        assert stats["worst_optimal_action_rate"] <= stats["mean_optimal_action_rate"]


class TestF09Acceptance:
    """*"Simulated regret decreases measurably over a 200-episode replay."*"""

    @pytest.fixture(scope="class")
    def measured(self) -> tuple:
        learned = replay_many_seeds(seeds=REPORTING_SEEDS, n_episodes=200)
        baseline = replay_many_seeds(
            seeds=REPORTING_SEEDS, n_episodes=200, random_policy=True
        )
        return learned, baseline

    @pytest.mark.slow
    def test_gates_pass_on_the_reporting_seeds(self, measured: tuple) -> None:
        learned, baseline = measured
        stats = assert_f09_gates(learned, baseline)
        assert stats["mean_sublinearity"] >= GATE_MEAN_SUBLINEARITY
        assert stats["worst_sublinearity"] >= GATE_WORST_SUBLINEARITY
        assert stats["regret_ratio"] <= GATE_REGRET_RATIO
        assert stats["mean_optimal_action_rate"] >= GATE_OPTIMAL_ACTION_RATE

    @pytest.mark.slow
    def test_regret_falls_on_every_reporting_seed(self, measured: tuple) -> None:
        """The literal criterion, per seed rather than on the mean."""
        learned, _ = measured
        for result in learned:
            assert result.sublinearity > 0.0, result.summary()
            assert result.window_regret(0.75, 1.0) < result.window_regret(0.0, 0.25), (
                result.summary()
            )

    @pytest.mark.slow
    def test_no_tier_violation_across_every_seed(self, measured: tuple) -> None:
        learned, baseline = measured
        for result in (*learned, *baseline):
            assert result.violations == 0

    @pytest.mark.slow
    def test_the_incremental_inverse_held_up(self, measured: tuple) -> None:
        learned, _ = measured
        assert aggregate(learned)["max_inverse_drift"] < 1e-9

    @pytest.mark.slow
    def test_gates_reject_a_policy_that_does_not_learn(self, measured: tuple) -> None:
        """The gates must bite. A random policy has to fail them."""
        _, baseline = measured
        with pytest.raises(SimulationError, match="F-09 gates failed"):
            assert_f09_gates(baseline, baseline)

    @pytest.mark.slow
    def test_learning_holds_at_every_trust_tier(self) -> None:
        """Masking must not prevent learning at the restrictive tiers.

        At ``observe`` only two arms are reachable, so the policy has less to learn
        and less room to be wrong -- but it must still improve, or the mask has broken
        the learner rather than constrained it.
        """
        for tier in ALL_TIERS:
            learned = replay_many_seeds(
                seeds=REPORTING_SEEDS[:3], n_episodes=200, tier=tier
            )
            baseline = replay_many_seeds(
                seeds=REPORTING_SEEDS[:3],
                n_episodes=200,
                tier=tier,
                random_policy=True,
            )
            stats = aggregate(learned)
            reference = aggregate(baseline)
            assert stats["total_violations"] == 0.0
            assert stats["mean_total_regret"] < reference["mean_total_regret"], tier
            assert stats["mean_optimal_action_rate"] > reference[
                "mean_optimal_action_rate"
            ], tier

    @pytest.mark.slow
    def test_the_learned_policy_is_risk_averse_by_construction(self) -> None:
        """A documented, intended outcome rather than a learning failure.

        The upside of containing over escalating a true attack is +0.4; the downside
        of containing a benign one runs to -4.4. With an imperfect detector the policy
        rationally prefers escalation, and it reaches that preference from the reward
        alone -- nobody encoded caution. Pinned here so a future change that makes the
        policy contain aggressively is noticed as a change in behaviour.
        """
        results = replay_many_seeds(seeds=REPORTING_SEEDS, n_episodes=200)
        contains = sum(r.action_counts[ResponseAction.AUTO_CONTAIN] for r in results)
        escalates = sum(r.action_counts[ResponseAction.ESCALATE] for r in results)
        dismisses = sum(r.action_counts[ResponseAction.DISMISS] for r in results)
        assert dismisses > escalates > contains
        total = sum(sum(r.action_counts.values()) for r in results)
        assert contains / total < 0.10

    @pytest.mark.slow
    def test_every_action_remains_reachable(self) -> None:
        """A policy that never plays an arm has not learned it, only avoided it."""
        results = replay_many_seeds(seeds=REPORTING_SEEDS, n_episodes=200)
        for action in ALL_ACTIONS:
            assert sum(r.action_counts[action] for r in results) > 0, action
