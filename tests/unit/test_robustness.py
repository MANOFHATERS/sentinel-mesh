"""Augmentation invariants and the calibration probe (:mod:`sentinel.ml.robustness`).

The classes that carry the weight:

*   :class:`TestAugmentationInvariants` — the three properties that keep the
    augmentation *measurement* honest: train-only, majority untouched, synthetic rows
    labelled. Each one, if violated, produces a better number and a worse system.
*   :class:`TestTheProbeHasTeeth` — the probe is only worth having if it fails on a
    model that deserves to fail. The raw softmax classifier does, and that is asserted
    directly, so the gates cannot quietly become unfalsifiable.
*   :class:`TestNoveltyGateFixesTheCollapse` — the measured remedy. Raw softmax peaks
    at +0.32 overconfidence; the gated confidence peaks at -0.004. Both halves are
    pinned, because the claim being made is comparative.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ml.anomaly import build_default_ensemble
from sentinel.ml.classify import FamilyClassifier
from sentinel.ml.diffusion import TabularDiffusion
from sentinel.ml.robustness import (
    BENIGN_FAMILY,
    DEFAULT_SCALES,
    GATE_MAX_OVERCONFIDENCE,
    NoveltyGate,
    RobustnessError,
    assert_calibration_holds,
    augment_training_set,
    boundary_adjacent_samples,
    calibration_report,
    expected_calibration_error,
    gated_calibration_report,
    gated_robustness_curve,
    perturb,
    robustness_curve,
)

RARE = ("web_attack", "botnet", "infiltration", "brute_force")


@pytest.fixture(scope="module")
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from sentinel.ml.datasets.synthetic import generate_alerts
    from sentinel.ml.featurestore import AlertVectorizer

    alerts = generate_alerts(16_000, seed=20260929)
    vectorizer = AlertVectorizer().fit(alerts)
    matrix = vectorizer.transform(alerts)
    families = np.asarray([a.ground_truth_label or "benign" for a in alerts])
    order = np.random.default_rng(7).permutation(len(families))
    cut = int(0.6 * len(order))
    train, test = order[:cut], order[cut:]
    return matrix[train], families[train], matrix[test], list(families[test])


@pytest.fixture(scope="module")
def generator(dataset) -> TabularDiffusion:
    x_train, f_train, _, _ = dataset
    mask = np.isin(f_train, RARE)
    return TabularDiffusion(
        n_steps=200, epochs=200, hidden=(160, 160), seed=1
    ).fit(x_train[mask], list(f_train[mask]))


@pytest.fixture(scope="module")
def classifier(dataset) -> FamilyClassifier:
    x_train, f_train, _, _ = dataset
    return FamilyClassifier(seed=5).fit(x_train, list(f_train))


@pytest.fixture(scope="module")
def support_detector(dataset):
    """Fitted on the classifier's *whole* training set, not benign only.

    The distinction is the finding. A benign-only detector answers "is this an attack",
    and every real attack in the classifier's training data scores at its ceiling, so
    its 99th percentile is 1.0 and a gate built on it has no headroom and stays inert.
    A detector fitted on the full training distribution answers "has the classifier
    seen anything like this", which is the question an out-of-distribution discount
    needs.
    """
    x_train, _, _, _ = dataset
    ensemble = build_default_ensemble()
    ensemble.fit(x_train)
    return ensemble


@pytest.fixture(scope="module")
def gate(dataset, support_detector) -> NoveltyGate:
    x_train, _, _, _ = dataset
    return NoveltyGate(quantile=0.99).fit(support_detector.score(x_train))


class TestAugmentationInvariants:
    def test_adds_rows_for_the_modelled_families(self, dataset, generator) -> None:
        x_train, f_train, _, _ = dataset
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=3.0,
            rng=np.random.default_rng(11),
        )
        assert result.n_synthetic > 0
        assert set(result.counts_added) <= set(generator.families_)

    def test_the_majority_class_is_untouched_bit_for_bit(self, dataset, generator) -> None:
        """PRD 5.5.5: *without touching the majority class distribution*."""
        x_train, f_train, _, _ = dataset
        labels = list(f_train)
        before = x_train[np.asarray(labels) == BENIGN_FAMILY]
        result = augment_training_set(
            x_train, labels, generator=generator, multiplier=3.0,
            rng=np.random.default_rng(11),
        )
        after = result.x[np.asarray(result.families) == BENIGN_FAMILY]
        assert after.shape == before.shape
        np.testing.assert_array_equal(after, before)

    def test_no_synthetic_row_is_labelled_benign(self, dataset, generator) -> None:
        x_train, f_train, _, _ = dataset
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=3.0,
            rng=np.random.default_rng(11),
        )
        synthetic_labels = np.asarray(result.families)[result.synthetic]
        assert BENIGN_FAMILY not in set(synthetic_labels)

    def test_real_rows_are_preserved_and_come_first(self, dataset, generator) -> None:
        x_train, f_train, _, _ = dataset
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=2.0,
            rng=np.random.default_rng(11),
        )
        assert result.n_real == x_train.shape[0]
        np.testing.assert_array_equal(result.real_rows(), x_train)
        assert not result.synthetic[: x_train.shape[0]].any()

    def test_the_synthetic_mask_is_returned_and_accurate(self, dataset, generator) -> None:
        """Without the mask a caller cannot honour any of the other invariants."""
        x_train, f_train, _, _ = dataset
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=2.0,
            rng=np.random.default_rng(11),
        )
        assert result.synthetic.shape == (result.x.shape[0],)
        assert result.n_real + result.n_synthetic == result.x.shape[0]
        assert result.n_synthetic == sum(result.counts_added.values())

    def test_a_generator_fitted_on_the_majority_class_is_refused(self, dataset) -> None:
        """The one configuration that would silently violate the PRD constraint."""
        x_train, f_train, _, _ = dataset
        mask = f_train == BENIGN_FAMILY
        benign_generator = TabularDiffusion(
            n_steps=20, epochs=30, hidden=(16,), seed=3
        ).fit(x_train[mask][:500], [BENIGN_FAMILY] * 500)
        with pytest.raises(RobustnessError, match="majority class"):
            augment_training_set(
                x_train, list(f_train), generator=benign_generator,
                rng=np.random.default_rng(1),
            )

    def test_multiplier_below_one_rejected(self, dataset, generator) -> None:
        x_train, f_train, _, _ = dataset
        with pytest.raises(RobustnessError, match="multiplier"):
            augment_training_set(
                x_train, list(f_train), generator=generator, multiplier=0.5,
                rng=np.random.default_rng(1),
            )

    def test_unfitted_generator_rejected(self, dataset) -> None:
        x_train, f_train, _, _ = dataset
        with pytest.raises(RobustnessError, match="fitted"):
            augment_training_set(
                x_train, list(f_train), generator=TabularDiffusion(),
                rng=np.random.default_rng(1),
            )

    def test_absolute_target_is_honoured(self, dataset, generator) -> None:
        x_train, f_train, _, _ = dataset
        labels = list(f_train)
        counts = {family: labels.count(family) for family in generator.families_}
        result = augment_training_set(
            x_train, labels, generator=generator, target_per_family=400,
            rng=np.random.default_rng(11),
        )
        for family, added in result.counts_added.items():
            assert counts[family] + added == 400

    def test_nothing_to_add_returns_the_input_unchanged(self, dataset, generator) -> None:
        x_train, f_train, _, _ = dataset
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=1.0,
            rng=np.random.default_rng(11),
        )
        assert result.n_synthetic == 0
        np.testing.assert_array_equal(result.x, x_train)

    def test_shape_mismatch_rejected(self, generator) -> None:
        with pytest.raises(RobustnessError, match="labels"):
            augment_training_set(
                np.zeros((4, generator.n_features)), ["a", "b"], generator=generator
            )

    def test_summary_is_readable(self, dataset, generator) -> None:
        x_train, f_train, _, _ = dataset
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=2.0,
            rng=np.random.default_rng(11),
        )
        assert "real" in result.summary()
        assert "synthetic" in result.summary()


class TestAugmentationStaysOutOfTheEvaluationSplit:
    def test_no_synthetic_row_appears_in_the_held_out_set(self, dataset, generator) -> None:
        """The property that makes the reported delta meaningful."""
        x_train, f_train, x_test, _ = dataset
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=3.0,
            rng=np.random.default_rng(11),
        )
        synthetic = result.x[result.synthetic]
        # Nearest-neighbour distance from every test row to every synthetic row must
        # be strictly positive: a synthetic row that leaked would be at distance zero.
        for row in synthetic[:100]:
            assert float(np.min(np.linalg.norm(x_test - row, axis=1))) > 1e-9

    def test_the_function_cannot_see_the_evaluation_split(self) -> None:
        """Structural, not behavioural: there is no parameter for it."""
        import inspect

        parameters = set(inspect.signature(augment_training_set).parameters)
        assert parameters == {
            "x",
            "families",
            "generator",
            "target_per_family",
            "multiplier",
            "rng",
            "benign_family",
        }


class TestExpectedCalibrationError:
    def test_perfect_calibration_scores_zero(self) -> None:
        confidences = np.asarray([0.95] * 100)
        correct = np.asarray([True] * 95 + [False] * 5)
        assert expected_calibration_error(confidences, correct, n_bins=10) < 0.02

    def test_total_overconfidence_scores_near_one(self) -> None:
        confidences = np.ones(100)
        correct = np.zeros(100, dtype=bool)
        assert expected_calibration_error(confidences, correct) > 0.9

    def test_total_underconfidence_also_scores_high(self) -> None:
        """ECE is an absolute gap; direction is reported by ``overconfidence``."""
        confidences = np.full(100, 0.1)
        correct = np.ones(100, dtype=bool)
        assert expected_calibration_error(confidences, correct) > 0.8

    def test_empty_bins_contribute_nothing(self) -> None:
        """A naive implementation counts them as perfectly calibrated."""
        confidences = np.full(50, 0.55)
        correct = np.asarray([True] * 28 + [False] * 22)
        coarse = expected_calibration_error(confidences, correct, n_bins=2)
        fine = expected_calibration_error(confidences, correct, n_bins=50)
        assert coarse == pytest.approx(fine, abs=0.02)

    def test_confidence_of_exactly_one_is_binned(self) -> None:
        """A half-open final bin would silently drop every saturated softmax output."""
        confidences = np.ones(10)
        correct = np.zeros(10, dtype=bool)
        assert expected_calibration_error(confidences, correct) > 0.9

    @pytest.mark.parametrize(
        "args,fragment",
        [
            ((np.zeros(3), np.zeros(4, dtype=bool)), "different lengths"),
            ((np.zeros(0), np.zeros(0, dtype=bool)), "empty"),
        ],
    )
    def test_malformed_input_rejected(self, args: tuple, fragment: str) -> None:
        with pytest.raises(RobustnessError, match=fragment):
            expected_calibration_error(*args)

    def test_too_few_bins_rejected(self) -> None:
        with pytest.raises(RobustnessError, match="n_bins"):
            expected_calibration_error(np.ones(4), np.ones(4, dtype=bool), n_bins=1)


class TestPerturb:
    def test_zero_scale_is_a_copy(self, dataset) -> None:
        _, _, x_test, _ = dataset
        out = perturb(x_test, scale=0.0, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(out, x_test)
        assert out is not x_test

    def test_perturbation_grows_with_scale(self, dataset) -> None:
        _, _, x_test, _ = dataset
        small = perturb(x_test, scale=0.1, rng=np.random.default_rng(1))
        large = perturb(x_test, scale=1.0, rng=np.random.default_rng(1))
        assert np.abs(large - x_test).mean() > np.abs(small - x_test).mean()

    def test_perturbation_is_in_units_of_column_spread(self) -> None:
        """A shared absolute nudge is meaningless across heterogeneous columns."""
        matrix = np.zeros((2000, 2))
        rng = np.random.default_rng(3)
        matrix[:, 0] = rng.normal(0.0, 1.0, size=2000)
        matrix[:, 1] = rng.normal(0.0, 100.0, size=2000)
        out = perturb(matrix, scale=0.5, rng=np.random.default_rng(5))
        deltas = np.abs(out - matrix).mean(axis=0)
        assert deltas[1] / deltas[0] > 50.0

    def test_negative_scale_rejected(self, dataset) -> None:
        _, _, x_test, _ = dataset
        with pytest.raises(RobustnessError, match="scale"):
            perturb(x_test, scale=-0.1, rng=np.random.default_rng(1))

    def test_wrong_column_scale_rejected(self, dataset) -> None:
        _, _, x_test, _ = dataset
        with pytest.raises(RobustnessError, match="column_scale"):
            perturb(
                x_test, scale=0.5, rng=np.random.default_rng(1),
                column_scale=np.ones(3),
            )


class TestBoundaryAdjacentSamples:
    def test_finds_the_requested_count(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        rows, left, right = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=50, rng=np.random.default_rng(17)
        )
        assert rows.shape[0] == 50
        assert len(left) == len(right) == 50

    def test_endpoint_labels_differ(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        _, left, right = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=50, rng=np.random.default_rng(17)
        )
        for a, b in zip(left, right, strict=True):
            assert a != b

    def test_boundary_points_are_genuinely_ambiguous(self, classifier, dataset) -> None:
        """The classifier's confidence on them must be far below clean-data confidence."""
        _, _, x_test, f_test = dataset
        rows, _, _ = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=200, rng=np.random.default_rng(17)
        )
        assert classifier.confidence(rows).mean() < classifier.confidence(x_test).mean()

    def test_points_lie_on_the_segment_between_real_samples(
        self, classifier, dataset
    ) -> None:
        """Interpolation, not an adversarial direction pointing off the manifold."""
        _, _, x_test, f_test = dataset
        rows, _, _ = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=30, rng=np.random.default_rng(19)
        )
        low, high = x_test.min(axis=0), x_test.max(axis=0)
        assert np.all(rows >= low - 1e-9)
        assert np.all(rows <= high + 1e-9)

    def test_deterministic_for_a_seed(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        first, _, _ = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=20, rng=np.random.default_rng(23)
        )
        second, _, _ = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=20, rng=np.random.default_rng(23)
        )
        np.testing.assert_array_equal(first, second)

    def test_a_single_class_classifier_is_refused(self, dataset) -> None:
        _, _, x_test, f_test = dataset

        class OneClass(FamilyClassifier):
            def predict(self, x):  # type: ignore[override]
                return tuple(["benign"] * len(x))

        model = OneClass()
        model.classes_ = ("benign", "ddos")
        with pytest.raises(RobustnessError, match="single class"):
            boundary_adjacent_samples(
                model, x_test, f_test, n_samples=5, rng=np.random.default_rng(1)
            )

    @pytest.mark.parametrize("kwargs", [{"n_samples": 0}, {"bisection_steps": 0}])
    def test_invalid_arguments_rejected(self, classifier, dataset, kwargs: dict) -> None:
        _, _, x_test, f_test = dataset
        with pytest.raises(RobustnessError):
            boundary_adjacent_samples(
                classifier, x_test, f_test, rng=np.random.default_rng(1), **kwargs
            )


class TestTheProbeHasTeeth:
    """A probe that passes everything measures nothing."""

    def test_the_raw_softmax_classifier_fails_the_gates(
        self, classifier, dataset
    ) -> None:
        """The measured defect: peak overconfidence +0.32, confidence drop only 0.029."""
        _, _, x_test, f_test = dataset
        curve = robustness_curve(
            classifier, x_test, f_test, rng=np.random.default_rng(13)
        )
        with pytest.raises(RobustnessError, match="calibration checks failed"):
            assert_calibration_holds(curve)

    def test_the_failure_names_the_overconfidence_peak(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        curve = robustness_curve(
            classifier, x_test, f_test, rng=np.random.default_rng(13)
        )
        with pytest.raises(RobustnessError) as caught:
            assert_calibration_holds(curve)
        assert "overconfidence peaks" in str(caught.value)

    def test_raw_confidence_is_u_shaped_not_monotone(self, classifier, dataset) -> None:
        """The reason the gate inspects the whole sweep and not just its endpoints.

        Confidence dips in the middle and recovers at extreme perturbation, so an
        endpoint comparison reads this curve as a fall and passes it.
        """
        _, _, x_test, f_test = dataset
        curve = robustness_curve(
            classifier, x_test, f_test, rng=np.random.default_rng(13)
        )
        confidences = [report.mean_confidence for report in curve]
        assert confidences[-1] > min(confidences), confidences

    def test_accuracy_does_fall_under_perturbation(self, classifier, dataset) -> None:
        """Establishes the perturbation is doing something, before judging confidence."""
        _, _, x_test, f_test = dataset
        curve = robustness_curve(
            classifier, x_test, f_test, rng=np.random.default_rng(13)
        )
        assert curve[0].accuracy - curve[-1].accuracy > 0.15

    def test_label_smoothing_does_not_fix_it(self, dataset) -> None:
        """The standard cheap remedy, measured. It makes peak overconfidence worse.

        Pinned because it is the obvious next thing to try, and trying it costs a
        training run and yields a worse model.
        """
        x_train, f_train, x_test, f_test = dataset
        sharp = FamilyClassifier(seed=5).fit(x_train, list(f_train))
        smooth = FamilyClassifier(seed=5, label_smoothing=0.1).fit(
            x_train, list(f_train)
        )
        sharp_peak = max(
            r.overconfidence
            for r in robustness_curve(
                sharp, x_test, f_test, rng=np.random.default_rng(13)
            )
        )
        smooth_peak = max(
            r.overconfidence
            for r in robustness_curve(
                smooth, x_test, f_test, rng=np.random.default_rng(13)
            )
        )
        assert smooth_peak > sharp_peak
        assert smooth_peak > GATE_MAX_OVERCONFIDENCE

    def test_gates_require_at_least_two_points(self, classifier, dataset) -> None:
        """A one-point sweep has nothing to compare, so judging it would be a fiction."""
        _, _, x_test, f_test = dataset
        single = robustness_curve(
            classifier, x_test[:200], f_test[:200], scales=(0.0,),
            rng=np.random.default_rng(1),
        )
        assert len(single) == 1
        with pytest.raises(RobustnessError, match="at least"):
            assert_calibration_holds(single)


class TestNoveltyGate:
    def test_quantile_must_be_sensible(self) -> None:
        for quantile in (0.4, 1.0, 1.5):
            with pytest.raises(RobustnessError, match="quantile"):
                NoveltyGate(quantile=quantile)

    def test_unfitted_gate_refuses(self) -> None:
        with pytest.raises(RobustnessError, match="not fitted"):
            NoveltyGate().excess(np.asarray([0.5]))

    def test_empty_fit_rejected(self) -> None:
        with pytest.raises(RobustnessError, match="empty"):
            NoveltyGate().fit(np.zeros(0))

    def test_non_finite_fit_rejected(self) -> None:
        with pytest.raises(RobustnessError, match="non-finite"):
            NoveltyGate().fit(np.asarray([0.1, np.nan]))

    def test_in_distribution_samples_are_not_discounted(self, gate) -> None:
        excess = gate.excess(np.asarray([0.0, 0.1, gate.reference_ - 0.01]))
        np.testing.assert_allclose(excess, 0.0)

    def test_fully_novel_samples_are_discounted_to_nothing(self, gate) -> None:
        assert gate.excess(np.asarray([1.0]))[0] == pytest.approx(1.0)
        assert gate.adjust(np.asarray([0.99]), np.asarray([1.0]))[0] == pytest.approx(
            0.0
        )

    def test_excess_is_bounded(self, gate) -> None:
        excess = gate.excess(np.linspace(-1.0, 2.0, 50))
        assert np.all(excess >= 0.0)
        assert np.all(excess <= 1.0)

    def test_adjust_is_multiplicative(self, gate) -> None:
        confidence = np.asarray([0.8, 0.8])
        novelty = np.asarray([0.0, 1.0])
        adjusted = gate.adjust(confidence, novelty)
        assert adjusted[0] == pytest.approx(0.8)
        assert adjusted[1] == pytest.approx(0.0)

    def test_a_saturated_detector_makes_the_gate_inert_not_zero(self) -> None:
        """A gate that discounts everything suppresses every alert.

        This is the failure mode found with a benign-only detector, whose 99th
        percentile on the classifier's training data is exactly 1.0.
        """
        saturated = NoveltyGate().fit(np.ones(100))
        assert saturated.reference_ == pytest.approx(1.0)
        np.testing.assert_allclose(saturated.excess(np.ones(5)), 0.0)
        np.testing.assert_allclose(
            saturated.adjust(np.full(5, 0.9), np.ones(5)), 0.9
        )

    def test_mismatched_lengths_rejected(self, gate) -> None:
        with pytest.raises(RobustnessError, match="entries"):
            gate.adjust(np.ones(3), np.ones(4))

    def test_the_support_detector_tracks_perturbation_monotonically(
        self, dataset, support_detector
    ) -> None:
        """The property the raw softmax lacks, and the reason the gate works."""
        _, _, x_test, _ = dataset
        spread = x_test.std(axis=0)
        column_scale = np.where(spread > 1e-12, spread, 1.0)
        means = [
            float(
                support_detector.score(
                    perturb(
                        x_test, scale=scale, rng=np.random.default_rng(13),
                        column_scale=column_scale,
                    )
                ).mean()
            )
            for scale in DEFAULT_SCALES
        ]
        assert means == sorted(means), means
        assert means[-1] - means[0] > 0.3


class TestNoveltyGateFixesTheCollapse:
    """The comparative claim, both halves pinned."""

    @pytest.fixture(scope="class")
    def curves(self, classifier, dataset, support_detector, gate):
        _, _, x_test, f_test = dataset
        raw = robustness_curve(
            classifier, x_test, f_test, rng=np.random.default_rng(13)
        )
        gated = gated_robustness_curve(
            classifier, x_test, f_test, novelty_fn=support_detector.score,
            gate=gate, rng=np.random.default_rng(13),
        )
        return raw, gated

    def test_gated_confidence_clears_the_gates(
        self, curves, classifier, dataset, support_detector, gate
    ) -> None:
        _, gated = curves
        _, _, x_test, f_test = dataset
        rows, left, _ = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=200, rng=np.random.default_rng(17)
        )
        boundary = gated_calibration_report(
            classifier, rows, left, novelty=support_detector.score(rows),
            gate=gate, label="boundary",
        )
        stats = assert_calibration_holds(gated, boundary=boundary)
        assert stats["peak_overconfidence"] <= GATE_MAX_OVERCONFIDENCE
        assert stats["confidence_drop"] > 0.30

    def test_gating_removes_the_overconfidence_peak(self, curves) -> None:
        raw, gated = curves
        raw_peak = max(r.overconfidence for r in raw)
        gated_peak = max(r.overconfidence for r in gated)
        assert raw_peak > 0.20, raw_peak
        assert gated_peak < 0.05, gated_peak
        assert gated_peak < raw_peak

    def test_gating_leaves_clean_data_almost_alone(self, curves) -> None:
        """An out-of-distribution discount must not tax in-distribution inputs."""
        raw, gated = curves
        assert abs(raw[0].mean_confidence - gated[0].mean_confidence) < 0.05
        assert gated[0].accuracy == pytest.approx(raw[0].accuracy)

    def test_gating_does_not_change_accuracy_anywhere(self, curves) -> None:
        """It discounts confidence, not predictions. Same argmax, same accuracy."""
        raw, gated = curves
        for a, b in zip(raw, gated, strict=True):
            assert a.accuracy == pytest.approx(b.accuracy)

    def test_gated_confidence_falls_monotonically(self, curves) -> None:
        """No U-turn: the defect the raw curve exhibits is gone, not reduced."""
        _, gated = curves
        confidences = [r.mean_confidence for r in gated]
        assert confidences == sorted(confidences, reverse=True), confidences

    def test_gated_confidence_errs_toward_caution(self, curves) -> None:
        """Under-confidence escalates to a human; over-confidence closes an alert."""
        _, gated = curves
        assert all(r.overconfidence <= GATE_MAX_OVERCONFIDENCE for r in gated)
        assert gated[-1].overconfidence < 0.0


class TestCalibrationReport:
    def test_reports_the_population(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        report = calibration_report(classifier, x_test, f_test, label="clean")
        assert report.n_samples == len(f_test)
        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.mean_confidence <= 1.0
        assert report.overconfidence == pytest.approx(
            report.mean_confidence - report.accuracy
        )

    def test_summary_is_readable(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        text = calibration_report(classifier, x_test, f_test, label="clean").summary()
        for token in ("acc=", "conf=", "ece=", "over="):
            assert token in text

    def test_empty_population_rejected(self, classifier) -> None:
        with pytest.raises(RobustnessError, match="empty"):
            calibration_report(classifier, np.zeros((0, 3)), [], label="empty")

    def test_report_is_immutable(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        report = calibration_report(classifier, x_test, f_test, label="clean")
        with pytest.raises(AttributeError):
            report.accuracy = 0.0  # type: ignore[misc]

    def test_empty_scales_rejected(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        with pytest.raises(RobustnessError, match="scales"):
            robustness_curve(
                classifier, x_test, f_test, scales=(), rng=np.random.default_rng(1)
            )

    def test_negative_scale_rejected(self, classifier, dataset) -> None:
        _, _, x_test, f_test = dataset
        with pytest.raises(RobustnessError, match="non-negative"):
            robustness_curve(
                classifier, x_test, f_test, scales=(0.0, -1.0),
                rng=np.random.default_rng(1),
            )
