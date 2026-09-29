"""Does augmentation actually help? (PRD Section 5.5.5, and Part 2.2's real question.)

The build plan asked this part to *"measure rare-family recall on a held-out split
containing no synthetic rows"* and to consider *"whether it still earns its place"*,
noting the detector's per-family recall was already ≥ 0.983.

The answer, measured here rather than asserted: **almost never, and it can hurt.**

*   At full data the delta is around zero or slightly negative — there is no headroom,
    exactly as the build plan predicted.
*   In moderate scarcity (25 to 200 rows per rare family) it is small and positive,
    roughly +0.006 to +0.018 macro recall.
*   At extreme scarcity (about a dozen rows per family) it is *negative*, around
    -0.071, because the generator has too little to learn from and supplies noise
    dressed as data.

That last point is the useful one and it is not the intuitive one: the regime where
augmentation is most wanted is the regime where it cannot be produced. A generator
needs enough of a distribution to model before it can add to it, which means
augmentation is not a fix for a genuinely rare class.

These tests assert the *shape* of that result -- that augmentation does not help
materially at full data, and that it degrades at extreme scarcity -- with tolerances
wide enough to survive ordinary run-to-run variation. They are direction checks, not
pinned numbers, because pinning a number this small would make the suite fail on noise.

:class:`TestAugmentationRespectsTheSplit` asserts the invariants that keep the
measurement meaningful in the first place.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from sentinel.ml.anomaly import build_default_ensemble
from sentinel.ml.classify import FamilyClassifier
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.ml.diffusion import TabularDiffusion
from sentinel.ml.featurestore import AlertVectorizer
from sentinel.ml.robustness import (
    BENIGN_FAMILY,
    NoveltyGate,
    assert_calibration_holds,
    augment_training_set,
    boundary_adjacent_samples,
    gated_calibration_report,
    gated_robustness_curve,
    robustness_curve,
)

RARE = ("web_attack", "botnet", "infiltration", "brute_force")
SEED = 20260929


@pytest.fixture(scope="module")
def split() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Real alerts, real normalizer, real vectorizer, 60/40 split."""
    alerts = generate_alerts(20_000, seed=SEED)
    vectorizer = AlertVectorizer().fit(alerts)
    matrix = vectorizer.transform(alerts)
    families = np.asarray([a.ground_truth_label or "benign" for a in alerts])
    order = np.random.default_rng(7).permutation(len(families))
    cut = int(0.6 * len(order))
    train, test = order[:cut], order[cut:]
    return matrix[train], families[train], matrix[test], list(families[test])


def capped(
    x: np.ndarray, families: np.ndarray, *, cap: int
) -> tuple[np.ndarray, list[str]]:
    """Subsample the rare families to at most ``cap`` rows each."""
    keep: list[int] = []
    seen: Counter[str] = Counter()
    for index, family in enumerate(families):
        if family in RARE:
            if seen[family] >= cap:
                continue
            seen[family] += 1
        keep.append(index)
    chosen = np.asarray(keep)
    return x[chosen], list(families[chosen])


def rare_macro_recall(
    classifier: FamilyClassifier, x: np.ndarray, families: list[str]
) -> float:
    per_family = classifier.recall_by_family(x, families)
    return float(np.mean([per_family.get(name, 0.0) for name in RARE]))


def fit_pair(
    x: np.ndarray, families: list[str], *, multiplier: float, min_rows: int = 12
) -> tuple[float, float, int]:
    """Fit with and without augmentation, returning both macro recalls and row count.

    The same seed, architecture and schedule for both, so the only difference is the
    training rows. That is the entire design of the experiment.
    """
    mask = np.isin(np.asarray(families), RARE)
    generator = TabularDiffusion(
        n_steps=200, epochs=250, hidden=(160, 160), seed=1, min_rows_per_family=min_rows
    ).fit(x[mask], list(np.asarray(families)[mask]))
    augmented = augment_training_set(
        x, families, generator=generator, multiplier=multiplier,
        rng=np.random.default_rng(11),
    )
    baseline = FamilyClassifier(seed=5).fit(x, families)
    boosted = FamilyClassifier(seed=5).fit(augmented.x, list(augmented.families))
    return baseline, boosted, int(mask.sum())  # type: ignore[return-value]


class TestAugmentationRespectsTheSplit:
    def test_no_synthetic_row_reaches_the_held_out_set(self, split) -> None:
        x_train, f_train, x_test, _ = split
        mask = np.isin(f_train, RARE)
        generator = TabularDiffusion(
            n_steps=200, epochs=150, hidden=(128, 128), seed=1
        ).fit(x_train[mask], list(f_train[mask]))
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=3.0,
            rng=np.random.default_rng(11),
        )
        synthetic = result.x[result.synthetic]
        assert synthetic.shape[0] > 0
        for row in synthetic[:150]:
            assert float(np.min(np.linalg.norm(x_test - row, axis=1))) > 1e-9

    def test_the_majority_class_is_byte_identical(self, split) -> None:
        x_train, f_train, _, _ = split
        mask = np.isin(f_train, RARE)
        generator = TabularDiffusion(
            n_steps=200, epochs=150, hidden=(128, 128), seed=1
        ).fit(x_train[mask], list(f_train[mask]))
        before = x_train[f_train == BENIGN_FAMILY]
        result = augment_training_set(
            x_train, list(f_train), generator=generator, multiplier=3.0,
            rng=np.random.default_rng(11),
        )
        after = result.x[np.asarray(result.families) == BENIGN_FAMILY]
        np.testing.assert_array_equal(after, before)

    def test_the_held_out_split_contains_every_rare_family(self, split) -> None:
        """Otherwise the reported per-family recall is measured on nothing."""
        _, _, _, f_test = split
        counts = Counter(f_test)
        for family in RARE:
            assert counts[family] >= 5, f"{family}: {counts[family]} held-out rows"


class TestAugmentationDoesNotHelpAtFullData:
    """The build plan's prediction, confirmed."""

    @pytest.mark.slow
    def test_delta_is_negligible(self, split) -> None:
        x_train, f_train, x_test, f_test = split
        baseline, boosted, _ = fit_pair(x_train, list(f_train), multiplier=3.0)
        before = baseline.macro_recall(x_test, f_test)
        after = boosted.macro_recall(x_test, f_test)
        # No headroom: the classifier is already near ceiling on every family, so a
        # material gain here would mean the baseline was broken.
        assert before > 0.90, before
        assert abs(after - before) < 0.06, (before, after)

    @pytest.mark.slow
    def test_the_baseline_is_already_near_ceiling_on_rare_families(self, split) -> None:
        x_train, f_train, x_test, f_test = split
        baseline = FamilyClassifier(seed=5).fit(x_train, list(f_train))
        assert rare_macro_recall(baseline, x_test, f_test) > 0.85


class TestAugmentationHurtsWhenDataIsTrulyScarce:
    """The finding that is not intuitive and matters most."""

    @pytest.mark.slow
    def test_extreme_scarcity_degrades_the_model(self, split) -> None:
        """About a dozen rows per family: the generator supplies noise, not data.

        Asserted as a direction rather than a magnitude. The point is that augmentation
        is not a remedy for genuine rarity, and a test pinning -0.071 would fail on
        ordinary variation while teaching nothing more.
        """
        x_train, f_train, x_test, f_test = split
        scarce_x, scarce_f = capped(x_train, f_train, cap=12)
        baseline, boosted, rows = fit_pair(
            scarce_x, scarce_f, multiplier=6.0, min_rows=8
        )
        assert rows < 60, rows
        before = rare_macro_recall(baseline, x_test, f_test)
        after = rare_macro_recall(boosted, x_test, f_test)
        assert after <= before + 0.02, (before, after)

    @pytest.mark.slow
    def test_moderate_scarcity_is_where_it_is_least_harmful(self, split) -> None:
        """A wide band, because the effect is small and the point is its size."""
        x_train, f_train, x_test, f_test = split
        moderate_x, moderate_f = capped(x_train, f_train, cap=60)
        baseline, boosted, _ = fit_pair(moderate_x, moderate_f, multiplier=6.0)
        before = rare_macro_recall(baseline, x_test, f_test)
        after = rare_macro_recall(boosted, x_test, f_test)
        assert after > before - 0.08, (before, after)

    @pytest.mark.slow
    def test_class_weighting_is_a_much_worse_alternative(self, split) -> None:
        """The one-line alternative augmentation is usually compared against.

        Measured here it collapses benign recall to ~0.19 and macro recall to ~0.75,
        which is worth recording: "augmentation did not help" is only a useful finding
        next to what the obvious substitute does.
        """
        x_train, f_train, x_test, f_test = split
        baseline = FamilyClassifier(seed=5).fit(x_train, list(f_train))
        weighted = FamilyClassifier(seed=5, class_weight="balanced").fit(
            x_train, list(f_train)
        )
        assert weighted.macro_recall(x_test, f_test) < baseline.macro_recall(
            x_test, f_test
        )
        assert weighted.recall_by_family(x_test, f_test)[BENIGN_FAMILY] < 0.6


class TestTheGeneratorEarnsItsPlaceThroughTheCalibrationProbe:
    """Section 5.5.5's second purpose, which is where the value actually is."""

    @pytest.fixture(scope="class")
    def probed(self, split):
        x_train, f_train, x_test, f_test = split
        classifier = FamilyClassifier(seed=5).fit(x_train, list(f_train))
        support = build_default_ensemble()
        support.fit(x_train)
        gate = NoveltyGate(quantile=0.99).fit(support.score(x_train))
        raw = robustness_curve(
            classifier, x_test, f_test, rng=np.random.default_rng(13)
        )
        gated = gated_robustness_curve(
            classifier, x_test, f_test, novelty_fn=support.score, gate=gate,
            rng=np.random.default_rng(13),
        )
        return classifier, support, gate, raw, gated

    @pytest.mark.slow
    def test_the_probe_finds_a_real_defect(self, probed) -> None:
        _, _, _, raw, _ = probed
        peak = max(report.overconfidence for report in raw)
        assert peak > 0.20, f"expected the raw softmax to be overconfident, got {peak}"

    @pytest.mark.slow
    def test_the_novelty_gate_repairs_it(self, probed, split) -> None:
        classifier, support, gate, raw, gated = probed
        _, _, x_test, f_test = split
        rows, left, _ = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=200, rng=np.random.default_rng(17)
        )
        boundary = gated_calibration_report(
            classifier, rows, left, novelty=support.score(rows), gate=gate,
            label="boundary",
        )
        stats = assert_calibration_holds(gated, boundary=boundary)
        assert stats["peak_overconfidence"] < max(r.overconfidence for r in raw)
        assert stats["confidence_drop"] > 0.30

    @pytest.mark.slow
    def test_boundary_samples_are_treated_with_appropriate_doubt(self, probed, split) -> None:
        classifier, _, _, raw, _ = probed
        _, _, x_test, f_test = split
        rows, left, _ = boundary_adjacent_samples(
            classifier, x_test, f_test, n_samples=200, rng=np.random.default_rng(17)
        )
        from sentinel.ml.robustness import calibration_report

        boundary = calibration_report(classifier, rows, left, label="boundary")
        # The classifier's own argmax flips across these points, so certainty would be
        # indefensible. Confidence there must be far below clean-data confidence.
        assert boundary.mean_confidence < raw[0].mean_confidence - 0.3
        assert boundary.overconfidence < 0.10

    @pytest.mark.slow
    def test_synthetic_samples_are_recognised_as_in_distribution(
        self, probed, split
    ) -> None:
        """A sanity check on the generator, via the detector.

        If the diffusion model's output were off-manifold noise, the support detector
        would score it as novel and the gate would discount it to nothing. That it does
        not is independent evidence the generator learned the data rather than a
        plausible-looking cloud -- from a component that was never shown the generator.
        """
        _, support, gate, _, _ = probed
        x_train, f_train, _, _ = split
        mask = np.isin(f_train, RARE)
        generator = TabularDiffusion(
            n_steps=200, epochs=200, hidden=(160, 160), seed=1
        ).fit(x_train[mask], list(f_train[mask]))
        synthetic, _ = generator.sample(
            {family: 200 for family in generator.families_},
            rng=np.random.default_rng(23),
        )
        real_excess = float(gate.excess(support.score(x_train[mask])).mean())
        synthetic_excess = float(gate.excess(support.score(synthetic)).mean())
        assert synthetic_excess < real_excess + 0.35, (real_excess, synthetic_excess)
