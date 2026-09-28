"""Anomaly ensemble: calibration, complementarity, and the F-03 AUC acceptance gate.

Two groups of tests matter most.

:class:`TestComplementarity` proves the two detectors are actually different models
rather than two spellings of the same one — a "diverse ensemble" of correlated
members is a false sense of coverage, and PRD Section 5.5.2's stated goal is that
*"a single model's blind spot does not silently define the system's sensitivity"*.
The tests construct the two blind spots directly: a pure marginal outlier that the
forest catches and PCA misses, and a correlation violation that PCA catches and the
forest misses.

:class:`TestAcceptanceCriteria` holds the F-03 gate at ROC-AUC >= 0.90, and holds
the weight tuning to only using the validation split.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.errors import ModelNotFittedError
from sentinel.ml.anomaly import (
    EmpiricalCdfCalibrator,
    IsolationForestDetector,
    PCAReconstructionDetector,
    WeightedEnsemble,
    build_default_ensemble,
)
from sentinel.ml.deep import DenoisingAutoencoderDetector
from sentinel.ml.featurestore import AlertVectorizer
from sentinel.ml.metrics import roc_auc, three_way_split

SEED = 20260928


class FastAutoencoder(DenoisingAutoencoderDetector):
    """The Part 2.1 autoencoder on a reduced epoch budget.

    It joins the generic detector suite below because that suite tests *protocol
    conformance* — bounded scores, refusal to score unfitted, feature-width
    checking, determinism — none of which depend on training to convergence. The
    full 80-epoch budget across these eight parametrised tests would add roughly a
    minute to a suite whose speed is the reason it gets run. Convergence quality
    and the F-03 numbers are tested in ``test_deep.py`` at the real budget.
    """

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("epochs", 12)
        kwargs.setdefault("patience", 4)
        super().__init__(**kwargs)  # type: ignore[arg-type]


#: Every detector implementing the AnomalyDetector protocol. Adding a detector
#: here subjects it to the whole conformance suite, which is the point.
DETECTOR_FACTORIES = [IsolationForestDetector, PCAReconstructionDetector, FastAutoencoder]


@pytest.fixture
def benign_matrix() -> np.ndarray:
    """Correlated benign data: three independent factors plus derived columns.

    Correlation is essential — an isotropic Gaussian has no structure for a
    subspace model to learn, so PCA would be trivially useless and the
    complementarity tests would be vacuous.
    """
    rng = np.random.default_rng(SEED)
    latent = rng.normal(size=(2000, 3))
    mixing = rng.normal(size=(3, 8))
    return latent @ mixing + rng.normal(scale=0.05, size=(2000, 8))


class TestCalibrator:
    def test_maps_training_scores_onto_the_unit_interval(self):
        rng = np.random.default_rng(SEED)
        scores = rng.normal(size=5000)
        calibrated = EmpiricalCdfCalibrator().fit(scores).transform(scores)
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_output_is_approximately_uniform_on_training_data(self):
        # The defining property: a calibrated score is the quantile of benign
        # traffic, so "0.99" means the same thing for every detector.
        rng = np.random.default_rng(SEED)
        scores = rng.lognormal(size=20000)
        calibrated = EmpiricalCdfCalibrator().fit(scores).transform(scores)
        for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert abs(float(np.mean(calibrated <= quantile)) - quantile) < 0.02

    def test_is_monotone(self):
        calibrator = EmpiricalCdfCalibrator().fit(np.linspace(0, 100, 1000))
        probe = np.array([0.0, 10.0, 50.0, 90.0, 100.0])
        transformed = calibrator.transform(probe)
        assert np.all(np.diff(transformed) >= 0)

    def test_clamps_beyond_the_training_range(self):
        calibrator = EmpiricalCdfCalibrator().fit(np.linspace(0, 1, 100))
        assert calibrator.transform(np.array([-100.0]))[0] == 0.0
        assert calibrator.transform(np.array([100.0]))[0] == 1.0

    def test_handles_a_constant_score_distribution(self):
        # A degenerate detector must not produce NaN, or the ensemble average dies.
        calibrated = EmpiricalCdfCalibrator().fit(np.full(500, 3.0)).transform(np.array([3.0, 9.0]))
        assert np.all(np.isfinite(calibrated))

    def test_unfitted_transform_is_refused(self):
        with pytest.raises(ModelNotFittedError):
            EmpiricalCdfCalibrator().transform(np.array([1.0]))

    def test_empty_fit_is_refused(self):
        with pytest.raises(ValueError, match="empty score vector"):
            EmpiricalCdfCalibrator().fit(np.array([]))

    def test_non_finite_scores_are_refused(self):
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalCdfCalibrator().fit(np.array([1.0, np.nan]))

    def test_too_few_quantiles_refused(self):
        with pytest.raises(ValueError, match="n_quantiles"):
            EmpiricalCdfCalibrator(n_quantiles=4)

    def test_state_round_trips(self):
        rng = np.random.default_rng(SEED)
        scores = rng.normal(size=1000)
        original = EmpiricalCdfCalibrator().fit(scores)
        restored = EmpiricalCdfCalibrator().load_state(original.state())
        assert np.allclose(original.transform(scores), restored.transform(scores))


class TestDetectors:
    @pytest.mark.parametrize(
        "factory", DETECTOR_FACTORIES
    )
    def test_scores_are_bounded(self, factory, benign_matrix):
        detector = factory(random_state=SEED).fit(benign_matrix)
        scores = detector.score(benign_matrix)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    @pytest.mark.parametrize(
        "factory", DETECTOR_FACTORIES
    )
    def test_obvious_outliers_score_higher_than_benign(self, factory, benign_matrix):
        detector = factory(random_state=SEED).fit(benign_matrix)
        outliers = np.full((20, benign_matrix.shape[1]), 50.0)
        assert detector.score(outliers).mean() > detector.score(benign_matrix).mean()

    @pytest.mark.parametrize(
        "factory", DETECTOR_FACTORIES
    )
    def test_unfitted_scoring_is_refused(self, factory, benign_matrix):
        with pytest.raises(ModelNotFittedError, match="must be fitted"):
            factory().score(benign_matrix)

    @pytest.mark.parametrize(
        "factory", DETECTOR_FACTORIES
    )
    def test_feature_count_change_is_refused(self, factory, benign_matrix):
        # Scoring with the wrong width would silently align the wrong columns.
        detector = factory(random_state=SEED).fit(benign_matrix)
        with pytest.raises(ValueError, match="feature spec changed"):
            detector.score(np.zeros((5, benign_matrix.shape[1] + 1)))

    @pytest.mark.parametrize(
        "factory", DETECTOR_FACTORIES
    )
    def test_single_sample_fit_is_refused(self, factory):
        with pytest.raises(ValueError, match="at least 2 samples"):
            factory().fit(np.zeros((1, 5)))

    @pytest.mark.parametrize(
        "factory", DETECTOR_FACTORIES
    )
    def test_non_finite_input_is_refused(self, factory, benign_matrix):
        detector = factory(random_state=SEED).fit(benign_matrix)
        bad = benign_matrix.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            detector.score(bad)

    @pytest.mark.parametrize(
        "factory", DETECTOR_FACTORIES
    )
    def test_deterministic_across_runs(self, factory, benign_matrix):
        first = factory(random_state=SEED).fit(benign_matrix).score(benign_matrix)
        second = factory(random_state=SEED).fit(benign_matrix).score(benign_matrix)
        assert np.array_equal(first, second)

    def test_pca_rank_is_selected_from_explained_variance(self, benign_matrix):
        # The data has 3 latent factors in 8 dimensions, so ~95% of variance should
        # be captured well below full rank.
        detector = PCAReconstructionDetector(variance_target=0.95, random_state=SEED)
        detector.fit(benign_matrix)
        assert 1 <= detector.n_components_ <= 4

    def test_pca_always_holds_a_dimension_out(self, benign_matrix):
        # A full-rank subspace reconstructs perfectly, making the score identically
        # zero and the detector useless.
        detector = PCAReconstructionDetector(variance_target=0.999999, random_state=SEED)
        detector.fit(benign_matrix)
        assert detector.n_components_ < benign_matrix.shape[1]
        assert detector.raw_score(benign_matrix).std() > 0

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_pca_variance_target_bounds(self, bad):
        with pytest.raises(ValueError, match="variance_target"):
            PCAReconstructionDetector(variance_target=bad)

    def test_raw_scores_are_on_different_scales(self, benign_matrix):
        """Why calibration is mandatory rather than cosmetic."""
        forest = IsolationForestDetector(random_state=SEED).fit(benign_matrix)
        pca = PCAReconstructionDetector(random_state=SEED).fit(benign_matrix)
        forest_spread = float(np.ptp(forest.raw_score(benign_matrix)))
        pca_spread = float(np.ptp(pca.raw_score(benign_matrix)))
        # Averaging these raw would let whichever has larger spread dominate, and
        # "tuned weights" would be absorbing a unit conversion.
        assert max(forest_spread, pca_spread) / min(forest_spread, pca_spread) > 2.0

    def test_calibration_puts_them_on_one_scale(self, benign_matrix):
        forest = IsolationForestDetector(random_state=SEED).fit(benign_matrix)
        pca = PCAReconstructionDetector(random_state=SEED).fit(benign_matrix)
        for detector in (forest, pca):
            scores = detector.score(benign_matrix)
            assert abs(float(scores.mean()) - 0.5) < 0.05


class TestComplementarity:
    """The two detectors' blind spots, constructed directly."""

    def test_pca_catches_a_correlation_violation_the_forest_misses(self, benign_matrix):
        forest = IsolationForestDetector(random_state=SEED).fit(benign_matrix)
        pca = PCAReconstructionDetector(random_state=SEED).fit(benign_matrix)

        # Build a point whose every marginal is ordinary but whose combination is
        # off-manifold: take a real benign row and permute its coordinates. Each
        # value still comes from the benign marginal distribution, so axis-aligned
        # partitioning sees nothing unusual; the correlation structure is destroyed.
        rng = np.random.default_rng(7)
        rows = benign_matrix[rng.choice(len(benign_matrix), 300, replace=False)]
        shuffled = np.array([rng.permutation(row) for row in rows])

        pca_lift = float(pca.score(shuffled).mean() - pca.score(benign_matrix).mean())
        forest_lift = float(forest.score(shuffled).mean() - forest.score(benign_matrix).mean())
        assert pca_lift > 0.2
        assert pca_lift > forest_lift

    def test_forest_catches_a_marginal_outlier_pca_underweights(self, benign_matrix):
        forest = IsolationForestDetector(random_state=SEED).fit(benign_matrix)
        pca = PCAReconstructionDetector(random_state=SEED).fit(benign_matrix)

        # An extreme value placed *along* the principal subspace: PCA reconstructs it
        # nearly perfectly because it lies on the learned manifold, while the forest
        # isolates it immediately because it is far out on every axis.
        direction = pca._model.components_[0]  # type: ignore[union-attr]
        along_manifold = np.outer(np.full(200, 12.0), direction)

        forest_lift = float(
            forest.score(along_manifold).mean() - forest.score(benign_matrix).mean()
        )
        pca_lift = float(pca.score(along_manifold).mean() - pca.score(benign_matrix).mean())
        assert forest_lift > 0.2
        assert forest_lift > pca_lift

    def test_detector_scores_are_not_redundant(self, benign_matrix):
        # If the two detectors correlated near 1.0, the ensemble would be one model
        # with extra steps.
        rng = np.random.default_rng(3)
        mixed = np.vstack([benign_matrix, rng.normal(scale=6.0, size=(400, 8))])
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        detail = ensemble.score_detail(mixed)
        scores = list(detail.per_detector.values())
        correlation = float(np.corrcoef(scores[0], scores[1])[0, 1])
        assert correlation < 0.97


class TestEnsemble:
    def test_scores_are_bounded(self, benign_matrix):
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        scores = ensemble.score(benign_matrix)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_equal_weights_average_the_members(self, benign_matrix):
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        detail = ensemble.score_detail(benign_matrix)
        expected = np.mean(list(detail.per_detector.values()), axis=0)
        assert np.allclose(detail.combined, expected)

    def test_weights_are_normalized(self):
        ensemble = WeightedEnsemble(
            detectors=[
                IsolationForestDetector(random_state=SEED),
                PCAReconstructionDetector(random_state=SEED),
            ],
            weights=[3.0, 1.0],
        )
        assert ensemble.weights.sum() == pytest.approx(1.0)
        assert ensemble.weights[0] == pytest.approx(0.75)

    def test_extreme_weight_reduces_to_one_detector(self, benign_matrix):
        ensemble = WeightedEnsemble(
            detectors=[
                IsolationForestDetector(random_state=SEED),
                PCAReconstructionDetector(random_state=SEED),
            ],
            weights=[1.0, 0.0],
        ).fit(benign_matrix)
        detail = ensemble.score_detail(benign_matrix)
        assert np.allclose(detail.combined, detail.per_detector["isolation_forest"])

    def test_explain_names_each_contribution(self, benign_matrix):
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        explanation = ensemble.score_detail(benign_matrix).explain(0)
        assert set(explanation) == {"combined", "isolation_forest", "pca_reconstruction"}

    def test_disagreement_is_reported(self, benign_matrix):
        # High disagreement is itself a triage signal: the two models see different
        # worlds, so the alert deserves a human rather than an averaged-away score.
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        disagreement = ensemble.score_detail(benign_matrix).disagreement()
        assert disagreement.shape == (len(benign_matrix),)
        assert disagreement.min() >= 0.0

    def test_empty_detector_list_refused(self):
        with pytest.raises(ValueError, match="at least one detector"):
            WeightedEnsemble(detectors=[])

    def test_duplicate_detector_names_refused(self):
        with pytest.raises(ValueError, match="unique"):
            WeightedEnsemble(
                detectors=[
                    IsolationForestDetector(random_state=SEED),
                    IsolationForestDetector(random_state=SEED),
                ]
            )

    def test_wrong_weight_count_refused(self):
        with pytest.raises(ValueError, match="expected 2 weights"):
            build_default_ensemble()._normalize_weights([1.0])

    def test_negative_weights_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            WeightedEnsemble(
                detectors=[
                    IsolationForestDetector(random_state=SEED),
                    PCAReconstructionDetector(random_state=SEED),
                ],
                weights=[-1.0, 2.0],
            )

    def test_zero_sum_weights_refused(self):
        with pytest.raises(ValueError, match="positive number"):
            WeightedEnsemble(
                detectors=[
                    IsolationForestDetector(random_state=SEED),
                    PCAReconstructionDetector(random_state=SEED),
                ],
                weights=[0.0, 0.0],
            )

    def test_threshold_for_fpr_hits_its_budget(self, benign_matrix):
        """The SOC-facing control: false positives per day, not an abstract score."""
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        for target in (0.01, 0.05, 0.10, 0.20):
            threshold = ensemble.threshold_for_fpr(benign_matrix, target)
            achieved = float(np.mean(ensemble.score(benign_matrix) >= threshold))
            assert abs(achieved - target) < 0.02, f"target {target} -> {achieved}"

    def test_lower_fpr_means_a_higher_threshold(self, benign_matrix):
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        assert ensemble.threshold_for_fpr(benign_matrix, 0.01) >= ensemble.threshold_for_fpr(
            benign_matrix, 0.20
        )

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
    def test_invalid_fpr_refused(self, benign_matrix, bad):
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        with pytest.raises(ValueError, match="target_fpr"):
            ensemble.threshold_for_fpr(benign_matrix, bad)

    def test_save_records_the_spec_fingerprint(self, benign_matrix, tmp_path):
        import json

        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        path = ensemble.save(tmp_path / "ens.json", spec_fingerprint="abc123")
        payload = json.loads(path.read_text())
        assert payload["spec_fingerprint"] == "abc123"
        assert payload["detectors"] == ["isolation_forest", "pca_reconstruction"]


class TestWeightTuning:
    @pytest.fixture
    def labelled(self, benign_matrix):
        rng = np.random.default_rng(11)
        attacks = rng.normal(scale=5.0, size=(400, benign_matrix.shape[1]))
        x = np.vstack([benign_matrix[:600], attacks])
        y = np.concatenate([np.zeros(600, dtype=int), np.ones(400, dtype=int)])
        return x, y

    def test_tuning_does_not_reduce_auc(self, benign_matrix, labelled):
        x, y = labelled
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        before = roc_auc(y, ensemble.score(x))
        ensemble.tune_weights(x, y)
        assert roc_auc(y, ensemble.score(x)) >= before - 1e-12

    def test_tuning_reports_the_weights_and_auc(self, benign_matrix, labelled):
        x, y = labelled
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        report = ensemble.tune_weights(x, y)
        assert set(report) == {"isolation_forest", "pca_reconstruction", "validation_auc"}
        assert ensemble.tuning_auc == pytest.approx(report["validation_auc"])

    def test_tuned_weights_still_sum_to_one(self, benign_matrix, labelled):
        x, y = labelled
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        ensemble.tune_weights(x, y)
        assert ensemble.weights.sum() == pytest.approx(1.0)

    def test_tuning_is_deterministic(self, benign_matrix, labelled):
        x, y = labelled

        def tune():
            ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
            ensemble.tune_weights(x, y)
            return ensemble.weights.tolist()

        assert tune() == tune()

    def test_tuning_an_unfitted_ensemble_is_refused(self, labelled):
        x, y = labelled
        with pytest.raises(ModelNotFittedError, match="fit the ensemble"):
            build_default_ensemble().tune_weights(x, y)

    def test_single_class_validation_split_is_refused(self, benign_matrix):
        # AUC is undefined, so every weight vector would look equally good and the
        # tuning would silently return whatever it saw first.
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        with pytest.raises(ValueError, match="only one class"):
            ensemble.tune_weights(benign_matrix[:100], np.zeros(100, dtype=int))

    def test_label_length_mismatch_is_refused(self, benign_matrix, labelled):
        x, y = labelled
        ensemble = build_default_ensemble(random_state=SEED).fit(benign_matrix)
        with pytest.raises(ValueError, match="do not match rows"):
            ensemble.tune_weights(x, y[:-5])

    def test_three_detector_ensemble_also_tunes(self, benign_matrix, labelled):
        x, y = labelled
        ensemble = WeightedEnsemble(
            detectors=[
                IsolationForestDetector(random_state=SEED),
                # Two PCA members at different subspace ranks; the name override is
                # what makes same-family members expressible at all.
                PCAReconstructionDetector(
                    variance_target=0.90, random_state=SEED, name="pca_90"
                ),
                PCAReconstructionDetector(
                    variance_target=0.99, random_state=SEED, name="pca_99"
                ),
            ],
        ).fit(benign_matrix)
        report = ensemble.tune_weights(x, y)
        assert set(report) == {"isolation_forest", "pca_90", "pca_99", "validation_auc"}
        assert report["validation_auc"] > 0.5
        assert ensemble.weights.sum() == pytest.approx(1.0)

    def test_blank_name_override_refused(self):
        with pytest.raises(ValueError, match="non-empty identifier"):
            PCAReconstructionDetector(name="  ")


class TestAcceptanceCriteria:
    """F-03: ROC-AUC >= 0.90 on a held-out split, on data from the real pipeline."""

    @pytest.fixture
    def pipeline(self, eval_alerts):
        y = np.array(
            [0 if a.ground_truth_label == "benign" else 1 for a in eval_alerts], dtype=int
        )
        split = three_way_split(y, seed=SEED)
        train = [eval_alerts[i] for i in split.train_benign]
        vectorizer = AlertVectorizer().fit(train)
        ensemble = build_default_ensemble(random_state=SEED).fit(vectorizer.transform(train))
        ensemble.tune_weights(
            vectorizer.transform([eval_alerts[i] for i in split.validation]),
            y[split.validation],
        )
        return vectorizer, ensemble, split, y

    def test_f03_auc_gate_on_held_out_test_split(self, pipeline, eval_alerts):
        vectorizer, ensemble, split, y = pipeline
        x_test = vectorizer.transform([eval_alerts[i] for i in split.test])
        auc = roc_auc(y[split.test], ensemble.score(x_test))
        assert auc >= 0.90, f"F-03 requires ROC-AUC >= 0.90 on held-out data; got {auc:.4f}"

    def test_ensemble_is_at_least_as_good_as_its_best_member(self, pipeline, eval_alerts):
        # The stated purpose of the ensemble. If it were worse than a single member,
        # the weighted average would be actively harmful.
        vectorizer, ensemble, split, y = pipeline
        x_test = vectorizer.transform([eval_alerts[i] for i in split.test])
        detail = ensemble.score_detail(x_test)
        combined = roc_auc(y[split.test], detail.combined)
        members = [roc_auc(y[split.test], s) for s in detail.per_detector.values()]
        assert combined >= max(members) - 0.02

    def test_training_split_contains_no_attacks(self, pipeline):
        # The detectors are semi-supervised novelty detectors: a single attack in
        # the benign training set teaches the model that attacks are normal.
        _, _, split, y = pipeline
        assert y[split.train_benign].sum() == 0

    def test_splits_are_disjoint(self, pipeline):
        _, _, split, _ = pipeline
        train = set(split.train_benign.tolist())
        validation = set(split.validation.tolist())
        test = set(split.test.tolist())
        assert train & validation == set()
        assert train & test == set()
        assert validation & test == set()

    def test_cross_dataset_generalisation_holds(self, pipeline):
        """PRD Section 10's mitigation for overfitting to one lab's traffic.

        Scored on UNSW-shaped data: a different schema, different units (seconds vs
        microseconds) and a different label vocabulary, with only the unified
        feature space in common. The honest caveat is that both corpora come from
        one campaign engine, so this validates the *unit and schema harmonisation*
        rather than a genuine distribution shift — which is exactly the bug class it
        is here to catch. Real distribution shift needs the real downloads
        (``docs/DATA.md``).
        """
        from sentinel.ml.datasets.synthetic import generate_alerts

        vectorizer, ensemble, _, _ = pipeline
        other = generate_alerts(4000, seed=SEED + 99, dataset="unsw")
        y_other = np.array(
            [0 if a.ground_truth_label == "benign" else 1 for a in other], dtype=int
        )
        auc = roc_auc(y_other, ensemble.score(vectorizer.transform(other)))
        assert auc >= 0.85, f"cross-dataset AUC collapsed to {auc:.4f}"

    def test_detection_is_reproducible_end_to_end(self, eval_alerts):
        def run() -> list[float]:
            y = np.array(
                [0 if a.ground_truth_label == "benign" else 1 for a in eval_alerts], dtype=int
            )
            split = three_way_split(y, seed=SEED)
            train = [eval_alerts[i] for i in split.train_benign]
            vectorizer = AlertVectorizer().fit(train)
            ensemble = build_default_ensemble(random_state=SEED).fit(
                vectorizer.transform(train)
            )
            return ensemble.score(
                vectorizer.transform([eval_alerts[i] for i in split.test[:200]])
            ).tolist()

        assert run() == run()
