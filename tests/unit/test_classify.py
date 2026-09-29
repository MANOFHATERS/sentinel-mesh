"""The supervised family classifier (:mod:`sentinel.ml.classify`).

This model exists to make augmentation measurable, so the properties that matter most
are the ones that keep the measurement valid:

*   :class:`TestFitIsReproducible` — the augmented and unaugmented runs differ only in
    their training rows. If the fit carried its own run-to-run variation, the
    difference between them would be unreadable, and any conclusion about augmentation
    would be a conclusion about the seed.
*   :class:`TestMacroRecallIsTheRightHeadline` — plain accuracy is 80% benign here, so
    a model predicting "benign" for everything scores 0.80. Macro recall is what the
    rare-class question needs, and the degenerate case is asserted rather than assumed.
*   :class:`TestGradientsAreCorrect` — the softmax cross entropy added to
    :mod:`sentinel.ml.nn` for this part is gradient-checked on the exact shipped
    composition.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ml.classify import ClassifierError, FamilyClassifier
from sentinel.ml.nn import gradient_check, softmax, softmax_cross_entropy_loss


@pytest.fixture(scope="module")
def dataset() -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Real alerts through the real vectorizer, split train/test."""
    from sentinel.ml.datasets.synthetic import generate_alerts
    from sentinel.ml.featurestore import AlertVectorizer

    alerts = generate_alerts(9000, seed=20260929)
    vectorizer = AlertVectorizer().fit(alerts)
    matrix = vectorizer.transform(alerts)
    families = np.asarray([a.ground_truth_label or "benign" for a in alerts])
    order = np.random.default_rng(7).permutation(len(families))
    cut = int(0.65 * len(order))
    train, test = order[:cut], order[cut:]
    return matrix[train], list(families[train]), matrix[test], list(families[test])


@pytest.fixture(scope="module")
def fitted(dataset) -> FamilyClassifier:
    x_train, f_train, _, _ = dataset
    return FamilyClassifier(seed=5).fit(x_train, f_train)


class TestConfiguration:
    @pytest.mark.parametrize(
        "kwargs,fragment",
        [
            ({"hidden": ()}, "hidden"),
            ({"hidden": (0,)}, "hidden"),
            ({"class_weight": "inverse"}, "class_weight"),
            ({"label_smoothing": 1.0}, "label_smoothing"),
            ({"label_smoothing": -0.1}, "label_smoothing"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict, fragment: str) -> None:
        with pytest.raises(ClassifierError, match=fragment):
            FamilyClassifier(**kwargs)

    def test_class_weighting_is_off_by_default(self) -> None:
        """It addresses the same problem as augmentation and would confound it."""
        assert FamilyClassifier().class_weight is None
        assert FamilyClassifier().label_smoothing == 0.0

    def test_unfitted_model_refuses_to_predict(self) -> None:
        for call in (
            lambda m: m.predict(np.zeros((1, 3))),
            lambda m: m.predict_proba(np.zeros((1, 3))),
            lambda m: m.logits(np.zeros((1, 3))),
            lambda m: m.training_report(),
        ):
            with pytest.raises(ClassifierError, match="not fitted"):
                call(FamilyClassifier())

    def test_single_class_rejected(self) -> None:
        with pytest.raises(ClassifierError, match="at least two classes"):
            FamilyClassifier().fit(np.zeros((10, 3)), ["only"] * 10)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ClassifierError, match="labels"):
            FamilyClassifier().fit(np.zeros((4, 3)), ["a", "b"])

    def test_non_finite_input_rejected(self) -> None:
        matrix = np.zeros((10, 3))
        matrix[0, 0] = np.inf
        with pytest.raises(ClassifierError, match="non-finite"):
            FamilyClassifier().fit(matrix, ["a"] * 5 + ["b"] * 5)

    def test_wrong_feature_count_at_predict_time_rejected(
        self, fitted: FamilyClassifier
    ) -> None:
        with pytest.raises(ClassifierError, match="features"):
            fitted.predict(np.zeros((2, 3)))


class TestGradientsAreCorrect:
    def test_the_shipped_composition_gradient_checks(self) -> None:
        model = FamilyClassifier(hidden=(24, 16), seed=9)
        net = model.build_network(n_features=6, n_classes=4)
        rng = np.random.default_rng(11)
        inputs = rng.normal(size=(5, 6))
        targets = np.eye(4)[rng.integers(0, 4, size=5)]
        error = gradient_check(
            net, inputs, targets, loss_fn=softmax_cross_entropy_loss, rng=rng
        )
        assert error < 1e-6, f"max relative gradient error {error:.2e}"

    def test_the_loss_gradient_is_the_textbook_form(self) -> None:
        rng = np.random.default_rng(13)
        logits = rng.normal(size=(7, 5))
        targets = np.eye(5)[rng.integers(0, 5, size=7)]
        _, gradient = softmax_cross_entropy_loss(logits, targets)
        np.testing.assert_allclose(gradient, (softmax(logits, axis=1) - targets) / 7)

    def test_the_loss_survives_extreme_logits(self) -> None:
        """An unshifted log-sum-exp overflows here and returns NaN."""
        logits = np.asarray([[900.0, -900.0], [-900.0, 900.0]])
        targets = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        loss, gradient = softmax_cross_entropy_loss(logits, targets)
        assert np.isfinite(loss)
        assert np.all(np.isfinite(gradient))

    def test_the_loss_accepts_soft_targets(self) -> None:
        """Label smoothing produces non-integral targets; an index API would refuse."""
        logits = np.zeros((2, 3))
        soft = np.full((2, 3), 1.0 / 3.0)
        loss, gradient = softmax_cross_entropy_loss(logits, soft)
        assert np.isfinite(loss)
        np.testing.assert_allclose(gradient, 0.0, atol=1e-12)

    def test_perfect_prediction_has_near_zero_loss(self) -> None:
        logits = np.asarray([[20.0, 0.0], [0.0, 20.0]])
        targets = np.eye(2)
        loss, _ = softmax_cross_entropy_loss(logits, targets)
        assert loss < 1e-8

    @pytest.mark.parametrize(
        "logits,targets,fragment",
        [
            (np.zeros((2, 3)), np.zeros((2, 4)), "shape mismatch"),
            (np.zeros(3), np.zeros(3), "2-D"),
            (np.zeros((0, 3)), np.zeros((0, 3)), "empty"),
            (np.zeros((2, 2)), -np.ones((2, 2)), "negative"),
        ],
    )
    def test_malformed_input_rejected(
        self, logits: np.ndarray, targets: np.ndarray, fragment: str
    ) -> None:
        with pytest.raises(ValueError, match=fragment):
            softmax_cross_entropy_loss(logits, targets)


class TestFitIsReproducible:
    """Without this, any augmentation delta is a statement about the seed."""

    def test_same_seed_same_predictions(self, dataset) -> None:
        x_train, f_train, x_test, _ = dataset
        first = FamilyClassifier(seed=5).fit(x_train, f_train)
        second = FamilyClassifier(seed=5).fit(x_train, f_train)
        assert first.predict(x_test) == second.predict(x_test)
        np.testing.assert_allclose(
            first.predict_proba(x_test), second.predict_proba(x_test)
        )

    def test_different_seeds_give_similar_quality(self, dataset) -> None:
        """Run-to-run spread must be far smaller than any effect being measured."""
        x_train, f_train, x_test, f_test = dataset
        scores = [
            FamilyClassifier(seed=seed).fit(x_train, f_train).macro_recall(x_test, f_test)
            for seed in (1, 2, 3)
        ]
        assert max(scores) - min(scores) < 0.10, scores


class TestPredictions:
    def test_probabilities_are_a_distribution(
        self, fitted: FamilyClassifier, dataset
    ) -> None:
        _, _, x_test, _ = dataset
        probabilities = fitted.predict_proba(x_test)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)

    def test_predict_agrees_with_argmax_of_proba(
        self, fitted: FamilyClassifier, dataset
    ) -> None:
        _, _, x_test, _ = dataset
        probabilities = fitted.predict_proba(x_test)
        expected = [fitted.classes_[i] for i in np.argmax(probabilities, axis=1)]
        assert list(fitted.predict(x_test)) == expected

    def test_confidence_is_the_max_probability(
        self, fitted: FamilyClassifier, dataset
    ) -> None:
        _, _, x_test, _ = dataset
        np.testing.assert_allclose(
            fitted.confidence(x_test), fitted.predict_proba(x_test).max(axis=1)
        )

    def test_classes_are_sorted_and_complete(self, fitted: FamilyClassifier, dataset) -> None:
        _, f_train, _, _ = dataset
        assert list(fitted.classes_) == sorted(set(f_train))

    def test_predictions_are_known_classes(self, fitted: FamilyClassifier, dataset) -> None:
        _, _, x_test, _ = dataset
        assert set(fitted.predict(x_test)) <= set(fitted.classes_)


class TestMacroRecallIsTheRightHeadline:
    def test_it_learns_the_task(self, fitted: FamilyClassifier, dataset) -> None:
        """F-02 asks for >= 85% agreement with ground-truth labels."""
        _, _, x_test, f_test = dataset
        assert fitted.accuracy(x_test, f_test) > 0.85
        assert fitted.macro_recall(x_test, f_test) > 0.70

    def test_a_majority_predictor_scores_well_on_accuracy_and_badly_on_macro(
        self, dataset
    ) -> None:
        """Why macro recall is the reported number and accuracy is not."""
        _, _, x_test, f_test = dataset

        class MajorityOnly(FamilyClassifier):
            def predict(self, x):  # type: ignore[override]
                return tuple(["benign"] * len(x))

        degenerate = MajorityOnly()
        degenerate.classes_ = ("benign", "ddos")
        assert degenerate.accuracy(x_test, f_test) > 0.7
        assert degenerate.macro_recall(x_test, f_test) < 0.2

    def test_recall_by_family_covers_every_present_family(
        self, fitted: FamilyClassifier, dataset
    ) -> None:
        _, _, x_test, f_test = dataset
        per_family = fitted.recall_by_family(x_test, f_test)
        assert set(per_family) == set(f_test)
        assert all(0.0 <= value <= 1.0 for value in per_family.values())

    def test_macro_recall_is_the_unweighted_mean(
        self, fitted: FamilyClassifier, dataset
    ) -> None:
        _, _, x_test, f_test = dataset
        per_family = fitted.recall_by_family(x_test, f_test)
        assert fitted.macro_recall(x_test, f_test) == pytest.approx(
            float(np.mean(list(per_family.values())))
        )


class TestClassWeighting:
    def test_balanced_weighting_changes_the_fit(self, dataset) -> None:
        x_train, f_train, x_test, f_test = dataset
        plain = FamilyClassifier(seed=5).fit(x_train, f_train)
        balanced = FamilyClassifier(seed=5, class_weight="balanced").fit(x_train, f_train)
        assert plain.predict(x_test) != balanced.predict(x_test)
        del f_test

    def test_explicit_sample_weight_is_honoured(self, dataset) -> None:
        x_train, f_train, x_test, _ = dataset
        weights = np.ones(len(f_train))
        weights[: len(weights) // 2] = 0.0
        weighted = FamilyClassifier(seed=5).fit(
            x_train, f_train, sample_weight=weights
        )
        assert weighted.is_fitted
        assert len(weighted.predict(x_test)) == x_test.shape[0]

    def test_negative_sample_weight_rejected(self, dataset) -> None:
        x_train, f_train, _, _ = dataset
        with pytest.raises(ClassifierError, match="non-negative"):
            FamilyClassifier().fit(
                x_train, f_train, sample_weight=-np.ones(len(f_train))
            )

    def test_wrong_length_sample_weight_rejected(self, dataset) -> None:
        x_train, f_train, _, _ = dataset
        with pytest.raises(ClassifierError, match="does not match"):
            FamilyClassifier().fit(x_train, f_train, sample_weight=np.ones(3))


class TestLabelSmoothing:
    def test_smoothing_lowers_clean_confidence(self, dataset) -> None:
        x_train, f_train, x_test, _ = dataset
        sharp = FamilyClassifier(seed=5).fit(x_train, f_train)
        smooth = FamilyClassifier(seed=5, label_smoothing=0.1).fit(x_train, f_train)
        assert smooth.confidence(x_test).mean() < sharp.confidence(x_test).mean()

    def test_smoothing_is_recorded_in_the_report(self, dataset) -> None:
        x_train, f_train, _, _ = dataset
        model = FamilyClassifier(seed=5, label_smoothing=0.05).fit(x_train, f_train)
        assert model.training_report()["label_smoothing"] == 0.05


class TestTrainingReport:
    def test_reports_the_fit(self, fitted: FamilyClassifier) -> None:
        report = fitted.training_report()
        assert report["final_loss"] < report["first_loss"]
        assert report["epochs_run"] >= 1
        assert isinstance(report["parameters"], int)
        assert report["classes"] == list(fitted.classes_)

    def test_report_is_json_serializable(self, fitted: FamilyClassifier) -> None:
        import json

        assert json.dumps(fitted.training_report())
