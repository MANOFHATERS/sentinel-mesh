"""The denoising autoencoder (PRD Section 5.5.2, Part 2.1).

Protocol conformance is covered by the shared parametrised suite in
``test_anomaly.py``, which this detector joins as ``FastAutoencoder``. What is
tested here is what is specific to it:

*   :class:`TestArchitectureGuards` — the identity-map failure mode. A bottleneck
    at or above the input width lets the network learn ``f(x) = x``, so
    reconstruction error collapses toward zero for attacks as well as benign
    traffic and the detector becomes noise with an excellent training loss. This
    is the single most dangerous way to get an autoencoder wrong, because every
    diagnostic you would normally look at says it is working.
*   :class:`TestDenoisingSemantics` — corruption on the input only, never the
    target, and never at inference.
*   :class:`TestBotnetGap` — the reason this part exists. Part 1 shipped botnet
    recall 0.608 as its documented weak spot; these tests measure whether the
    nonlinearity closes it.
*   :class:`TestEnsembleWeightCollapse` — the finding that an AUC-optimal tuner
    discards the Isolation Forest entirely once the autoencoder is in the
    ensemble, which is optimal for the metric and defeats the purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.errors import ModelNotFittedError
from sentinel.ml.anomaly import (
    IsolationForestDetector,
    PCAReconstructionDetector,
    WeightedEnsemble,
)
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.ml.deep import DenoisingAutoencoderDetector, build_deep_ensemble
from sentinel.ml.featurestore import AlertVectorizer
from sentinel.ml.metrics import detection_report, roc_auc, three_way_split
from sentinel.ml.nn import TrainingError

SEED = 20260928
BENIGN = "benign"


@pytest.fixture
def benign_matrix() -> np.ndarray:
    """Correlated benign data: 3 latent factors expressed in 8 columns.

    Correlated on purpose. On isotropic noise there is no manifold to learn, so
    reconstruction error carries no signal and every test below would be vacuous.
    """
    rng = np.random.default_rng(SEED)
    latent = rng.normal(size=(1500, 3))
    mixing = rng.normal(size=(3, 8))
    return latent @ mixing + rng.normal(scale=0.05, size=(1500, 8))


def fast(**kwargs: object) -> DenoisingAutoencoderDetector:
    """A detector with a small epoch budget, for tests about mechanics not accuracy."""
    kwargs.setdefault("epochs", 15)
    kwargs.setdefault("patience", 5)
    kwargs.setdefault("random_state", SEED)
    return DenoisingAutoencoderDetector(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Architecture guards
# --------------------------------------------------------------------------- #


class TestArchitectureGuards:
    def test_a_bottleneck_at_the_input_width_is_refused(self, benign_matrix) -> None:
        """The identity-map trap, blocked at fit time with an explanatory message."""
        with pytest.raises(ValueError, match="must be narrower than the input"):
            fast(bottleneck=8, hidden=16).fit(benign_matrix)

    def test_a_bottleneck_wider_than_the_input_is_refused(self, benign_matrix) -> None:
        with pytest.raises(ValueError, match="must be narrower than the input"):
            fast(bottleneck=12, hidden=32).fit(benign_matrix)

    def test_the_textbook_identity_collapse_does_not_actually_occur(
        self, benign_matrix
    ) -> None:
        """The measured counterpart to the guard, and it cuts against the folklore.

        The standard warning says a full-width bottleneck learns the identity map,
        so reconstruction error vanishes for anomalies too. Built by hand here — a
        *linear* full-rank autoencoder, the most favourable possible case for
        learning the identity, since it is exactly representable — and compared
        against a rank-2 bottleneck on inputs that violate the correlation
        structure at **normal magnitude** (each column resampled from its own
        benign marginal, so every individual value is ordinary and only the
        combination is impossible; that is the blind spot a reconstruction
        detector uniquely covers).

        Measured: the full-rank model separated violations from benign by 1141x,
        the rank-2 bottleneck by 5x. The full-rank model is *better*, because
        gradient descent from small initialisation converges to the projector onto
        the benign manifold rather than to the identity — nothing off-manifold
        appears in the training objective to push it there.

        This test exists so the guard is not mistaken for protection against a
        demonstrated failure. It is an architectural coherence check; the honest
        reasoning is in the module docstring.
        """
        from sentinel.ml.nn import Dense, Identity, Sequential, train

        width = benign_matrix.shape[1]
        mean, scale = benign_matrix.mean(axis=0), benign_matrix.std(axis=0)
        scaled = (benign_matrix - mean) / np.where(scale > 1e-12, scale, 1.0)

        shuffler = np.random.default_rng(7)
        violation = np.column_stack(
            [shuffler.permutation(scaled[:, j]) for j in range(width)]
        )[:300]
        # Same marginals and same magnitude as benign traffic — only the joint
        # structure is broken. Assert that, or the test proves nothing.
        assert float(np.mean(np.square(violation))) == pytest.approx(
            float(np.mean(np.square(scaled))), rel=0.10
        )

        def separation(bottleneck: int) -> float:
            generator = np.random.default_rng(SEED)
            net = Sequential(
                Dense(width, bottleneck, rng=generator, gain_for="linear", name="enc"),
                Identity(),
                Dense(bottleneck, width, rng=generator, gain_for="linear", name="dec"),
            )
            train(
                net,
                scaled,
                lambda rows, _: (rows, rows),
                rng=np.random.default_rng(SEED),
                epochs=400,
                batch_size=256,
                lr=1e-2,
            )
            benign_error = float(np.mean(np.square(scaled - net.forward(scaled))))
            violation_error = float(
                np.mean(np.square(violation - net.forward(violation)))
            )
            return violation_error / max(benign_error, 1e-15)

        full_rank = separation(width)
        rank_two = separation(2)
        assert full_rank > rank_two, (
            f"full-rank separated {full_rank:.1f}x vs rank-2 {rank_two:.1f}x — if the "
            "full-rank model has now collapsed, the module docstring's reasoning is "
            "stale and should be corrected"
        )

    def test_under_capacity_is_the_failure_that_actually_shows_up(
        self, benign_matrix
    ) -> None:
        """Too narrow a bottleneck cannot reconstruct benign traffic either.

        The benign fixture has 3 latent factors. At rank 2 the model underfits the
        manifold itself, so its error on *legitimate* data rises and swamps the
        contrast with anomalies. This is the realistic tuning mistake, and it is
        the opposite of the one the folklore warns about.
        """
        narrow = fast(epochs=60, bottleneck=2, hidden=16).fit(benign_matrix)
        adequate = fast(epochs=60, bottleneck=4, hidden=16).fit(benign_matrix)
        assert narrow.raw_score(benign_matrix).mean() > adequate.raw_score(
            benign_matrix
        ).mean(), "the rank-2 model did not underfit benign traffic as expected"

    def test_a_bottleneck_wider_than_the_hidden_layer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="wider than the hidden layer"):
            fast(hidden=4, bottleneck=8)

    def test_default_widths_scale_with_the_feature_count(self) -> None:
        """A hard-coded architecture silently becomes wrong when the spec moves.

        Part 1's session-context enricher already widened the feature space once.
        """
        narrow = fast().fit(np.random.default_rng(1).normal(size=(300, 9)))
        wide = fast().fit(np.random.default_rng(2).normal(size=(300, 60)))
        assert narrow.architecture_ is not None and wide.architecture_ is not None
        assert wide.architecture_[2] > narrow.architecture_[2]

    def test_compression_is_always_real(self) -> None:
        for width in (4, 9, 17, 32, 60):
            detector = fast(epochs=4).fit(
                np.random.default_rng(width).normal(size=(200, width))
            )
            assert detector.architecture_ is not None
            assert detector.architecture_[2] < width

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"noise_std": -0.1}, "non-negative"),
            ({"epochs": 1}, "at least 2 epochs|>= 2"),
            ({"validation_fraction": 0.5}, "validation_fraction"),
            ({"validation_fraction": -0.1}, "validation_fraction"),
            ({"hidden": 0}, "positive width"),
            ({"bottleneck": 0}, "positive width"),
        ],
    )
    def test_invalid_settings_are_refused(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            DenoisingAutoencoderDetector(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Denoising semantics
# --------------------------------------------------------------------------- #


class TestDenoisingSemantics:
    def test_scoring_is_deterministic_across_repeated_calls(self, benign_matrix) -> None:
        """No corruption at inference, or the demo is not reproducible.

        If the noise were applied in ``_raw_score``, a given alert's severity would
        depend on a random draw and two runs of the same scripted scenario would
        produce different alerts (PRD Section 8.3, hours 57-58).
        """
        detector = fast().fit(benign_matrix)
        first = detector.score(benign_matrix)
        second = detector.score(benign_matrix)
        np.testing.assert_array_equal(first, second)

    def test_noise_changes_the_learned_model(self, benign_matrix) -> None:
        """Denoising must actually be happening: noise_std=0 must differ from >0."""
        clean = fast(noise_std=0.0).fit(benign_matrix).score(benign_matrix)
        noisy = fast(noise_std=0.5).fit(benign_matrix).score(benign_matrix)
        assert not np.allclose(clean, noisy)

    def test_noise_std_zero_reduces_to_a_plain_autoencoder(self, benign_matrix) -> None:
        # Still a valid detector, just not a denoising one.
        detector = fast(noise_std=0.0).fit(benign_matrix)
        assert detector.is_fitted
        assert detector.training_report()["noise_std"] == 0.0

    def test_scores_stay_bounded_and_finite_on_extreme_input(
        self, benign_matrix
    ) -> None:
        detector = fast().fit(benign_matrix)
        extreme = np.full((10, benign_matrix.shape[1]), 1e6)
        scores = detector.score(extreme)
        assert np.all(np.isfinite(scores))
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_extreme_input_saturates_at_the_top_of_the_calibrated_range(
        self, benign_matrix
    ) -> None:
        # "More extreme than anything benign I have ever seen" is 1.0, by design.
        detector = fast().fit(benign_matrix)
        assert detector.score(np.full((5, benign_matrix.shape[1]), 1e6)).min() == 1.0


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #


class TestRobustness:
    def test_a_constant_column_does_not_produce_nan(self) -> None:
        """Zero-variance column: dividing by its std is inf, and inf poisons every
        gradient in the network from the first batch onward."""
        rng = np.random.default_rng(SEED)
        data = rng.normal(size=(400, 6))
        data[:, 2] = 3.0  # constant in benign traffic
        detector = fast().fit(data)
        scores = detector.score(data)
        assert np.all(np.isfinite(scores))

    def test_an_all_constant_matrix_is_handled(self) -> None:
        # Degenerate but must not crash: every column constant, so the model has
        # nothing to learn and reconstruction error is ~0 everywhere.
        data = np.tile(np.arange(5.0), (200, 1))
        detector = fast(epochs=4)
        # Either it fits and scores finite, or it refuses because nothing converged.
        try:
            scores = detector.fit(data).score(data)
        except TrainingError:
            return
        assert np.all(np.isfinite(scores))

    def test_training_that_does_not_converge_is_refused(self, benign_matrix) -> None:
        """A detector whose reconstruction error is untrained scores noise.

        Returning it would silently degrade the ensemble while every interface
        reported a fitted model, so fit() raises instead.
        """
        with pytest.raises(TrainingError, match="did not decrease"):
            # lr=1e-20 moves the weights far below float64 resolution relative to
            # the loss, so no learning happens. Note this only fails the convergence
            # check because `converged` requires a *relative* improvement: with a
            # bare `last < first` test this configuration passed, because reshuffling
            # a full batch changes float64 summation order and jitters the loss by
            # ~1e-16 in a random direction. See nn.MIN_RELATIVE_IMPROVEMENT.
            fast(lr=1e-20, epochs=5, noise_std=0.0, batch_size=10_000).fit(benign_matrix)

    def test_unfitted_training_report_is_refused(self) -> None:
        with pytest.raises(ModelNotFittedError):
            fast().training_report()

    def test_is_reproducible_across_runs(self, benign_matrix) -> None:
        first = fast().fit(benign_matrix).score(benign_matrix)
        second = fast().fit(benign_matrix).score(benign_matrix)
        np.testing.assert_array_equal(first, second)

    def test_a_different_seed_gives_a_different_model(self, benign_matrix) -> None:
        first = fast(random_state=1).fit(benign_matrix).score(benign_matrix)
        second = fast(random_state=2).fit(benign_matrix).score(benign_matrix)
        assert not np.array_equal(first, second)

    def test_training_report_describes_the_architecture(self, benign_matrix) -> None:
        report = fast(hidden=16, bottleneck=3).fit(benign_matrix).training_report()
        assert report["architecture"] == "8-16-3-16-8"
        assert report["compression_ratio"] == pytest.approx(3 / 8)
        assert report["n_parameters"] > 0
        assert report["epochs_run"] >= 1
        assert np.isfinite(report["final_train_loss"])


# --------------------------------------------------------------------------- #
# The botnet gap — why Part 2.1 exists
# --------------------------------------------------------------------------- #


def _pipeline(n: int, seed: int = SEED, dataset: str = "cic"):
    """Generate, split and vectorize exactly as ``scripts/evaluate.py`` does."""
    alerts = generate_alerts(n, seed=seed, dataset=dataset, separability=1.0)
    y = np.array([0 if a.ground_truth_label == BENIGN else 1 for a in alerts], dtype=int)
    families = [a.ground_truth_label for a in alerts]
    split = three_way_split(y, seed=seed)
    train_alerts = [alerts[i] for i in split.train_benign]
    vectorizer = AlertVectorizer().fit(train_alerts)
    return {
        "vectorizer": vectorizer,
        "x_train": vectorizer.transform(train_alerts),
        "x_validation": vectorizer.transform([alerts[i] for i in split.validation]),
        "x_test": vectorizer.transform([alerts[i] for i in split.test]),
        "y": y,
        "split": split,
        "families": families,
    }


def _pca_recall(data: dict) -> dict[str, float]:
    """Per-family recall for the Part 1 linear baseline."""
    return _family_recall(
        PCAReconstructionDetector(random_state=SEED).fit(data["x_train"]), data
    )


def _dae_recall(data: dict) -> dict[str, float]:
    """Per-family recall for the Part 2.1 nonlinear autoencoder, at the real budget."""
    return _family_recall(
        DenoisingAutoencoderDetector(random_state=SEED, epochs=80).fit(data["x_train"]),
        data,
    )


def _family_recall(detector, data: dict, target_fpr: float = 0.10) -> dict[str, float]:
    """Per-family recall for one detector at its own FPR budget on benign traffic."""
    threshold = float(np.quantile(detector.score(data["x_train"]), 1.0 - target_fpr))
    return detection_report(
        split="test",
        y_true=data["y"][data["split"].test],
        scores=detector.score(data["x_test"]),
        threshold=threshold,
        families=[data["families"][i] for i in data["split"].test],
    ).per_family_recall


@pytest.mark.slow
class TestBotnetGap:
    """Part 1's documented weak spot, measured at the scale the README quotes.

    Run at n=20,000 deliberately. At n=6,000 the test split holds only ~20 botnet
    flows, so per-family recall moves in 5% steps and PCA and the autoencoder
    measured *identically* (0.950 each) — small-sample noise, not parity. The gap
    is only measurable once there are enough botnet flows for the number to mean
    something, which is also the scale every figure in the README uses.
    """

    def test_the_autoencoder_closes_the_botnet_gap(self) -> None:
        data = _pipeline(20_000)
        pca = _pca_recall(data)
        dae = _dae_recall(data)
        assert "botnet" in pca and "botnet" in dae, "no botnet flows reached the test split"
        # Measured: PCA 0.608, autoencoder 1.000. Asserting a margin rather than the
        # exact figure so the test pins the *finding* (nonlinearity closes this gap)
        # rather than one run's arithmetic.
        assert dae["botnet"] > pca["botnet"] + 0.20, (
            f"botnet recall: PCA {pca['botnet']:.4f}, autoencoder {dae['botnet']:.4f} — "
            "the nonlinearity did not close the documented gap"
        )

    def test_no_family_regresses_versus_the_linear_baseline(self) -> None:
        """Closing one gap by opening another would be a wash, not progress."""
        data = _pipeline(20_000)
        pca = _pca_recall(data)
        dae = _dae_recall(data)
        regressions = {
            family: (pca[family], dae[family])
            for family in pca
            if dae.get(family, 0.0) < pca[family] - 0.01
        }
        assert not regressions, f"families regressed vs PCA: {regressions}"

    def test_the_deep_ensemble_meets_f03(self) -> None:
        """F-03: ROC-AUC >= 0.90 on the held-out split. Measured 0.9971 with the floor."""
        data = _pipeline(20_000)
        ensemble = build_deep_ensemble(random_state=SEED).fit(data["x_train"])
        ensemble.tune_weights(
            data["x_validation"], data["y"][data["split"].validation], min_weight=0.10
        )
        auc = roc_auc(data["y"][data["split"].test], ensemble.score(data["x_test"]))
        assert auc >= 0.90, f"F-03 failed: ROC-AUC {auc:.4f}"

    def test_the_deep_ensemble_beats_the_shallow_one(self) -> None:
        data = _pipeline(20_000)
        shallow = WeightedEnsemble(
            [
                IsolationForestDetector(random_state=SEED),
                PCAReconstructionDetector(random_state=SEED),
            ],
            weights=[0.5, 0.5],
        ).fit(data["x_train"])
        shallow.tune_weights(data["x_validation"], data["y"][data["split"].validation])

        deep = build_deep_ensemble(random_state=SEED).fit(data["x_train"])
        deep.tune_weights(
            data["x_validation"], data["y"][data["split"].validation], min_weight=0.10
        )

        y_test = data["y"][data["split"].test]
        shallow_auc = roc_auc(y_test, shallow.score(data["x_test"]))
        deep_auc = roc_auc(y_test, deep.score(data["x_test"]))
        assert deep_auc > shallow_auc, (
            f"deep {deep_auc:.4f} did not beat shallow {shallow_auc:.4f}"
        )

    @pytest.mark.parametrize("bottleneck", [4, 10, 24, 31])
    def test_detection_quality_is_flat_in_bottleneck_width(self, bottleneck: int) -> None:
        """Pins the sweep from the module docstring: 4 to 31 all score >= 0.99 AUC.

        Two things this protects. First, nobody needs to tune this hyperparameter —
        the optimum is shallow and the default sits in it. Second, if a future
        change *does* make the width matter, that is a signal something else broke
        (most likely the corruption, which is the actual regulariser).
        """
        data = _pipeline(20_000)
        detector = DenoisingAutoencoderDetector(
            random_state=SEED, epochs=80, bottleneck=bottleneck, hidden=64
        ).fit(data["x_train"])
        auc = roc_auc(data["y"][data["split"].test], detector.score(data["x_test"]))
        assert auc >= 0.99, f"bottleneck={bottleneck} gave ROC-AUC {auc:.4f}"

    def test_cross_dataset_generalisation_holds(self) -> None:
        """PRD Section 10: train on CIC, score UNSW. Guards against memorising one
        generator's quirks — and against the unit mismatches Part 1 found."""
        data = _pipeline(20_000)
        detector = DenoisingAutoencoderDetector(random_state=SEED, epochs=80).fit(
            data["x_train"]
        )
        other = generate_alerts(20_000, seed=SEED + 1, dataset="unsw", separability=1.0)
        y_other = np.array(
            [0 if a.ground_truth_label == BENIGN else 1 for a in other], dtype=int
        )
        auc = roc_auc(y_other, detector.score(data["vectorizer"].transform(other)))
        assert auc >= 0.90, f"cross-dataset ROC-AUC {auc:.4f}"


# --------------------------------------------------------------------------- #
# Ensemble weight collapse
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestEnsembleWeightCollapse:
    """An AUC-optimal tuner discards the Isolation Forest once the autoencoder is in.

    This is not a bug in the tuner — weight 0.0 genuinely maximises validation AUC.
    It is a conflict between the objective the tuner was given and the reason PRD
    Section 5.5.2 asks for an ensemble at all: *"so a single model's blind spot does
    not silently define the system's sensitivity."* Both halves are pinned here so
    the trade-off cannot quietly disappear in either direction.
    """

    def test_unconstrained_tuning_discards_a_detector(self) -> None:
        data = _pipeline(6_000)
        ensemble = build_deep_ensemble(random_state=SEED, epochs=40).fit(data["x_train"])
        report = ensemble.tune_weights(
            data["x_validation"], data["y"][data["split"].validation]
        )
        assert report["isolation_forest"] == pytest.approx(0.0, abs=1e-9), (
            f"expected the tuner to zero the forest, got {report}"
        )

    def test_a_floor_keeps_every_detector_contributing(self) -> None:
        data = _pipeline(6_000)
        ensemble = build_deep_ensemble(random_state=SEED, epochs=40).fit(data["x_train"])
        report = ensemble.tune_weights(
            data["x_validation"], data["y"][data["split"].validation], min_weight=0.10
        )
        for name in ("isolation_forest", "denoising_autoencoder"):
            assert report[name] >= 0.10 - 1e-9, f"{name} fell below the floor: {report}"
        assert sum(ensemble.weights) == pytest.approx(1.0)

    def test_the_floor_costs_little_auc(self) -> None:
        """Measured at n=20,000: 0.9988 free vs 0.9971 floored. The insurance is cheap."""
        data = _pipeline(6_000)
        ensemble = build_deep_ensemble(random_state=SEED, epochs=40).fit(data["x_train"])
        y_test = data["y"][data["split"].test]

        ensemble.tune_weights(data["x_validation"], data["y"][data["split"].validation])
        free_auc = roc_auc(y_test, ensemble.score(data["x_test"]))

        ensemble.tune_weights(
            data["x_validation"], data["y"][data["split"].validation], min_weight=0.10
        )
        floored_auc = roc_auc(y_test, ensemble.score(data["x_test"]))

        assert free_auc - floored_auc < 0.01, (
            f"the 0.10 floor cost {free_auc - floored_auc:.4f} AUC, more than expected"
        )


class TestWeightFloorMechanics:
    """Floor mechanics, on the fast shallow detectors — no training required."""

    @pytest.fixture
    def shallow(self) -> tuple[WeightedEnsemble, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(SEED)
        latent = rng.normal(size=(800, 3))
        benign = latent @ rng.normal(size=(3, 8)) + rng.normal(scale=0.05, size=(800, 8))
        ensemble = WeightedEnsemble(
            [
                IsolationForestDetector(random_state=SEED),
                PCAReconstructionDetector(random_state=SEED),
            ],
            weights=[0.5, 0.5],
        ).fit(benign)
        attacks = rng.normal(size=(200, 8)) * 8.0
        x = np.vstack([benign[:200], attacks])
        y = np.array([0] * 200 + [1] * 200)
        return ensemble, x, y

    def test_weights_respect_the_floor_and_still_sum_to_one(self, shallow) -> None:
        ensemble, x, y = shallow
        ensemble.tune_weights(x, y, min_weight=0.25)
        assert np.all(ensemble.weights >= 0.25 - 1e-12)
        assert float(ensemble.weights.sum()) == pytest.approx(1.0)

    def test_a_floor_at_one_over_k_is_refused(self, shallow) -> None:
        ensemble, x, y = shallow
        with pytest.raises(ValueError, match="nothing left to tune"):
            ensemble.tune_weights(x, y, min_weight=0.5)

    def test_a_negative_floor_is_refused(self, shallow) -> None:
        ensemble, x, y = shallow
        with pytest.raises(ValueError, match="min_weight must be in"):
            ensemble.tune_weights(x, y, min_weight=-0.1)

    def test_the_default_floor_is_zero_so_part_one_behaviour_is_unchanged(
        self, shallow
    ) -> None:
        ensemble, x, y = shallow
        before = ensemble.tune_weights(x, y)
        ensemble.tune_weights(x, y, min_weight=0.0)
        assert before == ensemble.tune_weights(x, y)

    def test_the_floor_applies_to_a_three_detector_ensemble(self) -> None:
        """Three members take the Dirichlet path, not the 2-D grid."""
        rng = np.random.default_rng(SEED)
        latent = rng.normal(size=(600, 3))
        benign = latent @ rng.normal(size=(3, 8)) + rng.normal(scale=0.05, size=(600, 8))
        ensemble = WeightedEnsemble(
            [
                IsolationForestDetector(random_state=SEED),
                PCAReconstructionDetector(variance_target=0.90, name="pca_tight"),
                PCAReconstructionDetector(variance_target=0.99, name="pca_loose"),
            ]
        ).fit(benign)
        x = np.vstack([benign[:150], rng.normal(size=(150, 8)) * 8.0])
        y = np.array([0] * 150 + [1] * 150)
        ensemble.tune_weights(x, y, min_weight=0.15)
        assert np.all(ensemble.weights >= 0.15 - 1e-12)
        assert float(ensemble.weights.sum()) == pytest.approx(1.0)


class TestBuildDeepEnsemble:
    def test_pairs_the_forest_with_the_autoencoder(self) -> None:
        names = [d.name for d in build_deep_ensemble().detectors]
        assert names == ["isolation_forest", "denoising_autoencoder"]

    def test_keep_pca_adds_a_third_member(self) -> None:
        names = [d.name for d in build_deep_ensemble(keep_pca=True).detectors]
        assert names == ["isolation_forest", "denoising_autoencoder", "pca_reconstruction"]

    def test_weights_start_uniform(self) -> None:
        ensemble = build_deep_ensemble()
        np.testing.assert_allclose(ensemble.weights, [0.5, 0.5])

    def test_explanation_names_every_detector(self, benign_matrix) -> None:
        """PRD Section 5.1: an analyst must be able to see which detector fired."""
        ensemble = build_deep_ensemble(epochs=6).fit(benign_matrix)
        explanation = ensemble.score_detail(benign_matrix).explain(0)
        assert set(explanation) == {
            "combined",
            "isolation_forest",
            "denoising_autoencoder",
        }
