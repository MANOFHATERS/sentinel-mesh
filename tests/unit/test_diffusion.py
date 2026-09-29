"""Tabular denoising diffusion (:mod:`sentinel.ml.diffusion`).

Generative models are unusually easy to test badly. "It produced numbers" passes on a
model that outputs noise, and "the loss went down" passes on one that has learned the
marginal mean. So the substantive tests here compare *distributions*: per-column means
and standard deviations against the real family, and the correlation structure between
columns, which is the first thing a model that has only learned marginals gets wrong.

:class:`TestScheduleAvoidsTheAmplificationTrap` and
:class:`TestSamplingIsStable` exist because of a real bug. Clipping ``alpha_bar`` from
below left one enormous consecutive ratio at the clip boundary — a per-step
amplification of 2.46 — which pushed the reverse trajectory off the region the
denoiser was trained on and produced samples with twenty times the real standard
deviation. Both the schedule property and the resulting sample moments are asserted so
that failure cannot return quietly.

:class:`TestGradientsAreCorrect` runs the checker on the network
:meth:`~sentinel.ml.diffusion.TabularDiffusion.build_network` actually ships, which is
Part 2's finding 6: a parameter-only gradient check on a hand-built stack says nothing
about whether the stack is wired correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ml.diffusion import (
    DEFAULT_TIMESTEPS,
    DiffusionError,
    TabularDiffusion,
    cosine_alpha_bar,
    timestep_features,
)
from sentinel.ml.nn import gradient_check, mse_loss

RARE = ("web_attack", "botnet", "infiltration", "brute_force")


@pytest.fixture(scope="module")
def minority() -> tuple[np.ndarray, list[str]]:
    """Real minority-family rows, through the real vectorizer."""
    from sentinel.ml.datasets.synthetic import generate_alerts
    from sentinel.ml.featurestore import AlertVectorizer

    alerts = generate_alerts(12_000, seed=20260929)
    vectorizer = AlertVectorizer().fit(alerts)
    matrix = vectorizer.transform(alerts)
    families = np.asarray([a.ground_truth_label or "benign" for a in alerts])
    mask = np.isin(families, RARE)
    return matrix[mask], list(families[mask])


@pytest.fixture(scope="module")
def fitted(minority: tuple[np.ndarray, list[str]]) -> TabularDiffusion:
    matrix, families = minority
    return TabularDiffusion(
        n_steps=200, epochs=200, hidden=(160, 160), seed=1
    ).fit(matrix, families)


class TestScheduleAvoidsTheAmplificationTrap:
    def test_starts_near_one_and_ends_near_zero(self) -> None:
        schedule = cosine_alpha_bar(200)
        assert schedule[0] > 0.999
        assert schedule[-1] < 1e-4

    def test_is_monotonically_decreasing(self) -> None:
        schedule = cosine_alpha_bar(200)
        assert np.all(np.diff(schedule) <= 0.0)

    def test_length_is_n_steps_plus_one(self) -> None:
        for n_steps in (1, 10, 200):
            assert cosine_alpha_bar(n_steps).shape == (n_steps + 1,)

    def test_stays_strictly_inside_the_unit_interval(self) -> None:
        schedule = cosine_alpha_bar(200)
        assert np.all(schedule > 0.0)
        assert np.all(schedule < 1.0)

    def test_no_consecutive_ratio_exceeds_the_beta_limit(self) -> None:
        """The property whose absence caused twenty-fold sample variance.

        The reverse step amplifies the state by ``1 / sqrt(alpha_t)``. Flooring
        ``alpha_bar`` leaves one huge ratio at the clip boundary; bounding ``beta``
        bounds every ratio.
        """
        for n_steps in (10, 50, 200, 500):
            schedule = cosine_alpha_bar(n_steps)
            ratios = schedule[1:] / schedule[:-1]
            assert np.all(ratios >= 1.0 - 0.999 - 1e-12), n_steps
            assert np.max(1.0 / np.sqrt(ratios)) < 32.0, n_steps

    def test_per_step_alpha_decreases_monotonically(self) -> None:
        """The property a clip boundary breaks, stated exactly.

        Under a cosine schedule the per-step ``alpha_t`` falls steadily: each step
        removes a little more signal than the last. Flooring ``alpha_bar`` flattens the
        tail to a constant and then drops it in one move, so the sequence stops being
        monotone right at the boundary. Asserting monotonicity is precise, where a
        threshold on the jump size would just be another arbitrary constant.
        """
        for n_steps in (50, 200, 500):
            schedule = cosine_alpha_bar(n_steps)
            ratios = schedule[1:] / schedule[:-1]
            assert np.all(np.diff(ratios) <= 1e-12), n_steps

    @pytest.mark.parametrize("n_steps", [0, -1])
    def test_invalid_step_count_rejected(self, n_steps: int) -> None:
        with pytest.raises(DiffusionError, match="n_steps"):
            cosine_alpha_bar(n_steps)

    def test_non_positive_offset_rejected(self) -> None:
        with pytest.raises(DiffusionError, match="offset"):
            cosine_alpha_bar(10, offset=0.0)


class TestTimestepFeatures:
    def test_shape(self) -> None:
        steps = np.asarray([1, 5, 50], dtype=np.int64)
        assert timestep_features(steps, n_features=16, n_steps=200).shape == (3, 16)

    def test_bounded(self) -> None:
        steps = np.arange(1, 201, dtype=np.int64)
        features = timestep_features(steps, n_features=16, n_steps=200)
        assert np.all(np.abs(features) <= 1.0 + 1e-12)

    def test_distinct_timesteps_get_distinct_embeddings(self) -> None:
        steps = np.arange(1, 201, dtype=np.int64)
        features = timestep_features(steps, n_features=16, n_steps=200)
        assert len({tuple(np.round(row, 9)) for row in features}) == 200

    def test_adjacent_steps_are_closer_than_distant_ones(self) -> None:
        """What the sinusoidal basis buys over feeding the raw integer."""
        features = timestep_features(
            np.asarray([10, 11, 180], dtype=np.int64), n_features=16, n_steps=200
        )
        near = np.linalg.norm(features[0] - features[1])
        far = np.linalg.norm(features[0] - features[2])
        assert near < far

    def test_deterministic(self) -> None:
        steps = np.asarray([3, 7], dtype=np.int64)
        np.testing.assert_array_equal(
            timestep_features(steps, n_features=8, n_steps=100),
            timestep_features(steps, n_features=8, n_steps=100),
        )

    @pytest.mark.parametrize("n_features", [0, 1, 3, 15])
    def test_odd_or_tiny_width_rejected(self, n_features: int) -> None:
        with pytest.raises(DiffusionError, match="even"):
            timestep_features(
                np.asarray([1], dtype=np.int64), n_features=n_features, n_steps=10
            )


class TestConfiguration:
    @pytest.mark.parametrize(
        "kwargs,fragment",
        [
            ({"n_steps": 0}, "n_steps"),
            ({"n_time_features": 7}, "n_time_features"),
            ({"n_time_features": 0}, "n_time_features"),
            ({"hidden": ()}, "hidden"),
            ({"hidden": (0,)}, "hidden"),
            ({"min_rows_per_family": 1}, "min_rows_per_family"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict, fragment: str) -> None:
        with pytest.raises(DiffusionError, match=fragment):
            TabularDiffusion(**kwargs)

    def test_default_step_count(self) -> None:
        assert TabularDiffusion().n_steps == DEFAULT_TIMESTEPS

    def test_unfitted_model_refuses_to_sample(self) -> None:
        with pytest.raises(DiffusionError, match="not fitted"):
            TabularDiffusion().sample({"web_attack": 1})

    def test_unfitted_model_has_no_report(self) -> None:
        with pytest.raises(DiffusionError, match="not fitted"):
            TabularDiffusion().training_report()


class TestFitting:
    def test_fits_and_reports(self, fitted: TabularDiffusion) -> None:
        report = fitted.training_report()
        assert report["converged"]
        assert report["final_loss"] < report["first_loss"]
        assert set(report["families"]) == set(RARE)
        assert isinstance(report["parameters"], int)

    def test_input_width_accounts_for_time_and_class_blocks(
        self, fitted: TabularDiffusion
    ) -> None:
        assert fitted.input_width == (
            fitted.n_features + fitted.n_time_features + len(fitted.families_)
        )

    def test_families_are_sorted(self, fitted: TabularDiffusion) -> None:
        assert list(fitted.families_) == sorted(fitted.families_)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(DiffusionError, match="family labels"):
            TabularDiffusion().fit(np.zeros((4, 3)), ["a", "b"])

    def test_one_dimensional_input_rejected(self) -> None:
        with pytest.raises(DiffusionError, match="2-D"):
            TabularDiffusion().fit(np.zeros(5), ["a"] * 5)

    def test_non_finite_input_rejected(self) -> None:
        matrix = np.zeros((20, 3))
        matrix[0, 0] = np.nan
        with pytest.raises(DiffusionError, match="non-finite"):
            TabularDiffusion().fit(matrix, ["a"] * 20)

    def test_families_below_the_floor_are_skipped_not_modelled_badly(self) -> None:
        """A model fitted on four rows produces samples worse than the rows."""
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(40, 4))
        families = ["common"] * 36 + ["vanishing"] * 4
        model = TabularDiffusion(
            n_steps=20, epochs=40, hidden=(16,), min_rows_per_family=12, seed=2
        ).fit(matrix, families)
        assert model.families_ == ("common",)
        assert model.skipped_families_ == ("vanishing",)

    def test_all_families_below_the_floor_is_an_error(self) -> None:
        with pytest.raises(DiffusionError, match="no family has"):
            TabularDiffusion(min_rows_per_family=100).fit(
                np.random.default_rng(1).normal(size=(20, 3)), ["a"] * 20
            )

    def test_constant_columns_do_not_divide_by_zero(self) -> None:
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(60, 4))
        matrix[:, 2] = 7.0
        model = TabularDiffusion(n_steps=20, epochs=40, hidden=(16,), seed=3).fit(
            matrix, ["a"] * 60
        )
        assert model.scale_ is not None
        assert np.all(np.isfinite(model.scale_))
        samples, _ = model.sample({"a": 20}, rng=np.random.default_rng(4))
        assert np.all(np.isfinite(samples))


class TestGradientsAreCorrect:
    def test_the_shipped_network_composition_gradient_checks(self) -> None:
        """Part 2 finding 6: check the composition, not just the parameters."""
        model = TabularDiffusion(n_steps=50, n_time_features=8, hidden=(24, 24), seed=9)
        net = model.build_network(n_features=6, n_classes=3)
        rng = np.random.default_rng(11)
        inputs = rng.normal(size=(5, 6 + 8 + 3))
        targets = rng.normal(size=(5, 6))
        error = gradient_check(net, inputs, targets, loss_fn=mse_loss, rng=rng)
        assert error < 1e-6, f"max relative gradient error {error:.2e}"

    def test_a_deeper_composition_also_checks(self) -> None:
        model = TabularDiffusion(n_steps=50, n_time_features=8, hidden=(16, 16, 16), seed=13)
        net = model.build_network(n_features=4, n_classes=2)
        rng = np.random.default_rng(17)
        error = gradient_check(
            net,
            rng.normal(size=(4, 4 + 8 + 2)),
            rng.normal(size=(4, 4)),
            loss_fn=mse_loss,
            rng=rng,
        )
        assert error < 1e-6

    def test_the_head_is_linear(self) -> None:
        """A squashed head would cap the epsilon the model can predict."""
        model = TabularDiffusion(hidden=(8,), seed=19)
        net = model.build_network(n_features=3, n_classes=2)
        rng = np.random.default_rng(23)
        big = 50.0 * rng.normal(size=(8, 3 + model.n_time_features + 2))
        output = net.forward(big)
        assert np.abs(output).max() > 1.5


class TestSamplingIsStable:
    """The regression guard for the twenty-fold variance bug."""

    @pytest.fixture(scope="class")
    def samples(
        self, fitted: TabularDiffusion
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix, families = fitted.sample(
            {family: 300 for family in fitted.families_},
            rng=np.random.default_rng(29),
        )
        return matrix, np.asarray(families)

    def test_shape_and_labels(
        self, fitted: TabularDiffusion, samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        matrix, labels = samples
        assert matrix.shape == (300 * len(fitted.families_), fitted.n_features)
        assert set(labels) == set(fitted.families_)
        for family in fitted.families_:
            assert int((labels == family).sum()) == 300

    def test_samples_are_finite(self, samples: tuple[np.ndarray, np.ndarray]) -> None:
        assert np.all(np.isfinite(samples[0]))

    def test_overall_scale_matches_the_real_data(
        self,
        fitted: TabularDiffusion,
        minority: tuple[np.ndarray, list[str]],
        samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """The bug produced a standard deviation twenty times too large."""
        real, _ = minority
        synthetic, _ = samples
        ratio = float(synthetic.std() / real.std())
        assert 0.5 < ratio < 2.0, f"synthetic/real std ratio {ratio:.2f}"

    def test_per_family_moments_match(
        self,
        fitted: TabularDiffusion,
        minority: tuple[np.ndarray, list[str]],
        samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        real, real_families = minority
        real_families = np.asarray(real_families)
        synthetic, labels = samples
        for family in fitted.families_:
            actual = real[real_families == family]
            generated = synthetic[labels == family]
            assert abs(generated.mean() - actual.mean()) < 0.6, family
            ratio = generated.std() / max(actual.std(), 1e-9)
            assert 0.4 < ratio < 2.5, f"{family} std ratio {ratio:.2f}"

    def test_samples_stay_near_the_real_data_range(
        self,
        fitted: TabularDiffusion,
        minority: tuple[np.ndarray, list[str]],
        samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Every column stays within a generous multiple of the real column's range.

        Not the clamp box exactly, and the distinction is worth being precise about:
        the clamp bounds the *x0 estimate* at each reverse step, while the step's
        output is a convex combination of that estimate and the current noisy state,
        plus the step's own noise. So the final sample can sit slightly outside the
        box, and a test asserting otherwise would be asserting something the algorithm
        does not promise.

        What the clamp does promise is that the trajectory cannot run away, and that is
        what this checks. The bug it guards produced values twenty times the real
        spread; this tolerance is nowhere near admitting that.
        """
        real, _ = minority
        synthetic, _ = samples
        span = real.max(axis=0) - real.min(axis=0)
        span = np.where(span > 1e-12, span, 1.0)
        assert np.all(synthetic >= real.min(axis=0) - 2.0 * span)
        assert np.all(synthetic <= real.max(axis=0) + 2.0 * span)

    def test_the_x0_clamp_bounds_are_derived_from_the_data(
        self, fitted: TabularDiffusion, minority: tuple[np.ndarray, list[str]]
    ) -> None:
        """The clamp must bracket the real data, or it would truncate real structure."""
        real, _ = minority
        assert fitted.clamp_low_ is not None and fitted.clamp_high_ is not None
        assert fitted.mean_ is not None and fitted.scale_ is not None
        standardized = (real - fitted.mean_) / fitted.scale_
        assert np.all(fitted.clamp_low_ <= standardized.min(axis=0))
        assert np.all(fitted.clamp_high_ >= standardized.max(axis=0))

    def test_correlation_structure_is_partly_preserved(
        self,
        fitted: TabularDiffusion,
        minority: tuple[np.ndarray, list[str]],
        samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """The test a marginals-only model fails.

        A generator that learned each column's mean and variance independently scores
        about zero here. It is not expected to reach 1.0 either: a few hundred rows
        per family does not determine a 32x32 correlation matrix.
        """
        real, real_families = minority
        real_families = np.asarray(real_families)
        synthetic, labels = samples
        for family in fitted.families_:
            actual = real[real_families == family]
            generated = synthetic[labels == family]
            with np.errstate(invalid="ignore", divide="ignore"):
                real_corr = np.corrcoef(actual.T)
                synthetic_corr = np.corrcoef(generated.T)
            usable = np.isfinite(real_corr) & np.isfinite(synthetic_corr)
            agreement = float(
                np.corrcoef(real_corr[usable].ravel(), synthetic_corr[usable].ravel())[0, 1]
            )
            assert agreement > 0.25, f"{family} correlation agreement {agreement:.3f}"

    def test_families_are_distinguishable(
        self, fitted: TabularDiffusion, samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Conditioning must do something: per-family means must differ."""
        synthetic, labels = samples
        centroids = np.stack(
            [synthetic[labels == family].mean(axis=0) for family in fitted.families_]
        )
        spread = float(np.std(centroids, axis=0).mean())
        assert spread > 0.05, "class conditioning has no measurable effect"

    def test_deterministic_for_a_seed(self, fitted: TabularDiffusion) -> None:
        first, labels_a = fitted.sample(
            {"botnet": 20}, rng=np.random.default_rng(31)
        )
        second, labels_b = fitted.sample(
            {"botnet": 20}, rng=np.random.default_rng(31)
        )
        np.testing.assert_array_equal(first, second)
        assert labels_a == labels_b

    def test_different_seeds_differ(self, fitted: TabularDiffusion) -> None:
        first, _ = fitted.sample({"botnet": 20}, rng=np.random.default_rng(31))
        second, _ = fitted.sample({"botnet": 20}, rng=np.random.default_rng(37))
        assert not np.array_equal(first, second)

    def test_samples_are_not_copies_of_training_rows(
        self, minority: tuple[np.ndarray, list[str]], fitted: TabularDiffusion
    ) -> None:
        """A generator that memorises is a lookup table with extra steps."""
        real, _ = minority
        synthetic, _ = fitted.sample({"brute_force": 50}, rng=np.random.default_rng(41))
        for row in synthetic:
            distances = np.linalg.norm(real - row, axis=1)
            assert distances.min() > 1e-6

    def test_guidance_sharpens_without_diverging(self, fitted: TabularDiffusion) -> None:
        for guidance in (0.5, 1.0, 1.5):
            matrix, _ = fitted.sample(
                {"web_attack": 40}, rng=np.random.default_rng(43), guidance=guidance
            )
            assert np.all(np.isfinite(matrix))

    def test_zero_counts_return_an_empty_matrix(self, fitted: TabularDiffusion) -> None:
        matrix, labels = fitted.sample({family: 0 for family in fitted.families_})
        assert matrix.shape == (0, fitted.n_features)
        assert labels == ()

    def test_unmodelled_family_rejected(self, fitted: TabularDiffusion) -> None:
        with pytest.raises(DiffusionError, match="unmodelled families"):
            fitted.sample({"benign": 5})

    def test_negative_counts_rejected(self, fitted: TabularDiffusion) -> None:
        with pytest.raises(DiffusionError, match="non-negative"):
            fitted.sample({"botnet": -1})

    def test_non_positive_guidance_rejected(self, fitted: TabularDiffusion) -> None:
        with pytest.raises(DiffusionError, match="guidance"):
            fitted.sample({"botnet": 5}, guidance=0.0)


class TestForwardProcess:
    """The noising direction, which the training objective depends on."""

    def test_noising_interpolates_between_data_and_noise(self) -> None:
        schedule = cosine_alpha_bar(200)
        rng = np.random.default_rng(47)
        clean = rng.normal(size=(500, 5))
        for step in (1, 50, 100, 199):
            noise = rng.normal(size=clean.shape)
            noisy = np.sqrt(schedule[step]) * clean + np.sqrt(1.0 - schedule[step]) * noise
            # Variance of the mixture is preserved at 1 when the data is standardized.
            assert abs(float(noisy.std()) - 1.0) < 0.15, step

    def test_late_steps_destroy_the_signal(self) -> None:
        schedule = cosine_alpha_bar(200)
        rng = np.random.default_rng(53)
        clean = rng.normal(size=(500, 5))
        noise = rng.normal(size=clean.shape)
        early = np.sqrt(schedule[1]) * clean + np.sqrt(1.0 - schedule[1]) * noise
        late = np.sqrt(schedule[200]) * clean + np.sqrt(1.0 - schedule[200]) * noise
        assert abs(float(np.corrcoef(early.ravel(), clean.ravel())[0, 1])) > 0.95
        assert abs(float(np.corrcoef(late.ravel(), clean.ravel())[0, 1])) < 0.10
