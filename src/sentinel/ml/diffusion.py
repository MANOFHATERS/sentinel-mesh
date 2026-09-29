"""Tabular denoising diffusion for rare attack families (PRD Section 5.5.5, 7.2).

What the PRD asks for
---------------------
Section 5.5.5: *"A tabular denoising diffusion model (a TabDDPM-style approach) is
trained on the minority attack classes and used to generate additional synthetic
samples, improving the anomaly detector's recall on rare classes without touching the
majority class distribution. The same generator doubles as a lightweight
adversarial-robustness check: perturbed synthetic samples near the decision boundary
are used to sanity-check that the Triage Agent's confidence calibration does not
collapse under slightly out-of-distribution input."*

Two purposes, and one of them does not survive contact with Section 5.5.2
-------------------------------------------------------------------------
The augmentation claim assumes a *supervised* detector whose decision boundary moves
when you add labelled minority examples. Section 5.5.2's detector is not one: the
denoising autoencoder and the Isolation Forest are semi-supervised novelty detectors
that :func:`~sentinel.ml.metrics.three_way_split` trains on **benign rows only**. They
never see an attack of any family during fitting, so synthetic attacks cannot move
their boundary, and adding them to the training set would break the benign-only
assumption the whole method rests on.

That is not a reason to skip the part. It is a reason to be precise about what the
generator is for here, and to measure rather than assume:

1.  **Where augmentation can help** is the supervised family classifier that the
    Triage Agent needs (F-02, *"≥ 85% agreement with dataset ground-truth labels"*).
    :mod:`sentinel.ml.classify` is that classifier, and
    ``scripts/evaluate.py --augment`` measures rare-family recall with and without
    synthetic rows on a held-out split that contains none of them.
2.  **Where it definitely helps** is Section 5.5.5's second purpose. A trained
    generator gives boundary-adjacent samples that are *on the data manifold* but
    off the training distribution, which is exactly the input a confidence estimate
    is most likely to be overconfident on. :mod:`sentinel.ml.robustness` uses it.

The measured effect of each is in ``docs/BUILD_PLAN.md``, including the case where it
is approximately zero.

The model
---------
Gaussian DDPM over standardized continuous features, with a cosine noise schedule and
one class-conditional denoiser rather than one model per family.

*   **Cosine schedule**, not linear. The linear schedule of the original DDPM work is
    tuned for hundreds of steps on images; at the 200 steps affordable here it
    destroys the signal too early, and the last third of the trajectory carries almost
    no information. The cosine schedule keeps ``alpha_bar`` away from zero for longer.
*   **Class conditioning via a one-hot block**, not a model per family. The rarest
    family in the mix is 0.8% of attacks, so a per-family model would be fitted on a
    few dozen rows; conditioning lets every family share the bulk of the network and
    is what makes the rare ones learnable at all.
*   **Sinusoidal timestep features.** Feeding the raw integer ``t`` makes the network
    spend capacity learning that 7 and 8 are adjacent. The sinusoidal basis hands it
    that for free, and it is four lines.
*   **Epsilon prediction**, not ``x_0`` prediction. Both are valid parameterisations;
    epsilon-prediction keeps the regression target unit-variance at every noise level,
    which matters when the optimiser's learning rate is shared across all timesteps.

The denoiser is built from :mod:`sentinel.ml.nn`'s ``Dense``/``ReLU``/``Sequential``
and trained by its ``train``, so it is gradient-checked at the depth it ships at —
Part 2's finding 6, that ``gradient_check`` is parameter-only and must be run on the
real composition, applies directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from sentinel.core.errors import SentinelError
from sentinel.ml.nn import (
    Dense,
    Identity,
    Module,
    ReLU,
    Sequential,
    TrainHistory,
    mse_loss,
    train,
)

__all__ = [
    "DEFAULT_TIMESTEPS",
    "DiffusionError",
    "TabularDiffusion",
    "cosine_alpha_bar",
    "timestep_features",
]

DTYPE: Final = np.float64

#: Diffusion steps. 200 is a deliberate compromise: the reverse process costs one
#: forward pass per step per sample, so 1000 steps (the original DDPM figure) would
#: make generating 20k rows a minute-scale operation inside a test suite. Sample
#: quality is measured in ``test_diffusion.py`` against the real distribution's
#: moments and correlations rather than assumed from the step count.
DEFAULT_TIMESTEPS: Final[int] = 200

#: Dimensionality of the sinusoidal timestep embedding. Must be even.
DEFAULT_TIME_FEATURES: Final[int] = 16

#: Largest permitted per-step noise ``beta_t``. This is the singularity guard, and it
#: is applied to ``beta`` rather than to ``alpha_bar`` -- which is the fix for a real
#: bug, not a stylistic preference.
#:
#: Clipping ``alpha_bar`` from below looks equivalent and is not. The reverse step
#: amplifies the state by ``1 / sqrt(alpha_t)`` where ``alpha_t = alpha_bar[t] /
#: alpha_bar[t-1]``, so flooring ``alpha_bar`` flattens the tail of the schedule and
#: leaves one enormous *ratio* at the boundary between the clipped and unclipped
#: regions. Measured here, the floor produced ``alpha_200 = 0.165`` -- an
#: amplification of 2.46 in a single step -- which pushed the trajectory off the
#: unit-variance region the denoiser was trained on and never recovered: samples came
#: back with twenty times the real standard deviation. Bounding ``beta`` bounds the
#: consecutive ratio directly, which is what the improved-DDPM schedule does and why.
_MAX_BETA: Final[float] = 0.999

#: ``alpha_bar`` is held just below 1.0 at ``t = 0``: at exactly 1.0 the forward
#: process adds no noise, so the epsilon target is unidentifiable.
_ALPHA_BAR_EPS: Final[float] = 1e-6


class DiffusionError(SentinelError):
    """The generator was configured, fitted or sampled from incorrectly."""


def cosine_alpha_bar(n_steps: int, *, offset: float = 0.008) -> npt.NDArray[np.float64]:
    """Cumulative signal retention ``alpha_bar[t]`` under a cosine schedule.

    ``alpha_bar[t] = f(t) / f(0)`` with ``f(t) = cos((t/T + s) / (1 + s) * pi/2)^2``,
    the form from Nichol and Dhariwal's improved-DDPM work. Returns ``n_steps + 1``
    values so ``alpha_bar[0] == 1`` (no noise) and index ``t`` is the state *after*
    ``t`` noising steps, which keeps the indexing in :meth:`TabularDiffusion.sample`
    free of off-by-one corrections.
    """
    if n_steps < 1:
        raise DiffusionError(f"n_steps must be at least 1, got {n_steps}")
    if offset <= 0.0:
        raise DiffusionError(f"offset must be positive, got {offset}")
    steps = np.arange(n_steps + 1, dtype=DTYPE) / n_steps
    values = np.cos((steps + offset) / (1.0 + offset) * (np.pi / 2.0)) ** 2
    raw = values / values[0]

    # Rebuild alpha_bar from beta-clipped per-step alphas, so no consecutive ratio can
    # exceed the schedule's own limit. Clipping alpha_bar directly would leave exactly
    # the boundary discontinuity this is here to remove; see :data:`_MAX_BETA`.
    alphas = np.clip(raw[1:] / raw[:-1], 1.0 - _MAX_BETA, 1.0)
    alpha_bar = np.empty(n_steps + 1, dtype=DTYPE)
    alpha_bar[0] = 1.0 - _ALPHA_BAR_EPS
    np.cumprod(alphas, out=alpha_bar[1:])
    alpha_bar[1:] *= alpha_bar[0]
    return alpha_bar


def timestep_features(
    timesteps: npt.NDArray[np.int64], *, n_features: int, n_steps: int
) -> npt.NDArray[np.float64]:
    """Sinusoidal embedding of ``timesteps``, shape ``(len(timesteps), n_features)``.

    Frequencies are geometrically spaced, so the embedding resolves both coarse
    ("early or late in the trajectory") and fine ("which of these two adjacent
    steps") distinctions. The alternative — one scalar ``t / T`` — forces the first
    layer to spend capacity discovering ordinality it could have been told.
    """
    if n_features < 2 or n_features % 2:
        raise DiffusionError(f"n_features must be even and >= 2, got {n_features}")
    scaled = np.asarray(timesteps, dtype=DTYPE).reshape(-1, 1) / float(n_steps)
    half = n_features // 2
    frequencies = np.exp(np.linspace(0.0, np.log(1000.0), half, dtype=DTYPE))
    angles = scaled * frequencies.reshape(1, -1) * np.pi
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


@dataclass
class TabularDiffusion:
    """Class-conditional Gaussian DDPM over standardized tabular features."""

    n_steps: int = DEFAULT_TIMESTEPS
    n_time_features: int = DEFAULT_TIME_FEATURES
    hidden: tuple[int, ...] = (128, 128)
    epochs: int = 120
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 0.0
    patience: int = 12
    seed: int = 20260929
    #: Minimum real rows a family needs before it will be modelled. Below this the
    #: generator would be fitting noise and its samples would be worse than the real
    #: rows they are meant to supplement, so the family is refused rather than
    #: silently modelled badly.
    min_rows_per_family: int = 12

    families_: tuple[str, ...] = field(default=(), repr=False)
    mean_: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    scale_: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    clamp_low_: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    clamp_high_: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    history_: TrainHistory | None = field(default=None, repr=False)
    skipped_families_: tuple[str, ...] = field(default=(), repr=False)
    _net: Module | None = field(default=None, repr=False)
    _alpha_bar: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    _n_features: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise DiffusionError("n_steps must be at least 1")
        if self.n_time_features < 2 or self.n_time_features % 2:
            raise DiffusionError("n_time_features must be even and at least 2")
        if not self.hidden:
            raise DiffusionError("hidden must name at least one layer width")
        if any(width < 1 for width in self.hidden):
            raise DiffusionError("hidden widths must be positive")
        if self.min_rows_per_family < 2:
            raise DiffusionError("min_rows_per_family must be at least 2")

    # --- properties ----------------------------------------------------------- #

    @property
    def is_fitted(self) -> bool:
        return self._net is not None

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def input_width(self) -> int:
        """Denoiser input width: features, then timestep embedding, then class block."""
        return self._n_features + self.n_time_features + len(self.families_)

    def _require_fitted(self) -> Module:
        if self._net is None or self._alpha_bar is None:
            raise DiffusionError("TabularDiffusion is not fitted; call fit() first")
        return self._net

    # --- fitting -------------------------------------------------------------- #

    def build_network(self, n_features: int, n_classes: int) -> Sequential:
        """The denoiser. Exposed so a test can gradient-check the real composition.

        Part 2's finding 6: ``gradient_check`` verifies a module's own backward pass
        against numerical gradients, which says nothing about whether the *stack* is
        wired correctly. Handing the exact shipped network to the checker is the only
        version of that test worth running, so this method exists rather than the
        network being built inline in :meth:`fit`.
        """
        rng = np.random.default_rng(self.seed)
        width = n_features + self.n_time_features + n_classes
        layers: list[Module] = []
        previous = width
        for size in self.hidden:
            layers.append(Dense(previous, size, rng=rng))
            layers.append(ReLU())
            previous = size
        # Linear head: the target is a standard normal noise vector, so squashing the
        # output would cap the epsilon the model can predict and bias every sample
        # toward the mean.
        layers.append(Dense(previous, n_features, rng=rng))
        layers.append(Identity())
        return Sequential(*layers)

    def fit(
        self,
        x: npt.NDArray[np.float64],
        families: Sequence[str],
        *,
        validation_fraction: float = 0.15,
    ) -> TabularDiffusion:
        """Fit on minority-family rows only.

        ``x`` and ``families`` must be the *minority* subset already; this method does
        not filter by frequency, because "which families are rare" is a decision about
        the dataset that belongs at the call site where the mix is known.
        """
        matrix = np.asarray(x, dtype=DTYPE)
        if matrix.ndim != 2:
            raise DiffusionError(f"expected a 2-D matrix, got shape {matrix.shape}")
        if matrix.shape[0] != len(families):
            raise DiffusionError(
                f"{matrix.shape[0]} rows but {len(families)} family labels"
            )
        if not np.all(np.isfinite(matrix)):
            raise DiffusionError("training matrix contains non-finite values")

        counts: dict[str, int] = {}
        for family in families:
            counts[family] = counts.get(family, 0) + 1
        kept = sorted(f for f, n in counts.items() if n >= self.min_rows_per_family)
        skipped = tuple(sorted(f for f, n in counts.items() if n < self.min_rows_per_family))
        if not kept:
            raise DiffusionError(
                f"no family has {self.min_rows_per_family} rows; counts were {counts}"
            )
        self.families_ = tuple(kept)
        self.skipped_families_ = skipped

        mask = np.asarray([f in self.families_ for f in families])
        rows = matrix[mask]
        labels = [f for f in families if f in self.families_]

        # Standardize per column. The forward process adds unit-variance noise, so a
        # column on a different scale is either drowned or dominant, and the shared
        # learning rate makes that unrecoverable. Columns with zero variance get a
        # scale of 1 rather than 0: a constant column is not informative, and
        # dividing by its standard deviation is a division by zero.
        self.mean_ = rows.mean(axis=0)
        spread = rows.std(axis=0)
        self.scale_ = np.where(spread > 1e-12, spread, 1.0)
        standardized = (rows - self.mean_) / self.scale_

        # The range the reverse process clamps its x0 estimate to. Taken from the data
        # with a margin rather than fixed at a few standard deviations: these columns
        # are standardized but emphatically not Gaussian -- packet counts and flow
        # durations are heavy-tailed -- so a symmetric sigma-based clamp would chop off
        # real structure at one end and leave slack at the other.
        span = standardized.max(axis=0) - standardized.min(axis=0)
        margin = np.where(span > 1e-12, 0.25 * span, 1.0)
        self.clamp_low_ = standardized.min(axis=0) - margin
        self.clamp_high_ = standardized.max(axis=0) + margin

        self._n_features = standardized.shape[1]
        self._alpha_bar = cosine_alpha_bar(self.n_steps)
        self._net = self.build_network(self._n_features, len(self.families_))

        one_hot = np.zeros((len(labels), len(self.families_)), dtype=DTYPE)
        for row, family in enumerate(labels):
            one_hot[row, self.families_.index(family)] = 1.0

        # ``train`` batches over rows of one matrix, so the class block travels with
        # the data rather than in a parallel array that batching could desynchronise.
        # Desynchronised conditioning is a bug that produces plausible samples of the
        # wrong class, which no loss curve reveals.
        bundled = np.concatenate([standardized, one_hot], axis=1)

        n_steps = self.n_steps
        n_time = self.n_time_features
        n_class = len(self.families_)
        alpha_bar = self._alpha_bar

        def make_batch(
            batch: np.ndarray, generator: np.random.Generator
        ) -> tuple[np.ndarray, np.ndarray]:
            clean = batch[:, :-n_class] if n_class else batch
            classes = batch[:, -n_class:] if n_class else np.zeros((batch.shape[0], 0))
            steps = generator.integers(1, n_steps + 1, size=batch.shape[0])
            noise = generator.normal(size=clean.shape)
            retained = np.sqrt(alpha_bar[steps]).reshape(-1, 1)
            residual = np.sqrt(1.0 - alpha_bar[steps]).reshape(-1, 1)
            noisy = retained * clean + residual * noise
            inputs = np.concatenate(
                [
                    noisy,
                    timestep_features(steps, n_features=n_time, n_steps=n_steps),
                    classes,
                ],
                axis=1,
            )
            # Target is the noise, not the clean row. See the module docstring for why
            # epsilon-prediction is preferred over x0-prediction here.
            return inputs, noise

        rng = np.random.default_rng(self.seed + 1)
        order = rng.permutation(bundled.shape[0])
        n_validation = round(validation_fraction * bundled.shape[0])
        if 0 < n_validation < bundled.shape[0]:
            validation = bundled[order[:n_validation]]
            training = bundled[order[n_validation:]]
        else:
            validation = None
            training = bundled

        self.history_ = train(
            self._net,
            training,
            make_batch,
            rng=rng,
            epochs=self.epochs,
            batch_size=min(self.batch_size, training.shape[0]),
            lr=self.lr,
            weight_decay=self.weight_decay,
            loss_fn=mse_loss,
            validation_data=validation,
            patience=self.patience,
        )
        if not self.history_.converged:
            raise DiffusionError(
                "diffusion training loss did not decrease "
                f"({self.history_.train_loss[0]:.6g} -> {self.history_.train_loss[-1]:.6g}); "
                "an untrained denoiser samples noise, and augmenting a training set "
                "with noise is worse than not augmenting it"
            )
        return self

    # --- sampling ------------------------------------------------------------- #

    def sample(
        self,
        counts: dict[str, int],
        *,
        rng: np.random.Generator | None = None,
        guidance: float = 1.0,
    ) -> tuple[npt.NDArray[np.float64], tuple[str, ...]]:
        """Generate rows per family by ancestral DDPM sampling.

        ``guidance`` scales the predicted noise. 1.0 is the plain reverse process;
        values slightly above 1 sharpen samples toward the conditional mode at the
        cost of diversity. It is exposed because :mod:`sentinel.ml.robustness` wants
        the opposite of sharpening — samples deliberately near the edge of the
        manifold — and a single knob covers both.
        """
        net = self._require_fitted()
        assert self._alpha_bar is not None and self.mean_ is not None
        assert self.scale_ is not None
        assert self.clamp_low_ is not None and self.clamp_high_ is not None
        if guidance <= 0.0:
            raise DiffusionError(f"guidance must be positive, got {guidance}")
        unknown = sorted(set(counts) - set(self.families_))
        if unknown:
            raise DiffusionError(
                f"cannot sample unmodelled families {unknown}; fitted families are "
                f"{list(self.families_)}"
            )
        if any(n < 0 for n in counts.values()):
            raise DiffusionError("counts must be non-negative")
        total = sum(counts.values())
        if total == 0:
            return np.zeros((0, self._n_features), dtype=DTYPE), ()

        generator = rng if rng is not None else np.random.default_rng(self.seed + 2)
        alpha_bar = self._alpha_bar

        labels: list[str] = []
        for family in self.families_:
            labels.extend([family] * counts.get(family, 0))
        one_hot = np.zeros((total, len(self.families_)), dtype=DTYPE)
        for row, family in enumerate(labels):
            one_hot[row, self.families_.index(family)] = 1.0

        state = generator.normal(size=(total, self._n_features))
        for step in range(self.n_steps, 0, -1):
            steps = np.full(total, step, dtype=np.int64)
            inputs = np.concatenate(
                [
                    state,
                    timestep_features(
                        steps, n_features=self.n_time_features, n_steps=self.n_steps
                    ),
                    one_hot,
                ],
                axis=1,
            )
            predicted_noise = net.forward(inputs) * guidance

            current = alpha_bar[step]
            previous = alpha_bar[step - 1]
            alpha = current / previous
            beta = 1.0 - alpha

            # Route the update through an explicit, clamped estimate of x0 rather than
            # applying the epsilon correction to the state directly. Algebraically the
            # two are identical; numerically they are not, and the difference decides
            # whether sampling works at all.
            #
            # The reverse process amplifies the state by 1/sqrt(alpha_t) per step,
            # compounding to sqrt(1/alpha_bar_T) -- a factor of hundreds -- and that is
            # cancelled only by an accurate epsilon. A denoiser fitted on a few hundred
            # rare-family rows is not that accurate, so the trajectory drifts; once it
            # leaves the unit-variance region the network is extrapolating and the
            # drift becomes divergence. Clamping x0 to the observed data range is the
            # standard remedy and it is a hard bound: every step lands somewhere the
            # denoiser has actually seen data, so a bad epsilon costs accuracy on that
            # step instead of destroying every step after it.
            x0_estimate = (
                state - np.sqrt(1.0 - current) * predicted_noise
            ) / np.sqrt(current)
            x0_estimate = np.clip(x0_estimate, self.clamp_low_, self.clamp_high_)

            # Posterior mean of q(x_{t-1} | x_t, x0), the standard DDPM form.
            state = (
                np.sqrt(previous) * beta / (1.0 - current) * x0_estimate
                + np.sqrt(alpha) * (1.0 - previous) / (1.0 - current) * state
            )
            if step > 1:
                # Posterior variance, not beta. The "fixed large" variant
                # over-disperses tabular samples noticeably at this step count; the
                # posterior form is tighter and is what the moment tests in
                # ``test_diffusion.py`` pass against.
                variance = beta * (1.0 - previous) / (1.0 - current)
                state = state + np.sqrt(max(variance, 0.0)) * generator.normal(
                    size=state.shape
                )

        return state * self.scale_ + self.mean_, tuple(labels)

    # --- diagnostics ---------------------------------------------------------- #

    def training_report(self) -> dict[str, Any]:
        """Shape of the fit, for the evaluation report and the audit trail."""
        if self.history_ is None:
            raise DiffusionError("not fitted")
        net = self._require_fitted()
        return {
            "families": list(self.families_),
            "skipped_families": list(self.skipped_families_),
            "n_features": self._n_features,
            "input_width": self.input_width,
            "n_steps": self.n_steps,
            "parameters": net.n_parameters(),
            "epochs_run": self.history_.epochs_run,
            "first_loss": self.history_.train_loss[0],
            "final_loss": self.history_.train_loss[-1],
            "converged": self.history_.converged,
        }
