"""The denoising autoencoder from PRD Section 5.5.2 (Part 2.1).

Part 1 shipped :class:`~sentinel.ml.anomaly.PCAReconstructionDetector` and named
it honestly: a *linear* autoencoder at its closed-form optimum. This module
delivers the nonlinear one the PRD actually specifies —

    *"A denoising autoencoder trained on benign-only flow features from
    CIC-IDS2017 and UNSW-NB15 produces a reconstruction-error anomaly score"*

— behind the same :class:`~sentinel.ml.anomaly.AnomalyDetector` protocol, so
:func:`~sentinel.ml.anomaly.build_default_ensemble` swaps one for the other and
nothing else in the platform changes.

Why nonlinearity is expected to matter here specifically
--------------------------------------------------------
Part 1's evaluation left one documented weak spot: **botnet recall 0.608** at the
10%-FPR threshold, against 0.97+ for every other family. The reason is structural
rather than a tuning miss. A C2 beacon is a *low-rate, highly regular* flow: each
individual flow is small and unremarkable, and what distinguishes it is the
interaction between several features at once — small payload **and** regular
inter-arrival **and** a long-lived destination **and** low byte variance. A rank-k
linear subspace can only represent the directions of largest benign variance, so
a conjunction like that is not a large residual in any single linear direction.
The forest sees no marginal outlier either, because no individual value is
extreme. A nonlinear bottleneck can carve a decision surface around the
conjunction, which is exactly the shape of the gap.

That is the hypothesis. Whether it holds is measured, not assumed:
``tests/unit/test_deep.py::TestBotnetGap`` compares this detector against PCA on
the botnet family specifically, and the number it produces is reported in
``docs/BUILD_PLAN.md`` whichever way it comes out.

Three design choices that are load-bearing
------------------------------------------
**Corruption is applied to the input only, never to the target.** That is the
entire difference between a denoising autoencoder and an ordinary one: the model
is forced to learn the manifold benign traffic lies on, rather than a compressed
copy of each row. Corrupting the target too would just add label noise.

**Scoring uses the clean input.** Training corrupts; inference must not, or the
score for one row would depend on a random draw and two runs of the demo would
disagree. The test suite asserts scoring is deterministic across calls.

**The bottleneck must be narrower than the input** — but for a weaker reason than
the textbook gives, and it is worth being precise about which.

What the textbook warning claims
-------------------------------
The standard warning is that with ``bottleneck >= n_features`` the network learns
the identity map, reconstruction error collapses to ~0 for attacks as well as
benign traffic, and the detector becomes noise with an excellent training loss.

**That collapse could not be reproduced here.** Measured on the real 32-feature
space, sweeping the bottleneck from 2 to 31 (n=20,000, 80 epochs):

===========  ========  ========  =============
bottleneck   ROC-AUC   PR-AUC    botnet recall
===========  ========  ========  =============
2            0.9986    0.9985    0.987
8            0.9988    0.9987    1.000
10           0.9988    0.9988    1.000
16           0.9983    0.9983    0.987
24           0.9989    0.9988    1.000
31           0.9989    0.9988    0.987
===========  ========  ========  =============

Detection quality is **flat** — 0.9978 to 0.9989 across the whole range — and a
near-full-rank bottleneck of 31 scores as well as anything. Two reasons, both
worth knowing before tuning this:

1.  **Gradient descent from small initialisation does not find the identity.**
    Benign traffic occupies a low-dimensional manifold, and on that manifold any
    map that acts as the identity is optimal — including the *projector onto the
    manifold*, which is what the low-norm solution near the initialisation
    actually is. Nothing in the training objective rewards being the identity
    off-manifold, because no training point is off-manifold. The identity map is
    reachable in principle and not what optimisation goes to in practice.
2.  **The corruption is the regulariser, not the bottleneck.** Denoising trains
    the model to map a perturbed point back onto the manifold, which is a
    contractive constraint the width of the bottleneck does not relax.

A control experiment made the same point more sharply: a *linear* full-rank
autoencoder (identity activations, ``bottleneck == n_features``) separated
correlation-violating inputs from benign ones by 1141x, while a rank-2 bottleneck
managed 5x. The narrow model was **worse**, because at rank 2 it could not
reconstruct benign traffic either — its error on legitimate data rose enough to
swamp the contrast. Under-capacity, not over-capacity, was the failure that
actually showed up.

So why keep the guard? Because ``bottleneck >= n_features`` is not a bottleneck
architecture, and a class that offers one should not silently accept a
configuration that has none — the collapse is not guaranteed, but neither is it
excluded on a feature space that does span its dimensions, and the cost of
refusing is zero. It is an architectural coherence check with a modest
justification, which is what it is documented as. The default width
(``n_features // 3``) sits at the measured optimum rather than at a guess.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from sentinel.core.errors import ModelNotFittedError
from sentinel.ml.anomaly import _CalibratedDetector
from sentinel.ml.nn import Dense, Sequential, Tanh, TrainHistory, TrainingError, train

__all__ = ["DenoisingAutoencoderDetector", "build_deep_ensemble"]

DTYPE: Final = np.float64


class DenoisingAutoencoderDetector(_CalibratedDetector):
    """Reconstruction error from a nonlinear, benign-trained denoising autoencoder.

    Architecture ``d -> hidden -> bottleneck -> hidden -> d`` with ``tanh``
    activations and a linear output layer. Tanh rather than ReLU because the
    encoder must represent *both* directions of a standardised feature — a feature
    two standard deviations below its benign mean is as informative as one two
    above, and a ReLU encoder discards the sign, which for network-flow features
    (small-vs-large packet, short-vs-long duration) throws away half the signal.

    ``noise_std`` is in units of the detector's internal per-feature standard
    deviation, so it means the same thing regardless of whether the caller passed
    standardised features.
    """

    name = "denoising_autoencoder"

    def __init__(
        self,
        *,
        hidden: int | None = None,
        bottleneck: int | None = None,
        noise_std: float = 0.15,
        epochs: int = 80,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        validation_fraction: float = 0.15,
        random_state: int = 20260928,
        n_quantiles: int = 1024,
        name: str | None = None,
    ) -> None:
        super().__init__(n_quantiles=n_quantiles, name=name)
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if epochs < 2:
            # Fewer than two epochs leaves no before/after to compare, so the
            # convergence check below could never pass. Refusing here gives a clear
            # message instead of a confusing "training did not decrease" at fit time.
            raise ValueError(
                f"epochs must be >= 2 to verify the loss decreased, got {epochs}"
            )
        if not 0.0 <= validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if hidden is not None and hidden < 1:
            raise ValueError("hidden must be a positive width")
        if bottleneck is not None and bottleneck < 1:
            raise ValueError("bottleneck must be a positive width")
        if hidden is not None and bottleneck is not None and bottleneck > hidden:
            raise ValueError(
                f"bottleneck ({bottleneck}) wider than the hidden layer ({hidden}) is not "
                "a bottleneck; the architecture would widen then narrow"
            )
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.noise_std = float(noise_std)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        self._net: Sequential | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.history_: TrainHistory | None = None
        self.architecture_: tuple[int, int, int] | None = None

    # --- architecture -------------------------------------------------------

    def _widths(self, n_features: int) -> tuple[int, int]:
        """Resolve (hidden, bottleneck), defaulting relative to the feature count.

        Defaults scale with the input rather than being hard-coded, because the
        feature spec is not fixed: Part 1's session-context enricher already
        widened it once, and a hard-coded ``(64, 16)`` becomes either a pointless
        expansion or an information bottleneck the moment the spec moves.

        ``n_features // 3`` is the measured optimum on the 32-column spec (10),
        not a convention — see the bottleneck sweep in the module docstring. It is
        a shallow optimum: anything from 8 to 31 performs within 0.001 AUC.
        """
        hidden = self.hidden if self.hidden is not None else max(4, min(64, n_features * 2))
        bottleneck = (
            self.bottleneck if self.bottleneck is not None else max(2, n_features // 3)
        )
        if bottleneck >= n_features:
            raise ValueError(
                f"bottleneck ({bottleneck}) must be narrower than the input "
                f"({n_features} features), or this is not a bottleneck architecture. "
                "Note this is a coherence check, not protection against a measured "
                "failure: a bottleneck of n_features-1 scores as well as the default "
                "on this feature space (see the sweep in the module docstring). If you "
                "want no dimensionality reduction, use a different detector rather than "
                "an autoencoder whose bottleneck does not bottleneck."
            )
        bottleneck = min(bottleneck, hidden)
        return hidden, bottleneck

    def _build(self, n_features: int, rng: np.random.Generator) -> Sequential:
        hidden, bottleneck = self._widths(n_features)
        self.architecture_ = (n_features, hidden, bottleneck)
        return Sequential(
            Dense(n_features, hidden, rng=rng, gain_for="tanh", name="enc.0"),
            Tanh(),
            Dense(hidden, bottleneck, rng=rng, gain_for="tanh", name="enc.1"),
            Tanh(),
            Dense(bottleneck, hidden, rng=rng, gain_for="tanh", name="dec.0"),
            Tanh(),
            Dense(hidden, n_features, rng=rng, gain_for="linear", name="dec.1"),
        )

    # --- standardisation ----------------------------------------------------

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._scale is None:
            raise ModelNotFittedError("autoencoder standardisation is not fitted")
        return (x - self._mean) / self._scale

    # --- fit / score --------------------------------------------------------

    def _fit_model(self, x: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)

        self._mean = x.mean(axis=0)
        deviation = x.std(axis=0)
        # A constant column has zero variance. Dividing by it yields inf/nan and
        # poisons every downstream gradient; substituting 1.0 maps the column to a
        # constant 0, which contributes nothing to reconstruction error — the
        # correct outcome, since a column that never varies in benign traffic
        # carries no reconstruction signal. The feature store reports such columns
        # via `degenerate_columns()`; this is defence in depth, not a substitute.
        self._scale = np.where(deviation > 1e-12, deviation, 1.0)
        scaled = self._standardize(x)

        n_rows = scaled.shape[0]
        n_validation = round(n_rows * self.validation_fraction)
        # Early stopping needs enough held-out rows for the loss to mean something.
        # Below ~8 the curve is dominated by which rows landed in the split, and
        # stopping on it is worse than not stopping at all.
        if n_validation >= 8 and n_rows - n_validation >= self.batch_size:
            order = rng.permutation(n_rows)
            validation = scaled[order[:n_validation]]
            training = scaled[order[n_validation:]]
        else:
            validation = None
            training = scaled

        self._net = self._build(scaled.shape[1], rng)
        noise_std = self.noise_std

        def make_batch(
            rows: np.ndarray, generator: np.random.Generator
        ) -> tuple[np.ndarray, np.ndarray]:
            if noise_std == 0.0:
                return rows, rows
            corrupted = rows + generator.normal(0.0, noise_std, size=rows.shape)
            # Input corrupted, target clean. This asymmetry *is* the denoising
            # objective: it forces the model to learn the benign manifold rather
            # than a compressed copy of each row.
            return corrupted, rows

        self.history_ = train(
            self._net,
            training,
            make_batch,
            rng=rng,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            validation_data=validation,
            patience=self.patience,
        )
        if not self.history_.converged:
            raise TrainingError(
                f"{self.name}: training loss did not decrease "
                f"({self.history_.train_loss[0]:.6g} -> {self.history_.train_loss[-1]:.6g}). "
                "A detector whose reconstruction error is untrained scores noise; "
                "refusing to return it rather than silently degrading the ensemble."
            )

    def _raw_score(self, x: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise ModelNotFittedError("autoencoder is not fitted")
        scaled = self._standardize(x)
        # No corruption at inference. Adding noise here would make the score for a
        # given row depend on a random draw, so two runs of the demo would produce
        # different alerts from identical input.
        reconstructed = self._net.forward(scaled)
        return np.mean(np.square(scaled - reconstructed), axis=1, dtype=DTYPE)

    # --- introspection ------------------------------------------------------

    def training_report(self) -> dict[str, Any]:
        """Architecture and training curve, for the evaluation report (F-12)."""
        if self.history_ is None or self.architecture_ is None:
            raise ModelNotFittedError(f"{self.name} is not fitted")
        n_features, hidden, bottleneck = self.architecture_
        return {
            "architecture": f"{n_features}-{hidden}-{bottleneck}-{hidden}-{n_features}",
            "n_parameters": self.history_.n_parameters,
            "epochs_run": self.history_.epochs_run,
            "stopped_early": self.history_.stopped_early,
            "final_train_loss": self.history_.train_loss[-1],
            "best_validation_loss": (
                self.history_.best_validation_loss
                if self.history_.validation_loss
                else None
            ),
            "compression_ratio": bottleneck / n_features,
            "noise_std": self.noise_std,
        }


def build_deep_ensemble(
    *,
    random_state: int = 20260928,
    epochs: int = 80,
    keep_pca: bool = False,
) -> Any:
    """The PRD Section 5.5.2 ensemble with the nonlinear autoencoder.

    ``keep_pca=True`` returns a three-detector ensemble instead. That is worth
    having rather than being indecisive: the linear and nonlinear reconstruction
    detectors fail on *different* rows, and
    :meth:`~sentinel.ml.anomaly.WeightedEnsemble.tune_weights` handles three
    members. It is off by default because the PRD specifies a two-model ensemble
    and three detectors triple the explanation an analyst has to read.
    """
    from sentinel.ml.anomaly import (
        IsolationForestDetector,
        PCAReconstructionDetector,
        WeightedEnsemble,
    )

    detectors: list[Any] = [
        IsolationForestDetector(random_state=random_state),
        DenoisingAutoencoderDetector(random_state=random_state, epochs=epochs),
    ]
    if keep_pca:
        detectors.append(PCAReconstructionDetector(random_state=random_state))
    return WeightedEnsemble(detectors=detectors, weights=[1.0] * len(detectors))
