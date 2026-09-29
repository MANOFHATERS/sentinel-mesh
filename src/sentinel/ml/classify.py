"""Supervised attack-family classifier (PRD F-02, and the measuring stick for 2.2).

Why this exists
---------------
PRD Section 5.5.5 claims augmentation improves *"the anomaly detector's recall on rare
classes"*. Section 5.5.2's detector cannot benefit: it is a semi-supervised novelty
detector trained on benign rows only, so no quantity of synthetic attacks moves its
boundary, and adding them would break the benign-only assumption the method rests on.
:mod:`sentinel.ml.diffusion` says more about that.

What *can* benefit is a supervised model over labelled families, which the Triage
Agent needs anyway — F-02 asks for *"≥ 85% agreement with dataset ground-truth
labels"*, and agreement on a label is a classification problem, not a novelty-detection
one. So this is both a component the system needs and the only honest way to answer
"does augmentation help?" with a number.

Deliberately small
------------------
One hidden layer, softmax output, AdamW, early stopping. It is not trying to be the
best classifier obtainable on this data; it is trying to be a *stable* one, because
the quantity being measured is a difference between two fits and a high-variance model
makes that difference unreadable. The same architecture, seed and schedule are used for
the augmented and unaugmented runs, so the only thing that differs is the training
rows — which is the whole design of the experiment.

Class weighting, and why it is off by default
---------------------------------------------
Inverse-frequency class weights are the standard cheap fix for imbalance and they
would confound this measurement completely: weighting and augmentation address the
same problem, so an augmented run with weighting on cannot show what augmentation did.
:attr:`FamilyClassifier.class_weight` therefore defaults to ``None``, and
``scripts/evaluate.py --augment`` reports weighting as a *separate* baseline, because
"augmentation helps" is only interesting next to "and it helps more than the one-line
alternative".
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
    softmax,
    softmax_cross_entropy_loss,
    train,
)

__all__ = ["ClassifierError", "FamilyClassifier"]

DTYPE: Final = np.float64


class ClassifierError(SentinelError):
    """The classifier was configured, fitted or queried incorrectly."""


@dataclass
class FamilyClassifier:
    """Multinomial classifier over attack families, on the shared numpy engine."""

    hidden: tuple[int, ...] = (64,)
    epochs: int = 150
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 20260929
    #: ``"balanced"`` applies inverse-frequency weights. Off by default because it
    #: addresses the same problem as augmentation and would confound the measurement.
    class_weight: str | None = None
    label_smoothing: float = 0.0

    classes_: tuple[str, ...] = field(default=(), repr=False)
    mean_: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    scale_: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    history_: TrainHistory | None = field(default=None, repr=False)
    _net: Module | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.hidden or any(width < 1 for width in self.hidden):
            raise ClassifierError("hidden must be non-empty with positive widths")
        if self.class_weight not in (None, "balanced"):
            raise ClassifierError(
                f"class_weight must be None or 'balanced', got {self.class_weight!r}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ClassifierError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )

    @property
    def is_fitted(self) -> bool:
        return self._net is not None

    def _require_fitted(self) -> Module:
        if self._net is None:
            raise ClassifierError("FamilyClassifier is not fitted; call fit() first")
        return self._net

    def build_network(self, n_features: int, n_classes: int) -> Sequential:
        """Exposed so a test can gradient-check the exact shipped composition."""
        rng = np.random.default_rng(self.seed)
        layers: list[Module] = []
        previous = n_features
        for width in self.hidden:
            layers.append(Dense(previous, width, rng=rng))
            layers.append(ReLU())
            previous = width
        layers.append(Dense(previous, n_classes, rng=rng))
        layers.append(Identity())  # logits; the loss owns the softmax
        return Sequential(*layers)

    def fit(
        self,
        x: npt.NDArray[np.float64],
        families: Sequence[str],
        *,
        validation_fraction: float = 0.15,
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> FamilyClassifier:
        """Fit on ``x`` with string family labels."""
        matrix = np.asarray(x, dtype=DTYPE)
        if matrix.ndim != 2:
            raise ClassifierError(f"expected a 2-D matrix, got {matrix.shape}")
        if matrix.shape[0] != len(families):
            raise ClassifierError(
                f"{matrix.shape[0]} rows but {len(families)} labels"
            )
        if matrix.shape[0] < 2:
            raise ClassifierError("need at least two rows to fit")
        if not np.all(np.isfinite(matrix)):
            raise ClassifierError("training matrix contains non-finite values")

        self.classes_ = tuple(sorted(set(families)))
        if len(self.classes_) < 2:
            raise ClassifierError(
                f"need at least two classes, got {list(self.classes_)}"
            )

        self.mean_ = matrix.mean(axis=0)
        spread = matrix.std(axis=0)
        self.scale_ = np.where(spread > 1e-12, spread, 1.0)
        standardized = (matrix - self.mean_) / self.scale_

        index = {name: position for position, name in enumerate(self.classes_)}
        n_classes = len(self.classes_)
        targets = np.zeros((matrix.shape[0], n_classes), dtype=DTYPE)
        for row, family in enumerate(families):
            targets[row, index[family]] = 1.0

        if self.label_smoothing:
            # Uniform smoothing. Relevant to 2.2 because it is the other standard way
            # to stop a classifier being overconfident, so the robustness probe can be
            # run against a smoothed and an unsmoothed fit and the two compared.
            targets = (
                targets * (1.0 - self.label_smoothing)
                + self.label_smoothing / n_classes
            )

        weights = self._row_weights(families, sample_weight)

        # Bundle features, targets and per-row weights into one matrix: ``train``
        # batches rows of a single array, and any parallel array would be
        # desynchronised by shuffling. A desynchronised label is a bug that shows up
        # only as mediocre accuracy.
        bundled = np.concatenate(
            [standardized, targets, weights.reshape(-1, 1)], axis=1
        )
        n_features = standardized.shape[1]
        self._net = self.build_network(n_features, n_classes)

        def make_batch(
            batch: np.ndarray, generator: np.random.Generator
        ) -> tuple[np.ndarray, np.ndarray]:
            del generator  # supervised: the batch is the batch
            inputs = batch[:, :n_features]
            batch_targets = batch[:, n_features : n_features + n_classes]
            batch_weights = batch[:, -1:]
            # Weighting is folded into the target rather than into the loss, which is
            # exact for cross entropy: scaling a row's one-hot target scales both its
            # loss contribution and its gradient by the same factor. It also means the
            # shared ``softmax_cross_entropy_loss`` needs no weighted variant.
            return inputs, batch_targets * batch_weights

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
            loss_fn=softmax_cross_entropy_loss,
            validation_data=validation,
            patience=self.patience,
        )
        return self

    def _row_weights(
        self,
        families: Sequence[str],
        sample_weight: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=DTYPE).ravel()
            if weights.shape[0] != len(families):
                raise ClassifierError("sample_weight length does not match rows")
            if np.any(weights < 0.0):
                raise ClassifierError("sample_weight must be non-negative")
            return weights
        if self.class_weight != "balanced":
            return np.ones(len(families), dtype=DTYPE)
        counts: dict[str, int] = {}
        for family in families:
            counts[family] = counts.get(family, 0) + 1
        total = len(families)
        n_classes = len(counts)
        return np.asarray(
            [total / (n_classes * counts[family]) for family in families], dtype=DTYPE
        )

    # --- inference ------------------------------------------------------------ #

    def _standardize(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self.mean_ is not None and self.scale_ is not None
        matrix = np.asarray(x, dtype=DTYPE)
        if matrix.ndim != 2:
            raise ClassifierError(f"expected a 2-D matrix, got {matrix.shape}")
        if matrix.shape[1] != self.mean_.shape[0]:
            raise ClassifierError(
                f"expected {self.mean_.shape[0]} features, got {matrix.shape[1]}"
            )
        return (matrix - self.mean_) / self.scale_

    def logits(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        net = self._require_fitted()
        return net.forward(self._standardize(x))

    def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Class probabilities, rows summing to one."""
        return softmax(self.logits(x), axis=1)

    def predict(self, x: npt.NDArray[np.float64]) -> tuple[str, ...]:
        return tuple(
            self.classes_[position] for position in np.argmax(self.logits(x), axis=1)
        )

    def confidence(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Probability assigned to the predicted class. The calibration probe's input."""
        return np.max(self.predict_proba(x), axis=1)

    def recall_by_family(
        self, x: npt.NDArray[np.float64], families: Sequence[str]
    ) -> dict[str, float]:
        """Per-family recall. The aggregate hides exactly the classes 2.2 is about."""
        predicted = self.predict(x)
        totals: dict[str, int] = {}
        hits: dict[str, int] = {}
        for truth, guess in zip(families, predicted, strict=True):
            totals[truth] = totals.get(truth, 0) + 1
            hits[truth] = hits.get(truth, 0) + int(truth == guess)
        return {family: hits[family] / totals[family] for family in sorted(totals)}

    def accuracy(self, x: npt.NDArray[np.float64], families: Sequence[str]) -> float:
        predicted = self.predict(x)
        return float(
            np.mean([truth == guess for truth, guess in zip(families, predicted, strict=True)])
        )

    def macro_recall(
        self, x: npt.NDArray[np.float64], families: Sequence[str]
    ) -> float:
        """Unweighted mean of per-family recall.

        The headline number for this part. Plain accuracy is dominated by the benign
        class, which is 80% of the data, so a model that predicts "benign" for
        everything scores 0.80 and has learned nothing about any attack.
        """
        per_family = self.recall_by_family(x, families)
        return float(np.mean(list(per_family.values())))

    def training_report(self) -> dict[str, Any]:
        if self.history_ is None:
            raise ClassifierError("not fitted")
        net = self._require_fitted()
        return {
            "classes": list(self.classes_),
            "parameters": net.n_parameters(),
            "epochs_run": self.history_.epochs_run,
            "first_loss": self.history_.train_loss[0],
            "final_loss": self.history_.train_loss[-1],
            "converged": self.history_.converged,
            "class_weight": self.class_weight,
            "label_smoothing": self.label_smoothing,
        }
