"""Layer 3 — the anomaly-detection ensemble (PRD F-03, Section 5.5.2).

The PRD specifies *"a denoising autoencoder trained on benign-only flow features
... an Isolation Forest trained on the same feature set ... combined with a simple
weighted-average ensemble (weights tuned on a held-out validation split) so a
single model's blind spot does not silently define the system's sensitivity."*
Acceptance criterion: **ROC-AUC >= 0.90 on a held-out split.**

What is built here, and what is honestly deferred
-------------------------------------------------
Two structurally different detectors, both trained **benign-only** (semi-supervised
novelty detection, not supervised classification — the whole point is catching
attacks with no signature and no label):

:class:`IsolationForestDetector`
    Axis-aligned random partitioning. Strong on marginal outliers, blind to
    correlation structure: a flow whose every individual feature is unremarkable
    but whose *combination* is impossible scores low.

:class:`PCAReconstructionDetector`
    Reconstruction error from a rank-``k`` linear subspace fitted on benign
    traffic. This is precisely the closed-form optimum of a linear autoencoder
    with squared loss — the same object the PRD's denoising autoencoder
    approximates, minus the nonlinearity. It is the complement of the forest: it
    sees correlation violations and is weak on pure marginal outliers.

    Calling it what it is matters. The nonlinear denoising autoencoder from
    Section 5.5.2 requires PyTorch and lands in Part 2 behind this same
    :class:`AnomalyDetector` protocol, with the swap costing one line in
    :func:`build_default_ensemble`. Shipping PCA and *labelling* it a deep
    autoencoder would be the kind of resume-driven dishonesty that gets caught in
    a technical interview.

:class:`WeightedEnsemble`
    Combines calibrated scores, with weights either fixed or tuned on a labelled
    validation split by :meth:`WeightedEnsemble.tune_weights`.

Calibration is not optional
---------------------------
Isolation Forest returns a negated path-length score around ``[-0.5, 0.5]``;
PCA returns a squared error in ``[0, inf)``. Averaging them raw would let whichever
happens to have larger numeric spread dominate, and the "tuned weights" would be
silently absorbing a unit conversion. Each detector is therefore wrapped in an
empirical-CDF calibrator fitted on **training scores only**, mapping every raw
score to its quantile within benign traffic. After calibration a score of 0.99
means "more anomalous than 99% of benign traffic" for both models, the weights
mean what they claim, and the ensemble threshold is interpretable as a
false-positive-rate budget — which is exactly how a SOC needs to reason about it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from sentinel.core.errors import ModelNotFittedError

__all__ = [
    "AnomalyDetector",
    "DetectorScores",
    "EmpiricalCdfCalibrator",
    "IsolationForestDetector",
    "PCAReconstructionDetector",
    "WeightedEnsemble",
    "build_default_ensemble",
]

DTYPE: Final = np.float64


@runtime_checkable
class AnomalyDetector(Protocol):
    """Every detector: fit on benign, score anything, higher means more anomalous."""

    name: str

    def fit(self, x: np.ndarray) -> AnomalyDetector: ...

    def score(self, x: np.ndarray) -> np.ndarray:
        """Calibrated anomaly scores in ``[0, 1]``, shape ``(n,)``."""
        ...


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


class EmpiricalCdfCalibrator:
    """Maps raw scores to their quantile within the fitted (benign) distribution.

    Implemented by storing a fixed-size sorted quantile grid rather than the whole
    training score vector: memory is bounded regardless of training-set size, and
    interpolation over the grid is O(log n) per query. With
    ``n_quantiles=1024`` the calibration error is below 0.1 percentile, far finer
    than any threshold a SOC actually sets.

    Scores beyond the training range clamp to 0.0 or 1.0. That is the correct
    behaviour — "more extreme than anything benign I have ever seen" is 1.0, and it
    keeps the output bounded so downstream code (a severity mapping, a bandit
    context) never receives a surprise.
    """

    def __init__(self, n_quantiles: int = 1024) -> None:
        if n_quantiles < 16:
            raise ValueError("n_quantiles must be >= 16 to calibrate usefully")
        self.n_quantiles = n_quantiles
        self._grid: np.ndarray | None = None
        self._levels: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._grid is not None

    def fit(self, raw_scores: np.ndarray) -> EmpiricalCdfCalibrator:
        scores = np.asarray(raw_scores, dtype=DTYPE).ravel()
        if scores.size == 0:
            raise ValueError("cannot calibrate on an empty score vector")
        if not np.all(np.isfinite(scores)):
            raise ValueError("raw scores contain non-finite values; fix the detector first")
        levels = np.linspace(0.0, 1.0, self.n_quantiles)
        grid = np.quantile(scores, levels, method="linear")
        # np.interp requires a non-decreasing x; ties are fine, inversions are not.
        self._grid = np.maximum.accumulate(grid)
        self._levels = levels
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._grid is None or self._levels is None:
            raise ModelNotFittedError("calibrator is not fitted")
        scores = np.asarray(raw_scores, dtype=DTYPE).ravel()
        return np.interp(scores, self._grid, self._levels, left=0.0, right=1.0)

    def state(self) -> dict[str, Any]:
        if self._grid is None or self._levels is None:
            raise ModelNotFittedError("calibrator is not fitted")
        return {"n_quantiles": self.n_quantiles, "grid": self._grid.tolist()}

    def load_state(self, state: dict[str, Any]) -> EmpiricalCdfCalibrator:
        self.n_quantiles = int(state["n_quantiles"])
        self._grid = np.asarray(state["grid"], dtype=DTYPE)
        self._levels = np.linspace(0.0, 1.0, self._grid.size)
        return self


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #


class _CalibratedDetector(ABC):
    """Shared fit/score plumbing: fit the model, then calibrate on its own output."""

    name: str = "detector"

    def __init__(self, *, n_quantiles: int = 1024, name: str | None = None) -> None:
        # An instance-level name override exists so an ensemble can hold two members
        # of the same family at different settings (e.g. a tight and a loose PCA
        # subspace), which the uniqueness check in WeightedEnsemble otherwise
        # forbids. Renaming after construction would be too late: the ensemble
        # validates names in its constructor.
        if name is not None:
            if not name or not name.strip():
                raise ValueError("detector name override must be a non-empty identifier")
            self.name = name
        self._calibrator = EmpiricalCdfCalibrator(n_quantiles=n_quantiles)
        self._n_features: int | None = None

    @abstractmethod
    def _fit_model(self, x: np.ndarray) -> None: ...

    @abstractmethod
    def _raw_score(self, x: np.ndarray) -> np.ndarray:
        """Uncalibrated score; higher must mean more anomalous."""

    @property
    def is_fitted(self) -> bool:
        return self._n_features is not None and self._calibrator.is_fitted

    def fit(self, x: np.ndarray) -> _CalibratedDetector:
        matrix = _as_matrix(x)
        if matrix.shape[0] < 2:
            raise ValueError(f"{self.name}: need at least 2 samples to fit, got {matrix.shape[0]}")
        self._n_features = int(matrix.shape[1])
        self._fit_model(matrix)
        self._calibrator.fit(self._raw_score(matrix))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} must be fitted before scoring")
        matrix = _as_matrix(x)
        if matrix.shape[1] != self._n_features:
            raise ValueError(
                f"{self.name} was fitted on {self._n_features} features but received "
                f"{matrix.shape[1]}; the feature spec changed under the model"
            )
        if matrix.shape[0] == 0:
            # A quiet feed produces an empty batch. scikit-learn refuses zero rows
            # with its own error, which surfaced as a library ValueError mid-pipeline
            # (Part 5 edge-case probe); an empty batch has an empty answer.
            return np.empty(0, dtype=DTYPE)
        return self._calibrator.transform(self._raw_score(matrix))

    def raw_score(self, x: np.ndarray) -> np.ndarray:
        """Uncalibrated score, for diagnostics and for explaining a decision."""
        if self._n_features is None:
            raise ModelNotFittedError(f"{self.name} must be fitted before scoring")
        return self._raw_score(_as_matrix(x))


class IsolationForestDetector(_CalibratedDetector):
    """Isolation Forest. Strong on marginal outliers, blind to correlation structure."""

    name = "isolation_forest"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_samples: int | float = 256,
        random_state: int = 20260928,
        n_quantiles: int = 1024,
        name: str | None = None,
    ) -> None:
        super().__init__(n_quantiles=n_quantiles, name=name)
        self.n_estimators = n_estimators
        # 256 is the subsample size from the original Liu et al. paper: the tree
        # depth needed to isolate a point stops improving beyond it, and larger
        # subsamples mainly cost time and blur the anomaly signal through swamping.
        self.max_samples = max_samples
        self.random_state = random_state
        self._model: IsolationForest | None = None

    def _fit_model(self, x: np.ndarray) -> None:
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, x.shape[0])
            if isinstance(self.max_samples, int)
            else self.max_samples,
            # Trained benign-only, so there is no contamination to estimate; fixing
            # it keeps the decision function stable across training-set sizes.
            contamination="auto",
            random_state=self.random_state,
            n_jobs=1,  # determinism over speed: a reproducible demo is the requirement
            bootstrap=False,
        )
        self._model.fit(x)

    def _raw_score(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelNotFittedError("isolation forest is not fitted")
        # score_samples is higher for *more normal* points, so negate it.
        return -np.asarray(self._model.score_samples(x), dtype=DTYPE)


class PCAReconstructionDetector(_CalibratedDetector):
    """Reconstruction error from a benign-fitted linear subspace.

    Equivalent to a linear autoencoder with squared loss at its global optimum. See
    the module docstring on why it is named for what it is; the nonlinear denoising
    autoencoder replaces it in Part 2 behind the same protocol.

    ``variance_target`` selects the subspace rank as the smallest ``k`` explaining
    that share of benign variance, which adapts to the feature spec instead of
    hard-coding a rank that silently becomes wrong when a column is added.
    """

    name = "pca_reconstruction"

    def __init__(
        self,
        *,
        variance_target: float = 0.95,
        max_components: int | None = None,
        random_state: int = 20260928,
        n_quantiles: int = 1024,
        name: str | None = None,
    ) -> None:
        super().__init__(n_quantiles=n_quantiles, name=name)
        if not 0.0 < variance_target < 1.0:
            raise ValueError("variance_target must be in (0, 1)")
        self.variance_target = variance_target
        self.max_components = max_components
        self.random_state = random_state
        self._model: PCA | None = None
        self.n_components_: int | None = None

    def _fit_model(self, x: np.ndarray) -> None:
        # Rank cannot exceed either dimension, and we need at least one component
        # held *out* of the subspace or reconstruction error is identically zero.
        max_rank = max(1, min(x.shape[0], x.shape[1]) - 1)
        if self.max_components is not None:
            max_rank = min(max_rank, self.max_components)

        probe = PCA(n_components=max_rank, random_state=self.random_state).fit(x)
        cumulative = np.cumsum(probe.explained_variance_ratio_)
        rank = int(np.searchsorted(cumulative, self.variance_target) + 1)
        rank = max(1, min(rank, max_rank))

        self.n_components_ = rank
        self._model = PCA(n_components=rank, random_state=self.random_state).fit(x)

    def _raw_score(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelNotFittedError("PCA detector is not fitted")
        reconstructed = self._model.inverse_transform(self._model.transform(x))
        # Mean squared error per sample: comparable across feature-space widths.
        return np.mean(np.square(x - reconstructed), axis=1, dtype=DTYPE)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DetectorScores:
    """Per-detector calibrated scores plus the combined score, for explainability.

    PRD Section 5.1 requires every agent decision to be inspectable. An analyst
    looking at an anomaly flag needs to know *which* detector fired: "reconstruction
    error high, forest normal" means the flow's feature combination is impossible
    even though each value is ordinary, and that reads very differently in an
    investigation than the reverse.
    """

    combined: np.ndarray
    per_detector: dict[str, np.ndarray]

    def explain(self, index: int) -> dict[str, float]:
        return {
            "combined": float(self.combined[index]),
            **{name: float(scores[index]) for name, scores in self.per_detector.items()},
        }

    def disagreement(self) -> np.ndarray:
        """Max pairwise gap between detectors, per sample.

        High disagreement is a useful triage signal in its own right: it means the
        two models see different worlds, so the alert deserves a human rather than
        an averaged-away middling score.
        """
        stacked = np.vstack(list(self.per_detector.values()))
        return stacked.max(axis=0) - stacked.min(axis=0)


class WeightedEnsemble:
    """Weighted average of calibrated detector scores (PRD Section 5.5.2)."""

    def __init__(
        self,
        detectors: Sequence[AnomalyDetector],
        weights: Sequence[float] | None = None,
    ) -> None:
        if not detectors:
            raise ValueError("an ensemble needs at least one detector")
        names = [d.name for d in detectors]
        if len(set(names)) != len(names):
            raise ValueError(f"detector names must be unique, got {names}")
        self.detectors = list(detectors)
        self.weights = self._normalize_weights(
            weights if weights is not None else [1.0] * len(detectors)
        )
        self._tuning_auc: float | None = None

    def _normalize_weights(self, weights: Sequence[float]) -> np.ndarray:
        array = np.asarray(weights, dtype=DTYPE)
        if array.size != len(self.detectors):
            raise ValueError(
                f"expected {len(self.detectors)} weights, got {array.size}"
            )
        if np.any(array < 0):
            raise ValueError("weights must be non-negative")
        total = float(array.sum())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        return array / total

    @property
    def is_fitted(self) -> bool:
        return all(getattr(d, "is_fitted", False) for d in self.detectors)

    def fit(self, x_benign: np.ndarray) -> WeightedEnsemble:
        """Fit every detector on benign-only traffic."""
        matrix = _as_matrix(x_benign)
        for detector in self.detectors:
            detector.fit(matrix)
        return self

    def score_detail(self, x: np.ndarray) -> DetectorScores:
        matrix = _as_matrix(x)
        per_detector = {d.name: d.score(matrix) for d in self.detectors}
        stacked = np.vstack([per_detector[d.name] for d in self.detectors])
        combined = np.tensordot(self.weights, stacked, axes=(0, 0))
        return DetectorScores(combined=combined, per_detector=per_detector)

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.score_detail(x).combined

    # --- weight tuning ------------------------------------------------------

    def tune_weights(
        self,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        grid_steps: int = 21,
        min_weight: float = 0.0,
    ) -> dict[str, float]:
        """Tune weights on a **labelled validation split** to maximise ROC-AUC.

        For the two-detector case this is an exact grid search over the simplex,
        which is the honest thing to do: with one free parameter, a 21-point grid
        resolves the weight to 0.05 and there is nothing a fancier optimiser could
        find that this misses. Three or more detectors fall back to a Dirichlet
        random search over the simplex, which is still unbiased and still cheap.

        The validation split must be **disjoint from both training and test**.
        Tuning on test data is the single most common way a security-ML benchmark
        becomes a work of fiction; :func:`~sentinel.ml.metrics.three_way_split`
        exists so callers do not have to get this right by hand.

        ``min_weight`` — why an AUC-optimal tuner needs a floor
        ------------------------------------------------------
        Unconstrained tuning is free to assign a detector **weight zero**, and
        against the Part 2 denoising autoencoder it does exactly that: measured on
        20,000 synthetic CIC-shaped flows, the tuner put ``isolation_forest`` at
        0.0 and the autoencoder at 1.0, because the autoencoder's validation AUC
        (0.9988) dominates across the whole ROC curve.

        That is the AUC-optimal answer and it quietly defeats the reason the PRD
        specifies an ensemble at all — Section 5.5.2 asks for two structurally
        different detectors *"so a single model's blind spot does not silently
        define the system's sensitivity."* A tuner that discards the second model
        has re-created the single point of failure the ensemble exists to remove,
        and it did so by optimising the metric it was told to optimise.

        A floor keeps the second signal alive. The cost is measured, not assumed:

        =============  ========  ========  =============
        floor          ROC-AUC   PR-AUC    botnet recall
        =============  ========  ========  =============
        0.00 (free)    0.9988    0.9988    1.000
        0.10           0.9971    0.9950    1.000
        0.20           0.9949    0.9901    0.987
        0.50 (equal)   0.9880    0.9759    0.987
        =============  ========  ========  =============

        0.10 costs 0.0017 AUC and keeps every family's recall intact; 0.20 starts
        eating the botnet recall this part exists to fix. Hence 0.10 in
        :mod:`scripts.evaluate`, and a default of 0.0 here so the parameter is an
        explicit decision at the call site rather than a hidden one.
        """
        from sentinel.ml.metrics import roc_auc

        if not self.is_fitted:
            raise ModelNotFittedError("fit the ensemble before tuning weights")
        n_detectors = len(self.detectors)
        if not 0.0 <= min_weight < 1.0 / n_detectors:
            raise ValueError(
                f"min_weight must be in [0, 1/{n_detectors}) = [0, "
                f"{1.0 / n_detectors:.4f}); got {min_weight}. At or above that "
                "every weight is pinned to the floor and there is nothing left to tune."
            )
        matrix = _as_matrix(x_validation)
        labels = np.asarray(y_validation).ravel().astype(int)
        if labels.size != matrix.shape[0]:
            raise ValueError(
                f"validation labels ({labels.size}) do not match rows ({matrix.shape[0]})"
            )
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "validation split contains only one class; AUC is undefined and any "
                "weights would appear equally good"
            )

        per_detector = {d.name: d.score(matrix) for d in self.detectors}
        stacked = np.vstack([per_detector[d.name] for d in self.detectors])

        # Every candidate is drawn from the *floored* simplex
        # ``{w : w_i >= min_weight, sum w = 1}`` by the affine map
        # ``w = floor + (1 - k*floor) * u`` for ``u`` on the unit simplex. Clipping
        # to the floor and renormalising afterwards would be the obvious approach
        # and is wrong: renormalising can push a clipped weight back below the
        # floor, so the constraint would hold only most of the time.
        headroom = 1.0 - n_detectors * min_weight
        candidates: list[np.ndarray] = []
        if n_detectors == 2:
            for step in range(grid_steps):
                u = step / (grid_steps - 1)
                candidates.append(
                    np.array(
                        [min_weight + headroom * u, min_weight + headroom * (1.0 - u)],
                        dtype=DTYPE,
                    )
                )
        else:
            rng = np.random.default_rng(20260928)
            simplex = [np.eye(n_detectors)[i] for i in range(n_detectors)]
            simplex.append(np.full(n_detectors, 1.0 / n_detectors))
            simplex += [rng.dirichlet(np.ones(n_detectors)) for _ in range(grid_steps * 10)]
            candidates = [min_weight + headroom * u for u in simplex]

        best_weights = self.weights
        best_auc = -1.0
        for candidate in candidates:
            total = float(candidate.sum())
            if total <= 0:
                continue
            normalized = candidate / total
            combined = np.tensordot(normalized, stacked, axes=(0, 0))
            auc = roc_auc(labels, combined)
            # Strict improvement only, so ties keep the earlier (more balanced or
            # lower-index) candidate instead of drifting with floating-point noise.
            if auc > best_auc:
                best_auc = auc
                best_weights = normalized

        self.weights = best_weights
        self._tuning_auc = best_auc
        return {
            **{d.name: float(w) for d, w in zip(self.detectors, self.weights, strict=True)},
            "validation_auc": float(best_auc),
        }

    @property
    def tuning_auc(self) -> float | None:
        """Validation AUC achieved at tuning time. ``None`` if never tuned."""
        return self._tuning_auc

    def weight_report(self) -> dict[str, float]:
        return {d.name: float(w) for d, w in zip(self.detectors, self.weights, strict=True)}

    # --- thresholds ---------------------------------------------------------

    def threshold_for_fpr(self, x_benign: np.ndarray, target_fpr: float) -> float:
        """Score threshold giving at most ``target_fpr`` on known-benign traffic.

        This is how a SOC actually configures a detector: not "what score is
        suspicious" but "how many false positives per day can my two analysts
        absorb". F-03 reports AUC, but PRD Section 9.1's 60% false-positive
        reduction target is a threshold decision, and this is where it is made
        explicit rather than buried in a magic constant.
        """
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        scores = self.score(x_benign)
        return float(np.quantile(scores, 1.0 - target_fpr))

    # --- persistence --------------------------------------------------------

    def save(self, path: str | Path, *, spec_fingerprint: str) -> Path:
        """Persist ensemble weights and the feature-spec fingerprint.

        Detector internals are not serialized here: pickling scikit-learn estimators
        across versions is a documented footgun, and the sprint's training run takes
        seconds. What *must* survive is the spec fingerprint, so a stale artifact
        cannot be paired with a changed feature spec (see
        :meth:`~sentinel.ml.featurestore.AlertVectorizer.assert_compatible`).
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "detectors": [d.name for d in self.detectors],
            "weights": self.weights.tolist(),
            "validation_auc": self._tuning_auc,
            "spec_fingerprint": spec_fingerprint,
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


def _as_matrix(x: np.ndarray) -> np.ndarray:
    matrix = np.asarray(x, dtype=DTYPE)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {matrix.shape}")
    if matrix.size and not np.all(np.isfinite(matrix)):
        bad = int(np.count_nonzero(~np.isfinite(matrix)))
        raise ValueError(
            f"feature matrix contains {bad} non-finite value(s); the feature store must "
            "not emit NaN or Inf (check the normalizer's rate columns)"
        )
    return matrix


def build_default_ensemble(*, random_state: int = 20260928) -> WeightedEnsemble:
    """The **shallow baseline** ensemble: Isolation Forest + PCA reconstruction.

    Part 2.1 added the nonlinear denoising autoencoder PRD Section 5.5.2 specifies,
    and it is strictly better on this data (ROC-AUC 0.9988 vs 0.9894, botnet recall
    1.000 vs 0.608). It lives in
    :func:`~sentinel.ml.deep.build_deep_ensemble` and is what
    ``scripts/evaluate.py`` reports.

    This factory was *not* changed to return it, which is a deliberate choice
    rather than an oversight:

    *   **It is the fast path.** This ensemble fits in milliseconds; the deep one
        takes 0.7-4s depending on training-set size. Roughly fifteen unit and
        integration tests fit an ensemble, several of them repeatedly, so
        substituting here would add minutes to a suite whose speed is why it gets
        run.
    *   **Composition is part of its contract.** Tests assert this ensemble
        explains itself as exactly ``{isolation_forest, pca_reconstruction}``.
        Silently changing what a factory named "default" returns would invalidate
        those assertions for a reason unrelated to what they test.
    *   **A linear baseline that stays measurable is worth keeping.** The
        0.9894-vs-0.9988 comparison is only reproducible while both ensembles
        still exist and are both reachable from one command
        (``python scripts/evaluate.py --shallow``).
    """
    return WeightedEnsemble(
        detectors=[
            IsolationForestDetector(random_state=random_state),
            PCAReconstructionDetector(random_state=random_state),
        ],
        weights=[0.5, 0.5],
    )
