"""Augmentation harness and the adversarial calibration probe (PRD Section 5.5.5).

Two things live here because they share a generator and nothing else does.

Augmentation, with the invariants enforced
------------------------------------------
:func:`augment_training_set` adds synthetic minority rows to a training split. Three
properties are enforced rather than trusted, because each is an easy mistake that
inflates the result it is supposed to measure:

1.  **Train only.** The function takes the training split and returns a training
    split. Nothing here can see the held-out set, so synthetic rows cannot reach it.
    ``test_robustness.py`` additionally asserts no synthetic row appears in the
    evaluation split.
2.  **The majority class is untouched.** PRD Section 5.5.5 says *"without touching the
    majority class distribution"*. Benign rows are passed through bit-identically and
    :func:`augment_training_set` refuses to generate them at all. Quietly
    down-sampling benign rows would produce a better rare-class recall number and
    would not be augmentation.
3.  **Synthetic rows are labelled as synthetic.** The returned mask is the only way a
    caller can honour (1) and (2) downstream, so it is returned, not optional.

The calibration probe
---------------------
Section 5.5.5's second purpose: *"perturbed synthetic samples near the decision
boundary are used to sanity-check that the Triage Agent's confidence calibration does
not collapse under slightly out-of-distribution input."*

The failure being probed is specific, and it is the one that matters operationally. A
model that is wrong is a nuisance. A model that is wrong *and certain* is a hazard,
because the whole autonomy ladder in PRD Section 5.7 is driven by confidence: a
confident dismissal closes an alert, and Part 1 made the confidence floor a schema
invariant precisely because confidence is load-bearing. So the question is not "is
accuracy preserved under perturbation" — it will not be — but "does confidence fall
when accuracy does".

:func:`robustness_curve` answers it by measuring accuracy and mean confidence together
at increasing perturbation, and :func:`assert_calibration_holds` gates on the
*relationship* between them rather than on either alone.

Why interpolation finds the boundary
------------------------------------
:func:`boundary_adjacent_samples` locates near-boundary points by bisecting along the
straight line between two samples the classifier assigns to different classes. No
gradients are needed, which matters because the engine here is a hand-written forward
pass and an input-gradient path would be a second backward implementation to get
wrong. The points found are genuinely ambiguous — the classifier's own argmax flips
across them — and they are on the segment between two real data points, so they are
on the data manifold rather than in an adversarial direction pointing off it. That is
the distinction between this and an attack: Section 5.5.5 asks for a *sanity check*
on calibration, not a worst-case bound.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Final

import numpy as np
import numpy.typing as npt

from sentinel.core.errors import SentinelError
from sentinel.ml.classify import FamilyClassifier
from sentinel.ml.diffusion import TabularDiffusion

__all__ = [
    "BENIGN_FAMILY",
    "DEFAULT_SCALES",
    "GATE_CLEAN_ECE",
    "GATE_CONFIDENCE_DROP",
    "GATE_MAX_OVERCONFIDENCE",
    "AugmentationResult",
    "CalibrationReport",
    "NoveltyGate",
    "RobustnessError",
    "assert_calibration_holds",
    "augment_training_set",
    "boundary_adjacent_samples",
    "expected_calibration_error",
    "gated_calibration_report",
    "gated_robustness_curve",
    "perturb",
    "robustness_curve",
]

DTYPE: Final = np.float64

#: The majority label, which augmentation must never generate or alter.
BENIGN_FAMILY: Final[str] = "benign"

#: Perturbation scales for the sweep, in units of each column's standard deviation.
#: The range extends to 4 sigma deliberately: the interesting failure only appears
#: beyond 1 sigma, and a sweep that stops at 1 reports a model that looks fine.
DEFAULT_SCALES: Final[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

#: Expected calibration error gate on clean held-out data. Measured 0.005 for the
#: novelty-gated confidence. Without this the rest is uninterpretable: a model already
#: miscalibrated in distribution says nothing about out-of-distribution behaviour.
GATE_CLEAN_ECE: Final[float] = 0.05

#: Minimum fall in mean confidence between clean data and the most perturbed setting.
#: This is the gate that encodes "calibration does not collapse": the model may lose
#: accuracy out of distribution and may not stay equally sure of itself while doing so.
#: Measured 0.900 gated, against 0.029 for the raw softmax -- so the gate separates the
#: two by a wide margin rather than sitting in the noise between them.
GATE_CONFIDENCE_DROP: Final[float] = 0.30

#: Ceiling on overconfidence (``mean_confidence - accuracy``) at *any* point on the
#: perturbation sweep, not just at the ends.
#:
#: The sweep matters because the raw-softmax failure is U-shaped: confidence dips
#: around 1 sigma and climbs back to 0.970 at 4 sigma while accuracy sits at 0.650.
#: A gate comparing only the clean and worst points scores that curve as a confidence
#: *fall* and passes it. Measured peak overconfidence is +0.320 raw and -0.004 gated.
GATE_MAX_OVERCONFIDENCE: Final[float] = 0.05


class RobustnessError(SentinelError):
    """Augmentation or the calibration probe was used incorrectly."""


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    """An augmented training split, with the provenance needed to audit it."""

    x: npt.NDArray[np.float64]
    families: tuple[str, ...]
    #: True for rows produced by the generator. Returned rather than optional: without
    #: it a caller cannot tell real rows from synthetic ones, and every invariant this
    #: module enforces depends on that distinction.
    synthetic: npt.NDArray[np.bool_]
    counts_added: dict[str, int] = field(default_factory=dict)
    skipped_families: tuple[str, ...] = ()

    @property
    def n_real(self) -> int:
        return int((~self.synthetic).sum())

    @property
    def n_synthetic(self) -> int:
        return int(self.synthetic.sum())

    def real_rows(self) -> npt.NDArray[np.float64]:
        return self.x[~self.synthetic]

    def summary(self) -> str:
        added = ", ".join(f"{k}+{v}" for k, v in sorted(self.counts_added.items()))
        return (
            f"{self.n_real} real + {self.n_synthetic} synthetic "
            f"({added or 'nothing added'})"
        )


def augment_training_set(
    x: npt.NDArray[np.float64],
    families: Sequence[str],
    *,
    generator: TabularDiffusion,
    target_per_family: int | None = None,
    multiplier: float = 3.0,
    rng: np.random.Generator | None = None,
    benign_family: str = BENIGN_FAMILY,
) -> AugmentationResult:
    """Add synthetic rows for the generator's fitted families. Train split only.

    ``target_per_family`` sets an absolute target count per modelled family;
    ``multiplier`` is used when it is ``None`` and scales each family's existing count.
    A multiplier is usually the better choice: an absolute target hands the rarest
    family the largest synthetic-to-real ratio, so the family the generator knows
    least about ends up dominating its own training rows.
    """
    matrix = np.asarray(x, dtype=DTYPE)
    if matrix.ndim != 2:
        raise RobustnessError(f"expected a 2-D matrix, got {matrix.shape}")
    if matrix.shape[0] != len(families):
        raise RobustnessError(f"{matrix.shape[0]} rows but {len(families)} labels")
    if not generator.is_fitted:
        raise RobustnessError("generator must be fitted before augmenting")
    if benign_family in generator.families_:
        raise RobustnessError(
            f"generator was fitted on {benign_family!r}; PRD Section 5.5.5 requires the "
            "majority class distribution to be left untouched, so the majority class "
            "must not be modelled"
        )
    if multiplier < 1.0:
        raise RobustnessError(
            f"multiplier must be at least 1.0, got {multiplier}; values below 1 would "
            "remove rows rather than add them"
        )
    if target_per_family is not None and target_per_family < 0:
        raise RobustnessError("target_per_family must be non-negative")

    labels = list(families)
    counts: dict[str, int] = {}
    for family in labels:
        counts[family] = counts.get(family, 0) + 1

    wanted: dict[str, int] = {}
    for family in generator.families_:
        present = counts.get(family, 0)
        if present == 0:
            continue
        target = (
            target_per_family
            if target_per_family is not None
            else round(multiplier * present)
        )
        extra = max(0, target - present)
        if extra:
            wanted[family] = extra

    if not wanted:
        return AugmentationResult(
            x=matrix.copy(),
            families=tuple(labels),
            synthetic=np.zeros(matrix.shape[0], dtype=bool),
            counts_added={},
            skipped_families=generator.skipped_families_,
        )

    synthetic_x, synthetic_families = generator.sample(wanted, rng=rng)
    if not np.all(np.isfinite(synthetic_x)):
        raise RobustnessError(
            "generator produced non-finite rows; training on them would poison the "
            "classifier silently, so they are refused rather than filtered"
        )

    combined = np.concatenate([matrix, synthetic_x], axis=0)
    combined_families = (*labels, *synthetic_families)
    mask = np.concatenate(
        [
            np.zeros(matrix.shape[0], dtype=bool),
            np.ones(synthetic_x.shape[0], dtype=bool),
        ]
    )

    # The invariant, checked rather than documented: the majority rows that come out
    # are exactly the majority rows that went in, byte for byte.
    benign_before = matrix[np.asarray(labels) == benign_family]
    benign_after = combined[np.asarray(combined_families) == benign_family]
    if benign_after.shape != benign_before.shape or not np.array_equal(
        benign_after, benign_before
    ):
        raise RobustnessError(
            "the majority class changed during augmentation, which PRD Section 5.5.5 "
            "forbids"
        )

    return AugmentationResult(
        x=combined,
        families=combined_families,
        synthetic=mask,
        counts_added=wanted,
        skipped_families=generator.skipped_families_,
    )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    confidences: npt.NDArray[np.float64],
    correct: npt.NDArray[np.bool_],
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error: mean ``|accuracy - confidence|`` over equal-width bins.

    Weighted by bin population, so empty bins contribute nothing rather than being
    counted as perfectly calibrated — which is the usual way an ECE implementation
    flatters a model whose confidences are all clustered in one bin.
    """
    scores = np.asarray(confidences, dtype=DTYPE).ravel()
    hits = np.asarray(correct, dtype=bool).ravel()
    if scores.shape != hits.shape:
        raise RobustnessError("confidences and correctness have different lengths")
    if scores.size == 0:
        raise RobustnessError("expected_calibration_error on an empty sample")
    if n_bins < 2:
        raise RobustnessError(f"n_bins must be at least 2, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lower, upper in pairwise(edges):
        # Half-open bins, with the last one closed so confidence exactly 1.0 lands
        # somewhere. Confidence of exactly 1.0 is common with a softmax head.
        in_bin = (scores > lower) & (scores <= upper) if lower > 0 else (scores <= upper)
        population = int(in_bin.sum())
        if not population:
            continue
        gap = abs(float(hits[in_bin].mean()) - float(scores[in_bin].mean()))
        total += gap * population / scores.size
    return float(total)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Accuracy and confidence measured together, which is the only useful way."""

    label: str
    n_samples: int
    accuracy: float
    mean_confidence: float
    ece: float
    #: ``mean_confidence - accuracy``. Positive means overconfident, which is the
    #: direction that matters: an under-confident model escalates unnecessarily, an
    #: over-confident one closes alerts it should not.
    overconfidence: float

    def summary(self) -> str:
        return (
            f"{self.label:<22} n={self.n_samples:<6} acc={self.accuracy:.3f} "
            f"conf={self.mean_confidence:.3f} ece={self.ece:.3f} "
            f"over={self.overconfidence:+.3f}"
        )


def calibration_report(
    classifier: FamilyClassifier,
    x: npt.NDArray[np.float64],
    families: Sequence[str],
    *,
    label: str,
    n_bins: int = 10,
) -> CalibrationReport:
    """Measure one population."""
    if len(families) == 0:
        raise RobustnessError("calibration_report on an empty sample")
    probabilities = classifier.predict_proba(x)
    predicted = np.asarray(
        [classifier.classes_[position] for position in np.argmax(probabilities, axis=1)]
    )
    confidences = np.max(probabilities, axis=1)
    correct = predicted == np.asarray(families)
    accuracy = float(correct.mean())
    mean_confidence = float(confidences.mean())
    return CalibrationReport(
        label=label,
        n_samples=len(families),
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        ece=expected_calibration_error(confidences, correct, n_bins=n_bins),
        overconfidence=mean_confidence - accuracy,
    )


def perturb(
    x: npt.NDArray[np.float64],
    *,
    scale: float,
    rng: np.random.Generator,
    column_scale: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Additive Gaussian perturbation, in units of each column's own spread.

    Perturbing in raw units would be meaningless across a feature spec whose columns
    span packet counts and normalised ratios: the same absolute nudge is imperceptible
    on one and catastrophic on another. ``column_scale`` defaults to the per-column
    standard deviation of ``x``, so ``scale=0.1`` means "a tenth of a standard
    deviation" for every column.
    """
    matrix = np.asarray(x, dtype=DTYPE)
    if scale < 0.0:
        raise RobustnessError(f"scale must be non-negative, got {scale}")
    if scale == 0.0:
        return matrix.copy()
    if column_scale is None:
        spread = matrix.std(axis=0)
        column_scale = np.where(spread > 1e-12, spread, 1.0)
    if column_scale.shape != (matrix.shape[1],):
        raise RobustnessError(
            f"column_scale must have {matrix.shape[1]} entries, got {column_scale.shape}"
        )
    return matrix + rng.normal(0.0, scale, size=matrix.shape) * column_scale


def boundary_adjacent_samples(
    classifier: FamilyClassifier,
    x: npt.NDArray[np.float64],
    families: Sequence[str],
    *,
    n_samples: int = 200,
    rng: np.random.Generator,
    bisection_steps: int = 12,
) -> tuple[npt.NDArray[np.float64], tuple[str, ...], tuple[str, ...]]:
    """Points on the classifier's own decision boundary, found by bisection.

    Returns ``(rows, family_a, family_b)``: each row sits between a sample the
    classifier calls ``family_a`` and one it calls ``family_b``, close enough to the
    flip that the prediction is genuinely uncertain. Both endpoint labels are returned
    because a boundary point has no single correct answer — which is the point. The
    right behaviour on these inputs is low confidence, and that is what
    :func:`robustness_curve` measures.
    """
    matrix = np.asarray(x, dtype=DTYPE)
    if matrix.shape[0] != len(families):
        raise RobustnessError("row count does not match label count")
    if n_samples < 1:
        raise RobustnessError("n_samples must be at least 1")
    if bisection_steps < 1:
        raise RobustnessError("bisection_steps must be at least 1")

    predicted = np.asarray(classifier.predict(matrix))
    by_class: dict[str, npt.NDArray[np.int64]] = {
        name: np.flatnonzero(predicted == name) for name in set(predicted.tolist())
    }
    usable = [name for name, rows in by_class.items() if rows.size]
    if len(usable) < 2:
        raise RobustnessError(
            "the classifier predicts a single class on this sample, so it has no "
            "decision boundary to probe"
        )

    rows: list[npt.NDArray[np.float64]] = []
    left_labels: list[str] = []
    right_labels: list[str] = []
    attempts = 0
    while len(rows) < n_samples and attempts < n_samples * 20:
        attempts += 1
        first, second = rng.choice(len(usable), size=2, replace=False)
        name_a, name_b = usable[int(first)], usable[int(second)]
        row_a = matrix[rng.choice(by_class[name_a])]
        row_b = matrix[rng.choice(by_class[name_b])]

        low, high = 0.0, 1.0
        for _ in range(bisection_steps):
            middle = 0.5 * (low + high)
            candidate = (1.0 - middle) * row_a + middle * row_b
            if classifier.predict(candidate.reshape(1, -1))[0] == name_a:
                low = middle
            else:
                high = middle
        crossing = 0.5 * (low + high)
        rows.append((1.0 - crossing) * row_a + crossing * row_b)
        left_labels.append(name_a)
        right_labels.append(name_b)

    if not rows:  # pragma: no cover - requires a degenerate classifier
        raise RobustnessError("failed to locate any boundary point")
    return np.stack(rows), tuple(left_labels), tuple(right_labels)


def robustness_curve(
    classifier: FamilyClassifier,
    x: npt.NDArray[np.float64],
    families: Sequence[str],
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    rng: np.random.Generator,
) -> tuple[CalibrationReport, ...]:
    """Accuracy and confidence at increasing perturbation.

    Both are reported at every scale because the interesting quantity is their
    *relationship*. Accuracy falling is expected and fine. Accuracy falling while
    confidence holds is the failure, and it is invisible if either is measured alone.
    """
    if not scales:
        raise RobustnessError("scales must not be empty")
    if any(scale < 0.0 for scale in scales):
        raise RobustnessError("scales must be non-negative")
    matrix = np.asarray(x, dtype=DTYPE)
    spread = matrix.std(axis=0)
    column_scale = np.where(spread > 1e-12, spread, 1.0)
    return tuple(
        calibration_report(
            classifier,
            perturb(matrix, scale=scale, rng=rng, column_scale=column_scale),
            families,
            label=f"perturb={scale:g}",
        )
        for scale in scales
    )


def assert_calibration_holds(
    curve: Sequence[CalibrationReport],
    *,
    boundary: CalibrationReport | None = None,
    clean_ece_gate: float = GATE_CLEAN_ECE,
    confidence_drop_gate: float = GATE_CONFIDENCE_DROP,
    max_overconfidence_gate: float = GATE_MAX_OVERCONFIDENCE,
) -> dict[str, float]:
    """Raise unless confidence degrades with accuracy across the whole sweep.

    Four checks, and the third is the one that does the work:

    1.  The classifier is calibrated on clean data at all. Without this the rest is
        uninterpretable.
    2.  Mean confidence *falls* measurably from clean to most-perturbed. An
        out-of-distribution input must make the model less sure.
    3.  Overconfidence stays under its ceiling at **every** point on the sweep, not
        just at the ends. This is deliberate: the raw-softmax failure is U-shaped, so
        an endpoint-only comparison reads it as a confidence fall and passes a curve
        whose middle reaches +0.18 and whose tail reaches +0.32.
    4.  Boundary-adjacent samples are not predicted with clean-data confidence. The
        classifier's own argmax flips across those points, so certainty about them is
        indefensible whatever the perturbation numbers say.
    """
    if len(curve) < 2:
        raise RobustnessError("need at least a clean and a perturbed report")
    clean, worst = curve[0], curve[-1]
    peak = max(report.overconfidence for report in curve)
    peak_label = max(curve, key=lambda report: report.overconfidence).label

    failures: list[str] = []
    if clean.ece > clean_ece_gate:
        failures.append(
            f"clean ECE {clean.ece:.3f} > gate {clean_ece_gate:.2f}; the classifier is "
            "not calibrated in distribution, so its out-of-distribution behaviour is "
            "not interpretable"
        )
    drop = clean.mean_confidence - worst.mean_confidence
    if drop < confidence_drop_gate:
        failures.append(
            f"mean confidence fell only {drop:.3f} from clean to {worst.label} "
            f"(gate {confidence_drop_gate:.2f}); the model stays about as sure of "
            "itself on out-of-distribution input as on real data"
        )
    if peak > max_overconfidence_gate:
        failures.append(
            f"overconfidence peaks at {peak:+.3f} ({peak_label}) > gate "
            f"{max_overconfidence_gate:.2f}; the model is confidently wrong somewhere "
            "on the sweep"
        )
    if boundary is not None and boundary.mean_confidence >= clean.mean_confidence:
        failures.append(
            f"boundary-adjacent samples are predicted with confidence "
            f"{boundary.mean_confidence:.3f}, at or above the clean-data "
            f"{clean.mean_confidence:.3f}; the classifier is certain about inputs its "
            "own argmax flips on"
        )
    if failures:
        raise RobustnessError("calibration checks failed: " + "; ".join(failures))

    return {
        "clean_accuracy": clean.accuracy,
        "clean_confidence": clean.mean_confidence,
        "clean_ece": clean.ece,
        "worst_accuracy": worst.accuracy,
        "worst_confidence": worst.mean_confidence,
        "worst_ece": worst.ece,
        "confidence_drop": drop,
        "accuracy_drop": clean.accuracy - worst.accuracy,
        "peak_overconfidence": peak,
        "boundary_confidence": (
            boundary.mean_confidence if boundary is not None else float("nan")
        ),
    }


# --------------------------------------------------------------------------- #
# The novelty gate
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class NoveltyGate:
    """Discount a classifier's confidence by how far off-manifold its input is.

    Why this exists, and why it is not a calibration trick
    -----------------------------------------------------
    :func:`robustness_curve` found a real defect in the classifier, and it is not a
    defect peculiar to this classifier. Measured on held-out data, accuracy falls from
    0.998 to 0.650 as perturbation grows, while mean confidence goes 0.999 -> 0.971 and
    then back *up* to 0.970 at the largest perturbation, peaking at +0.32
    overconfidence. The curve is U-shaped: the model is least sure in the middle and
    recovers its certainty as inputs become absurd.

    That is the well-known behaviour of a softmax head, not a bug in the fit. Far from
    the training data one logit dominates and the softmax saturates, so confidence is
    high in exactly the region where the model knows least. Label smoothing, the
    standard cheap remedy, was measured across five settings and made it *worse*: it
    lowers clean confidence without lowering far-out-of-distribution confidence, so
    peak overconfidence rose from +0.32 to +0.46.

    The conclusion is architectural rather than numerical: a classifier's own output
    cannot be its out-of-distribution detector, because the quantity that should fall
    is computed from the same saturating function that rises. Something measuring
    distance from the training manifold is needed, and this system already has one --
    Section 5.5.2's anomaly ensemble, whose whole job is to score novelty. Measured on
    the same perturbation sweep, its score rises monotonically, 0.601 -> 0.999, with no
    U-turn.

    How the reference quantile works
    --------------------------------
    Raw novelty cannot be used as the discount, because in-distribution attacks
    legitimately score high: the detector is fitted on benign rows only, so a real
    attack *should* look novel to it. What matters is novelty beyond what the
    classifier's own training data exhibited. :meth:`fit` therefore records a high
    quantile of novelty over the training rows, and :meth:`excess` reports how far
    above that a sample sits, on a 0-to-1 scale. A sample no stranger than the
    strangest few percent of training data is not discounted at all.
    """

    #: Quantile of training novelty treated as the edge of "normal". 0.99 rather than
    #: 1.0 because the maximum over tens of thousands of rows is an outlier, and
    #: anchoring to an outlier makes the gate fire on almost nothing.
    quantile: float = 0.99
    reference_: float = field(default=0.0, repr=False)
    fitted_: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.5 <= self.quantile < 1.0:
            raise RobustnessError(
                f"quantile must be in [0.5, 1.0), got {self.quantile}"
            )

    def fit(self, training_novelty: npt.NDArray[np.float64]) -> NoveltyGate:
        """Record the novelty quantile of the classifier's own training rows."""
        scores = np.asarray(training_novelty, dtype=DTYPE).ravel()
        if scores.size == 0:
            raise RobustnessError("cannot fit a novelty gate on an empty sample")
        if not np.all(np.isfinite(scores)):
            raise RobustnessError("training novelty contains non-finite values")
        self.reference_ = float(np.quantile(scores, self.quantile))
        self.fitted_ = True
        return self

    def excess(self, novelty: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """How far above the reference each sample sits, in ``[0, 1]``."""
        if not self.fitted_:
            raise RobustnessError("NoveltyGate is not fitted; call fit() first")
        scores = np.asarray(novelty, dtype=DTYPE).ravel()
        headroom = 1.0 - self.reference_
        if headroom <= 1e-12:
            # Every training row already scores at the ceiling, so the detector cannot
            # distinguish anything and the gate must be inert rather than saturated --
            # a gate that discounts everything is a gate that suppresses every alert.
            return np.zeros_like(scores)
        return np.clip((scores - self.reference_) / headroom, 0.0, 1.0)

    def adjust(
        self,
        confidence: npt.NDArray[np.float64],
        novelty: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Gated confidence: ``confidence * (1 - excess)``.

        Multiplicative, so an in-distribution sample keeps its confidence exactly and
        a wholly novel one is reduced to zero. An additive penalty would either leave
        confident-and-novel samples above the dismissal floor or push confident
        in-distribution ones below it, depending on the constant; multiplication needs
        no constant.
        """
        values = np.asarray(confidence, dtype=DTYPE).ravel()
        excess = self.excess(novelty)
        if values.shape != excess.shape:
            raise RobustnessError(
                f"confidence has {values.shape[0]} entries, novelty has {excess.shape[0]}"
            )
        return values * (1.0 - excess)


def gated_calibration_report(
    classifier: FamilyClassifier,
    x: npt.NDArray[np.float64],
    families: Sequence[str],
    *,
    novelty: npt.NDArray[np.float64],
    gate: NoveltyGate,
    label: str,
    n_bins: int = 10,
) -> CalibrationReport:
    """:func:`calibration_report`, but with confidence passed through ``gate``."""
    if len(families) == 0:
        raise RobustnessError("gated_calibration_report on an empty sample")
    probabilities = classifier.predict_proba(x)
    predicted = np.asarray(
        [classifier.classes_[position] for position in np.argmax(probabilities, axis=1)]
    )
    confidences = gate.adjust(np.max(probabilities, axis=1), novelty)
    correct = predicted == np.asarray(families)
    accuracy = float(correct.mean())
    mean_confidence = float(confidences.mean())
    return CalibrationReport(
        label=label,
        n_samples=len(families),
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        ece=expected_calibration_error(confidences, correct, n_bins=n_bins),
        overconfidence=mean_confidence - accuracy,
    )


def gated_robustness_curve(
    classifier: FamilyClassifier,
    x: npt.NDArray[np.float64],
    families: Sequence[str],
    *,
    novelty_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    gate: NoveltyGate,
    scales: Sequence[float] = DEFAULT_SCALES,
    rng: np.random.Generator,
) -> tuple[CalibrationReport, ...]:
    """The perturbation sweep with novelty-gated confidence.

    ``novelty_fn`` is the detector's scoring function, passed as a callable so this
    module does not depend on which detector is in use -- the ensemble, a single
    autoencoder or a future replacement all satisfy the same one-line contract.
    """
    if not scales:
        raise RobustnessError("scales must not be empty")
    if any(scale < 0.0 for scale in scales):
        raise RobustnessError("scales must be non-negative")
    matrix = np.asarray(x, dtype=DTYPE)
    spread = matrix.std(axis=0)
    column_scale = np.where(spread > 1e-12, spread, 1.0)
    reports: list[CalibrationReport] = []
    for scale in scales:
        perturbed = perturb(matrix, scale=scale, rng=rng, column_scale=column_scale)
        reports.append(
            gated_calibration_report(
                classifier,
                perturbed,
                families,
                novelty=novelty_fn(perturbed),
                gate=gate,
                label=f"perturb={scale:g}",
            )
        )
    return tuple(reports)
