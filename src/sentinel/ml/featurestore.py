"""Layer 3 — the feature store (PRD Section 7.3).

*"The same normalized-alert schema feeds both the real-time agent pipeline and the
offline training/evaluation jobs, so there is exactly one definition of what an
alert looks like."*

This module is where that stops being a good intention. Three structural guards:

1.  **One input type.** :class:`AlertVectorizer` accepts only
    :class:`~sentinel.core.schemas.Alert` objects. There is no ``fit(dataframe)``
    overload, so there is no path by which a training job could read a CSV column
    the serving path never sees. The training script must go through the same
    normalizer the live pipeline uses.

2.  **One code path for batch and single.** :meth:`AlertVectorizer.transform`
    (batch) and :meth:`AlertVectorizer.transform_one` (serving) call the same
    per-column extraction. ``test_batch_and_single_paths_are_bit_identical``
    asserts equality at full float64 precision, not ``allclose`` — the classic
    skew bug is a batch path that vectorizes with pandas and a serving path that
    loops, and it shows up in the last few bits long before it shows up in
    accuracy.

3.  **Fingerprinted specs.** A fitted vectorizer carries a
    :attr:`FeatureSpec.fingerprint` over its column names *and* their transform
    identities. A model artifact stores the fingerprint it was trained under and
    :meth:`AlertVectorizer.assert_compatible` refuses inference when it differs.
    Silently scoring with a reordered feature vector is the failure mode that
    produces a confidently wrong SOC.

Transform choices are not arbitrary. Flow byte and packet counts are heavy-tailed
over several orders of magnitude, so they are ``log1p``-compressed before
standardization; a raw-scale standardizer would let one 400 MB transfer dominate
the variance of every other feature. Ports are *not* ordinal — port 443 is not
"more" than port 80 — so they are expanded into semantic indicators rather than
fed in as integers. Protocol is one-hot over the three protocols that carry
essentially all of both datasets, plus an explicit "other" bucket so an unseen
protocol is represented rather than silently zeroed.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from sentinel.core.errors import FeatureError, ModelNotFittedError
from sentinel.core.schemas import Alert

__all__ = [
    "DEFAULT_SPEC",
    "AlertVectorizer",
    "Column",
    "FeatureSpec",
    "Identity",
    "Indicator",
    "Log1p",
    "OneHot",
    "default_spec",
]

DTYPE: Final = np.float64
"""float64 throughout. A security model that disagrees with itself between the
training notebook and the serving process because one used float32 is not a
debugging session anybody enjoys."""


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


class Column(ABC):
    """One scalar output column derived from an :class:`Alert`."""

    def __init__(self, name: str, source: str) -> None:
        if not name:
            raise ValueError("column name must be non-empty")
        self.name = name
        self.source = source

    @abstractmethod
    def extract(self, alert: Alert) -> float:
        """Compute this column's value. Must be total: never raise on a valid Alert."""

    def identity(self) -> str:
        """A stable description used in the spec fingerprint."""
        return f"{type(self).__name__}({self.name}<-{self.source})"

    def __repr__(self) -> str:
        return f"<{self.identity()}>"


def _numeric(alert: Alert, source: str) -> float:
    """Read a numeric feature, treating missing/non-numeric as 0.0.

    Non-finite values cannot reach here: :class:`Alert` rejects them at
    construction, and the normalizer sanitizes them with a counted substitution.
    The guard stays anyway, because a feature pipeline that can emit ``NaN`` will
    eventually emit ``NaN``, and a NaN in a StandardScaler poisons every
    subsequent batch silently.
    """
    raw = alert.features.get(source)
    if raw is None or isinstance(raw, str):
        return 0.0
    value = float(raw)
    return value if math.isfinite(value) else 0.0


class Identity(Column):
    """Pass a numeric feature through unchanged."""

    def extract(self, alert: Alert) -> float:
        return _numeric(alert, self.source)


class Log1p(Column):
    """``log1p`` of a non-negative count or rate.

    Negative inputs are clipped to 0 rather than producing ``NaN``: a negative
    byte count is a source-data defect, and turning it into a NaN would propagate
    the defect into every model that touches the batch.
    """

    def extract(self, alert: Alert) -> float:
        return math.log1p(max(_numeric(alert, self.source), 0.0))


class Indicator(Column):
    """A 0/1 flag computed by a named predicate over a numeric feature."""

    def __init__(self, name: str, source: str, predicate: str, threshold: float = 0.0) -> None:
        super().__init__(name, source)
        if predicate not in _PREDICATES:
            raise ValueError(f"unknown predicate {predicate!r}; known: {sorted(_PREDICATES)}")
        self.predicate = predicate
        self.threshold = threshold

    def extract(self, alert: Alert) -> float:
        holds = _PREDICATES[self.predicate](_numeric(alert, self.source), self.threshold)
        return 1.0 if holds else 0.0

    def identity(self) -> str:
        return f"Indicator({self.name}<-{self.source},{self.predicate},{self.threshold!r})"


_PREDICATES: Final[dict[str, Any]] = {
    "gt": lambda value, threshold: value > threshold,
    "gte": lambda value, threshold: value >= threshold,
    "lt": lambda value, threshold: value < threshold,
    "lte": lambda value, threshold: value <= threshold,
    "eq": lambda value, threshold: value == threshold,
    "in_range_exclusive_upper": lambda value, threshold: 0.0 <= value < threshold,
}


class OneHot(Column):
    """One level of a one-hot expansion over an integer-valued feature."""

    def __init__(self, name: str, source: str, level: int | None, *, is_other: bool = False,
                 known_levels: Sequence[int] = ()) -> None:
        super().__init__(name, source)
        if is_other and not known_levels:
            raise ValueError("the 'other' level needs the known levels to test against")
        self.level = level
        self.is_other = is_other
        self.known_levels = tuple(known_levels)

    def extract(self, alert: Alert) -> float:
        raw = alert.features.get(self.source)
        if raw is None or isinstance(raw, str):
            # Missing protocol is genuinely "other", not "tcp". Encoding it as all
            # zeros would make a missing value indistinguishable from an unseen one.
            return 1.0 if self.is_other else 0.0
        value = int(float(raw))
        if self.is_other:
            return 1.0 if value not in self.known_levels else 0.0
        return 1.0 if value == self.level else 0.0

    def identity(self) -> str:
        tag = "other" if self.is_other else str(self.level)
        return f"OneHot({self.name}<-{self.source},{tag},known={self.known_levels})"


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """An ordered, fingerprinted list of columns."""

    name: str
    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("a feature spec needs at least one column")
        names = [column.name for column in self.columns]
        duplicates = sorted({item for item in names if names.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate column names in spec: {duplicates}")

    @property
    def width(self) -> int:
        return len(self.columns)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def fingerprint(self) -> str:
        """SHA-256 over the ordered column identities.

        Order is included on purpose: two specs with the same columns in a
        different order produce different vectors and must not be interchangeable.
        """
        material = json.dumps(
            {"spec": self.name, "columns": [column.identity() for column in self.columns]},
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def describe(self) -> str:
        lines = [f"FeatureSpec {self.name!r} ({self.width} columns, {self.fingerprint[:12]}…)"]
        lines += [f"  {index:>2}. {column.identity()}" for index, column in enumerate(self.columns)]
        return "\n".join(lines)


#: Protocols carrying essentially all traffic in both datasets, by IANA number.
PROTOCOL_LEVELS: Final[tuple[int, ...]] = (6, 17, 1)  # tcp, udp, icmp


def default_spec() -> FeatureSpec:
    """The sprint's flow-feature spec over :data:`~sentinel.ingest.normalizer.UNIFIED_FEATURES`.

    Deliberately built on the unified cross-dataset feature space, so a detector
    trained on CIC-IDS2017 can be validated on UNSW-NB15 — the mitigation PRD
    Section 10 lists for "overfitting evaluation to the same datasets used for
    training". A 78-column CIC-specific spec would score better on CIC and tell us
    nothing about generalisation.
    """
    columns: list[Column] = [
        Log1p("log_duration_seconds", "duration_seconds"),
        Log1p("log_src_bytes", "src_bytes"),
        Log1p("log_dst_bytes", "dst_bytes"),
        Log1p("log_src_packets", "src_packets"),
        Log1p("log_dst_packets", "dst_packets"),
        Log1p("log_total_bytes", "total_bytes"),
        Log1p("log_total_packets", "total_packets"),
        Log1p("log_bytes_per_second", "bytes_per_second"),
        Log1p("log_packets_per_second", "packets_per_second"),
        Log1p("log_src_mean_packet_bytes", "src_mean_packet_bytes"),
        Log1p("log_dst_mean_packet_bytes", "dst_mean_packet_bytes"),
        # Already a bounded ratio in [0, 1]; compressing it would destroy the
        # signal that distinguishes an upload from a download.
        Identity("bytes_ratio_src_to_total", "bytes_ratio_src_to_total"),
        # Asymmetry matters on its own: a flow with zero bytes returned is a very
        # different event from a balanced conversation, and the ratio alone maps
        # both "no response" and "tiny response" close to 1.0.
        Indicator("no_response", "dst_bytes", "eq", 0.0),
        Indicator("no_request_payload", "src_bytes", "eq", 0.0),
        Indicator("single_packet_flow", "total_packets", "lte", 1.0),
        # Port semantics, not port magnitude.
        Indicator("port_is_system", "dst_port", "in_range_exclusive_upper", 1024.0),
        Indicator("port_is_registered", "dst_port", "gte", 1024.0),
        Indicator("port_is_ephemeral", "dst_port", "gte", 49152.0),
        Indicator("port_is_unset", "dst_port", "lte", 0.0),
        *[
            OneHot(f"proto_{level}", "protocol_number", level)
            for level in PROTOCOL_LEVELS
        ],
        OneHot("proto_other", "protocol_number", None, is_other=True, known_levels=PROTOCOL_LEVELS),
        # --- session context (sentinel.ingest.enrich) -----------------------
        #
        # Added after the first honest evaluation of the per-flow-only spec came in
        # at ROC-AUC 0.839 against F-03's 0.90 target, with brute_force recall 0.02
        # and web_attack 0.12. Those two families are not separable from a single
        # flow's byte statistics, because a single brute-force login attempt *is* a
        # login attempt. What separates them is fan-out over a time window, so the
        # window features are part of the contract rather than an optional extra.
        Log1p("log_src_flow_count_window", "src_flow_count_window"),
        Log1p("log_src_distinct_dst_ports_window", "src_distinct_dst_ports_window"),
        Log1p("log_src_distinct_dst_ips_window", "src_distinct_dst_ips_window"),
        Log1p("log_src_bytes_sum_window", "src_bytes_sum_window"),
        Log1p("log_src_interarrival_mean_window", "src_interarrival_mean_window"),
        # Not compressed: the coefficient of variation is already a bounded-ish
        # ratio, and it is the *low* end that matters — a beacon's CV near zero is
        # the signal, and log1p would squash exactly that region.
        Identity("src_interarrival_cv_window", "src_interarrival_cv_window"),
        # Fan-in. The only feature that can see a distributed attack at all: a DDoS
        # draws a fresh source per flow, so every per-source count stays at ~1.
        Log1p("log_dst_flow_count_window", "dst_flow_count_window"),
        Log1p("log_dst_distinct_src_ips_window", "dst_distinct_src_ips_window"),
        # Machine-regular timing at low volume: the C2 beacon signature, which
        # neither the volume features nor the CV alone isolate.
        Indicator("beacon_like_regularity", "src_interarrival_cv_window", "lt", 0.25),
    ]
    return FeatureSpec(name="unified-flow-context-v1", columns=tuple(columns))


DEFAULT_SPEC: Final[FeatureSpec] = default_spec()


# --------------------------------------------------------------------------- #
# Vectorizer
# --------------------------------------------------------------------------- #


class AlertVectorizer:
    """Turns alerts into standardized float64 matrices.

    Standardization statistics are computed on ``fit`` and frozen. A zero-variance
    column gets scale 1.0 rather than dividing by zero — and it is *reported*
    through :attr:`degenerate_columns`, because a constant feature in production
    usually means an upstream field stopped being populated, which is a detection
    gap disguised as a modelling detail.
    """

    def __init__(self, spec: FeatureSpec | None = None, *, standardize: bool = True) -> None:
        self.spec = spec or DEFAULT_SPEC
        self.standardize = standardize
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._n_fitted: int = 0

    # --- fitting ------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    def fit(self, alerts: Iterable[Alert]) -> AlertVectorizer:
        """Fit standardization statistics. Returns ``self`` for chaining."""
        raw = self._extract_matrix(alerts)
        if raw.shape[0] == 0:
            raise FeatureError("cannot fit a vectorizer on zero alerts")
        self._mean = raw.mean(axis=0)
        std = raw.std(axis=0, ddof=0)
        self._scale = np.where(std > _STD_FLOOR, std, 1.0)
        self._n_fitted = int(raw.shape[0])
        return self

    @property
    def degenerate_columns(self) -> tuple[str, ...]:
        """Columns that had (near-)zero variance at fit time."""
        if self._scale is None:
            raise ModelNotFittedError("vectorizer is not fitted")
        return tuple(
            name for name, value in zip(self.spec.column_names, self._scale, strict=True)
            if value == 1.0
        )

    # --- transforming -------------------------------------------------------

    def transform(self, alerts: Iterable[Alert]) -> np.ndarray:
        """Vectorize a batch. Shape ``(n, spec.width)``."""
        return self._apply(self._extract_matrix(alerts))

    def transform_one(self, alert: Alert) -> np.ndarray:
        """Vectorize a single alert for the serving path. Shape ``(spec.width,)``.

        Calls exactly the same extraction and scaling as :meth:`transform`, so the
        two cannot drift. Returns 1-D for ergonomics; ``transform([alert])[0]`` is
        bit-identical and a test proves it.
        """
        return self._apply(self._extract_row(alert).reshape(1, -1))[0]

    def fit_transform(self, alerts: Sequence[Alert]) -> np.ndarray:
        return self.fit(alerts).transform(alerts)

    def _extract_row(self, alert: Alert) -> np.ndarray:
        if not isinstance(alert, Alert):
            raise FeatureError(
                f"AlertVectorizer only accepts canonical Alert objects, got "
                f"{type(alert).__name__}. Route source data through a Normalizer first — "
                "that is what keeps training and serving on one definition (PRD 7.3)."
            )
        row = np.empty(self.spec.width, dtype=DTYPE)
        for index, column in enumerate(self.spec.columns):
            row[index] = column.extract(alert)
        return row

    def _extract_matrix(self, alerts: Iterable[Alert]) -> np.ndarray:
        rows = [self._extract_row(alert) for alert in alerts]
        if not rows:
            return np.empty((0, self.spec.width), dtype=DTYPE)
        return np.vstack(rows)

    def _apply(self, raw: np.ndarray) -> np.ndarray:
        if raw.shape[1] != self.spec.width:
            raise FeatureError(
                f"extracted {raw.shape[1]} columns but spec declares {self.spec.width}"
            )
        if not self.standardize:
            return raw
        if self._mean is None or self._scale is None:
            raise ModelNotFittedError(
                "vectorizer must be fitted before transforming; call fit() on the training "
                "split only, never on validation or test data"
            )
        return (raw - self._mean) / self._scale

    # --- persistence --------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Serializable fitted state, carrying the spec fingerprint."""
        if self._mean is None or self._scale is None:
            raise ModelNotFittedError("nothing to save: vectorizer is not fitted")
        return {
            "spec_name": self.spec.name,
            "spec_fingerprint": self.spec.fingerprint,
            "column_names": list(self.spec.column_names),
            "standardize": self.standardize,
            "n_fitted": self._n_fitted,
            "mean": self._mean.tolist(),
            "scale": self._scale.tolist(),
        }

    def load_state(self, state: dict[str, Any]) -> AlertVectorizer:
        """Restore fitted state, refusing a mismatched spec."""
        stored = str(state.get("spec_fingerprint", ""))
        if stored != self.spec.fingerprint:
            raise FeatureError(
                "feature spec fingerprint mismatch: artifact was fitted under "
                f"{stored[:12]}… but this process builds {self.spec.fingerprint[:12]}…. "
                "The columns or their order changed; scoring anyway would silently "
                "feed the model the wrong features."
            )
        self.standardize = bool(state["standardize"])
        self._mean = np.asarray(state["mean"], dtype=DTYPE)
        self._scale = np.asarray(state["scale"], dtype=DTYPE)
        self._n_fitted = int(state.get("n_fitted", 0))
        if self._mean.shape != (self.spec.width,) or self._scale.shape != (self.spec.width,):
            raise FeatureError("stored statistics do not match the spec width")
        return self

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, spec: FeatureSpec | None = None) -> AlertVectorizer:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(spec or DEFAULT_SPEC).load_state(state)

    def assert_compatible(self, fingerprint: str) -> None:
        """Raise unless ``fingerprint`` matches this vectorizer's spec."""
        if fingerprint != self.spec.fingerprint:
            raise FeatureError(
                f"model was trained under spec {fingerprint[:12]}… but the live spec is "
                f"{self.spec.fingerprint[:12]}…; refusing to score"
            )


_STD_FLOOR: Final[float] = 1e-12
