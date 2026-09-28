"""Layer 2 — streaming session-context enrichment.

Why this module exists
---------------------
The first honest evaluation of the F-03 ensemble on a purely per-flow feature
space produced ROC-AUC 0.839 against a 0.90 target, and the per-family recall
breakdown said exactly why: ``brute_force`` 0.02, ``web_attack`` 0.12. That is not
a tuning failure, and no amount of weight search fixes it. **A single SSH
brute-force flow is indistinguishable from a legitimate SSH login by its byte and
packet statistics, because that is what it is.** What distinguishes it is that
four hundred of them arrive from one host in ninety seconds.

Per-flow features cannot express that. So the detector was not the problem — the
feature space was, and the fix is to give the model the temporal structure the
attack actually lives in:

============================  ==================================================
Attack family                 The signal that actually separates it
============================  ==================================================
Brute force                   many flows, one source, one destination port
Port scan / recon             many flows, one source, many destination ports
DDoS                          many flows, many sources, **one** destination (fan-in)
Slowloris DoS                 many concurrent long-lived flows to one port
Botnet C2                     low-rate, highly regular inter-arrival to a rare port
Infiltration / exfil          byte volume — visible per-flow already
============================  ==================================================

:class:`SessionContextEnricher` computes those in one pass over the live stream.

Two properties make it correct rather than merely useful
-------------------------------------------------------
**Strictly backward-looking.** A window feature is computed from events already
seen, never from the current event or any later one. An enricher that included the
current flow in its own count, or that was built from a full-dataset ``groupby``,
would be leaking the future into the features — and would score beautifully
offline and collapse in production, because at serving time the future does not
exist yet. :func:`test_enricher_is_strictly_causal` pins this by asserting that
enriching a prefix of the stream gives identical features to enriching the whole
stream and taking the prefix.

**Bounded memory.** A SOC stream is unbounded and an attacker can choose the
cardinality of the key space — spoofing a million source addresses is free. The
window state is therefore capped by :attr:`SessionContextEnricher.max_tracked_keys`
with least-recently-used eviction, and evictions are counted. Unbounded
``defaultdict(list)`` state keyed on attacker-controlled data is a denial-of-service
in the detection path.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from sentinel.core.schemas import Alert

__all__ = [
    "CONTEXT_FEATURES",
    "SessionContextEnricher",
    "WindowStats",
]

CONTEXT_FEATURES: Final[tuple[str, ...]] = (
    "src_flow_count_window",
    "src_distinct_dst_ports_window",
    "src_distinct_dst_ips_window",
    "src_bytes_sum_window",
    "src_interarrival_mean_window",
    "src_interarrival_cv_window",
    "dst_flow_count_window",
    "dst_distinct_src_ips_window",
)
"""Context features appended to every alert. Order is part of the feature contract."""


@dataclass(slots=True)
class _KeyWindow:
    """Backward-looking event window for one key (a source or destination address)."""

    timestamps: deque[float] = field(default_factory=deque)
    ports: deque[int] = field(default_factory=deque)
    peers: deque[str] = field(default_factory=deque)
    byte_counts: deque[float] = field(default_factory=deque)

    def evict_before(self, cutoff: float) -> None:
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
            self.ports.popleft()
            self.peers.popleft()
            self.byte_counts.popleft()

    def append(self, at: float, port: int, peer: str, byte_count: float, cap: int) -> None:
        self.timestamps.append(at)
        self.ports.append(port)
        self.peers.append(peer)
        self.byte_counts.append(byte_count)
        # Hard cap per key as well as per stream: one host emitting a million
        # flows inside the window must not grow this deque without limit.
        while len(self.timestamps) > cap:
            self.timestamps.popleft()
            self.ports.popleft()
            self.peers.popleft()
            self.byte_counts.popleft()

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True, slots=True)
class WindowStats:
    """Computed window statistics for one alert, before they become features."""

    flow_count: int
    distinct_ports: int
    distinct_peers: int
    byte_sum: float
    interarrival_mean: float
    interarrival_cv: float


class SessionContextEnricher:
    """Adds sliding-window context features to a live alert stream.

    Used identically in the training pipeline and the serving pipeline — the same
    object, the same method, one pass, in stream order. That is the whole reason it
    is in the ingestion layer rather than in a training notebook.
    """

    #: Window length in seconds. 60s is chosen to match the tempo of the attacks in
    #: scope: CIC-IDS2017's Patator brute-force runs and its port scan both produce
    #: hundreds of flows per minute, while ordinary user sessions produce a handful.
    #: Much shorter and a slow scan hides inside it; much longer and normal
    #: busy-host traffic starts to look like a campaign.
    DEFAULT_WINDOW_SECONDS: Final[float] = 60.0

    #: Per-key event cap, and the value every count feature saturates at. Beyond a
    #: few thousand flows per minute from one host the exact number carries no extra
    #: information — it is already far outside anything benign.
    DEFAULT_PER_KEY_CAP: Final[int] = 4096

    def __init__(
        self,
        *,
        window_seconds: float | None = None,
        max_tracked_keys: int = 50_000,
        per_key_cap: int | None = None,
    ) -> None:
        if window_seconds is not None and window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_tracked_keys < 1:
            raise ValueError("max_tracked_keys must be positive")
        self.window_seconds = window_seconds or self.DEFAULT_WINDOW_SECONDS
        self.max_tracked_keys = max_tracked_keys
        self.per_key_cap = per_key_cap or self.DEFAULT_PER_KEY_CAP
        # OrderedDict as an LRU: move_to_end on touch, popitem(last=False) to evict.
        self._by_source: OrderedDict[str, _KeyWindow] = OrderedDict()
        self._by_destination: OrderedDict[str, _KeyWindow] = OrderedDict()
        self.evictions = 0
        self.processed = 0
        self._last_timestamp: float | None = None
        self.out_of_order_events = 0

    # --- public API ---------------------------------------------------------

    def enrich(self, alert: Alert) -> Alert:
        """Return ``alert`` with :data:`CONTEXT_FEATURES` added.

        The returned alert's context features describe traffic seen **strictly
        before** this alert. The alert is added to the window afterwards, so it
        never contributes to its own counts.
        """
        at = alert.timestamp.timestamp()
        if self._last_timestamp is not None and at < self._last_timestamp:
            # Real streams reorder slightly. Counting it is better than either
            # crashing or silently producing window stats from a scrambled order.
            self.out_of_order_events += 1
        self._last_timestamp = max(at, self._last_timestamp or at)

        source_key = self._source_key(alert)
        destination_key = self._destination_key(alert)
        cutoff = at - self.window_seconds

        source_stats = self._stats(self._by_source, source_key, cutoff)
        destination_stats = self._stats(self._by_destination, destination_key, cutoff)

        features: dict[str, float | int | bool | str | None] = dict(alert.features)
        features.update(
            {
                "src_flow_count_window": float(source_stats.flow_count),
                "src_distinct_dst_ports_window": float(source_stats.distinct_ports),
                "src_distinct_dst_ips_window": float(source_stats.distinct_peers),
                "src_bytes_sum_window": source_stats.byte_sum,
                "src_interarrival_mean_window": source_stats.interarrival_mean,
                "src_interarrival_cv_window": source_stats.interarrival_cv,
                "dst_flow_count_window": float(destination_stats.flow_count),
                "dst_distinct_src_ips_window": float(destination_stats.distinct_peers),
            }
        )

        total_bytes = _as_float(alert.features.get("total_bytes"))
        port = int(alert.dst_port or 0)

        self._observe(self._by_source, source_key, at, port, destination_key, total_bytes)
        self._observe(self._by_destination, destination_key, at, port, source_key, total_bytes)
        self.processed += 1

        # updated() rather than model_copy(): it re-runs Alert's validators, so a
        # context feature that somehow came out non-finite is caught here rather
        # than silently breaking the canonical hash of every downstream record.
        return alert.updated(features=features)

    def enrich_all(self, alerts: list[Alert]) -> list[Alert]:
        """Enrich a stream in order. Order matters and is the caller's responsibility.

        Sorting internally would be wrong: at serving time the stream arrives in
        arrival order and cannot be sorted, so sorting here would make the training
        features unreproducible in production — the exact skew this layer exists to
        prevent.
        """
        return [self.enrich(alert) for alert in alerts]

    def reset(self) -> None:
        self._by_source.clear()
        self._by_destination.clear()
        self.evictions = 0
        self.processed = 0
        self.out_of_order_events = 0
        self._last_timestamp = None

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _source_key(alert: Alert) -> str:
        return alert.src_ip or f"asset:{alert.asset_id}"

    @staticmethod
    def _destination_key(alert: Alert) -> str:
        return alert.dst_ip or f"asset:{alert.asset_id}"

    def _stats(
        self, index: OrderedDict[str, _KeyWindow], key: str, cutoff: float
    ) -> WindowStats:
        window = index.get(key)
        if window is None:
            return WindowStats(0, 0, 0, 0.0, 0.0, 0.0)
        window.evict_before(cutoff)
        count = len(window)
        if count == 0:
            return WindowStats(0, 0, 0, 0.0, 0.0, 0.0)

        timestamps = list(window.timestamps)
        gaps = [b - a for a, b in itertools.pairwise(timestamps)]
        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap > 0 and len(gaps) > 1:
                variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                # Coefficient of variation: near zero means machine-regular timing,
                # which is the signature of a beacon rather than a human.
                cv = (variance**0.5) / mean_gap
            else:
                cv = 0.0
        else:
            mean_gap = 0.0
            cv = 0.0

        return WindowStats(
            flow_count=count,
            distinct_ports=len(set(window.ports)),
            distinct_peers=len(set(window.peers)),
            byte_sum=float(sum(window.byte_counts)),
            interarrival_mean=mean_gap,
            interarrival_cv=cv,
        )

    def _observe(
        self,
        index: OrderedDict[str, _KeyWindow],
        key: str,
        at: float,
        port: int,
        peer: str,
        byte_count: float,
    ) -> None:
        window = index.get(key)
        if window is None:
            window = _KeyWindow()
            index[key] = window
        else:
            index.move_to_end(key)
        window.append(at, port, peer, byte_count, self.per_key_cap)

        while len(index) > self.max_tracked_keys:
            index.popitem(last=False)
            self.evictions += 1

    # --- observability ------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "tracked_sources": len(self._by_source),
            "tracked_destinations": len(self._by_destination),
            "evictions": self.evictions,
            "out_of_order_events": self.out_of_order_events,
        }


def _as_float(value: object) -> float:
    if value is None or isinstance(value, str):
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def window_cutoff(at: datetime, window_seconds: float) -> float:
    """The epoch-second cutoff for a window ending at ``at``. Exposed for tests."""
    return at.timestamp() - window_seconds
