"""Layer 1 — the alert/log replay service (PRD F-01).

Replays labelled public intrusion datasets as if they were a live SIEM/EDR feed:
read a row, normalize it to a canonical :class:`~sentinel.core.schemas.Alert`,
publish it to the event bus, and record an ``ALERT_INGESTED`` event in the audit
chain.

Two time models, and the distinction matters more than it looks:

``TimeModel.WALL_CLOCK``
    Paces publication against real time to hit a target alerts-per-second. This is
    what the live demo runs, because a judge watching a dashboard needs to see
    alerts arrive, not appear instantly.

``TimeModel.VIRTUAL``
    Advances an injected :class:`~sentinel.core.clock.FrozenClock` instead of
    sleeping. A 100,000-row replay finishes in the time it takes to compute, and
    every timestamp is reproducible to the microsecond. This is what the offline
    evaluation and the test suite use. PRD Section 9.1 measures simulated MTTD as
    a timestamp delta in the pipeline logs — a metric that would otherwise be a
    measurement of how busy the laptop was.

F-01's acceptance criterion is *1,000+ alerts/min sustained with zero
schema-validation failures*. :class:`ReplayStats` reports the achieved rate and
the normalizer's rejection count so the claim is measured, not asserted; a test
holds the pipeline to a floor two orders of magnitude above the requirement.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.clock import Clock, FrozenClock, SystemClock
from sentinel.core.schemas import Alert, AuditEventType
from sentinel.ingest.bus import EventBus
from sentinel.ingest.normalizer import Normalizer

__all__ = [
    "CsvReplaySource",
    "IterableReplaySource",
    "ReplayService",
    "ReplaySource",
    "ReplayStats",
    "TimeModel",
]


class TimeModel(StrEnum):
    WALL_CLOCK = "wall_clock"
    VIRTUAL = "virtual"


@runtime_checkable
class ReplaySource(Protocol):
    """Anything that yields raw source rows in order."""

    name: str

    def rows(self) -> Iterator[Mapping[str, Any]]:
        ...


@dataclass(slots=True)
class IterableReplaySource:
    """Replay an in-memory sequence. Used by tests and the synthetic generator."""

    records: Iterable[Mapping[str, Any]]
    name: str = "iterable"

    def rows(self) -> Iterator[Mapping[str, Any]]:
        yield from self.records


@dataclass(slots=True)
class CsvReplaySource:
    """Stream a dataset CSV row by row without loading it into memory.

    CIC-IDS2017's largest single CSV is several hundred megabytes and the full set
    is ~2.8M flows, so :mod:`csv` streaming is the right tool rather than
    ``pandas.read_csv``. ``encoding`` defaults to ``utf-8-sig`` with a
    ``latin-1`` fallback because the widely-mirrored copies of these files are
    inconsistently encoded — the same reason the label normalizer folds the
    cp1252 mojibake dash.
    """

    path: Path
    name: str = ""
    encoding: str = "utf-8-sig"
    limit: int | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.name:
            self.name = self.path.name

    def rows(self) -> Iterator[Mapping[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Place the dataset CSVs under data/raw/ — see "
                "docs/DATA.md for the download links; nothing in this repo ships them."
            )
        for encoding in (self.encoding, "latin-1"):
            try:
                with self.path.open("r", encoding=encoding, newline="") as handle:
                    reader = csv.DictReader(handle)
                    for index, row in enumerate(reader):
                        if self.limit is not None and index >= self.limit:
                            return
                        yield row
                return
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(  # pragma: no cover - both encodings failing is pathological
            "csv", b"", 0, 1, f"could not decode {self.path} as utf-8 or latin-1"
        )


@dataclass(slots=True)
class ReplayStats:
    """What the replay actually achieved."""

    source: str
    rows_read: int = 0
    alerts_published: int = 0
    rows_rejected: int = 0
    elapsed_seconds: float = 0.0
    first_alert_at: str | None = None
    last_alert_at: str | None = None
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def alerts_per_second(self) -> float:
        return 0.0 if self.elapsed_seconds <= 0 else self.alerts_published / self.elapsed_seconds

    @property
    def alerts_per_minute(self) -> float:
        return self.alerts_per_second * 60.0

    @property
    def meets_f01_throughput(self) -> bool:
        """F-01 requires 1,000+ alerts/min sustained."""
        return self.alerts_per_minute >= 1000.0

    @property
    def schema_failure_free(self) -> bool:
        """F-01 requires zero schema-validation failures."""
        return self.rows_rejected == 0

    def summary(self) -> str:
        return (
            f"{self.source}: {self.alerts_published} alerts from {self.rows_read} rows in "
            f"{self.elapsed_seconds:.3f}s ({self.alerts_per_minute:,.0f}/min), "
            f"{self.rows_rejected} rejected; "
            f"F-01 throughput={'PASS' if self.meets_f01_throughput else 'FAIL'}, "
            f"schema={'PASS' if self.schema_failure_free else 'FAIL'}"
        )


class ReplayService:
    """Turns a dataset into a live alert stream on the bus.

    The service owns no threads. Callers drive it with :meth:`run`, which is a
    plain loop — easier to reason about, to bound in tests, and to interrupt than a
    background producer, and the orchestrator in Part 3 consumes from the bus
    independently anyway.
    """

    #: Interval at which virtual time advances per alert, when no rate is given.
    DEFAULT_VIRTUAL_STEP_SECONDS: float = 0.05

    def __init__(
        self,
        *,
        normalizer: Normalizer,
        bus: EventBus,
        topic: str = "alerts.raw",
        clock: Clock | None = None,
        time_model: TimeModel = TimeModel.VIRTUAL,
        target_rate_per_second: float | None = None,
        audit_log: HashChainedAuditLog | None = None,
        audit_every: int = 1,
    ) -> None:
        if target_rate_per_second is not None and target_rate_per_second <= 0:
            raise ValueError("target_rate_per_second must be positive or None")
        if audit_every < 1:
            raise ValueError("audit_every must be >= 1")
        if time_model is TimeModel.VIRTUAL and clock is None:
            clock = FrozenClock()
        if time_model is TimeModel.VIRTUAL and not isinstance(clock, FrozenClock):
            raise ValueError(
                "TimeModel.VIRTUAL requires a FrozenClock; a real clock cannot be advanced "
                "deterministically, which is the whole point of virtual replay"
            )

        self.normalizer = normalizer
        self.bus = bus
        self.topic = topic
        self.clock: Clock = clock or SystemClock()
        self.time_model = time_model
        self.target_rate = target_rate_per_second
        self.audit_log = audit_log
        self.audit_every = audit_every

    # --- main loop ----------------------------------------------------------

    def run(
        self,
        source: ReplaySource,
        *,
        max_alerts: int | None = None,
        on_alert: Any | None = None,
    ) -> ReplayStats:
        """Replay ``source`` until exhausted or ``max_alerts`` alerts are published.

        ``on_alert`` is an optional callback receiving each published
        :class:`Alert`, used by the offline evaluation to collect ground truth
        without a second pass over the data.
        """
        stats = ReplayStats(source=source.name)
        step = (
            1.0 / self.target_rate
            if self.target_rate is not None
            else self.DEFAULT_VIRTUAL_STEP_SECONDS
        )
        started_wall = time.perf_counter()
        started_virtual = self.clock.now()
        next_due = time.perf_counter()

        for row_index, row in enumerate(source.rows()):
            if max_alerts is not None and stats.alerts_published >= max_alerts:
                break
            stats.rows_read += 1

            observed_at = self.clock.now()
            alert = self.normalizer.normalize(
                row,
                row_index=row_index,
                observed_at=observed_at,
                ingested_at=observed_at,
            )
            if alert is None:
                stats.rows_rejected += 1
                self._advance(step, next_due)
                next_due += step
                continue

            self.bus.publish(self.topic, self._envelope(alert))
            stats.alerts_published += 1
            label = alert.ground_truth_label or "unlabelled"
            stats.labels[label] = stats.labels.get(label, 0) + 1
            if stats.first_alert_at is None:
                stats.first_alert_at = alert.ingested_at.isoformat()
            stats.last_alert_at = alert.ingested_at.isoformat()

            if self.audit_log is not None and stats.alerts_published % self.audit_every == 0:
                self.audit_log.append(
                    AuditEventType.ALERT_INGESTED,
                    actor="replay_service",
                    tenant_id=alert.tenant_id,
                    subject_id=alert.alert_id,
                    payload={
                        "dataset": alert.dataset,
                        "source": alert.source.value,
                        "row_index": row_index,
                        "alert_hash": alert.canonical_hash(),
                        # The payload hash, not the payload: the audit log must not
                        # become a second copy of attacker-controlled text.
                        "raw_payload_sha256": alert.raw_payload.digest,
                        "injection_verdict": alert.injection_scan.verdict.value,
                    },
                )

            if on_alert is not None:
                on_alert(alert)

            self._advance(step, next_due)
            next_due += step

        if self.time_model is TimeModel.VIRTUAL:
            stats.elapsed_seconds = (self.clock.now() - started_virtual).total_seconds()
            # A virtual replay of zero duration still took wall-clock work to do;
            # reporting 0.0 would make throughput infinite and the F-01 check
            # meaningless, so fall back to measured wall time.
            if stats.elapsed_seconds <= 0:
                stats.elapsed_seconds = max(time.perf_counter() - started_wall, 1e-9)
        else:
            stats.elapsed_seconds = max(time.perf_counter() - started_wall, 1e-9)
        return stats

    def wall_clock_throughput(self, stats: ReplayStats, wall_seconds: float) -> float:
        """Alerts per minute measured against real time, whatever the time model."""
        return 0.0 if wall_seconds <= 0 else stats.alerts_published / wall_seconds * 60.0

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _envelope(alert: Alert) -> dict[str, Any]:
        """Bus payload. Alerts cross the bus as canonical JSON-ready dicts.

        The alert's own hash travels with it so a consumer can detect a payload
        mangled in transit without re-deriving it from a mutable source.
        """
        return {
            "kind": "alert",
            "schema": "sentinel.alert.v1",
            "alert": alert.model_dump(mode="json"),
            "alert_hash": alert.canonical_hash(),
        }

    def _advance(self, step: float, next_due: float) -> None:
        if self.time_model is TimeModel.VIRTUAL:
            assert isinstance(self.clock, FrozenClock)  # guaranteed in __init__
            self.clock.advance(step)
            return
        if self.target_rate is None:
            return
        remaining = next_due + step - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def decode(envelope: Mapping[str, Any]) -> Alert:
        """Rebuild an :class:`Alert` from a bus envelope, verifying its hash.

        A mismatch means the payload changed between publish and consume. That is
        either corruption or tampering; either way it must not be validated into a
        plausible-looking alert and quietly analysed.
        """
        if envelope.get("kind") != "alert":
            raise ValueError(f"not an alert envelope: kind={envelope.get('kind')!r}")
        alert = Alert.model_validate(envelope["alert"])
        expected = envelope.get("alert_hash")
        if expected is not None and alert.canonical_hash() != expected:
            raise ValueError(
                f"alert {alert.alert_id} failed hash verification on the bus "
                f"(expected {str(expected)[:16]}…, computed {alert.canonical_hash()[:16]}…)"
            )
        return alert

    def virtual_span(self, stats: ReplayStats) -> timedelta:
        """Virtual time covered by a replay, for MTTD/MTTC arithmetic."""
        return timedelta(seconds=stats.elapsed_seconds)
