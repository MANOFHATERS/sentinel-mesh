"""Injectable clocks.

Time is an input, not an ambient fact. Every component that stamps a timestamp
takes a :class:`Clock`, so tests can assert exact values instead of tolerating
windows — and so replayed historical data can be stamped with *its own* time
rather than wall-clock time.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Anything that can report the current instant and a monotonic tick."""

    def now(self) -> datetime:
        """Return the current instant as a timezone-aware UTC datetime."""
        ...

    def monotonic(self) -> float:
        """Return a monotonically non-decreasing seconds counter for durations."""
        ...


class SystemClock:
    """The real clock. UTC-aware by construction; never returns naive datetimes."""

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class FrozenClock:
    """A deterministic clock for tests and for deterministic demo replays.

    Starts at ``start`` and only moves when :meth:`advance` is called, so a test
    can assert a timestamp equals an exact expected value.
    """

    __slots__ = ("_mono", "_now")

    def __init__(self, start: datetime | None = None) -> None:
        base = start if start is not None else datetime(2026, 1, 1, tzinfo=UTC)
        if base.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware start datetime")
        self._now = base.astimezone(UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float = 1.0) -> datetime:
        """Move the clock forward and return the new instant."""
        if seconds < 0:
            raise ValueError("FrozenClock cannot move backwards")
        self._now = self._now + timedelta(seconds=seconds)
        self._mono += seconds
        return self._now


class SimulationClock:
    """Real elapsed time on a chosen origin, plus explicit jumps.

    The clock the offline evaluation needs, and neither of the other two is it.

    :class:`SystemClock` gives real durations but no way to simulate the analyst
    who takes twelve seconds to click approve, so PRD Section 9.1's MTTC — which
    is defined to *include* that click — cannot be measured with it.
    :class:`FrozenClock` can inject the click but reports every node as taking
    zero seconds, so MTTD comes out as 0.00s and its 30-second budget passes
    without measuring anything. A gate that cannot fail is not a gate.

    This clock advances by genuine monotonic time as work happens, so node
    durations are real, and :meth:`advance` adds a simulated interval on top for
    the parts of the timeline a human occupies. The origin is fixed, so a replay
    of a historical corpus can still be stamped on its own timeline.
    """

    __slots__ = ("_origin", "_skew", "_start_mono")

    def __init__(self, start: datetime | None = None) -> None:
        base = start if start is not None else datetime(2026, 1, 1, tzinfo=UTC)
        if base.tzinfo is None:
            raise ValueError("SimulationClock requires a timezone-aware start datetime")
        self._origin = base.astimezone(UTC)
        self._start_mono = time.monotonic()
        self._skew = 0.0

    def _elapsed(self) -> float:
        return (time.monotonic() - self._start_mono) + self._skew

    def now(self) -> datetime:
        return self._origin + timedelta(seconds=self._elapsed())

    def monotonic(self) -> float:
        return self._elapsed()

    def advance(self, seconds: float = 1.0) -> datetime:
        """Add a simulated interval — a human's thinking time, or a feed gap."""
        if seconds < 0:
            raise ValueError("SimulationClock cannot move backwards")
        self._skew += seconds
        return self.now()


def to_utc(value: datetime) -> datetime:
    """Normalize any datetime to timezone-aware UTC.

    Naive datetimes are rejected rather than assumed-local: silently guessing a
    timezone is how incident timelines end up hours apart across components.
    """
    if value.tzinfo is None:
        raise ValueError("naive datetime rejected; attach a timezone (UTC preferred)")
    return value.astimezone(UTC)
