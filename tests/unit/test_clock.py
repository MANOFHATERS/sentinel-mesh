"""Injectable clocks.

:class:`SimulationClock` gets the most attention here because it is the one the
offline evaluation's PRD Section 9.1 gates depend on, and because the reason it
exists is that the other two each make one of those gates unmeasurable.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from sentinel.core.clock import (
    Clock,
    FrozenClock,
    SimulationClock,
    SystemClock,
    to_utc,
)

START = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


class TestProtocolConformance:
    @pytest.mark.parametrize(
        "clock", [SystemClock(), FrozenClock(START), SimulationClock(START)]
    )
    def test_every_clock_satisfies_the_protocol(self, clock) -> None:
        assert isinstance(clock, Clock)

    @pytest.mark.parametrize(
        "clock", [SystemClock(), FrozenClock(START), SimulationClock(START)]
    )
    def test_now_is_always_timezone_aware_utc(self, clock) -> None:
        """A naive timestamp in an audit record is a timeline nobody can align."""
        assert clock.now().tzinfo is not None
        assert clock.now().utcoffset() == timedelta(0)

    @pytest.mark.parametrize(
        "clock", [SystemClock(), FrozenClock(START), SimulationClock(START)]
    )
    def test_monotonic_never_decreases(self, clock) -> None:
        readings = [clock.monotonic() for _ in range(5)]
        assert readings == sorted(readings)


class TestFrozenClock:
    def test_does_not_move_on_its_own(self) -> None:
        clock = FrozenClock(START)
        first = clock.now()
        time.sleep(0.01)
        assert clock.now() == first

    def test_advance_moves_both_readings(self) -> None:
        clock = FrozenClock(START)
        assert clock.advance(2.5) == START + timedelta(seconds=2.5)
        assert clock.monotonic() == pytest.approx(2.5)

    def test_refuses_to_move_backwards(self) -> None:
        with pytest.raises(ValueError, match="backwards"):
            FrozenClock(START).advance(-1.0)

    def test_refuses_a_naive_start(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            FrozenClock(datetime(2026, 1, 1))

    def test_reports_zero_duration_for_real_work(self) -> None:
        """The property that makes it wrong for the MTTD gate, stated as a test.

        A frozen clock reports every node as instantaneous, so a 30-second budget
        measured on one passes regardless of how slow the pipeline is. That is
        why :class:`SimulationClock` exists.
        """
        clock = FrozenClock(START)
        before = clock.now()
        assert sum(range(200_000)) > 0
        assert (clock.now() - before).total_seconds() == 0.0


class TestSimulationClock:
    def test_starts_at_its_origin(self) -> None:
        clock = SimulationClock(START)
        assert clock.now() - START < timedelta(seconds=1)

    def test_real_work_shows_up_as_real_duration(self) -> None:
        """The whole point: a node that takes time is recorded as taking time."""
        clock = SimulationClock(START)
        before = clock.now()
        time.sleep(0.02)
        assert (clock.now() - before).total_seconds() >= 0.015

    def test_advance_adds_simulated_time_on_top(self) -> None:
        clock = SimulationClock(START)
        before = clock.now()
        clock.advance(12.0)
        assert (clock.now() - before).total_seconds() >= 12.0

    def test_advance_and_real_time_compose(self) -> None:
        clock = SimulationClock(START)
        before = clock.now()
        clock.advance(5.0)
        time.sleep(0.02)
        clock.advance(5.0)
        assert (clock.now() - before).total_seconds() >= 10.015

    def test_refuses_to_move_backwards(self) -> None:
        with pytest.raises(ValueError, match="backwards"):
            SimulationClock(START).advance(-1.0)

    def test_refuses_a_naive_start(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            SimulationClock(datetime(2026, 1, 1))

    def test_monotonic_tracks_now(self) -> None:
        clock = SimulationClock(START)
        before_now, before_mono = clock.now(), clock.monotonic()
        clock.advance(3.0)
        assert (clock.now() - before_now).total_seconds() == pytest.approx(
            clock.monotonic() - before_mono, abs=0.05
        )

    def test_the_origin_is_independent_of_wall_clock(self) -> None:
        """So a historical replay is stamped on its own timeline, not on today's."""
        past = datetime(2017, 7, 3, 8, 0, 0, tzinfo=UTC)
        assert SimulationClock(past).now().year == 2017


class TestToUtc:
    def test_converts_an_offset_datetime(self) -> None:
        value = datetime(2026, 9, 28, 14, 0, tzinfo=timezone(timedelta(hours=2)))
        assert to_utc(value) == datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

    def test_rejects_a_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="naive datetime"):
            to_utc(datetime(2026, 9, 28, 12, 0))

    def test_is_idempotent(self) -> None:
        assert to_utc(to_utc(START)) == START
