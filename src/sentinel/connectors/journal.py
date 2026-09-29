"""Exactly-once-ish execution, and a ceiling on how much damage a loop can do.

Two controls that sit between the router and the connectors, and that neither the
graph nor the schema can provide.

The execution journal
---------------------
F-04's promise is that a run can pause and resume anywhere. The checkpoint is written
*after* a node returns, so a process that dies between the connector call and the
checkpoint resumes by running the execution node again — and without a journal, a
host is isolated twice, a second pull request is opened, an account is disabled that
a human had re-enabled in the meantime. The journal records ``started`` before the
call and ``completed`` after it, keyed by ``action_id``:

*   ``completed`` found → the recorded outcome is returned, marked ``replayed``, and
    the remote system is not called.
*   ``started`` found without ``completed`` → the call is made again. Every real
    connector here is written to be idempotent against its remote system (the Git
    connector looks for its branch and pull request before creating them; SCIM
    ``active=false`` is a state assignment; the webhook carries the action id as its
    delivery id), so re-issuing is correct, and refusing would strand the action.

The SQLite implementation survives the process, which is the case that matters.

The blast-radius limiter
------------------------
A policy bug, a replayed feed or a prompt-injected investigation can turn one bad
decision into a hundred identical ones, each individually approved by an analyst
clicking through a queue. :class:`BlastRadiusLimiter` caps destructive executions per
tenant in a sliding window. Past the cap the router refuses, the action fails with a
reason naming the limit, and the audit chain records it. It is deliberately a
*count* rather than a rate: the question it answers is "how many hosts can this
system take off the network in an hour before a human notices", and the answer
should be a number someone chose.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from sentinel.connectors.base import ExecutionOutcome
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import GuardrailViolation

__all__ = [
    "BlastRadiusExceeded",
    "BlastRadiusLimiter",
    "ExecutionJournal",
    "JournalEntry",
    "MemoryJournal",
    "SqliteJournal",
]


class BlastRadiusExceeded(GuardrailViolation):
    """Too many destructive actions for one tenant inside the window."""


@dataclass(frozen=True, slots=True)
class JournalEntry:
    action_id: str
    connector: str
    completed: bool
    outcome: ExecutionOutcome | None


class ExecutionJournal(Protocol):
    def lookup(self, action_id: str) -> JournalEntry | None: ...

    def start(self, action_id: str, *, connector: str) -> None: ...

    def complete(self, action_id: str, outcome: ExecutionOutcome) -> None: ...


@dataclass(slots=True)
class MemoryJournal:
    """An in-process journal. Correct within a process; forgets on exit."""

    _entries: dict[str, JournalEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def lookup(self, action_id: str) -> JournalEntry | None:
        with self._lock:
            return self._entries.get(action_id)

    def start(self, action_id: str, *, connector: str) -> None:
        with self._lock:
            existing = self._entries.get(action_id)
            if existing is not None and existing.completed:
                return
            self._entries[action_id] = JournalEntry(action_id, connector, False, None)

    def complete(self, action_id: str, outcome: ExecutionOutcome) -> None:
        with self._lock:
            existing = self._entries.get(action_id)
            connector = existing.connector if existing else "unknown"
            self._entries[action_id] = JournalEntry(action_id, connector, True, outcome)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_journal (
    action_id  TEXT PRIMARY KEY,
    connector  TEXT NOT NULL,
    completed  INTEGER NOT NULL DEFAULT 0,
    succeeded  INTEGER,
    detail     TEXT,
    reference  TEXT
);
"""


class SqliteJournal:
    """A journal that survives the process — the case a resume actually needs."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def lookup(self, action_id: str) -> JournalEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT connector, completed, succeeded, detail, reference "
                "FROM execution_journal WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        connector, completed, succeeded, detail, reference = row
        outcome = (
            ExecutionOutcome(succeeded=bool(succeeded), detail=detail or "", reference=reference)
            if completed
            else None
        )
        return JournalEntry(action_id, connector, bool(completed), outcome)

    def start(self, action_id: str, *, connector: str) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT INTO execution_journal (action_id, connector, completed) "
                "VALUES (?, ?, 0) ON CONFLICT(action_id) DO NOTHING",
                (action_id, connector),
            )

    def complete(self, action_id: str, outcome: ExecutionOutcome) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT INTO execution_journal "
                "(action_id, connector, completed, succeeded, detail, reference) "
                "VALUES (?, 'unknown', 1, ?, ?, ?) "
                "ON CONFLICT(action_id) DO UPDATE SET completed = 1, "
                "succeeded = excluded.succeeded, detail = excluded.detail, "
                "reference = excluded.reference",
                (action_id, int(outcome.succeeded), outcome.detail, outcome.reference),
            )


@dataclass(slots=True)
class BlastRadiusLimiter:
    """At most ``max_actions`` destructive executions per tenant per ``window``."""

    max_actions: int = 25
    window: timedelta = timedelta(hours=1)
    clock: Clock = field(default_factory=SystemClock)
    _history: dict[str, deque[datetime]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.max_actions < 1:
            raise ValueError("max_actions must be at least 1; to forbid execution, "
                             "remove the connector")
        if self.window <= timedelta(0):
            raise ValueError("window must be positive")

    def _trim(self, tenant_id: str, now: datetime) -> deque[datetime]:
        history = self._history.setdefault(tenant_id, deque())
        while history and now - history[0] >= self.window:
            history.popleft()
        return history

    def remaining(self, tenant_id: str) -> int:
        with self._lock:
            return self.max_actions - len(self._trim(tenant_id, self.clock.now()))

    def acquire(self, tenant_id: str) -> None:
        """Reserve one slot or raise :class:`BlastRadiusExceeded`."""
        with self._lock:
            now = self.clock.now()
            history = self._trim(tenant_id, now)
            if len(history) >= self.max_actions:
                raise BlastRadiusExceeded(
                    f"tenant {tenant_id} has reached {self.max_actions} destructive "
                    f"action(s) in the last {self.window}; refusing more until a human "
                    "reviews what is happening"
                )
            history.append(now)

    def release(self, tenant_id: str) -> None:
        """Return the most recent slot — used when the call never reached the wire."""
        with self._lock:
            history = self._history.get(tenant_id)
            if history:
                history.pop()
