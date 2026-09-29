"""Checkpoint storage for the orchestration graph (PRD F-04, Section 5.3).

F-04's acceptance criterion is *"any node can pause for human input and resume
with full context intact"*. "Intact" is the whole criterion, and it is the part
that is easy to claim and hard to mean, so it is defined here as three properties
that are each tested rather than asserted in prose:

1.  **Complete.** The checkpoint holds the entire :class:`IncidentState`, so a
    resume needs nothing from the writing process's memory. The
    ``test_runtime.py`` suite proves this the only way it can be proven: it
    resumes from a checkpointer that was serialized to bytes and reloaded in a
    fresh object graph, and asserts the finished state is hash-identical to an
    uninterrupted run.

2.  **Verified.** Every checkpoint stores the canonical hash of the state it
    holds, and :meth:`Checkpointer.get` recomputes it on read. A checkpoint that
    was edited in the store is refused rather than resumed. This matters more
    here than for a general workflow engine: the state carries approval status,
    so an unverified checkpoint store is an approval-forgery path that bypasses
    every invariant :mod:`sentinel.core.schemas` enforces at construction time.

3.  **Chained.** Each checkpoint names its parent's hash, so the sequence for a
    thread is itself a hash chain. Deleting an intermediate checkpoint — the
    cheapest way to hide that an approval was asked for and denied — breaks the
    chain and is detected by :meth:`Checkpointer.verify`. This is the same
    construction as :mod:`sentinel.audit.log`, and deliberately so: there is one
    tamper-evidence idea in this system, not two.

``audit_head`` binds a checkpoint to a position in the audit chain. A checkpoint
written at audit head ``(n, h)`` asserts that every event this run had produced
by that point is at or below sequence ``n``. That is what makes "the run resumed
from here" and "the log says this happened" the same claim rather than two
claims that happen to agree.

Two backends, one protocol: :class:`InMemoryCheckpointer` for tests and the
in-process demo, :class:`SqliteCheckpointer` for a run that must survive the
process. The PRD calls for SQLite at this scale (Section 5.6) and the audit log
already brings the dependency.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Protocol, Self, runtime_checkable

from pydantic import Field

from sentinel.agents.state import IncidentState
from sentinel.core.canonical import canonical_bytes, sha256_hex
from sentinel.core.errors import SentinelError
from sentinel.core.schemas import GENESIS_HASH, SHA256_HEX_PATTERN, Contract, UtcDatetime

__all__ = [
    "Checkpoint",
    "CheckpointError",
    "Checkpointer",
    "InMemoryCheckpointer",
    "SqliteCheckpointer",
]

SCHEMA_VERSION: Final[int] = 1


class CheckpointError(SentinelError):
    """A checkpoint could not be written, read back, or verified."""


class Checkpoint(Contract):
    """One saved position in a run.

    ``node`` is the node that will run *next*, not the one that just ran. That is
    the direction resume needs, and naming it after the completed step is the
    off-by-one that makes a resumed run re-execute its last node — which for the
    Containment Agent means proposing the same action twice.
    """

    thread_id: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=0)
    node: str = Field(min_length=1, max_length=64)
    state: IncidentState
    state_hash: str = Field(min_length=64, max_length=64)
    parent_hash: str = Field(min_length=64, max_length=64)
    audit_head_seq: int | None = Field(default=None, ge=0)
    audit_head_hash: str | None = Field(default=None, min_length=64, max_length=64)
    created_at: UtcDatetime

    @classmethod
    def of(
        cls,
        *,
        thread_id: str,
        step: int,
        node: str,
        state: IncidentState,
        parent_hash: str = GENESIS_HASH,
        audit_head: tuple[int, str] | None = None,
    ) -> Checkpoint:
        """Build a checkpoint with its hash derived rather than supplied."""
        return cls(
            thread_id=thread_id,
            step=step,
            node=node,
            state=state,
            state_hash=state.canonical_hash(),
            parent_hash=parent_hash,
            audit_head_seq=None if audit_head is None else audit_head[0],
            audit_head_hash=None if audit_head is None else audit_head[1],
            created_at=state.updated_at,
        )

    @property
    def audit_head(self) -> tuple[int, str] | None:
        if self.audit_head_seq is None or self.audit_head_hash is None:
            return None
        return (self.audit_head_seq, self.audit_head_hash)

    @property
    def link_hash(self) -> str:
        """This checkpoint's position in the thread's chain.

        Covers the parent link, so recomputing it over a store with a deleted
        intermediate checkpoint does not reproduce the stored value.
        """
        return sha256_hex(
            canonical_bytes(
                {
                    "thread_id": self.thread_id,
                    "step": self.step,
                    "node": self.node,
                    "state_hash": self.state_hash,
                    "parent_hash": self.parent_hash,
                    "audit_head_seq": self.audit_head_seq,
                    "audit_head_hash": self.audit_head_hash,
                }
            )
        )

    def verify(self) -> None:
        """Raise if the stored hash does not describe the stored state."""
        recomputed = self.state.canonical_hash()
        if recomputed != self.state_hash:
            raise CheckpointError(
                f"checkpoint {self.thread_id}#{self.step} does not verify: stored hash "
                f"{self.state_hash[:12]}... but the state hashes to {recomputed[:12]}.... "
                "The checkpoint store has been modified since it was written; refusing "
                "to resume from it."
            )


@runtime_checkable
class Checkpointer(Protocol):
    """Where a run's checkpoints live."""

    def put(self, checkpoint: Checkpoint) -> None:
        """Persist ``checkpoint``. Must reject a duplicate ``(thread_id, step)``."""
        ...

    def latest(self, thread_id: str) -> Checkpoint | None:
        """The newest verified checkpoint for ``thread_id``, or ``None``."""
        ...

    def history(self, thread_id: str) -> tuple[Checkpoint, ...]:
        """Every checkpoint for ``thread_id``, oldest first, each verified."""
        ...

    def threads(self) -> tuple[str, ...]:
        """Every thread id with at least one checkpoint, in insertion order."""
        ...


def verify_chain(checkpoints: tuple[Checkpoint, ...]) -> None:
    """Raise unless ``checkpoints`` form an unbroken, correctly linked chain.

    Shared by both backends so there is one definition of what a valid chain is.
    """
    expected_parent = GENESIS_HASH
    for position, checkpoint in enumerate(checkpoints):
        checkpoint.verify()
        if checkpoint.step != position:
            raise CheckpointError(
                f"checkpoint chain for {checkpoint.thread_id!r} jumps from step "
                f"{position} to {checkpoint.step}; a missing checkpoint is how a "
                "denied approval gets erased"
            )
        if checkpoint.parent_hash != expected_parent:
            raise CheckpointError(
                f"checkpoint {checkpoint.thread_id}#{checkpoint.step} claims parent "
                f"{checkpoint.parent_hash[:12]}... but the previous checkpoint links to "
                f"{expected_parent[:12]}..."
            )
        expected_parent = checkpoint.link_hash


class InMemoryCheckpointer:
    """Process-local storage. Thread-safe; loses everything on exit."""

    def __init__(self) -> None:
        self._threads: dict[str, list[Checkpoint]] = {}
        self._lock = threading.RLock()

    def put(self, checkpoint: Checkpoint) -> None:
        checkpoint.verify()
        with self._lock:
            saved = self._threads.setdefault(checkpoint.thread_id, [])
            if any(existing.step == checkpoint.step for existing in saved):
                raise CheckpointError(
                    f"checkpoint {checkpoint.thread_id}#{checkpoint.step} already exists; "
                    "checkpoints are append-only so a resumed run cannot overwrite the "
                    "position it resumed from"
                )
            expected_parent = GENESIS_HASH if not saved else saved[-1].link_hash
            if checkpoint.parent_hash != expected_parent:
                raise CheckpointError(
                    f"checkpoint {checkpoint.thread_id}#{checkpoint.step} does not link to "
                    "the current head of its thread"
                )
            saved.append(checkpoint)

    def latest(self, thread_id: str) -> Checkpoint | None:
        with self._lock:
            saved = self._threads.get(thread_id)
            if not saved:
                return None
            head = saved[-1]
        head.verify()
        return head

    def history(self, thread_id: str) -> tuple[Checkpoint, ...]:
        with self._lock:
            saved = tuple(self._threads.get(thread_id, ()))
        verify_chain(saved)
        return saved

    def threads(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._threads)

    def __len__(self) -> int:
        with self._lock:
            return sum(len(saved) for saved in self._threads.values())


_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id   TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    node        TEXT    NOT NULL,
    state_json  TEXT    NOT NULL,
    state_hash  TEXT    NOT NULL,
    parent_hash TEXT    NOT NULL,
    link_hash   TEXT    NOT NULL,
    audit_seq   INTEGER,
    audit_hash  TEXT,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (thread_id, step)
);
CREATE INDEX IF NOT EXISTS checkpoints_thread ON checkpoints(thread_id, step);
CREATE TABLE IF NOT EXISTS checkpoint_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class SqliteCheckpointer:
    """Durable storage. A run survives the process that started it.

    Uses the same ``PRAGMA synchronous = FULL`` posture as the audit log: a lost
    tail here is a run that silently forgets it was waiting for an approval.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._closed = False
        if str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path), isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.executescript(_DDL)
        self._conn.execute(
            "INSERT OR IGNORE INTO checkpoint_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        stored = self._conn.execute(
            "SELECT value FROM checkpoint_meta WHERE key = 'schema_version'"
        ).fetchone()
        if stored is not None and int(stored["value"]) != SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint store at {self._path} is schema version {stored['value']}, "
                f"this build writes version {SCHEMA_VERSION}"
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise CheckpointError("checkpoint store is closed")

    def put(self, checkpoint: Checkpoint) -> None:
        self._require_open()
        checkpoint.verify()
        payload = json.dumps(
            checkpoint.state.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                head = self._conn.execute(
                    "SELECT step, link_hash FROM checkpoints WHERE thread_id = ? "
                    "ORDER BY step DESC LIMIT 1",
                    (checkpoint.thread_id,),
                ).fetchone()
                expected_parent = GENESIS_HASH if head is None else str(head["link_hash"])
                if checkpoint.parent_hash != expected_parent:
                    raise CheckpointError(
                        f"checkpoint {checkpoint.thread_id}#{checkpoint.step} does not link "
                        "to the current head of its thread"
                    )
                self._conn.execute(
                    "INSERT INTO checkpoints (thread_id, step, node, state_json, state_hash, "
                    "parent_hash, link_hash, audit_seq, audit_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        checkpoint.thread_id,
                        checkpoint.step,
                        checkpoint.node,
                        payload,
                        checkpoint.state_hash,
                        checkpoint.parent_hash,
                        checkpoint.link_hash,
                        checkpoint.audit_head_seq,
                        checkpoint.audit_head_hash,
                        checkpoint.created_at.isoformat(),
                    ),
                )
                self._conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise CheckpointError(
                    f"checkpoint {checkpoint.thread_id}#{checkpoint.step} already exists; "
                    "checkpoints are append-only"
                ) from exc
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def latest(self, thread_id: str) -> Checkpoint | None:
        self._require_open()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        checkpoint = _from_row(row)
        checkpoint.verify()
        return checkpoint

    def history(self, thread_id: str) -> tuple[Checkpoint, ...]:
        self._require_open()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY step ASC",
                (thread_id,),
            ).fetchall()
        chain = tuple(_from_row(row) for row in rows)
        verify_chain(chain)
        return chain

    def threads(self) -> tuple[str, ...]:
        self._require_open()
        with self._lock:
            rows = self._conn.execute(
                "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MIN(rowid)"
            ).fetchall()
        return tuple(str(row["thread_id"]) for row in rows)

    def __len__(self) -> int:
        self._require_open()
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM checkpoints").fetchone()
        return int(row["n"])

    def __iter__(self) -> Iterator[str]:
        return iter(self.threads())


def _from_row(row: sqlite3.Row) -> Checkpoint:
    """Rebuild a checkpoint, reporting the stored hashes rather than recomputing them.

    The stored ``state_hash`` is carried through deliberately: reconstructing it
    from the loaded state would make :meth:`Checkpoint.verify` a tautology, which
    is exactly the mistake that turns tamper-evidence into decoration.
    """
    stored_hash = str(row["state_hash"])
    if not SHA256_HEX_PATTERN.match(stored_hash):
        raise CheckpointError(
            f"checkpoint {row['thread_id']}#{row['step']} stores a malformed state hash"
        )
    try:
        payload: dict[str, Any] = json.loads(str(row["state_json"]))
    except json.JSONDecodeError as exc:
        raise CheckpointError(
            f"checkpoint {row['thread_id']}#{row['step']} holds unparseable state"
        ) from exc
    try:
        state = IncidentState.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError, or anything a validator raises
        raise CheckpointError(
            f"checkpoint {row['thread_id']}#{row['step']} holds a state that no longer "
            f"satisfies its contract: {exc}"
        ) from exc
    audit_seq = row["audit_seq"]
    audit_hash = row["audit_hash"]
    return Checkpoint(
        thread_id=str(row["thread_id"]),
        step=int(row["step"]),
        node=str(row["node"]),
        state=state,
        state_hash=stored_hash,
        parent_hash=str(row["parent_hash"]),
        audit_head_seq=None if audit_seq is None else int(audit_seq),
        audit_head_hash=None if audit_hash is None else str(audit_hash),
        created_at=state.updated_at,
    )
