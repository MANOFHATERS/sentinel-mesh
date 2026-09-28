"""Tamper-evident, append-only audit log (PRD F-11, Section 5.7).

Every agent decision, tool call, model score and human approval lands here. The
acceptance criterion is *"chain-verification detects any injected tampering in
under 1 second"*, which this module meets by streaming a single ordered scan.

Threat model, stated honestly
----------------------------
A plain SHA-256 chain is **tamper-evident against an attacker who cannot
recompute it** — someone who edits a row with a hex editor, a buggy migration, a
corrupted disk, or an insider who does not realise the rows are linked. It is
*not* evidence against an attacker who has write access to the database file and
knows the scheme: they can rewrite every subsequent row's hash and the chain
verifies again. Calling that "immutable" would be theatre.

Three mechanisms close the gap, in increasing order of strength:

1.  **DB-enforced append-only.** ``BEFORE UPDATE`` and ``BEFORE DELETE`` triggers
    ``RAISE(ABORT)``. An ordinary bug or a careless ``UPDATE`` through this
    connection cannot rewrite history — it has to be a deliberate attack on the
    file with the triggers dropped.

2.  **Keyed chaining (HMAC-SHA256).** With ``hmac_key`` set, the row hash is an
    HMAC under a key that lives outside the database — in the sprint, an
    environment variable; in production, a KMS. Rewriting the chain now requires
    the key as well as file write access.

3.  **External anchoring.** :meth:`HashChainedAuditLog.checkpoint` emits the head
    ``(seq, hash)`` pair for publication somewhere the platform cannot reach
    (an object-store WORM bucket, a compliance mailbox, a partner's system).
    Any rewrite of history before an anchored point is then detectable even by an
    attacker-with-the-key, because the anchored hash will not match.

What each mechanism does and does not buy is asserted by tests in
``tests/unit/test_audit_log.py``, including a test that deliberately forges a
full chain to prove mechanism 2 is what stops it.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from json.encoder import encode_basestring as _escape_json_string
from pathlib import Path
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.canonical import canonical_bytes
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import (
    AppendOnlyViolation,
    AuditError,
    CanonicalizationError,
    ChainIntegrityError,
)
from sentinel.core.schemas import GENESIS_HASH, AuditEventType

__all__ = [
    "AuditRecord",
    "Finding",
    "FindingKind",
    "HashChainedAuditLog",
    "VerificationResult",
    "compute_row_hash",
    "row_material",
]

SCHEMA_VERSION: Final[int] = 1

#: Fields that participate in a row's hash, in a fixed order. Adding a column
#: without adding it here would leave that column unprotected, so the list is
#: asserted against the table definition by ``test_hash_covers_every_column``.
HASHED_FIELDS: Final[tuple[str, ...]] = (
    "seq",
    "recorded_at",
    "event_type",
    "actor",
    "tenant_id",
    "subject_id",
    "payload_hash",
    "prev_hash",
)

_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq          INTEGER PRIMARY KEY,
    recorded_at  TEXT    NOT NULL,
    event_type   TEXT    NOT NULL,
    actor        TEXT    NOT NULL,
    tenant_id    TEXT    NOT NULL,
    subject_id   TEXT    NOT NULL,
    payload_json TEXT    NOT NULL,
    payload_hash TEXT    NOT NULL,
    prev_hash    TEXT    NOT NULL,
    row_hash     TEXT    NOT NULL UNIQUE
) STRICT;

CREATE TABLE IF NOT EXISTS audit_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_audit_tenant_seq  ON audit_log(tenant_id, seq);
CREATE INDEX IF NOT EXISTS idx_audit_subject     ON audit_log(subject_id);
CREATE INDEX IF NOT EXISTS idx_audit_event_type  ON audit_log(event_type);

-- Append-only, enforced by the database rather than by convention.
CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE is forbidden');
END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: DELETE is forbidden');
END;
"""


class AuditRecord(BaseModel):
    """One committed row of the chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=1)
    recorded_at: str
    event_type: AuditEventType
    actor: str
    tenant_id: str
    subject_id: str
    payload: dict[str, Any]
    payload_hash: str
    prev_hash: str
    row_hash: str


class FindingKind(StrEnum):
    """Why verification failed. Each maps to a distinct real-world cause."""

    EMPTY = "empty_chain"
    BAD_GENESIS = "bad_genesis"
    SEQ_GAP = "seq_gap"
    SEQ_OUT_OF_ORDER = "seq_out_of_order"
    BROKEN_LINK = "broken_link"
    ROW_HASH_MISMATCH = "row_hash_mismatch"
    PAYLOAD_MISMATCH = "payload_mismatch"
    HEAD_MISMATCH = "head_mismatch"
    TRIGGERS_MISSING = "append_only_triggers_missing"


@dataclass(frozen=True, slots=True)
class Finding:
    """A single integrity problem, located precisely enough to investigate."""

    kind: FindingKind
    seq: int | None
    detail: str

    def __str__(self) -> str:
        where = f"seq={self.seq}" if self.seq is not None else "chain"
        return f"[{self.kind.value}] {where}: {self.detail}"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The outcome of a full chain scan."""

    ok: bool
    rows_checked: int
    head_seq: int | None
    head_hash: str | None
    findings: tuple[Finding, ...] = ()
    duration_seconds: float = 0.0

    def raise_if_broken(self) -> None:
        """Raise :class:`ChainIntegrityError` describing the first problem found."""
        if self.ok:
            return
        first = self.findings[0]
        extra = f" (+{len(self.findings) - 1} more)" if len(self.findings) > 1 else ""
        raise ChainIntegrityError(f"{first}{extra}", seq=first.seq)

    def summary(self) -> str:
        if self.ok:
            return (
                f"OK: {self.rows_checked} rows verified in {self.duration_seconds * 1000:.1f} ms "
                f"(head seq={self.head_seq})"
            )
        detail = "; ".join(str(f) for f in self.findings[:5])
        return (
            f"BROKEN: {len(self.findings)} finding(s) across {self.rows_checked} rows: "
            f"{detail}"
        )


def row_material(
    *,
    seq: int,
    recorded_at: str,
    event_type: str,
    actor: str,
    tenant_id: str,
    subject_id: str,
    payload_hash: str,
    prev_hash: str,
) -> bytes:
    """Canonical bytes for one row, via a fixed-schema fast path.

    Semantically identical to ``canonical_bytes({...})`` over the same mapping.
    The generic canonicalizer walks the structure, type-dispatches every value and
    UTF-8 probes every string for lone surrogates, which is the right behaviour for
    arbitrary payloads but costs ~45 us per row. Profiling a 50,000-row verification
    put 83% of the wall time there and pushed the F-11 sub-second budget to 1.8s on
    a contended machine.

    This row schema is fixed, flat, and entirely ``str`` plus one ``int``, so the
    canonical JSON can be emitted directly with keys pre-sorted. String escaping
    uses :func:`json.encoder.encode_basestring`, which is exactly the encoder
    ``json.dumps(..., ensure_ascii=False)`` uses, so the bytes match rather than
    merely resemble. Lone surrogates still fail, on the final ``encode``, and are
    re-raised as :class:`CanonicalizationError` to match the generic path.

    Any value of an unexpected type falls back to the generic canonicalizer rather
    than being guessed at: a fast path that is *usually* equivalent would be a
    silent forgery generator.
    ``test_fast_path_matches_the_generic_canonicalizer`` asserts equality across
    adversarial inputs, so the two cannot drift.
    """
    fields = (recorded_at, event_type, actor, tenant_id, subject_id, payload_hash, prev_hash)
    if type(seq) is not int or not all(type(field) is str for field in fields):
        return canonical_bytes(
            {
                "seq": seq,
                "recorded_at": recorded_at,
                "event_type": event_type,
                "actor": actor,
                "tenant_id": tenant_id,
                "subject_id": subject_id,
                "payload_hash": payload_hash,
                "prev_hash": prev_hash,
                "schema_version": SCHEMA_VERSION,
            }
        )

    quote = _escape_json_string
    # Keys in codepoint order, exactly as sort_keys=True would emit them.
    text = (
        '{"actor":' + quote(actor)
        + ',"event_type":' + quote(event_type)
        + ',"payload_hash":' + quote(payload_hash)
        + ',"prev_hash":' + quote(prev_hash)
        + ',"recorded_at":' + quote(recorded_at)
        + f',"schema_version":{SCHEMA_VERSION:d}'
        + f',"seq":{seq:d}'
        + ',"subject_id":' + quote(subject_id)
        + ',"tenant_id":' + quote(tenant_id)
        + "}"
    )
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError(
            f"audit row contains un-encodable codepoints ({exc.reason}); decode source "
            "bytes strictly before hashing"
        ) from exc


def compute_row_hash(
    *,
    seq: int,
    recorded_at: str,
    event_type: str,
    actor: str,
    tenant_id: str,
    subject_id: str,
    payload_hash: str,
    prev_hash: str,
    hmac_key: bytes | None = None,
) -> str:
    """Compute a row's hash from exactly the fields in :data:`HASHED_FIELDS`.

    Kept as a free function so a verifier written independently of the writer can
    import it — and so tests can forge rows the way an attacker would, proving the
    keyed mode is what stops them.
    """
    material = row_material(
        seq=seq,
        recorded_at=recorded_at,
        event_type=event_type,
        actor=actor,
        tenant_id=tenant_id,
        subject_id=subject_id,
        payload_hash=payload_hash,
        prev_hash=prev_hash,
    )
    if hmac_key is not None:
        return hmac.new(hmac_key, material, hashlib.sha256).hexdigest()
    return hashlib.sha256(material).hexdigest()


class HashChainedAuditLog:
    """An append-only, hash-chained log over SQLite.

    Usage::

        with HashChainedAuditLog(path) as log:
            log.append(AuditEventType.TRIAGE_DECIDED, actor="triage_agent",
                       tenant_id="acme", subject_id=alert.alert_id,
                       payload={"severity": "high", "confidence": 0.91})
            log.verify().raise_if_broken()

    Thread-safe: appends hold a lock so the read-head/compute/insert sequence is
    atomic. Use one instance per process; SQLite handles cross-process writers
    with WAL, but the ``seq`` allocation would need a shared transaction, so
    cross-process writing is explicitly unsupported and documented rather than
    silently racy.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Clock | None = None,
        hmac_key: bytes | None = None,
    ) -> None:
        if hmac_key is not None and len(hmac_key) < 16:
            raise ValueError("hmac_key must be at least 16 bytes to be worth having")
        self._path = Path(path)
        self._clock = clock or SystemClock()
        self._hmac_key = hmac_key
        self._lock = threading.RLock()
        self._closed = False

        if str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,  # explicit transactions only
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        # Durability over speed: this is the audit log, and a lost tail is a
        # lost approval record.
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_DDL)
        self._conn.execute(
            "INSERT OR IGNORE INTO audit_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO audit_meta(key, value) VALUES ('keyed', ?)",
            ("1" if hmac_key is not None else "0",),
        )
        stored_keyed = self._meta("keyed")
        if stored_keyed != ("1" if hmac_key is not None else "0"):
            raise AuditError(
                f"this log was written in {'keyed' if stored_keyed == '1' else 'unkeyed'} mode "
                f"but was opened in {'keyed' if hmac_key is not None else 'unkeyed'} mode; "
                "every row would fail verification. Open it the way it was written."
            )

    # --- lifecycle ----------------------------------------------------------

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
            raise AuditError("audit log is closed")

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM audit_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    # --- writing ------------------------------------------------------------

    def append(
        self,
        event_type: AuditEventType,
        *,
        actor: str,
        tenant_id: str,
        subject_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        """Append one event and return the committed record.

        ``payload`` is canonicalized before hashing, so a payload containing
        ``NaN`` or an unserializable object is rejected *before* anything is
        written rather than corrupting the chain.
        """
        self._require_open()
        for name, value in (("actor", actor), ("tenant_id", tenant_id), ("subject_id", subject_id)):
            if not value or not value.strip():
                raise ValueError(f"{name} must be a non-empty identifier")

        body: dict[str, Any] = dict(payload or {})
        payload_bytes = canonical_bytes(body)  # raises CanonicalizationError early
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        recorded_at = self._clock.now().strftime("%Y-%m-%dT%H:%M:%S.") + (
            f"{self._clock.now().microsecond:06d}Z"
        )

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                head = self._conn.execute(
                    "SELECT seq, row_hash FROM audit_log ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                seq = 1 if head is None else int(head["seq"]) + 1
                prev_hash = GENESIS_HASH if head is None else str(head["row_hash"])

                row_hash = compute_row_hash(
                    seq=seq,
                    recorded_at=recorded_at,
                    event_type=str(event_type.value),
                    actor=actor,
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    payload_hash=payload_hash,
                    prev_hash=prev_hash,
                    hmac_key=self._hmac_key,
                )
                self._conn.execute(
                    "INSERT INTO audit_log (seq, recorded_at, event_type, actor, tenant_id, "
                    "subject_id, payload_json, payload_hash, prev_hash, row_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        seq,
                        recorded_at,
                        str(event_type.value),
                        actor,
                        tenant_id,
                        subject_id,
                        payload_bytes.decode("utf-8"),
                        payload_hash,
                        prev_hash,
                        row_hash,
                    ),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

        return AuditRecord(
            seq=seq,
            recorded_at=recorded_at,
            event_type=event_type,
            actor=actor,
            tenant_id=tenant_id,
            subject_id=subject_id,
            payload=body,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
            row_hash=row_hash,
        )

    # --- reading ------------------------------------------------------------

    def count(self) -> int:
        self._require_open()
        row = self._conn.execute("SELECT COUNT(*) AS n FROM audit_log").fetchone()
        return int(row["n"])

    def head(self) -> tuple[int, str] | None:
        """The ``(seq, row_hash)`` of the newest row, or ``None`` on an empty log."""
        self._require_open()
        row = self._conn.execute(
            "SELECT seq, row_hash FROM audit_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return None if row is None else (int(row["seq"]), str(row["row_hash"]))

    def checkpoint(self) -> dict[str, Any]:
        """An anchor to publish outside the platform (see module docstring, §3)."""
        head = self.head()
        return {
            "schema_version": SCHEMA_VERSION,
            "keyed": self._hmac_key is not None,
            "head_seq": None if head is None else head[0],
            "head_hash": GENESIS_HASH if head is None else head[1],
            "rows": self.count(),
            "anchored_at": self._clock.now(),
        }

    def read(
        self,
        *,
        since_seq: int = 0,
        limit: int | None = None,
        tenant_id: str | None = None,
        event_type: AuditEventType | None = None,
        subject_id: str | None = None,
    ) -> list[AuditRecord]:
        """Read records in chain order with optional filters."""
        self._require_open()
        clauses = ["seq > ?"]
        params: list[Any] = [since_seq]
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(str(event_type.value))
        if subject_id is not None:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        sql = f"SELECT * FROM audit_log WHERE {' AND '.join(clauses)} ORDER BY seq ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_record_from_row(row) for row in self._conn.execute(sql, params)]

    def iter_records(self, *, batch_size: int = 2048) -> Iterator[AuditRecord]:
        """Stream every record in chain order without materializing the table."""
        self._require_open()
        cursor = self._conn.execute("SELECT * FROM audit_log ORDER BY seq ASC")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                return
            for row in rows:
                yield _record_from_row(row)

    # --- verification -------------------------------------------------------

    def verify(
        self,
        *,
        expected_head: tuple[int, str] | None = None,
        stop_after: int | None = None,
        batch_size: int = 4096,
    ) -> VerificationResult:
        """Recompute the whole chain and report every discrepancy.

        A single ordered scan: each row's payload is re-hashed, its row hash is
        recomputed from :data:`HASHED_FIELDS`, and its ``prev_hash`` is compared to
        the previous row's computed hash. Deletions surface as ``seq`` gaps *and*
        as broken links, so a row cannot be removed quietly even from the middle.

        ``expected_head`` compares against a previously published
        :meth:`checkpoint`, which is what detects a fully-rewritten chain.
        ``stop_after`` bounds the scan for a fast liveness check on a huge log.
        """
        self._require_open()
        start = self._clock.monotonic()
        findings: list[Finding] = []

        if not self._append_only_triggers_present():
            findings.append(
                Finding(
                    FindingKind.TRIGGERS_MISSING,
                    None,
                    "append-only triggers are absent; the file has been altered outside "
                    "this application",
                )
            )

        prev_hash = GENESIS_HASH
        prev_seq = 0
        checked = 0
        head_seq: int | None = None
        head_hash: str | None = None

        cursor = self._conn.execute("SELECT * FROM audit_log ORDER BY seq ASC")
        done = False
        while not done:
            rows: Sequence[sqlite3.Row] = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                seq = int(row["seq"])
                checked += 1

                if seq <= prev_seq:
                    findings.append(
                        Finding(FindingKind.SEQ_OUT_OF_ORDER, seq, f"follows seq={prev_seq}")
                    )
                elif seq != prev_seq + 1:
                    findings.append(
                        Finding(
                            FindingKind.SEQ_GAP,
                            seq,
                            f"expected seq={prev_seq + 1}; {seq - prev_seq - 1} row(s) removed",
                        )
                    )

                payload_json = row["payload_json"]
                stored_payload_hash = row["payload_hash"]
                actual_payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                if actual_payload_hash != stored_payload_hash:
                    findings.append(
                        Finding(
                            FindingKind.PAYLOAD_MISMATCH,
                            seq,
                            "payload_json does not match payload_hash; the record body was edited",
                        )
                    )

                if row["prev_hash"] != prev_hash:
                    findings.append(
                        Finding(
                            FindingKind.BROKEN_LINK,
                            seq,
                            f"prev_hash={str(row['prev_hash'])[:16]}… but previous row hashes to "
                            f"{prev_hash[:16]}…",
                        )
                    )
                    if seq == 1:
                        findings.append(
                            Finding(
                                FindingKind.BAD_GENESIS, 1, "first row must chain to the zero hash"
                            )
                        )

                recomputed = compute_row_hash(
                    seq=seq,
                    recorded_at=row["recorded_at"],
                    event_type=row["event_type"],
                    actor=row["actor"],
                    tenant_id=row["tenant_id"],
                    subject_id=row["subject_id"],
                    payload_hash=stored_payload_hash,
                    prev_hash=row["prev_hash"],
                    hmac_key=self._hmac_key,
                )
                stored = row["row_hash"]
                if not hmac.compare_digest(recomputed, stored):
                    findings.append(
                        Finding(
                            FindingKind.ROW_HASH_MISMATCH,
                            seq,
                            "stored row_hash does not match the recomputed hash of this row's "
                            "own fields",
                        )
                    )

                # Chain forward on the *stored* hash: using the recomputed value
                # would mask a single-row edit as one finding instead of showing
                # the break, and using it after a mismatch would cascade
                # spurious BROKEN_LINK findings down the rest of the log.
                prev_hash = stored
                prev_seq = seq
                head_seq, head_hash = seq, stored

                if stop_after is not None and checked >= stop_after:
                    done = True
                    break

        if checked == 0:
            findings.append(Finding(FindingKind.EMPTY, None, "log contains no rows"))

        if expected_head is not None:
            want_seq, want_hash = expected_head
            if head_seq is None or head_seq < want_seq:
                findings.append(
                    Finding(
                        FindingKind.HEAD_MISMATCH,
                        want_seq,
                        f"anchored checkpoint expects at least seq={want_seq}; log head is "
                        f"{head_seq}. History has been truncated.",
                    )
                )
            else:
                anchored = self._conn.execute(
                    "SELECT row_hash FROM audit_log WHERE seq = ?", (want_seq,)
                ).fetchone()
                if anchored is None or not hmac.compare_digest(
                    str(anchored["row_hash"]), want_hash
                ):
                    findings.append(
                        Finding(
                            FindingKind.HEAD_MISMATCH,
                            want_seq,
                            "row at the anchored seq does not match the published hash; the "
                            "chain was rewritten (this is what an attacker-with-the-key looks "
                            "like)",
                        )
                    )

        duration = self._clock.monotonic() - start
        # Report findings in chain order so the first one is the earliest damage.
        findings.sort(key=lambda f: (f.seq if f.seq is not None else -1, f.kind.value))
        return VerificationResult(
            ok=not findings,
            rows_checked=checked,
            head_seq=head_seq,
            head_hash=head_hash,
            findings=tuple(findings),
            duration_seconds=duration,
        )

    def _append_only_triggers_present(self) -> bool:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'audit_log'"
        ).fetchall()
        names = {str(row["name"]) for row in rows}
        return {"audit_log_no_update", "audit_log_no_delete"} <= names

    # --- deliberate misuse surface (tests and forensics only) ---------------

    def _raw_connection(self) -> sqlite3.Connection:
        """Escape hatch for tests that must simulate tampering. Not for product code."""
        return self._conn

    def try_update(self, seq: int, column: str, value: Any) -> None:
        """Attempt an in-place edit. Always raises :class:`AppendOnlyViolation`.

        Exists so the append-only guarantee is demonstrable rather than claimed.
        """
        if column not in {*HASHED_FIELDS, "payload_json", "row_hash"}:
            raise ValueError(f"unknown audit column {column!r}")
        try:
            self._conn.execute(
                f"UPDATE audit_log SET {column} = ? WHERE seq = ?",
                (value, seq),
            )
        except sqlite3.IntegrityError as exc:
            raise AppendOnlyViolation(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            raise AppendOnlyViolation(str(exc)) from exc
        raise AppendOnlyViolation(
            "UPDATE on audit_log unexpectedly succeeded; the append-only triggers are gone"
        )


def _record_from_row(row: sqlite3.Row) -> AuditRecord:
    import json

    return AuditRecord(
        seq=int(row["seq"]),
        recorded_at=str(row["recorded_at"]),
        event_type=AuditEventType(str(row["event_type"])),
        actor=str(row["actor"]),
        tenant_id=str(row["tenant_id"]),
        subject_id=str(row["subject_id"]),
        payload=json.loads(str(row["payload_json"])),
        payload_hash=str(row["payload_hash"]),
        prev_hash=str(row["prev_hash"]),
        row_hash=str(row["row_hash"]),
    )
