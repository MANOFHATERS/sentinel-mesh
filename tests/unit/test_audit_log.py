"""Audit log: F-11 — tamper detection, append-only enforcement, and honest limits.

The acceptance criterion is *"chain-verification script detects any injected
tampering in under 1 second"*. That is tested three ways: every distinct class of
tampering is detected, the detection is precise about *where*, and the scan is timed
against a log an order of magnitude larger than the demo will produce.

The most important test in this file is
:meth:`TestThreatModel.test_unkeyed_chain_can_be_fully_rewritten`, which proves the
*limit* of the plain SHA-256 chain rather than its strength. A security control whose
boundary is untested is a control nobody can reason about.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
import time

import pytest

from sentinel.audit.log import (
    HASHED_FIELDS,
    SCHEMA_VERSION,
    FindingKind,
    HashChainedAuditLog,
    compute_row_hash,
    row_material,
)
from sentinel.core.canonical import canonical_bytes
from sentinel.core.errors import (
    AppendOnlyViolation,
    AuditError,
    CanonicalizationError,
    ChainIntegrityError,
)
from sentinel.core.schemas import GENESIS_HASH, AuditEventType


def append_n(log: HashChainedAuditLog, n: int, *, tenant: str = "acme") -> None:
    for index in range(n):
        log.append(
            AuditEventType.TRIAGE_DECIDED,
            actor="triage_agent",
            tenant_id=tenant,
            subject_id=f"alert-{index}",
            payload={"severity": "high", "confidence": 0.9, "index": index},
        )


class TestAppendAndRead:
    def test_first_row_chains_to_genesis(self, audit_log):
        record = audit_log.append(
            AuditEventType.ALERT_INGESTED,
            actor="replay_service",
            tenant_id="acme",
            subject_id="alert-0",
        )
        assert record.seq == 1
        assert record.prev_hash == GENESIS_HASH

    def test_sequence_is_dense_and_monotonic(self, audit_log):
        append_n(audit_log, 10)
        assert [r.seq for r in audit_log.read()] == list(range(1, 11))

    def test_each_row_links_to_the_previous(self, audit_log):
        append_n(audit_log, 5)
        records = audit_log.read()
        for previous, current in itertools.pairwise(records):
            assert current.prev_hash == previous.row_hash

    def test_payload_round_trips(self, audit_log):
        payload = {"severity": "critical", "score": 0.75, "fields": ["a", "b"], "ok": True}
        audit_log.append(
            AuditEventType.MODEL_SCORED,
            actor="anomaly_ensemble",
            tenant_id="acme",
            subject_id="alert-1",
            payload=payload,
        )
        assert audit_log.read()[0].payload == payload

    def test_empty_payload_is_allowed(self, audit_log):
        record = audit_log.append(
            AuditEventType.ALERT_INGESTED, actor="x", tenant_id="t", subject_id="s"
        )
        assert record.payload == {}

    def test_head_tracks_the_newest_row(self, audit_log):
        assert audit_log.head() is None
        append_n(audit_log, 3)
        seq, digest = audit_log.head()
        assert seq == 3
        assert digest == audit_log.read()[-1].row_hash

    def test_count(self, audit_log):
        append_n(audit_log, 7)
        assert audit_log.count() == 7

    @pytest.mark.parametrize("field", ["actor", "tenant_id", "subject_id"])
    def test_blank_identity_fields_rejected(self, audit_log, field):
        kwargs = {"actor": "a", "tenant_id": "t", "subject_id": "s", field: "  "}
        with pytest.raises(ValueError, match=field):
            audit_log.append(AuditEventType.ALERT_INGESTED, **kwargs)

    def test_unhashable_payload_is_rejected_before_writing(self, audit_log):
        # A NaN must not reach the table: it would be written, then fail to
        # canonicalize on verification, so the log would be permanently broken.
        with pytest.raises(CanonicalizationError):
            audit_log.append(
                AuditEventType.MODEL_SCORED,
                actor="m",
                tenant_id="t",
                subject_id="s",
                payload={"score": float("nan")},
            )
        assert audit_log.count() == 0

    def test_closed_log_refuses_work(self, tmp_path):
        log = HashChainedAuditLog(tmp_path / "a.sqlite")
        log.close()
        with pytest.raises(AuditError, match="closed"):
            log.append(AuditEventType.ALERT_INGESTED, actor="a", tenant_id="t", subject_id="s")

    def test_close_is_idempotent(self, tmp_path):
        log = HashChainedAuditLog(tmp_path / "a.sqlite")
        log.close()
        log.close()

    def test_reopening_continues_the_chain(self, tmp_path):
        path = tmp_path / "persist.sqlite"
        with HashChainedAuditLog(path) as log:
            append_n(log, 3)
            head = log.head()
        with HashChainedAuditLog(path) as reopened:
            record = reopened.append(
                AuditEventType.APPROVAL_GRANTED,
                actor="analyst",
                tenant_id="acme",
                subject_id="action-1",
            )
            assert record.seq == 4
            assert record.prev_hash == head[1]
            assert reopened.verify().ok


class TestFiltering:
    def test_filter_by_tenant(self, audit_log):
        append_n(audit_log, 3, tenant="acme")
        append_n(audit_log, 2, tenant="globex")
        assert len(audit_log.read(tenant_id="globex")) == 2

    def test_filter_by_event_type(self, audit_log):
        append_n(audit_log, 2)
        audit_log.append(
            AuditEventType.APPROVAL_GRANTED,
            actor="analyst",
            tenant_id="acme",
            subject_id="action-1",
        )
        found = audit_log.read(event_type=AuditEventType.APPROVAL_GRANTED)
        assert len(found) == 1
        assert found[0].actor == "analyst"

    def test_filter_by_subject(self, audit_log):
        append_n(audit_log, 5)
        assert len(audit_log.read(subject_id="alert-3")) == 1

    def test_since_seq_is_exclusive(self, audit_log):
        append_n(audit_log, 5)
        assert [r.seq for r in audit_log.read(since_seq=3)] == [4, 5]

    def test_limit(self, audit_log):
        append_n(audit_log, 10)
        assert len(audit_log.read(limit=4)) == 4

    def test_iter_records_streams_in_order(self, audit_log):
        append_n(audit_log, 50)
        assert [r.seq for r in audit_log.iter_records(batch_size=7)] == list(range(1, 51))


class TestAppendOnlyEnforcement:
    def test_update_is_blocked_by_the_database(self, audit_log):
        append_n(audit_log, 3)
        with pytest.raises(AppendOnlyViolation, match="append-only"):
            audit_log.try_update(2, "actor", "attacker")

    def test_delete_is_blocked_by_the_database(self, audit_log):
        append_n(audit_log, 3)
        connection = audit_log._raw_connection()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM audit_log WHERE seq = 2")

    def test_blocked_update_leaves_the_chain_intact(self, audit_log):
        append_n(audit_log, 3)
        with pytest.raises(AppendOnlyViolation):
            audit_log.try_update(2, "payload_json", '{"tampered":true}')
        assert audit_log.verify().ok

    def test_unknown_column_is_refused(self, audit_log):
        append_n(audit_log, 1)
        with pytest.raises(ValueError, match="unknown audit column"):
            audit_log.try_update(1, "not_a_column", "x")


class TestTamperDetection:
    """Every distinct tampering class, applied by dropping the triggers first.

    Dropping the triggers is what a real attacker with file access would have to do,
    and it is itself reported as a finding.
    """

    @staticmethod
    def _disable_triggers(log: HashChainedAuditLog) -> sqlite3.Connection:
        connection = log._raw_connection()
        connection.execute("DROP TRIGGER audit_log_no_update")
        connection.execute("DROP TRIGGER audit_log_no_delete")
        return connection

    def test_clean_log_verifies(self, audit_log):
        append_n(audit_log, 25)
        result = audit_log.verify()
        assert result.ok
        assert result.rows_checked == 25
        assert "OK" in result.summary()

    def test_edited_payload_is_detected(self, audit_log):
        append_n(audit_log, 10)
        connection = self._disable_triggers(audit_log)
        connection.execute(
            "UPDATE audit_log SET payload_json = ? WHERE seq = 5",
            (json.dumps({"severity": "low"}, separators=(",", ":")),),
        )
        result = audit_log.verify()
        assert not result.ok
        kinds = {f.kind for f in result.findings}
        assert FindingKind.PAYLOAD_MISMATCH in kinds
        assert any(f.seq == 5 for f in result.findings)

    def test_edited_actor_is_detected(self, audit_log):
        # An insider rewriting "who approved this" is the canonical audit attack.
        append_n(audit_log, 10)
        connection = self._disable_triggers(audit_log)
        connection.execute("UPDATE audit_log SET actor = 'someone_else' WHERE seq = 4")
        result = audit_log.verify()
        assert FindingKind.ROW_HASH_MISMATCH in {f.kind for f in result.findings}

    def test_edited_payload_hash_is_detected(self, audit_log):
        # Updating the payload *and* its hash still fails, because the row hash
        # covers payload_hash.
        append_n(audit_log, 10)
        connection = self._disable_triggers(audit_log)
        body = json.dumps({"severity": "low"}, separators=(",", ":"))
        connection.execute(
            "UPDATE audit_log SET payload_json = ?, payload_hash = ? WHERE seq = 6",
            (body, hashlib.sha256(body.encode()).hexdigest()),
        )
        result = audit_log.verify()
        assert FindingKind.ROW_HASH_MISMATCH in {f.kind for f in result.findings}

    def test_deleted_middle_row_is_detected(self, audit_log):
        append_n(audit_log, 10)
        connection = self._disable_triggers(audit_log)
        connection.execute("DELETE FROM audit_log WHERE seq = 5")
        result = audit_log.verify()
        kinds = {f.kind for f in result.findings}
        # Detected twice over: the sequence has a hole and the link is broken.
        assert FindingKind.SEQ_GAP in kinds
        assert FindingKind.BROKEN_LINK in kinds

    def test_deleted_tail_row_is_detected_by_the_anchor(self, audit_log):
        append_n(audit_log, 10)
        anchor = audit_log.checkpoint()
        connection = self._disable_triggers(audit_log)
        connection.execute("DELETE FROM audit_log WHERE seq = 10")

        # A truncated tail leaves a *self-consistent chain*: the remaining rows still
        # link correctly and every row hash still recomputes, so the chain arithmetic
        # alone finds nothing wrong. (The only complaint is that the triggers are
        # gone, which is a statement about the file, not the chain.) That is
        # precisely the gap the published anchor closes.
        chain_only = {
            f.kind
            for f in audit_log.verify().findings
            if f.kind is not FindingKind.TRIGGERS_MISSING
        }
        assert chain_only == set()

        anchored = audit_log.verify(expected_head=(anchor["head_seq"], anchor["head_hash"]))
        assert not anchored.ok
        assert FindingKind.HEAD_MISMATCH in {f.kind for f in anchored.findings}

    def test_broken_link_is_detected(self, audit_log):
        append_n(audit_log, 6)
        connection = self._disable_triggers(audit_log)
        connection.execute("UPDATE audit_log SET prev_hash = ? WHERE seq = 3", ("0" * 64,))
        result = audit_log.verify()
        assert FindingKind.BROKEN_LINK in {f.kind for f in result.findings}

    def test_forged_genesis_is_detected(self, audit_log):
        append_n(audit_log, 3)
        connection = self._disable_triggers(audit_log)
        connection.execute("UPDATE audit_log SET prev_hash = ? WHERE seq = 1", ("f" * 64,))
        result = audit_log.verify()
        assert FindingKind.BAD_GENESIS in {f.kind for f in result.findings}

    def test_reordered_rows_are_detected(self, audit_log):
        append_n(audit_log, 6)
        connection = self._disable_triggers(audit_log)
        # Swap two rows' sequence numbers via a temporary value.
        connection.execute("UPDATE audit_log SET seq = 999 WHERE seq = 3")
        connection.execute("UPDATE audit_log SET seq = 3 WHERE seq = 4")
        connection.execute("UPDATE audit_log SET seq = 4 WHERE seq = 999")
        result = audit_log.verify()
        assert not result.ok

    def test_missing_triggers_are_themselves_a_finding(self, audit_log):
        append_n(audit_log, 3)
        self._disable_triggers(audit_log)
        result = audit_log.verify()
        assert FindingKind.TRIGGERS_MISSING in {f.kind for f in result.findings}

    def test_empty_log_is_reported(self, audit_log):
        result = audit_log.verify()
        assert not result.ok
        assert FindingKind.EMPTY in {f.kind for f in result.findings}

    def test_findings_are_reported_earliest_first(self, audit_log):
        append_n(audit_log, 10)
        connection = self._disable_triggers(audit_log)
        connection.execute("UPDATE audit_log SET actor = 'x' WHERE seq = 8")
        connection.execute("UPDATE audit_log SET actor = 'y' WHERE seq = 3")
        result = audit_log.verify()
        located = [f.seq for f in result.findings if f.seq is not None]
        assert located == sorted(located)

    def test_findings_locate_the_tampered_row(self, audit_log):
        append_n(audit_log, 5)
        connection = self._disable_triggers(audit_log)
        connection.execute("UPDATE audit_log SET actor = 'x' WHERE seq = 2")
        result = audit_log.verify()
        row_findings = [f for f in result.findings if f.kind is FindingKind.ROW_HASH_MISMATCH]
        assert [f.seq for f in row_findings] == [2]

    def test_raise_if_broken_raises_and_counts_the_rest(self, audit_log):
        append_n(audit_log, 5)
        connection = self._disable_triggers(audit_log)
        connection.execute("UPDATE audit_log SET actor = 'x' WHERE seq = 2")
        with pytest.raises(ChainIntegrityError, match=r"\+1 more"):
            audit_log.verify().raise_if_broken()

    def test_raise_if_broken_is_silent_on_a_clean_log(self, audit_log):
        append_n(audit_log, 3)
        audit_log.verify().raise_if_broken()

    def test_summary_describes_the_breakage(self, audit_log):
        append_n(audit_log, 5)
        connection = self._disable_triggers(audit_log)
        connection.execute("UPDATE audit_log SET actor = 'x' WHERE seq = 2")
        assert "BROKEN" in audit_log.verify().summary()


class TestFastPathEquivalence:
    """The fixed-schema serializer must be byte-identical to the generic one.

    ``row_material`` emits canonical JSON directly instead of walking the generic
    canonicalizer. That took 50,000-row verification from 2.9s to 0.41s and brought
    the F-11 budget back inside its bound with headroom — it had reached 1.8s on a
    contended machine. A fast path that is merely *usually* equivalent would be a
    silent forgery generator, so equality is asserted over adversarial strings rather
    than assumed from inspection.
    """

    #: Strings chosen to break naive hand-rolled JSON escaping: quotes, backslashes,
    #: control characters, combining marks, astral-plane codepoints, the bidi
    #: override, a zero-width space, JSON punctuation and a non-breaking space.
    NASTY: tuple[str, ...] = (
        "",
        "a",
        'quote"x',
        "back\\slash",
        "new\nline",
        "tab\tx",
        "ctrl\x01\x1f",
        "emoji\U0001f600",
        "café",
        "é",
        "‮bidi",
        "​zero-width",
        "x" * 300,
        "{}[]:,",
        " nbsp",
        "ünïcödé",
        "中文",
        "\U0001f6e1️",
    )

    BASE: dict[str, object] = {
        "seq": 1,
        "recorded_at": "2026-01-01T00:00:00.000000Z",
        "event_type": "triage_decided",
        "actor": "triage_agent",
        "tenant_id": "acme",
        "subject_id": "alert-1",
        "payload_hash": "0" * 64,
        "prev_hash": "f" * 64,
    }

    @staticmethod
    def _generic(**kwargs: object) -> bytes:
        return canonical_bytes({**kwargs, "schema_version": SCHEMA_VERSION})

    def test_random_combinations_match(self):
        import random

        rng = random.Random(20260928)
        mismatches = []
        for index in range(3000):
            kwargs: dict[str, object] = {"seq": rng.randint(0, 10**9)}
            for field in (
                "recorded_at",
                "event_type",
                "actor",
                "tenant_id",
                "subject_id",
                "payload_hash",
                "prev_hash",
            ):
                kwargs[field] = rng.choice(self.NASTY)
            if row_material(**kwargs) != self._generic(**kwargs):
                mismatches.append((index, kwargs))
        assert mismatches == [], f"{len(mismatches)} divergences; first: {mismatches[:1]}"

    def test_every_nasty_string_in_every_field_matches(self):
        """Exhaustive per-position coverage, so no single field can hide a bug."""
        string_fields = [key for key in self.BASE if key != "seq"]
        for field, value in itertools.product(string_fields, self.NASTY):
            kwargs = {**self.BASE, field: value}
            assert row_material(**kwargs) == self._generic(**kwargs), (field, repr(value))

    @pytest.mark.parametrize("seq", [0, 1, 7, 10**6, 2**62])
    def test_integer_seq_values_match(self, seq):
        kwargs = {**self.BASE, "seq": seq}
        assert row_material(**kwargs) == self._generic(**kwargs)

    def test_unexpected_type_falls_back_to_the_generic_path(self):
        # A float seq must not be emitted as an int. Falling back keeps the bytes
        # correct even for inputs the schema does not expect.
        kwargs = {**self.BASE, "seq": 1.0}
        assert row_material(**kwargs) == self._generic(**kwargs)
        assert b'"seq":1.0' in row_material(**kwargs)

    def test_bool_seq_is_not_silently_an_int(self):
        # `type(x) is int` rather than isinstance, so True does not become 1.
        kwargs = {**self.BASE, "seq": True}
        assert row_material(**kwargs) == self._generic(**kwargs)

    def test_non_string_field_falls_back(self):
        kwargs = {**self.BASE, "actor": 42}
        assert row_material(**kwargs) == self._generic(**kwargs)

    def test_lone_surrogate_is_rejected_like_the_generic_path(self):
        kwargs = {**self.BASE, "actor": "bad" + chr(0xD800) + "actor"}
        with pytest.raises(CanonicalizationError):
            row_material(**kwargs)
        with pytest.raises(CanonicalizationError):
            self._generic(**kwargs)

    def test_keys_are_emitted_in_codepoint_order(self):
        material = row_material(**self.BASE).decode()
        positions = [
            (material.index(f'"{key}":'), key)
            for key in (
                "actor",
                "event_type",
                "payload_hash",
                "prev_hash",
                "recorded_at",
                "schema_version",
                "seq",
                "subject_id",
                "tenant_id",
            )
        ]
        assert [key for _, key in sorted(positions)] == sorted(key for _, key in positions)

    def test_hash_matches_the_generic_serialization(self):
        """A chain written before the fast path existed must still verify.

        If the canonical form of a row ever changes, every previously written audit
        log stops verifying — which must be a deliberate ``SCHEMA_VERSION`` bump
        rather than an accident.
        """
        digest = compute_row_hash(**self.BASE)
        assert digest == hashlib.sha256(self._generic(**self.BASE)).hexdigest()
        assert len(digest) == 64


class TestThreatModel:
    """What the plain chain does and does not prove. Both directions matter."""

    @staticmethod
    def _forge_whole_chain(log: HashChainedAuditLog, *, tamper_seq: int = 3) -> None:
        """Rewrite every row's hash as an attacker with file access would."""
        connection = log._raw_connection()
        connection.execute("DROP TRIGGER audit_log_no_update")
        connection.execute("DROP TRIGGER audit_log_no_delete")

        rows = list(connection.execute("SELECT * FROM audit_log ORDER BY seq"))
        forged_body = json.dumps({"severity": "low"}, separators=(",", ":"))
        forged_payload_hash = hashlib.sha256(forged_body.encode()).hexdigest()
        prev = GENESIS_HASH
        for row in rows:
            tampered = row["seq"] == tamper_seq
            payload_hash = forged_payload_hash if tampered else row["payload_hash"]
            payload_json = forged_body if tampered else row["payload_json"]
            # Recomputed with no key — the best an attacker without it can do.
            row_hash = compute_row_hash(
                seq=row["seq"],
                recorded_at=row["recorded_at"],
                event_type=row["event_type"],
                actor=row["actor"],
                tenant_id=row["tenant_id"],
                subject_id=row["subject_id"],
                payload_hash=payload_hash,
                prev_hash=prev,
                hmac_key=None,
            )
            connection.execute(
                "UPDATE audit_log SET payload_json=?, payload_hash=?, prev_hash=?, row_hash=? "
                "WHERE seq=?",
                (payload_json, payload_hash, prev, row_hash, row["seq"]),
            )
            prev = row_hash

    def test_unkeyed_chain_can_be_fully_rewritten(self, audit_log):
        """The honest limit: an attacker who knows the scheme can forge the chain.

        This is why the module documents keyed mode and external anchoring rather
        than calling a SHA-256 chain "immutable". An untested boundary is a boundary
        nobody can reason about.
        """
        append_n(audit_log, 5)
        self._forge_whole_chain(audit_log)
        result = audit_log.verify()
        # The only surviving evidence is that the triggers are gone.
        assert {f.kind for f in result.findings} == {FindingKind.TRIGGERS_MISSING}

    def test_keyed_chain_resists_the_same_rewrite(self, keyed_audit_log):
        """Mechanism 2: without the key, the identical forgery does not verify."""
        append_n(keyed_audit_log, 5)
        self._forge_whole_chain(keyed_audit_log)
        result = keyed_audit_log.verify()
        assert not result.ok
        assert FindingKind.ROW_HASH_MISMATCH in {f.kind for f in result.findings}

    def test_keyed_and_unkeyed_hashes_differ(self):
        args = {
            "seq": 1,
            "recorded_at": "2026-01-01T00:00:00.000000Z",
            "event_type": "triage_decided",
            "actor": "a",
            "tenant_id": "t",
            "subject_id": "s",
            "payload_hash": "0" * 64,
            "prev_hash": GENESIS_HASH,
        }
        assert compute_row_hash(**args) != compute_row_hash(**args, hmac_key=b"k" * 16)

    def test_opening_a_keyed_log_unkeyed_is_refused(self, tmp_path):
        path = tmp_path / "keyed.sqlite"
        with HashChainedAuditLog(path, hmac_key=b"k" * 16) as log:
            append_n(log, 2)
        with pytest.raises(AuditError, match="keyed"):
            HashChainedAuditLog(path)

    def test_opening_an_unkeyed_log_keyed_is_refused(self, tmp_path):
        path = tmp_path / "plain.sqlite"
        with HashChainedAuditLog(path) as log:
            append_n(log, 2)
        with pytest.raises(AuditError, match="unkeyed"):
            HashChainedAuditLog(path, hmac_key=b"k" * 16)

    def test_short_hmac_key_refused(self, tmp_path):
        with pytest.raises(ValueError, match="at least 16 bytes"):
            HashChainedAuditLog(tmp_path / "x.sqlite", hmac_key=b"short")

    def test_keyed_log_verifies_normally(self, keyed_audit_log):
        append_n(keyed_audit_log, 20)
        assert keyed_audit_log.verify().ok


class TestCheckpointing:
    def test_checkpoint_reports_the_head(self, audit_log):
        append_n(audit_log, 4)
        checkpoint = audit_log.checkpoint()
        assert checkpoint["head_seq"] == 4
        assert checkpoint["rows"] == 4
        assert checkpoint["keyed"] is False

    def test_empty_checkpoint_uses_the_genesis_hash(self, audit_log):
        checkpoint = audit_log.checkpoint()
        assert checkpoint["head_seq"] is None
        assert checkpoint["head_hash"] == GENESIS_HASH

    def test_matching_anchor_verifies(self, audit_log):
        append_n(audit_log, 5)
        checkpoint = audit_log.checkpoint()
        append_n(audit_log, 3)
        result = audit_log.verify(
            expected_head=(checkpoint["head_seq"], checkpoint["head_hash"])
        )
        assert result.ok

    def test_wrong_anchor_hash_is_detected(self, audit_log):
        append_n(audit_log, 5)
        result = audit_log.verify(expected_head=(3, "a" * 64))
        assert FindingKind.HEAD_MISMATCH in {f.kind for f in result.findings}


class TestPerformanceAndStructure:
    def test_hash_covers_every_column_that_matters(self, audit_log):
        # If a column is added to the table but not to HASHED_FIELDS, it would be
        # freely editable without breaking the chain.
        append_n(audit_log, 1)
        columns = {
            row[1]
            for row in audit_log._raw_connection().execute("PRAGMA table_info(audit_log)")
        }
        unhashed = columns - set(HASHED_FIELDS) - {"payload_json", "row_hash"}
        assert unhashed == set(), f"columns outside the hash: {unhashed}"

    def test_payload_json_is_covered_transitively(self, audit_log):
        # payload_json is not hashed directly, but payload_hash is, and payload_hash
        # is verified against payload_json — so editing the body is still detected.
        append_n(audit_log, 1)
        connection = audit_log._raw_connection()
        connection.execute("DROP TRIGGER audit_log_no_update")
        connection.execute("UPDATE audit_log SET payload_json = '{}' WHERE seq = 1")
        assert FindingKind.PAYLOAD_MISMATCH in {f.kind for f in audit_log.verify().findings}

    @pytest.mark.slow
    def test_f11_verification_of_50k_rows_under_one_second(self, tmp_path):
        with HashChainedAuditLog(tmp_path / "big.sqlite") as log:
            # synchronous=FULL makes 50k individual commits slow, which is the right
            # trade for a log. Relax it while building the fixture only; the chain
            # itself is written and verified exactly as in production.
            log._raw_connection().execute("PRAGMA synchronous = OFF")
            append_n(log, 50_000)
            started = time.perf_counter()
            result = log.verify()
            elapsed = time.perf_counter() - started
        assert result.ok
        assert result.rows_checked == 50_000
        assert elapsed < 1.0, f"F-11 requires sub-second verification; took {elapsed:.3f}s"

    def test_stop_after_bounds_the_scan(self, audit_log):
        append_n(audit_log, 100)
        assert audit_log.verify(stop_after=10).rows_checked == 10

    def test_concurrent_appends_produce_a_valid_chain(self, tmp_path):
        import threading

        # Appends must be atomic across the read-head/compute/insert sequence, or two
        # threads produce duplicate seq values and the chain forks.
        with HashChainedAuditLog(tmp_path / "threads.sqlite") as log:

            def worker(worker_id: int) -> None:
                for index in range(40):
                    log.append(
                        AuditEventType.MODEL_SCORED,
                        actor=f"worker-{worker_id}",
                        tenant_id="acme",
                        subject_id=f"alert-{worker_id}-{index}",
                        payload={"i": index},
                    )

            threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert log.count() == 160
            assert log.verify().ok
            assert [r.seq for r in log.read()] == list(range(1, 161))
