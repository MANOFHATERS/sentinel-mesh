"""F-07 end to end: repository -> findings -> CVEs -> gated draft PR -> audit chain.

This drives the real code-scan graph over the real seeded fixture with the real audit
log, so every claim here is about the code path the evaluation script uses. Four
things are asserted that unit tests cannot reach:

*   **F-07's acceptance criterion**, on the fixture, through the graph: three or more
    seeded vulnerabilities detected and a validated patch for each.
*   **F-08 still holds on a second graph.** The Human Approval Gate and the audit
    chain are shared machinery, so a new graph is exactly where "zero actions
    executed without a logged approval" could quietly stop being true.
*   **F-04 on this graph**: interrupt, checkpoint, resume in a *fresh process image*
    over a real SQLite file, with the finished state hash-identical to an
    uninterrupted run.
*   **The engine cannot reach the diff**, driven through the live graph with
    ``HostileEngine`` rather than asserted at the unit boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.agents.checkpoint import SqliteCheckpointer
from sentinel.agents.codescan import (
    NODE_APPROVE,
    NODE_OPEN_PR,
    NODE_SCAN,
    CodeScanAgent,
    DraftPullRequestConnector,
    build_code_scan_graph,
    code_scan_timings,
    new_code_scan_incident,
    synthesize_scan_alert,
)
from sentinel.agents.contain import verify_no_ungated_execution
from sentinel.agents.engine import HostileEngine, NullEngine
from sentinel.agents.state import HumanDecision, IncidentStatus
from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.clock import SimulationClock
from sentinel.core.schemas import ActionType, ApprovalStatus, AuditEventType
from sentinel.kb.retrieve import KnowledgeBase
from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.seeded import FIXTURE_DIR, load_seed_manifest, score_scan

START = datetime(2026, 9, 29, 9, 0, 0, tzinfo=UTC)
F07_MINIMUM = 3


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build()


@pytest.fixture(scope="module")
def snapshot() -> RepoSnapshot:
    return RepoSnapshot.from_dir(FIXTURE_DIR)


def _run(kb, snapshot, tmp_path, *, engine=None, approve=True, checkpointer=None):
    """Drive one scan to completion. Returns everything a test might assert on."""
    clock = SimulationClock(START)
    connector = DraftPullRequestConnector()
    agent = CodeScanAgent(kb=kb, clock=clock, engine=engine or NullEngine())
    graph = build_code_scan_graph(
        agent=agent,
        snapshot=snapshot,
        connector=connector,
        checkpointer=checkpointer,
    )
    log = HashChainedAuditLog(tmp_path / "audit.sqlite", clock=clock)
    alert = synthesize_scan_alert(
        snapshot,
        tenant_id="acme",
        repository="acme/billing",
        at=clock.now(),
        commit="abc123",
    )
    result = graph.invoke(
        new_code_scan_incident(alert, at=clock.now()), clock=clock, audit=log
    )
    interrupted = result.interrupted
    if interrupted:
        clock.advance(25.0)
        result = graph.resume(
            result.state.incident_id,
            HumanDecision(
                approver="dev@acme", approved=approve, decided_at=clock.now()
            ),
            clock=clock,
            audit=log,
        )
    return graph, result, connector, log, interrupted, clock


class TestF07AcceptanceCriterion:
    def test_three_or_more_seeded_vulnerabilities_are_detected(self, snapshot):
        manifest = load_seed_manifest(snapshot)
        result = AstAnalyzer().scan(snapshot)
        score = score_scan(manifest, result.findings)
        assert len(score.detected) >= F07_MINIMUM
        assert not score.missed, score.describe()

    def test_each_one_gets_a_syntactically_valid_patch(self, snapshot):
        import ast

        from sentinel.scan.patch import apply_unified_diff

        result = AstAnalyzer().scan(snapshot)
        assert len(result.valid_patches) >= F07_MINIMUM
        for patch in result.valid_patches:
            ast.parse(patch.after)
            assert apply_unified_diff(patch.before, patch.diff) == patch.after

    def test_the_gated_pr_reaches_the_connector_only_after_approval(
        self, kb, snapshot, tmp_path
    ):
        _g, result, connector, _log, interrupted, _c = _run(kb, snapshot, tmp_path)
        assert interrupted, "OPEN_PATCH_PR is destructive; the gate must fire"
        assert result.state.status is IncidentStatus.COMPLETED
        assert len(connector.opened) == 1
        _action, draft = connector.opened[0]
        assert len(draft.patches) >= F07_MINIMUM

    def test_the_findings_are_mapped_to_cves_in_the_report(self, kb, snapshot, tmp_path):
        from sentinel.core.schemas import EvidenceKind

        _graph, result, _connector, _log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        report = result.state.report
        assert report is not None
        kinds = {item.kind for item in report.evidence}
        assert EvidenceKind.CODE_FINDING in kinds
        assert EvidenceKind.CVE_RECORD in kinds

    def test_every_claim_in_the_report_is_grounded(self, kb, snapshot, tmp_path):
        _graph, result, _connector, _log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        report = result.state.report
        assert report is not None and report.is_grounded
        known = {item.ref for item in report.evidence}
        for _statement, refs in report.claims:
            assert refs and set(refs) <= known

    def test_the_timings_come_from_the_run_history(self, kb, snapshot, tmp_path):
        _graph, result, _connector, _log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        scan_seconds, to_pr = code_scan_timings(result.state)
        assert scan_seconds is not None and scan_seconds >= 0.0
        # The scripted 25s analyst click is inside the time-to-PR, by definition.
        assert to_pr is not None and to_pr >= 25.0


class TestF08OnASecondGraph:
    def test_no_action_executed_without_a_logged_approval(self, kb, snapshot, tmp_path):
        # Read back from the audit chain, not from the ActionRequest objects: those
        # enforce the rule, so asking them whether it held would be circular.
        _graph, _result, _connector, log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        assert verify_no_ungated_execution(log) == ()
        log.close()

    def test_the_audit_chain_verifies(self, kb, snapshot, tmp_path):
        _graph, _result, _connector, log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        assert log.verify().findings == ()
        log.close()

    def test_a_rejected_patch_never_reaches_the_connector(self, kb, snapshot, tmp_path):
        _graph, result, connector, log, interrupted, _clock = _run(
            kb, snapshot, tmp_path, approve=False
        )
        assert interrupted
        assert connector.opened == []
        action = result.state.actions[-1]
        assert action.approval_status is ApprovalStatus.REJECTED
        assert verify_no_ungated_execution(log) == ()
        log.close()

    def test_the_approval_is_recorded_before_the_execution(self, kb, snapshot, tmp_path):
        _graph, _result, _connector, log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        events = [
            (record.seq, record.event_type)
            for record in log.iter_records()
            if record.event_type
            in (AuditEventType.APPROVAL_GRANTED, AuditEventType.ACTION_EXECUTED)
        ]
        granted = next(seq for seq, kind in events if kind is AuditEventType.APPROVAL_GRANTED)
        executed = next(seq for seq, kind in events if kind is AuditEventType.ACTION_EXECUTED)
        # An approval logged *after* the execution is not an approval.
        assert granted < executed
        log.close()

    def test_the_audit_log_never_carries_a_source_line(self, kb, snapshot, tmp_path):
        # The log is exported to a customer's SIEM. Copying an attacker-influenced
        # source line into it would make the tamper-evident record a second delivery
        # channel for whatever was in that line.
        _graph, _result, _connector, log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        for record in log.iter_records():
            if record.event_type is AuditEventType.CODE_SCAN_COMPLETED:
                assert "excerpt" not in record.payload
                assert not any(
                    isinstance(value, str) and "hashlib" in value
                    for value in record.payload.values()
                )
        log.close()

    def test_the_diff_digest_binds_the_approval_to_the_reviewed_bytes(
        self, kb, snapshot, tmp_path
    ):
        import hashlib

        _graph, _result, connector, log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        _action, draft = connector.opened[0]
        expected = hashlib.sha256(draft.combined_diff.encode("utf-8")).hexdigest()
        proposed = [
            record.payload.get("diff_sha256")
            for record in log.iter_records()
            if record.event_type is AuditEventType.ACTION_PROPOSED
        ]
        assert expected in proposed
        log.close()

    def test_the_proposal_is_an_open_patch_pr_and_it_is_destructive(
        self, kb, snapshot, tmp_path
    ):
        _graph, result, _connector, log, _interrupted, _clock = _run(
            kb, snapshot, tmp_path
        )
        action = result.state.actions[-1]
        assert action.action_type is ActionType.OPEN_PATCH_PR
        assert action.action_type.is_destructive
        assert action.requires_human_approval
        log.close()


class TestF04OnASecondGraph:
    def test_the_gate_interrupts_before_the_acting_node_runs(
        self, kb, snapshot, tmp_path
    ):
        clock = SimulationClock(START)
        connector = DraftPullRequestConnector()
        graph = build_code_scan_graph(
            agent=CodeScanAgent(kb=kb, clock=clock),
            snapshot=snapshot,
            connector=connector,
        )
        alert = synthesize_scan_alert(
            snapshot, tenant_id="acme", repository="r", at=clock.now(), commit="c"
        )
        result = graph.invoke(
            new_code_scan_incident(alert, at=clock.now()), clock=clock
        )
        # Interrupted *before* approve, so open_pr has not run.
        assert result.state.visited == (NODE_SCAN,)
        assert result.state.interrupt is not None
        assert result.state.interrupt.node == NODE_APPROVE
        assert connector.opened == []

    def test_resume_across_a_fresh_process_image_is_hash_identical(
        self, kb, snapshot, tmp_path
    ):
        """F-04's criterion on this graph, over a real SQLite file.

        The second half rebuilds the graph, the agent and the checkpointer from
        scratch — a different object graph reading the same file, which is what a
        worker restart actually looks like — and the finished report must hash the
        same as an uninterrupted run's.

        A :class:`FrozenClock` rather than a ``SimulationClock``, because the subject
        is *state* identity: a simulation clock advances by real monotonic time, so two
        runs stamp timestamps microseconds apart and the hashes would differ for a
        reason that has nothing to do with resume fidelity. Forward-moving time is
        covered by :meth:`test_resume_after_a_real_delay_completes`.
        """
        from sentinel.core.clock import FrozenClock
        from sentinel.core.schemas import RiskTier

        db = tmp_path / "checkpoints.sqlite"
        alert = synthesize_scan_alert(
            snapshot,
            tenant_id="acme",
            repository="acme/billing",
            at=START,
            commit="abc123",
        )

        # 1. An uninterrupted reference run, at a tier that permits unattended
        #    execution so no gate fires.
        reference = build_code_scan_graph(
            agent=CodeScanAgent(kb=kb, clock=FrozenClock(START)),
            snapshot=snapshot,
            connector=DraftPullRequestConnector(),
        )
        unattended = reference.invoke(
            new_code_scan_incident(
                alert, at=START, trust_tier=RiskTier.AUTONOMOUS
            ),
            clock=FrozenClock(START),
        )
        assert not unattended.interrupted, "the autonomous tier needs no gate"
        assert unattended.state.status is IncidentStatus.COMPLETED
        assert unattended.state.visited == (NODE_SCAN, NODE_OPEN_PR)

        # 2. A gated run, interrupted and persisted.
        with SqliteCheckpointer(db) as first_store:
            first = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=FrozenClock(START)),
                snapshot=snapshot,
                connector=DraftPullRequestConnector(),
                checkpointer=first_store,
            )
            paused = first.invoke(
                new_code_scan_incident(alert, at=START), clock=FrozenClock(START)
            )
            assert paused.interrupted
            thread = paused.state.incident_id

        # 3. A completely fresh object graph over the same file.
        with SqliteCheckpointer(db) as second_store:
            connector = DraftPullRequestConnector()
            second = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=FrozenClock(START)),
                snapshot=snapshot,
                connector=connector,
                checkpointer=second_store,
            )
            finished = second.resume(
                thread,
                HumanDecision(approver="dev@acme", approved=True, decided_at=START),
                clock=FrozenClock(START),
            )

        assert finished.state.status is IncidentStatus.COMPLETED
        assert finished.state.visited == (NODE_SCAN, NODE_APPROVE, NODE_OPEN_PR)
        # The draft survived the process boundary: the open node re-derived it from the
        # snapshot and verified it against the digest the approval was recorded with.
        assert len(connector.opened) == 1
        assert finished.state.report is not None
        assert unattended.state.report is not None
        assert (
            finished.state.report.canonical_hash()
            == unattended.state.report.canonical_hash()
        )

    def test_resume_after_a_real_delay_completes(self, kb, snapshot, tmp_path):
        """The realistic case: the worker comes back minutes later.

        Worth its own test because the schema refuses an approval timestamped before
        the action was created, so a resume that reconstructs its clock from a fixed
        origin fails the run rather than silently accepting a backwards decision. That
        is the invariant working — an approval that predates the request it approves is
        not an approval — and this asserts the forward-moving path still completes.
        """
        from datetime import timedelta

        db = tmp_path / "delayed.sqlite"
        alert = synthesize_scan_alert(
            snapshot, tenant_id="acme", repository="r", at=START, commit="c"
        )
        with SqliteCheckpointer(db) as store:
            clock = SimulationClock(START)
            graph = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=clock),
                snapshot=snapshot,
                connector=DraftPullRequestConnector(),
                checkpointer=store,
            )
            paused = graph.invoke(
                new_code_scan_incident(alert, at=clock.now()), clock=clock
            )
            thread = paused.state.incident_id
            proposed_at = paused.state.actions[-1].created_at

        with SqliteCheckpointer(db) as store:
            later = SimulationClock(START + timedelta(minutes=5))
            connector = DraftPullRequestConnector()
            graph = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=later),
                snapshot=snapshot,
                connector=connector,
                checkpointer=store,
            )
            finished = graph.resume(
                thread,
                HumanDecision(
                    approver="dev@acme", approved=True, decided_at=later.now()
                ),
                clock=later,
            )

        assert finished.state.status is IncidentStatus.COMPLETED
        assert len(connector.opened) == 1
        action = finished.state.actions[-1]
        assert action.approval_status is ApprovalStatus.EXECUTED
        assert action.decided_at is not None and action.decided_at > proposed_at

    def test_a_backwards_approval_is_refused_rather_than_accepted(
        self, kb, snapshot, tmp_path
    ):
        """The negative half of the above: a backwards clock cannot resume at all.

        The refusal lands earlier than one might expect, and earlier is better.
        ``IncidentState`` validates ``updated_at >= created_at``, so a resume whose
        clock reads before the pause is rejected while *rebuilding the state* — before
        any node runs, before the approval is recorded, and before the connector is
        reachable. A system that instead recorded the decision and only noticed on the
        way out would have an approval in its audit log that predates the request it
        approves.
        """
        from pydantic import ValidationError
        db = tmp_path / "backwards.sqlite"
        alert = synthesize_scan_alert(
            snapshot, tenant_id="acme", repository="r", at=START, commit="c"
        )
        with SqliteCheckpointer(db) as store:
            clock = SimulationClock(START)
            graph = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=clock),
                snapshot=snapshot,
                connector=DraftPullRequestConnector(),
                checkpointer=store,
            )
            paused = graph.invoke(
                new_code_scan_incident(alert, at=clock.now()), clock=clock
            )
            thread = paused.state.incident_id

        with SqliteCheckpointer(db) as store:
            from datetime import timedelta

            backwards = SimulationClock(START - timedelta(hours=1))
            connector = DraftPullRequestConnector()
            graph = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=backwards),
                snapshot=snapshot,
                connector=connector,
                checkpointer=store,
            )
            with pytest.raises(ValidationError, match="precedes created_at"):
                graph.resume(
                    thread,
                    HumanDecision(
                        approver="dev@acme", approved=True, decided_at=backwards.now()
                    ),
                    clock=backwards,
                )
        assert connector.opened == []

    def test_a_tampered_checkpoint_is_refused_on_resume(self, kb, snapshot, tmp_path):
        """An edited checkpoint is refused, not resumed.

        The attack this closes: the trust tier lives in the checkpointed state, and
        ``autonomous`` permits unattended execution. An attacker with write access to
        the checkpoint file who could promote ``recommend`` to ``autonomous`` would
        turn a gated pull request into an unattended one without ever touching the
        approval gate. The checkpoint's ``state_hash`` covers the canonical bytes, so
        the edit is detected on load and the resume refuses.
        """
        import sqlite3

        from sentinel.agents.checkpoint import CheckpointError

        db = tmp_path / "tampered.sqlite"
        alert = synthesize_scan_alert(
            snapshot, tenant_id="acme", repository="r", at=START, commit="c"
        )
        with SqliteCheckpointer(db) as store:
            clock = SimulationClock(START)
            graph = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=clock),
                snapshot=snapshot,
                connector=DraftPullRequestConnector(),
                checkpointer=store,
            )
            paused = graph.invoke(
                new_code_scan_incident(alert, at=clock.now()), clock=clock
            )
            assert paused.interrupted
            thread = paused.state.incident_id

        with sqlite3.connect(db) as connection:
            changed = connection.execute(
                "UPDATE checkpoints SET state_json = REPLACE("
                "state_json, '\"recommend\"', '\"autonomous\"')"
            ).rowcount
        assert changed, "the tamper must actually have changed a row"

        with SqliteCheckpointer(db) as store:
            connector = DraftPullRequestConnector()
            later = SimulationClock(START)
            later.advance(60.0)
            graph = build_code_scan_graph(
                agent=CodeScanAgent(kb=kb, clock=later),
                snapshot=snapshot,
                connector=connector,
                checkpointer=store,
            )
            with pytest.raises(CheckpointError):
                graph.resume(
                    thread,
                    HumanDecision(
                        approver="attacker", approved=True, decided_at=later.now()
                    ),
                    clock=later,
                )
        assert connector.opened == []


class TestHostileEngineThroughTheLiveGraph:
    def test_a_hostile_engine_cannot_change_the_pushed_diff(
        self, kb, snapshot, tmp_path
    ):
        _g1, _r1, safe_connector, log1, _i1, _c1 = _run(
            kb, snapshot, tmp_path / "safe", engine=NullEngine()
        )
        (tmp_path / "hostile").mkdir(parents=True, exist_ok=True)
        _g2, _r2, hostile_connector, log2, _i2, _c2 = _run(
            kb, snapshot, tmp_path / "hostile", engine=HostileEngine()
        )
        assert safe_connector.opened and hostile_connector.opened
        assert (
            safe_connector.opened[0][1].combined_diff
            == hostile_connector.opened[0][1].combined_diff
        )
        log1.close()
        log2.close()

    def test_a_hostile_engine_cannot_get_a_pr_opened_without_approval(
        self, kb, snapshot, tmp_path
    ):
        _graph, _result, connector, log, interrupted, _clock = _run(
            kb, snapshot, tmp_path, engine=HostileEngine(), approve=False
        )
        assert interrupted
        assert connector.opened == []
        assert verify_no_ungated_execution(log) == ()
        log.close()

    def test_a_hostile_engine_cannot_reduce_the_finding_count(
        self, kb, snapshot, tmp_path
    ):
        _g1, safe, _c1, log1, _i1, _cl1 = _run(
            kb, snapshot, tmp_path / "s", engine=NullEngine()
        )
        (tmp_path / "h").mkdir(parents=True, exist_ok=True)
        _g2, hostile, _c2, log2, _i2, _cl2 = _run(
            kb, snapshot, tmp_path / "h", engine=HostileEngine()
        )
        assert safe.state.report is not None and hostile.state.report is not None
        assert len(hostile.state.report.claims) >= len(safe.state.report.claims)
        log1.close()
        log2.close()


class TestFixtureCreatesNoSurprises:
    def test_the_scan_is_clean_against_every_claim_the_fixture_makes(self, snapshot):
        score = score_scan(load_seed_manifest(snapshot), AstAnalyzer().scan(snapshot).findings)
        assert score.clean, score.describe()

    def test_no_patch_was_built_and_then_rejected(self, snapshot):
        result = AstAnalyzer().scan(snapshot)
        assert not result.rejected_patches, [
            (p.rule_id, p.rejection) for p in result.rejected_patches
        ]
