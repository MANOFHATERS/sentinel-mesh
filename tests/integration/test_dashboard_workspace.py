"""The dashboard's live mesh (Part 5): scenarios, decisions, restarts and crashes.

Everything here goes through :class:`~sentinel.dashboard.workspace.Workspace` with
the real agents, the real three graphs, the real router and the live loopback
emulators. What the tests assert is read back from the *remote systems* (the Wazuh
and GitHub emulators) and the audit chain, not from the workspace's own report.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.agents.state import IncidentStatus
from sentinel.core.clock import SimulationClock
from sentinel.core.schemas import ActionType, ApprovalStatus, AuditEventType
from sentinel.core.untrusted import InjectionVerdict
from sentinel.dashboard.scenarios import ScenarioName
from sentinel.dashboard.workspace import (
    Conflict,
    GraphKind,
    NotFound,
    Workspace,
    WorkspaceError,
    scenario_progress,
)

MAYA = "maya@acme.example"


@pytest.fixture
def clock():
    return SimulationClock(datetime(2026, 9, 29, 9, 0, tzinfo=UTC))


@pytest.fixture
def workspace(mesh_models, tmp_path, clock):
    with Workspace(mesh_models, tenant_id="acme", workdir=tmp_path / "acme", clock=clock) as ws:
        yield ws


def pending(ws, thread_id):
    state = ws.state(thread_id)
    assert state.is_waiting, state.status
    return state.pending_action


def decide(ws, thread_id, approved, *, clock=None, who=MAYA, note=""):
    if clock is not None:
        clock.advance(15.0)
    action = pending(ws, thread_id)
    return ws.decide(thread_id, action_id=action.action_id, approved=approved,
                     approver=who, note=note)


class TestScenarioOnePhishingToLateralMovement:
    def test_three_escalated_incidents_each_gated(self, workspace):
        ids = workspace.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)
        assert len(ids) == 3
        expected = [
            (ActionType.ISOLATE_HOST, "10.20.4.17"),
            (ActionType.BLOCK_IP, "10.20.4.17"),
            (ActionType.BLOCK_IP, "10.20.0.5"),
        ]
        for thread_id, (action_type, target) in zip(ids, expected, strict=True):
            state = workspace.state(thread_id)
            assert state.triage.decision.value == "escalate"
            assert state.report is not None and state.report.is_grounded
            action = pending(workspace, thread_id)
            assert (action.action_type, action.target) == (action_type, target)
            assert action.requires_human_approval
        # Nothing reached the remote before anyone approved.
        assert workspace.sandbox.wazuh.executed == []

    def test_the_designed_walkthrough(self, workspace, clock):
        isolate, block_ws, block_jump = workspace.launch(
            ScenarioName.PHISHING_LATERAL, launched_by=MAYA
        )
        decide(workspace, isolate, True, clock=clock)
        decide(workspace, block_ws, False, clock=clock, note="host already isolated")
        final = decide(workspace, block_jump, True, clock=clock)

        # Remote state is exactly the approved-and-permitted set.
        assert workspace.sandbox.wazuh.isolated_hosts() == {"10.20.4.17"}
        assert workspace.sandbox.wazuh.blocked_addresses() == set()

        # The jump-host block was approved, then refused by the router: a FAILED
        # action inside a COMPLETED run, with the reason on the action.
        assert final.status is IncidentStatus.COMPLETED
        failed = final.actions[-1]
        assert failed.approval_status is ApprovalStatus.FAILED
        assert failed.approved_by == MAYA
        assert "protected" in failed.failure_reason
        refusals = workspace.refusal_records()
        assert len(refusals) == 1
        assert refusals[0].subject_id == failed.action_id
        assert "10.20.0.0/28" in refusals[0].payload["reason"]

        progress = scenario_progress(workspace, ScenarioName.PHISHING_LATERAL)
        assert progress.complete
        assert progress.failed_actions == 1 and progress.failed_runs == 0
        assert workspace.ungated() == ()
        assert workspace.verify_audit().ok

    def test_every_decision_names_its_approver_in_the_chain(self, workspace, clock):
        ids = workspace.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)
        for thread_id, approve in zip(ids, (True, False, True), strict=True):
            decide(workspace, thread_id, approve, clock=clock, note=f"n-{approve}")
        rows = [r for r in workspace.records()
                if r.event_type in (AuditEventType.APPROVAL_GRANTED,
                                    AuditEventType.APPROVAL_DENIED)]
        assert len(rows) == 3
        assert {r.payload["approver"] for r in rows} == {MAYA}
        assert [r.payload["note"] for r in rows] == ["n-True", "n-False", "n-True"]


class TestScenarioTwoVendorCve:
    def test_review_and_code_scan_both_gate(self, workspace, mesh_models):
        review, scan = workspace.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)
        assert workspace.ref(review).kind == GraphKind.SUPPLY_CHAIN
        assert workspace.ref(scan).kind == GraphKind.CODE_SCAN
        action = pending(workspace, review)
        assert action.action_type is ActionType.OPEN_PATCH_PR
        assert action.target == mesh_models.cve_package, "the review is about its advisory"
        assert pending(workspace, scan).action_type is ActionType.OPEN_PATCH_PR
        github = workspace.sandbox.github
        assert github.issues == [] and github.pulls == []

    def test_the_fourth_order_path_is_the_evidence(self, workspace, mesh_models):
        review, _scan = workspace.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)
        state = workspace.state(review)
        refs = [e.ref for e in pending(workspace, review).evidence]
        assert any(ref.endswith("#intrinsic") for ref in refs)
        published = workspace.advisory("CVE-2026-41822")
        assert published.advisory.package_id == mesh_models.cve_package
        # Organisations in scope are reached at exactly four hops.
        explained = workspace.explain(published.scope[-1])
        assert state.report.is_grounded
        assert explained.node_id in published.scope
        orgs = [n for n in published.scope if n.startswith("org-")]
        assert orgs
        for org in orgs:
            scoped = workspace.explain(org, advisory_id="CVE-2026-41822")
            assert scoped.paths, org
            assert {p.source for p in scoped.paths} == {mesh_models.cve_package}
            assert min(p.hops for p in scoped.paths) == 4
            # On the whole graph the same organisation is dominated by nearer sources,
            # which is why the advisory view scopes the explanation.
            whole = workspace.explain(org)
            assert whole.risk_score == scoped.risk_score
            assert whole.neighbourhood_share == scoped.neighbourhood_share

    def test_approvals_open_an_issue_and_a_draft_pr_and_never_merge(self, workspace, clock):
        review, scan = workspace.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)
        decide(workspace, review, True, clock=clock)
        decide(workspace, scan, True, clock=clock)
        github = workspace.sandbox.github
        assert len(github.issues) == 1
        assert len(github.pulls) == 1 and github.pulls[0]["draft"] is True
        assert github.merges == 0 and github.pull_edits == 0
        assert scenario_progress(workspace, ScenarioName.VENDOR_CVE).complete
        # The diff the reviewer saw is byte-identical to the one the approval binds.
        _ref, assessment = workspace.code_scan()
        assert assessment.draft is not None

    def test_the_advisory_changes_scores_but_not_the_base_graph(self, workspace, mesh_models):
        before = mesh_models.graph.node(mesh_models.cve_package)
        workspace.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)
        assert mesh_models.graph.node(mesh_models.cve_package) == before
        assert workspace.graph.node(mesh_models.cve_package) != before
        index = workspace.graph.index_of(mesh_models.cve_package)
        assert workspace.scores[index] != pytest.approx(mesh_models.base_scores[index])


class TestScenarioThreeMaliciousPackage:
    def test_injection_is_flagged_and_forced_to_escalate(self, workspace):
        _review, exfil = workspace.launch(ScenarioName.MALICIOUS_PACKAGE, launched_by=MAYA)
        state = workspace.state(exfil)
        assert state.triage.injection_verdict is InjectionVerdict.LIKELY_INJECTION
        assert state.triage.decision.value == "escalate"
        injection_rows = [r for r in workspace.records([state.alert.alert_id])
                          if r.event_type is AuditEventType.INJECTION_DETECTED]
        assert len(injection_rows) == 1
        # The audit row carries a digest, never the attacker's text.
        assert "ignore all previous" not in repr(injection_rows[0].payload).lower()
        action = pending(workspace, exfil)
        assert (action.action_type, action.target) == (ActionType.ISOLATE_HOST, "10.20.8.30")

    def test_walkthrough_isolates_the_runner_and_files_the_issue(self, workspace, clock,
                                                                 mesh_models):
        review, exfil = workspace.launch(ScenarioName.MALICIOUS_PACKAGE, launched_by=MAYA)
        assert pending(workspace, review).target == mesh_models.malicious_package
        decide(workspace, exfil, True, clock=clock)
        decide(workspace, review, True, clock=clock)
        assert workspace.sandbox.wazuh.isolated_hosts() == {"10.20.8.30"}
        assert len(workspace.sandbox.github.issues) == 1
        assert scenario_progress(workspace, ScenarioName.MALICIOUS_PACKAGE).complete


class TestAllThreeTogether:
    def test_launch_order_does_not_change_what_each_review_proposes(
        self, mesh_models, tmp_path, clock
    ):
        targets = {}
        for order in ((ScenarioName.VENDOR_CVE, ScenarioName.MALICIOUS_PACKAGE),
                      (ScenarioName.MALICIOUS_PACKAGE, ScenarioName.VENDOR_CVE)):
            with Workspace(mesh_models, tenant_id="acme", clock=clock,
                           workdir=tmp_path / "-".join(o.value for o in order)) as ws:
                for name in order:
                    ws.launch(name, launched_by=MAYA)
                targets[order] = sorted(
                    ws.state(ref.thread_id).pending_action.target
                    for ref in ws.refs() if ref.kind == GraphKind.SUPPLY_CHAIN
                )
        first, second = targets.values()
        assert first == second == sorted([mesh_models.cve_package,
                                          mesh_models.malicious_package])

    def test_f08_holds_across_every_graph(self, workspace, clock):
        for name in ScenarioName:
            workspace.launch(name, launched_by=MAYA)
        for ref in workspace.refs():
            if workspace.state(ref.thread_id).is_waiting:
                decide(workspace, ref.thread_id, True, clock=clock)
        assert workspace.ungated() == ()
        # No connector touched the wire for a gated action before its approval row.
        granted = {}
        for record in workspace.records():
            if record.event_type is AuditEventType.APPROVAL_GRANTED:
                granted.setdefault(record.subject_id, record.seq)
            elif (record.event_type is AuditEventType.CONNECTOR_CALLED
                  and record.payload.get("requires_human_approval")):
                assert granted.get(record.subject_id, 10**9) < record.seq
        assert all(scenario_progress(workspace, n).complete for n in ScenarioName)
        chain = workspace.verify_audit()
        assert chain.ok and chain.rows_checked > 50


class TestDecisionsAreForWhatWasReviewed:
    def test_a_stale_action_id_is_refused(self, workspace):
        thread_id = workspace.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)[0]
        with pytest.raises(Conflict, match="not the one you reviewed"):
            workspace.decide(thread_id, action_id="some-other-action", approved=True,
                             approver=MAYA)
        assert workspace.state(thread_id).is_waiting
        assert workspace.sandbox.wazuh.executed == []

    def test_a_second_decision_is_a_conflict_not_a_second_execution(self, workspace):
        thread_id = workspace.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)[0]
        action = pending(workspace, thread_id)
        workspace.decide(thread_id, action_id=action.action_id, approved=True, approver=MAYA)
        with pytest.raises(Conflict, match="no pending decision"):
            workspace.decide(thread_id, action_id=action.action_id, approved=True,
                             approver=MAYA)
        assert len(workspace.sandbox.wazuh.executed) == 1

    def test_an_approver_must_be_named(self, workspace):
        thread_id = workspace.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)[0]
        action = pending(workspace, thread_id)
        with pytest.raises(ValueError):
            workspace.decide(thread_id, action_id=action.action_id, approved=True,
                             approver="   ")
        assert workspace.state(thread_id).is_waiting

    def test_a_scenario_launches_once(self, workspace):
        workspace.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)
        with pytest.raises(Conflict, match="already been launched"):
            workspace.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)

    def test_unknown_ids(self, workspace):
        with pytest.raises(NotFound):
            workspace.state("nope")
        with pytest.raises(NotFound):
            workspace.explain("pkg-9999")
        with pytest.raises(NotFound):
            workspace.advisory("CVE-0000-0000")

    def test_recover_is_only_for_stalled_runs(self, workspace):
        thread_id = workspace.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)[0]
        with pytest.raises(Conflict, match="stalled"):
            workspace.recover(thread_id)


class TestBackgroundFeed:
    def test_replay_counts_and_advances(self, workspace):
        before = workspace.feed_remaining()
        outcome = workspace.replay(60)
        assert outcome["ingested"] == 60
        assert sum(outcome[k] for k in ("dismissed", "handled", "gated", "failed")) == 60
        assert outcome["failed"] == 0
        assert workspace.feed_remaining() == before - 60
        states = [workspace.state(r.thread_id) for r in workspace.refs()]
        assert all(s.tenant_id == "acme" for s in states)
        # Re-stamped at ingestion, so MTTD is the pipeline's latency, not the capture's age.
        assert all((s.created_at - s.alert.ingested_at).total_seconds() < 5 for s in states)

    @pytest.mark.parametrize("count", [0, -1, 501])
    def test_replay_bounds(self, workspace, count):
        with pytest.raises(WorkspaceError):
            workspace.replay(count)


class TestTenants:
    def test_two_tenants_share_models_and_nothing_else(self, mesh_models, tmp_path, clock):
        with Workspace(mesh_models, tenant_id="acme", workdir=tmp_path / "a",
                       clock=clock) as acme, Workspace(
            mesh_models, tenant_id="globex", workdir=tmp_path / "g", clock=clock
        ) as globex:
            acme_ids = acme.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)
            globex_ids = globex.launch(ScenarioName.PHISHING_LATERAL, launched_by="ops@globex")
            assert not set(acme_ids) & set(globex_ids), "ids are tenant-derived"
            with pytest.raises(NotFound):
                globex.state(acme_ids[0])
            with pytest.raises(NotFound):
                globex.decide(acme_ids[0], action_id="x", approved=True, approver="ops")
            decide(acme, acme_ids[0], True)
            assert acme.sandbox.wazuh.isolated_hosts() == {"10.20.4.17"}
            assert globex.sandbox.wazuh.executed == []
            assert all(r.tenant_id == "globex" for r in globex.records())


class TestRestart:
    def test_a_waiting_run_survives_a_restart_and_resumes(self, mesh_models, tmp_path, clock):
        workdir = tmp_path / "acme"
        with Workspace(mesh_models, tenant_id="acme", workdir=workdir, clock=clock) as first:
            first.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)
            first.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)
            waiting = {r.thread_id: first.state(r.thread_id).pending_action.action_id
                       for r in first.refs()}
            head = first.audit.head()
        with Workspace(mesh_models, tenant_id="acme", workdir=workdir, clock=clock) as second:
            assert {r.thread_id for r in second.refs()} == set(waiting)
            assert set(second.scenario_runs()) == {ScenarioName.VENDOR_CVE,
                                                   ScenarioName.PHISHING_LATERAL}
            assert second.audit.head() == head
            for thread_id, action_id in waiting.items():
                assert second.state(thread_id).pending_action.action_id == action_id
                ok, count, _ = second.checkpoint_chain_ok(thread_id)
                assert ok and count >= 1
            # The rebuilt review graph resumes to the same outcome.
            review = next(r for r in second.refs() if r.kind == GraphKind.SUPPLY_CHAIN)
            final = decide(second, review.thread_id, True, clock=clock)
            assert final.status is IncidentStatus.COMPLETED
            assert final.actions[-1].approval_status is ApprovalStatus.EXECUTED
            assert len(second.sandbox.github.issues) == 1
            assert second.code_scan() is not None
            with pytest.raises(Conflict):
                second.launch(ScenarioName.VENDOR_CVE, launched_by=MAYA)
            assert second.verify_audit().ok


class _Crash(BaseException):
    """A process death: not an Exception, so no node's error handling catches it."""


class _CrashOnce:
    """Wraps the router; dies right after the first gated call reaches the remote."""

    def __init__(self, inner):
        self.inner = inner
        self.armed = True

    def execute(self, action):
        outcome = self.inner.execute(action)
        if self.armed and action.requires_human_approval:
            self.armed = False
            raise _Crash
        return outcome

    def open_draft(self, action, draft):
        return self.inner.open_draft(action, draft)


class TestCrashRecovery:
    def test_a_crash_after_the_wire_recovers_exactly_once(self, mesh_models, tmp_path, clock):
        workdir = tmp_path / "acme"
        first = Workspace(mesh_models, tenant_id="acme", workdir=workdir, clock=clock,
                          connector_wrapper=_CrashOnce)
        thread_id = first.launch(ScenarioName.PHISHING_LATERAL, launched_by=MAYA)[0]
        action = pending(first, thread_id)
        with pytest.raises(_Crash):
            first.decide(thread_id, action_id=action.action_id, approved=True, approver=MAYA)
        # The remote did act, once, before the "process" died.
        assert first.sandbox.wazuh.isolated_hosts() == {"10.20.4.17"}
        assert len(first.sandbox.wazuh.executed) == 1
        stalled = first.state(thread_id)
        assert stalled.status is IncidentStatus.RUNNING and not stalled.is_waiting
        first.close()

        with Workspace(mesh_models, tenant_id="acme", workdir=workdir, clock=clock) as second:
            recovered = second.recover(thread_id)
            assert recovered.status is IncidentStatus.COMPLETED
            assert recovered.actions[-1].approval_status is ApprovalStatus.EXECUTED
            # The journal replayed the completed call: the new process sent nothing.
            assert second.sandbox.wazuh.executed == []
            replayed = [x for x in second.router.executions
                        if x.action.action_id == action.action_id]
            assert len(replayed) == 1 and replayed[0].outcome.replayed
            assert second.ungated() == ()
            assert second.verify_audit().ok
