"""Parts 1-4 end to end: every graph, the real connector layer, live sockets, one log.

``test_all_agents.py`` proved that three graphs share one Human Approval Gate and one
audit chain, with stand-in connectors. This drives the same three graphs through
:class:`~sentinel.connectors.router.ConnectorRouter` in front of the Wazuh, SCIM,
GitHub and Slack connectors, talking HTTP to the sandbox emulators — so every
approved action in this file became real requests on a real socket, and every
assertion about an effect is made against the *remote system's state*:

*   the hosts Wazuh reports isolated are exactly the approved ``isolate_host``
    targets, and the addresses it blocked are exactly the approved ``block_ip`` ones;
*   the pull request exists, is a draft, was never merged, and its files re-scan with
    the patched findings gone;
*   the supply-chain remediation became an issue, not a code change;
*   ``verify_no_ungated_execution`` — unmodified since Part 3 — still returns empty,
    and a stronger check holds: **no connector touched the wire for a gated action
    before its approval row.**

Then the crash test. A process dies after the connector call and before the
execution node returns; a fresh process image recovers the run from its SQLite
checkpoint. With the SQLite execution journal the host is isolated once. The control
repeats it with a fresh in-memory journal and the host is isolated **twice** — which
is the measurement that says the journal is load-bearing, not decorative.

Marked slow: it fits a triage model, a GNN and a knowledge base.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from sentinel.agents.checkpoint import SqliteCheckpointer
from sentinel.agents.codescan import (
    CodeScanAgent,
    build_code_scan_graph,
    new_code_scan_incident,
    synthesize_scan_alert,
)
from sentinel.agents.contain import ContainmentAgent, verify_no_ungated_execution
from sentinel.agents.investigate import InvestigationAgent
from sentinel.agents.orchestrator import build_incident_graph, new_incident
from sentinel.agents.state import HumanDecision, IncidentStatus
from sentinel.agents.supplychain import (
    SupplyChainAgent,
    SupplyChainMonitor,
    build_supply_chain_review_graph,
)
from sentinel.agents.triage import TriageAgent, TriageModel
from sentinel.audit.log import HashChainedAuditLog
from sentinel.connectors.journal import BlastRadiusLimiter, MemoryJournal, SqliteJournal
from sentinel.connectors.sandbox import Sandbox
from sentinel.core.clock import SimulationClock
from sentinel.core.schemas import ActionType, ApprovalStatus, AuditEventType
from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
from sentinel.graph.synthetic import SyntheticGraphGenerator
from sentinel.kb.retrieve import KnowledgeBase
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.ml.metrics import four_way_split
from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.seeded import FIXTURE_DIR

pytestmark = pytest.mark.slow

SEED = 20260928
START = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
MAX_INCIDENTS = 700
TARGET_GATED = 12  # drive incidents until this many have stopped at the gate
REJECT_EVERY = 4  # every fourth gated incident is rejected, so the "no" path is live


@pytest.fixture(scope="module")
def models():
    alerts = generate_alerts(4000, seed=SEED)
    labels = np.array([0 if a.ground_truth_label == "benign" else 1 for a in alerts])
    split = four_way_split(labels, seed=SEED)
    triage_model = TriageModel.fit(
        [alerts[i] for i in split.train_benign],
        train_labelled=[alerts[i] for i in split.train_labelled],
        validation=[alerts[i] for i in split.validation],
        seed=SEED,
    )
    test_alerts = [alerts[i] for i in split.test][:MAX_INCIDENTS]
    graph, truth = SyntheticGraphGenerator(seed=SEED).generate()
    node_ids = graph.node_ids()
    node_labels = truth.labels(node_ids)
    gnn = SupplyChainGNN(random_state=SEED).fit(
        graph, node_labels, GraphSplit.stratified(node_labels, seed=SEED),
        exposure=truth.risk_vector(node_ids),
    )
    return {
        "triage": triage_model,
        "alerts": test_alerts,
        "kb": KnowledgeBase.build(),
        "graph": graph,
        "gnn": gnn,
        "scores": gnn.risk_scores(graph),
        "snapshot": RepoSnapshot.from_dir(FIXTURE_DIR),
        # One router serves one tenant. The synthetic corpus is a single tenant, and
        # routing it through another tenant's router is refused (tested below).
        "tenant": test_alerts[0].tenant_id,
    }


@pytest.fixture(scope="module")
def mesh(models, tmp_path_factory):
    """All three graphs, one sandbox, one router, one audit chain."""
    clock = SimulationClock(START)
    root = tmp_path_factory.mktemp("mesh4")
    log = HashChainedAuditLog(root / "audit.sqlite", clock=clock)
    snapshot = models["snapshot"]
    sandbox = Sandbox(
        repo_files={f.path: f.text for f in snapshot.files},
        hosts=[a.asset_id for a in models["alerts"]],
        clock=clock,
    ).start()
    journal = SqliteJournal(root / "journal.sqlite")
    tenant = models["tenant"]
    router = sandbox.router(
        tenant_id=tenant,
        audit=log,
        journal=journal,
        limiter=BlastRadiusLimiter(max_actions=200, window=timedelta(hours=1), clock=clock),
    )
    kb = models["kb"]

    # --- graph 1: incidents -------------------------------------------------
    incident_graph = build_incident_graph(
        triage=TriageAgent(model=models["triage"], clock=clock),
        investigation=InvestigationAgent(kb=kb, clock=clock),
        containment=ContainmentAgent(clock=clock),
        connector=router,
    )
    counts = {"gated": 0, "approved": 0, "rejected": 0, "completed": 0, "failed": 0,
              "dismissed": 0, "incidents": 0}
    failed_actions: list[tuple[str, str]] = []
    approved_targets: dict[ActionType, set[str]] = {}
    rejected_action_ids: list[str] = []
    for alert in models["alerts"]:
        if counts["gated"] >= TARGET_GATED:
            break
        counts["incidents"] += 1
        run = incident_graph.invoke(new_incident(alert, at=clock.now()), clock=clock, audit=log)
        if run.interrupted:
            counts["gated"] += 1
            approve = counts["gated"] % REJECT_EVERY != 0
            pending = run.state.actions[-1]
            clock.advance(12.0)
            run = incident_graph.resume(
                run.state.incident_id,
                HumanDecision(approver="soc@acme", approved=approve, decided_at=clock.now()),
                clock=clock, audit=log,
            )
            if approve:
                counts["approved"] += 1
                approved_targets.setdefault(pending.action_type, set()).add(pending.target)
            else:
                counts["rejected"] += 1
                rejected_action_ids.append(pending.action_id)
        failed_actions += [
            (a.action_type.value, a.failure_reason or "")
            for a in run.state.actions
            if a.approval_status is ApprovalStatus.FAILED
        ]
        status = run.state.status
        counts["completed"] += status is IncidentStatus.COMPLETED
        counts["failed"] += status is IncidentStatus.FAILED
        counts["dismissed"] += status is IncidentStatus.DISMISSED

    # --- graph 2: a code scan -------------------------------------------------
    code_scan_graph = build_code_scan_graph(
        agent=CodeScanAgent(kb=kb, clock=clock), snapshot=snapshot, connector=router
    )
    scan_alert = synthesize_scan_alert(snapshot, tenant_id=tenant, repository="acme/billing",
                                       at=clock.now(), commit="HEAD")
    draft = CodeScanAgent(kb=kb, clock=clock).assess(
        snapshot, alert=scan_alert, now=clock.now()
    ).draft
    scan_run = code_scan_graph.invoke(new_code_scan_incident(scan_alert, at=clock.now()),
                                      clock=clock, audit=log)
    scan_gated = scan_run.interrupted
    prs_before_approval = len(sandbox.github.pulls)
    clock.advance(25.0)
    scan_run = code_scan_graph.resume(
        scan_run.state.incident_id,
        HumanDecision(approver="dev@acme", approved=True, decided_at=clock.now()),
        clock=clock, audit=log,
    )

    # --- graph 3: a scheduled supply-chain review ------------------------------
    agent = SupplyChainAgent(kb=kb, clock=clock)
    review_graph = build_supply_chain_review_graph(
        agent=agent, graph=models["graph"], model=models["gnn"], scores=models["scores"],
        connector=router,
    )
    monitor = SupplyChainMonitor(agent=agent, graph=models["graph"], model=models["gnn"],
                                 scores=models["scores"], clock=clock)
    vendor_run = monitor.tick(tenant_id=tenant, compiled=review_graph, audit=log,
                              assessment_id="2026-09-29")
    vendor_gated = vendor_run.interrupted
    issues_before_approval = len(sandbox.github.issues)
    if vendor_gated:
        clock.advance(45.0)
        vendor_run = review_graph.resume(
            vendor_run.state.incident_id,
            HumanDecision(approver="vciso@acme", approved=True, decided_at=clock.now()),
            clock=clock, audit=log,
        )

    yield {
        "log": log,
        "records": tuple(log.iter_records()),
        "sandbox": sandbox,
        "router": router,
        "counts": counts,
        "failed_actions": failed_actions,
        "draft": draft,
        "approved_targets": approved_targets,
        "rejected_action_ids": rejected_action_ids,
        "scan_run": scan_run,
        "scan_gated": scan_gated,
        "prs_before_approval": prs_before_approval,
        "vendor_run": vendor_run,
        "vendor_gated": vendor_gated,
        "issues_before_approval": issues_before_approval,
        "snapshot": snapshot,
    }
    journal.close()
    sandbox.stop()
    log.close()


class TestIncidentsReachTheRemoteSystems:
    def test_the_gate_fired_and_both_answers_were_given(self, mesh):
        counts = mesh["counts"]
        assert counts["gated"] == TARGET_GATED, counts
        assert counts["approved"] > 0 and counts["rejected"] > 0
        assert counts["failed"] == 0

    def test_no_action_failed_and_the_router_refused_nothing(self, mesh):
        # A run can finish COMPLETED with its action FAILED, so the incident status
        # alone would hide a connector layer that refused everything.
        assert mesh["failed_actions"] == []
        assert mesh["router"].refusals == []

    def test_wazuh_isolated_exactly_the_approved_hosts(self, mesh):
        expected = mesh["approved_targets"].get(ActionType.ISOLATE_HOST, set())
        assert mesh["sandbox"].wazuh.isolated_hosts() == expected

    def test_wazuh_blocked_exactly_the_approved_addresses(self, mesh):
        expected = mesh["approved_targets"].get(ActionType.BLOCK_IP, set())
        assert expected, "the corpus should produce at least one approved block"
        assert mesh["sandbox"].wazuh.blocked_addresses() == expected

    def test_blocks_ran_on_the_firewall_agent_and_isolations_on_the_host(self, mesh):
        sandbox = mesh["sandbox"]
        for run in sandbox.wazuh.executed:
            if run["command"] == "!firewall-drop":
                assert run["agent"]["id"] == sandbox.firewall_agent
            else:
                assert run["agent"]["ip"] != "10.255.0.1"

    def test_a_rejected_action_never_reached_any_connector(self, mesh):
        rejected = set(mesh["rejected_action_ids"])
        assert rejected
        touched = {
            r.subject_id for r in mesh["records"]
            if r.event_type is AuditEventType.CONNECTOR_CALLED
        }
        assert not (rejected & touched)
        routed = {item.action.action_id for item in mesh["router"].executions}
        assert not (rejected & routed)

    def test_non_destructive_actions_used_their_own_connectors(self, mesh):
        connectors = {item.connector for item in mesh["router"].executions}
        assert "wazuh" in connectors
        assert connectors <= {"wazuh", "slack", "enrichment", "github", "scim"}


class TestTheCodeScanReachedGitHubAsADraft:
    def test_gated_then_opened_one_draft(self, mesh):
        assert mesh["scan_gated"] and mesh["prs_before_approval"] == 0
        assert mesh["scan_run"].state.status is IncidentStatus.COMPLETED
        pulls = mesh["sandbox"].github.pulls
        assert len(pulls) == 1 and pulls[0]["draft"] is True

    def test_nothing_was_merged_or_edited(self, mesh):
        github = mesh["sandbox"].github
        assert github.merges == 0 and github.pull_edits == 0
        assert {r.method for r in github.requests} <= {"GET", "POST"}

    def test_the_pushed_branch_rescans_with_every_patched_finding_gone(self, mesh):
        # The draft carries only patches at or above the agent's minimum severity,
        # so the expectation comes from the draft, not from every validated patch.
        github = mesh["sandbox"].github
        draft = mesh["draft"]
        assert github.pulls[0]["head"]["ref"] == draft.branch
        analyzer = AstAnalyzer()
        for file in mesh["snapshot"].files:
            pushed = github.file_at(draft.branch, file.path)
            ast.parse(pushed)
            before = analyzer.rule_counts(file)
            after = analyzer.rule_counts(file.with_text(pushed))
            patches = [p for p in draft.patches if p.path == file.path]
            for rule_id in {p.rule_id for p in patches}:
                fixed = sum(1 for p in patches if p.rule_id == rule_id)
                assert after.get(rule_id, 0) == before[rule_id] - fixed, (file.path, rule_id)
            assert not {r for r, n in after.items() if n > before.get(r, 0)}, file.path
            assert github.file_at("main", file.path) == file.text

    def test_the_low_severity_finding_was_left_for_a_human(self, mesh):
        # bind-all-interfaces is LOW: validated, but below the draft's threshold.
        draft = mesh["draft"]
        assert "python.bind-all-interfaces" not in {p.rule_id for p in draft.patches}
        assert draft.unpatched_refs

    def test_the_executed_row_carries_the_same_diff_digest_that_was_approved(self, mesh):
        rows = [r for r in mesh["records"] if r.actor == "code_scan_agent"]
        proposed = next(r for r in rows if r.event_type is AuditEventType.ACTION_PROPOSED)
        executed = next(r for r in rows if r.event_type is AuditEventType.ACTION_EXECUTED)
        assert proposed.payload["diff_sha256"] == executed.payload["diff_sha256"]
        assert "draft PR" in executed.payload["detail"]


class TestTenantIsolationEndToEnd:
    def test_another_tenants_router_refuses_the_whole_feed(self, models, tmp_path):
        """The finding that shaped this file: routed as the wrong tenant, nothing ran."""
        clock = SimulationClock(START)
        with HashChainedAuditLog(tmp_path / "a.sqlite", clock=clock) as log, Sandbox(
            hosts=[a.asset_id for a in models["alerts"]], clock=clock
        ) as sandbox:
            router = sandbox.router(tenant_id="some-other-customer", audit=log)
            graph = build_incident_graph(
                triage=TriageAgent(model=models["triage"], clock=clock),
                investigation=InvestigationAgent(kb=models["kb"], clock=clock),
                containment=ContainmentAgent(clock=clock),
                connector=router,
            )
            run = _gated_incident(models, graph, clock, log)
            clock.advance(12.0)
            run = graph.resume(run.state.incident_id,
                               HumanDecision(approver="soc", approved=True,
                                             decided_at=clock.now()),
                               clock=clock, audit=log)
            action = run.state.actions[-1]
            assert action.approval_status is ApprovalStatus.FAILED
            assert "refused an action for tenant" in action.failure_reason
            assert sandbox.wazuh.requests == []
            assert verify_no_ungated_execution(log) == ()


class TestTheSupplyChainReviewBecameAnIssue:
    def test_gated_then_one_issue_and_no_code_change(self, mesh):
        assert mesh["vendor_gated"] and mesh["issues_before_approval"] == 0
        assert mesh["vendor_run"].state.status is IncidentStatus.COMPLETED
        github = mesh["sandbox"].github
        assert len(github.issues) == 1 and len(github.pulls) == 1  # the code-scan PR only
        assert github.issues[0]["labels"] == ["sentinel-mesh"]


class TestF08OnTheWire:
    def test_the_part_3_reader_is_unmodified_and_still_empty(self, mesh):
        assert verify_no_ungated_execution(mesh["log"]) == ()

    def test_no_gated_action_touched_the_wire_before_its_approval(self, mesh):
        granted: dict[str, int] = {}
        for record in mesh["records"]:
            if record.event_type is AuditEventType.APPROVAL_GRANTED:
                granted.setdefault(record.subject_id, record.seq)
            elif (record.event_type is AuditEventType.CONNECTOR_CALLED
                  and record.payload["requires_human_approval"]):
                assert record.subject_id in granted, record.subject_id
                assert granted[record.subject_id] < record.seq
                assert record.payload["approval_status"] == ApprovalStatus.APPROVED.value

    def test_every_executed_action_has_wire_evidence_or_is_local(self, mesh):
        called = {r.subject_id for r in mesh["records"]
                  if r.event_type is AuditEventType.CONNECTOR_CALLED}
        for record in mesh["records"]:
            if record.event_type is not AuditEventType.ACTION_EXECUTED:
                continue
            if record.payload["action_type"] == ActionType.ENRICH_ONLY.value:
                assert record.subject_id not in called  # no side effect, no request
            else:
                assert record.subject_id in called, record.payload["action_type"]

    def test_the_shared_chain_verifies(self, mesh):
        assert mesh["log"].verify().findings == ()

    def test_no_credential_reached_the_log(self, mesh):
        rendered = " ".join(repr(r.payload) for r in mesh["records"])
        for secret in mesh["sandbox"]._secrets.values():
            assert secret not in rendered
        assert "Bearer " not in rendered and "Basic " not in rendered

    def test_no_attacker_controlled_text_reached_the_log(self, mesh):
        needles = ("Pr0d-Postgres-2024!", "fixture-only-billing-key-not-a-real-credential",
                   "hashlib.md5", '" Label"')
        rendered = " ".join(repr(r.payload) for r in mesh["records"])
        for needle in needles:
            assert needle not in rendered, needle


# --------------------------------------------------------------------------- #
# A process death between the connector call and the checkpoint
# --------------------------------------------------------------------------- #


class _Crash(BaseException):
    """Not an Exception: the execution node's handler must not swallow it."""


class _CrashAfterCall:
    """Forwards to the router, then kills the process once, after a gated side effect."""

    def __init__(self, inner):
        self.inner = inner
        self.armed = True

    def execute(self, action):
        outcome = self.inner.execute(action)
        # Only the approved, destructive action dies mid-flight; the notifications
        # executed while looking for a gated incident pass through.
        if self.armed and action.requires_human_approval:
            self.armed = False
            raise _Crash
        return outcome


def _gated_incident(models, graph, clock, log):
    for alert in models["alerts"]:
        run = graph.invoke(new_incident(alert, at=clock.now()), clock=clock, audit=log)
        if run.interrupted:
            return run
    pytest.fail("no incident reached the approval gate")


def _crash_and_recover(models, tmp_path, *, durable_journal: bool):
    clock = SimulationClock(START)
    log = HashChainedAuditLog(tmp_path / "audit.sqlite", clock=clock)
    checkpoints = tmp_path / "checkpoints.sqlite"
    journal_path = tmp_path / "journal.sqlite"
    sandbox = Sandbox(hosts=[a.asset_id for a in models["alerts"]], clock=clock).start()

    def graph_with(connector, store):
        return build_incident_graph(
            triage=TriageAgent(model=models["triage"], clock=clock),
            investigation=InvestigationAgent(kb=models["kb"], clock=clock),
            containment=ContainmentAgent(clock=clock),
            connector=connector,
            checkpointer=store,
        )

    try:
        # --- process 1: approve, execute, die before the node returns --------
        journal = SqliteJournal(journal_path) if durable_journal else MemoryJournal()
        with SqliteCheckpointer(checkpoints) as store:
            router = sandbox.router(tenant_id=models["tenant"], audit=log, journal=journal)
            graph = graph_with(_CrashAfterCall(router), store)
            paused = _gated_incident(models, graph, clock, log)
            incident_id = paused.state.incident_id
            action_id = paused.state.actions[-1].action_id
            clock.advance(12.0)
            with pytest.raises(_Crash):
                graph.resume(incident_id, HumanDecision(approver="soc@acme", approved=True,
                                                        decided_at=clock.now()),
                             clock=clock, audit=log)
        if durable_journal:
            journal.close()
        wazuh_runs_after_crash = len(sandbox.wazuh.executed)

        # --- process 2: fresh objects, the same files ---------------------------
        journal = SqliteJournal(journal_path) if durable_journal else MemoryJournal()
        with SqliteCheckpointer(checkpoints) as store:
            router = sandbox.router(tenant_id=models["tenant"], audit=log, journal=journal)
            graph = graph_with(router, store)
            recovered = graph.recover(incident_id, clock=clock, audit=log)
        if durable_journal:
            journal.close()
        executed_rows = [r for r in log.iter_records()
                         if r.event_type is AuditEventType.ACTION_EXECUTED
                         and r.subject_id == action_id]
        return {
            "recovered": recovered,
            "wazuh_runs_after_crash": wazuh_runs_after_crash,
            "wazuh_runs_total": len(sandbox.wazuh.executed),
            "executed_rows": executed_rows,
            "ungated": verify_no_ungated_execution(log),
            "chain": log.verify(),
        }
    finally:
        sandbox.stop()
        log.close()


class TestExactlyOnceAcrossACrash:
    def test_with_the_durable_journal_the_action_runs_once(self, models, tmp_path):
        result = _crash_and_recover(models, tmp_path, durable_journal=True)
        assert result["recovered"].state.status is IncidentStatus.COMPLETED
        assert result["wazuh_runs_after_crash"] == 1
        assert result["wazuh_runs_total"] == 1  # the recovery replayed, it did not re-run
        assert len(result["executed_rows"]) == 1
        assert "replayed" in result["executed_rows"][0].payload["detail"]
        assert result["ungated"] == () and result["chain"].findings == ()

    def test_control_without_it_the_action_runs_twice(self, models, tmp_path):
        # The measurement that makes the journal load-bearing rather than decorative.
        result = _crash_and_recover(models, tmp_path, durable_journal=False)
        assert result["recovered"].state.status is IncidentStatus.COMPLETED
        assert result["wazuh_runs_total"] == 2
