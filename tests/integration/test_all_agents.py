"""All five agents, three graphs, one audit chain: the whole of Parts 1-3 at once.

Every other test in this suite verifies one layer or one criterion. This one exists to
verify the claim that ties them together, which no single-layer test can reach:

    Three graphs share one state machine, one Human Approval Gate and one audit chain,
    so **F-08 is implemented once and therefore checked once**.

That claim is what makes adding a graph cheap and safe. The way it could quietly stop
being true is for a new graph to grow its own approval path — and the symptom would not
be a failing test, it would be a second implementation of the guarantee with no tests of
its own. So this drives all three graphs into a *single* ``HashChainedAuditLog`` and then
asks ``verify_no_ungated_execution`` — one function, unmodified — whether the rule held
across all of them.

The three triggers are deliberately all different, because that is the reason there are
three graphs rather than one:

*   an **alert** arrives from the feed (PRD Figure 3),
*   a **commit** is pushed (F-07),
*   a **schedule** fires with nothing having happened at all (F-06).

Marked slow: it fits four models and drives sixty incidents.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from sentinel.agents.codescan import (
    CodeScanAgent,
    DraftPullRequestConnector,
    build_code_scan_graph,
    new_code_scan_incident,
    synthesize_scan_alert,
)
from sentinel.agents.contain import (
    ContainmentAgent,
    SimulatedConnector,
    verify_no_ungated_execution,
)
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
from sentinel.core.clock import SimulationClock
from sentinel.core.schemas import AgentName, AuditEventType
from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
from sentinel.graph.synthetic import SyntheticGraphGenerator
from sentinel.kb.retrieve import KnowledgeBase
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.ml.metrics import four_way_split
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.seeded import FIXTURE_DIR

pytestmark = pytest.mark.slow

SEED = 20260928
START = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
N_INCIDENTS = 60


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    """Drive all three graphs through one shared audit log.

    Module-scoped: this fits a triage model, a GNN and a knowledge base, and the whole
    point is that the *same log* sees all three runs, so splitting it per test would
    defeat the thing being asserted.
    """
    clock = SimulationClock(START)
    kb = KnowledgeBase.build()
    log_path = tmp_path_factory.mktemp("mesh") / "shared-audit.sqlite"
    log = HashChainedAuditLog(log_path, clock=clock)

    # --- graph 1: an incident. The trigger is an alert. --------------------
    alerts = generate_alerts(4000, seed=SEED)
    labels = np.array(
        [0 if alert.ground_truth_label == "benign" else 1 for alert in alerts]
    )
    split = four_way_split(labels, seed=SEED)
    triage_model = TriageModel.fit(
        [alerts[i] for i in split.train_benign],
        train_labelled=[alerts[i] for i in split.train_labelled],
        validation=[alerts[i] for i in split.validation],
        seed=SEED,
    )
    edr = SimulatedConnector()
    incident_graph = build_incident_graph(
        triage=TriageAgent(model=triage_model, clock=clock),
        investigation=InvestigationAgent(kb=kb, clock=clock),
        containment=ContainmentAgent(clock=clock),
        connector=edr,
    )
    incidents = {"gated": 0, "dismissed": 0, "completed": 0, "failed": 0}
    for alert in [alerts[i] for i in split.test][:N_INCIDENTS]:
        run = incident_graph.invoke(
            new_incident(alert, at=clock.now()), clock=clock, audit=log
        )
        if run.state.status is IncidentStatus.DISMISSED:
            incidents["dismissed"] += 1
        if run.state.status is IncidentStatus.FAILED:
            incidents["failed"] += 1
        if run.interrupted:
            incidents["gated"] += 1
            clock.advance(12.0)
            run = incident_graph.resume(
                run.state.incident_id,
                HumanDecision(
                    approver="soc@acme", approved=True, decided_at=clock.now()
                ),
                clock=clock,
                audit=log,
            )
        if run.state.status is IncidentStatus.COMPLETED:
            incidents["completed"] += 1

    # --- graph 2: a code scan. The trigger is a commit. --------------------
    snapshot = RepoSnapshot.from_dir(FIXTURE_DIR)
    pull_requests = DraftPullRequestConnector()
    code_scan_graph = build_code_scan_graph(
        agent=CodeScanAgent(kb=kb, clock=clock),
        snapshot=snapshot,
        connector=pull_requests,
    )
    scan_alert = synthesize_scan_alert(
        snapshot,
        tenant_id="acme",
        repository="acme/billing",
        at=clock.now(),
        commit="HEAD",
    )
    scan_run = code_scan_graph.invoke(
        new_code_scan_incident(scan_alert, at=clock.now()), clock=clock, audit=log
    )
    scan_gated = scan_run.interrupted
    if scan_gated:
        clock.advance(25.0)
        scan_run = code_scan_graph.resume(
            scan_run.state.incident_id,
            HumanDecision(approver="dev@acme", approved=True, decided_at=clock.now()),
            clock=clock,
            audit=log,
        )

    # --- graph 3: an assessment. The trigger is a schedule. ----------------
    graph, truth = SyntheticGraphGenerator(seed=SEED).generate()
    node_ids = graph.node_ids()
    node_labels = truth.labels(node_ids)
    gnn = SupplyChainGNN(random_state=SEED).fit(
        graph,
        node_labels,
        GraphSplit.stratified(node_labels, seed=SEED),
        exposure=truth.risk_vector(node_ids),
    )
    scores = gnn.risk_scores(graph)
    vendor_connector = SimulatedConnector()
    supply_agent = SupplyChainAgent(kb=kb, clock=clock)
    review_graph = build_supply_chain_review_graph(
        agent=supply_agent,
        graph=graph,
        model=gnn,
        scores=scores,
        connector=vendor_connector,
    )
    monitor = SupplyChainMonitor(
        agent=supply_agent, graph=graph, model=gnn, scores=scores, clock=clock
    )
    vendor_run = monitor.tick(
        tenant_id="acme", compiled=review_graph, audit=log, assessment_id="2026-09-29"
    )
    vendor_gated = vendor_run.interrupted
    if vendor_gated:
        clock.advance(45.0)
        vendor_run = review_graph.resume(
            vendor_run.state.incident_id,
            HumanDecision(approver="vciso@acme", approved=True, decided_at=clock.now()),
            clock=clock,
            audit=log,
        )

    yield {
        "log": log,
        "records": tuple(log.iter_records()),
        "incidents": incidents,
        "edr": edr,
        "pull_requests": pull_requests,
        "vendor_connector": vendor_connector,
        "scan_run": scan_run,
        "scan_gated": scan_gated,
        "vendor_run": vendor_run,
        "vendor_gated": vendor_gated,
    }
    log.close()


class TestAllThreeTriggersRun:
    def test_the_alert_driven_graph_ran_and_gated(self, mesh):
        assert mesh["incidents"]["gated"] > 0
        assert mesh["incidents"]["failed"] == 0
        assert mesh["edr"].executed

    def test_the_commit_driven_graph_ran_and_gated(self, mesh):
        assert mesh["scan_gated"], "OPEN_PATCH_PR is destructive; the gate must fire"
        assert mesh["scan_run"].state.status is IncidentStatus.COMPLETED
        assert len(mesh["pull_requests"].opened) == 1

    def test_the_schedule_driven_graph_ran_and_gated(self, mesh):
        assert mesh["vendor_gated"]
        assert mesh["vendor_run"].state.status is IncidentStatus.COMPLETED
        assert len(mesh["vendor_connector"].executed) == 1

    def test_all_five_agents_appear_in_the_one_log(self, mesh):
        # The five PRD agents, plus the orchestrator recording the approvals. If one is
        # missing, an agent is running without an audit trail.
        actors = {record.actor for record in mesh["records"]}
        expected = {
            AgentName.TRIAGE.value,
            AgentName.INVESTIGATION.value,
            AgentName.CONTAINMENT.value,
            AgentName.CODE_SCAN.value,
            AgentName.SUPPLY_CHAIN.value,
            AgentName.ORCHESTRATOR.value,
        }
        assert expected <= actors, sorted(expected - actors)


class TestF08HoldsOnceAcrossAllThree:
    def test_no_action_executed_without_a_logged_approval_anywhere(self, mesh):
        # One function, unmodified, reading one log that saw all three graphs. This is
        # the claim: the guarantee has one implementation, so it has one check.
        assert verify_no_ungated_execution(mesh["log"]) == ()

    def test_the_shared_chain_verifies(self, mesh):
        assert mesh["log"].verify().findings == ()

    def test_every_gated_execution_was_preceded_by_its_own_approval(self, mesh):
        # Stronger than the aggregate check: per action, the grant must come first.
        # An approval logged after an execution is not an approval, it is a cover story.
        granted: dict[str, int] = {}
        for record in mesh["records"]:
            if record.event_type is AuditEventType.APPROVAL_GRANTED:
                granted.setdefault(record.subject_id, record.seq)
            elif record.event_type is AuditEventType.ACTION_EXECUTED and record.payload.get(
                "requires_human_approval"
            ):
                assert record.subject_id in granted, record.subject_id
                assert granted[record.subject_id] < record.seq

    def test_every_gate_that_fired_produced_an_approval_request(self, mesh):
        requested = {
            record.subject_id
            for record in mesh["records"]
            if record.event_type is AuditEventType.APPROVAL_REQUESTED
        }
        proposed_gated = {
            record.subject_id
            for record in mesh["records"]
            if record.event_type is AuditEventType.ACTION_PROPOSED
            and record.payload.get("requires_human_approval")
        }
        # A gated proposal with no request is an action waiting on a human nobody asked.
        assert proposed_gated <= requested

    def test_the_new_graphs_contribute_their_own_events(self, mesh):
        kinds = {record.event_type for record in mesh["records"]}
        assert AuditEventType.CODE_SCAN_COMPLETED in kinds
        assert AuditEventType.SUPPLY_CHAIN_ASSESSED in kinds


class TestTheLogLeaksNothing:
    def test_no_attacker_controlled_text_is_copied_into_the_log(self, mesh):
        """The log is exported to a customer's SIEM.

        Copying a payload or a source line into it would make the tamper-evident record
        a second delivery channel for whatever was in that text. The needles here are
        strings that exist verbatim in the corpus and in the F-07 fixture, so if any
        path ever starts echoing content this fails.
        """
        needles = (
            "Pr0d-Postgres-2024!",  # the fixture's planted credential
            "fixture-only-billing-key-not-a-real-credential",
            "hashlib.md5",
            "ignore all previous instructions",
            '" Label"',  # CIC-IDS2017's ground-truth column
        )
        for record in mesh["records"]:
            rendered = repr(record.payload)
            for needle in needles:
                assert needle not in rendered, (record.event_type, needle)

    def test_the_code_scan_row_carries_counts_and_rule_ids_only(self, mesh):
        row = next(
            record
            for record in mesh["records"]
            if record.event_type is AuditEventType.CODE_SCAN_COMPLETED
        )
        assert isinstance(row.payload["rules"], list)
        assert row.payload["findings"] > 0
        assert "excerpt" not in row.payload
        assert "diff" not in row.payload
