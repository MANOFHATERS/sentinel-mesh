"""The Supply-Chain Agent end to end, including both routing paths.

Two things only an integration test can assert.

**Both routes are real.** A flagged *package* proposes ``OPEN_PATCH_PR``, which is
destructive, so the run stops at the Human Approval Gate. A flagged *vendor* proposes
``NOTIFY_ANALYST``, which is not, so the same graph routes straight through with no
gate. That the route depends on *what was found* rather than on a configuration flag
is the design, and a unit test on either half would miss it.

**The trigger works.** Supply-chain risk is continuous, so the run is started by a
scheduled monitor that mints its own ``AlertSource.VENDOR_FEED`` alert. Re-running the
same assessment must resume rather than fork, which is what makes an MSSP's nightly
job idempotent.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from sentinel.agents.contain import SimulatedConnector, verify_no_ungated_execution
from sentinel.agents.runtime import OrchestrationError
from sentinel.agents.state import HumanDecision, IncidentStatus
from sentinel.agents.supplychain import (
    NODE_ASSESS,
    NODE_REMEDIATE,
    SupplyChainAgent,
    SupplyChainMonitor,
    build_supply_chain_review_graph,
    supply_chain_timings,
    synthesize_vendor_alert,
)
from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.clock import SimulationClock
from sentinel.core.schemas import (
    ActionType,
    ApprovalStatus,
    AuditEventType,
    EvidenceKind,
    RiskTier,
)
from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
from sentinel.graph.schema import NodeKind
from sentinel.graph.synthetic import SyntheticGraphGenerator
from sentinel.kb.retrieve import KnowledgeBase

START = datetime(2026, 9, 29, 3, 0, 0, tzinfo=UTC)
SEED = 20260928


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build()


@pytest.fixture(scope="module")
def scored():
    graph, truth = SyntheticGraphGenerator(seed=SEED).generate()
    node_ids = graph.node_ids()
    labels = truth.labels(node_ids)
    split = GraphSplit.stratified(labels, seed=SEED)
    model = SupplyChainGNN(random_state=SEED).fit(
        graph, labels, split, exposure=truth.risk_vector(node_ids)
    )
    return graph, model, model.risk_scores(graph)


def _vendor_only_scores(graph, scores: np.ndarray) -> np.ndarray:
    """Scores that rank vendors and organisations at the top.

    Used to exercise the ``NOTIFY_ANALYST`` route. Constructed rather than found,
    because whether the trained model happens to rank a vendor first is a property of
    the seed, and a test whose route depends on that is a test that changes meaning
    when the model improves.
    """
    forced = np.zeros_like(scores)
    for index, node in enumerate(graph.nodes):
        forced[index] = 0.99 if node.kind is not NodeKind.PACKAGE else 0.01
    return forced


def _drive(kb, graph, model, scores, tmp_path, *, approve=True, tier=RiskTier.RECOMMEND):
    clock = SimulationClock(START)
    connector = SimulatedConnector()
    agent = SupplyChainAgent(kb=kb, clock=clock)
    compiled = build_supply_chain_review_graph(
        agent=agent, graph=graph, model=model, scores=scores, connector=connector
    )
    log = HashChainedAuditLog(tmp_path / "audit.sqlite", clock=clock)
    monitor = SupplyChainMonitor(
        agent=agent, graph=graph, model=model, scores=scores, clock=clock
    )
    from sentinel.agents.supplychain import new_vendor_risk_incident

    alert = synthesize_vendor_alert(
        tenant_id="acme", graph=graph, at=clock.now(), assessment_id="2026-09-29"
    )
    result = compiled.invoke(
        new_vendor_risk_incident(alert, at=clock.now(), trust_tier=tier),
        clock=clock,
        audit=log,
    )
    interrupted = result.interrupted
    if interrupted:
        clock.advance(45.0)
        result = compiled.resume(
            result.state.incident_id,
            HumanDecision(
                approver="vciso@acme", approved=approve, decided_at=clock.now()
            ),
            clock=clock,
            audit=log,
        )
    return compiled, result, connector, log, interrupted, monitor, clock


class TestPackageRouteIsGated:
    def test_a_flagged_package_proposes_a_gated_patch_pr(self, kb, scored, tmp_path):
        graph, model, scores = scored
        _c, result, connector, log, interrupted, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        assert interrupted, "OPEN_PATCH_PR is destructive; the gate must fire"
        action = result.state.actions[-1]
        assert action.action_type is ActionType.OPEN_PATCH_PR
        assert action.requires_human_approval
        assert result.state.status is IncidentStatus.COMPLETED
        assert len(connector.executed) == 1
        log.close()

    def test_a_rejected_remediation_never_reaches_the_connector(
        self, kb, scored, tmp_path
    ):
        graph, model, scores = scored
        _c, result, connector, log, interrupted, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path, approve=False
        )
        assert interrupted
        assert connector.executed == []
        assert result.state.actions[-1].approval_status is ApprovalStatus.REJECTED
        assert verify_no_ungated_execution(log) == ()
        log.close()

    def test_the_rationale_carries_the_exposure_path(self, kb, scored, tmp_path):
        # An approval prompt that says only "risk 0.97" is a prompt the reviewer
        # cannot answer. F-06's guardrail reaches the gate, not just the report.
        graph, model, scores = scored
        _c, result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        rationale = result.state.actions[-1].rationale
        assert "rank #" in rationale
        assert "unmaintained" in rationale or "reaches" in rationale
        log.close()

    def test_the_action_carries_the_graph_path_evidence(self, kb, scored, tmp_path):
        graph, model, scores = scored
        _c, result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        action = result.state.actions[-1]
        assert action.evidence
        assert any(item.kind is EvidenceKind.GRAPH_PATH for item in action.evidence)
        log.close()

    def test_the_patch_rationale_admits_it_drafts_no_manifest_edit(
        self, kb, scored, tmp_path
    ):
        # The sprint graph is synthetic, so a version pin would be invented. Saying so
        # in the rationale is the difference between a limitation and a fabrication.
        graph, model, scores = scored
        _c, result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        assert "not drafted here" in result.state.actions[-1].rationale
        log.close()


class TestVendorRouteIsNotGated:
    def test_a_flagged_vendor_notifies_without_a_gate(self, kb, scored, tmp_path):
        # NOTIFY_ANALYST has no side effect on a monitored system, so gating it would
        # be asking a human to approve telling a human.
        graph, _model, scores = scored
        forced = _vendor_only_scores(graph, scores)
        _c, result, connector, log, interrupted, _m, _clock = _drive(
            kb, graph, None, forced, tmp_path
        )
        assert not interrupted, "a notification must not stop for approval"
        assert result.state.status is IncidentStatus.COMPLETED
        action = result.state.actions[-1]
        assert action.action_type is ActionType.NOTIFY_ANALYST
        assert not action.requires_human_approval
        assert action.approval_status is ApprovalStatus.EXECUTED
        assert len(connector.executed) == 1
        log.close()

    def test_f08_still_holds_on_the_ungated_route(self, kb, scored, tmp_path):
        # An ungated execution of a non-destructive action is not an F-08 violation,
        # and the log-based check must agree rather than flagging it.
        graph, _model, scores = scored
        forced = _vendor_only_scores(graph, scores)
        _c, _result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, None, forced, tmp_path
        )
        assert verify_no_ungated_execution(log) == ()
        assert log.verify().findings == ()
        log.close()

    def test_the_notification_rationale_says_a_vendor_cannot_be_patched(
        self, kb, scored, tmp_path
    ):
        graph, _model, scores = scored
        forced = _vendor_only_scores(graph, scores)
        _c, result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, None, forced, tmp_path
        )
        assert "cannot be patched" in result.state.actions[-1].rationale
        log.close()

    def test_an_organisation_is_evaluated_by_ranking_not_flagged_for_patching(
        self, kb, scored, tmp_path
    ):
        graph, _model, scores = scored
        forced = _vendor_only_scores(graph, scores)
        _c, result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, None, forced, tmp_path
        )
        report = result.state.report
        assert report is not None
        assert ActionType.OPEN_PATCH_PR not in report.recommended_actions
        log.close()


class TestTheTrigger:
    def test_the_monitor_starts_a_run(self, kb, scored, tmp_path):
        graph, model, scores = scored
        clock = SimulationClock(START)
        agent = SupplyChainAgent(kb=kb, clock=clock)
        compiled = build_supply_chain_review_graph(
            agent=agent, graph=graph, model=model, scores=scores
        )
        monitor = SupplyChainMonitor(
            agent=agent, graph=graph, model=model, scores=scores, clock=clock
        )
        result = monitor.tick(
            tenant_id="acme", compiled=compiled, assessment_id="2026-09-29"
        )
        assert result.state.visited == (NODE_ASSESS,)
        assert result.interrupted

    def test_re_running_the_same_assessment_does_not_fork_the_incident(
        self, kb, scored, tmp_path
    ):
        # The idempotence that makes a nightly job safe to retry. Starting a thread
        # that already has checkpoints is refused rather than silently forked.
        graph, model, scores = scored
        clock = SimulationClock(START)
        agent = SupplyChainAgent(kb=kb, clock=clock)
        compiled = build_supply_chain_review_graph(
            agent=agent, graph=graph, model=model, scores=scores
        )
        monitor = SupplyChainMonitor(
            agent=agent, graph=graph, model=model, scores=scores, clock=clock
        )
        monitor.tick(tenant_id="acme", compiled=compiled, assessment_id="2026-09-29")
        with pytest.raises(OrchestrationError, match="already has checkpoints"):
            monitor.tick(
                tenant_id="acme", compiled=compiled, assessment_id="2026-09-29"
            )

    def test_a_different_day_is_a_different_run(self, kb, scored, tmp_path):
        graph, model, scores = scored
        clock = SimulationClock(START)
        agent = SupplyChainAgent(kb=kb, clock=clock)
        compiled = build_supply_chain_review_graph(
            agent=agent, graph=graph, model=model, scores=scores
        )
        monitor = SupplyChainMonitor(
            agent=agent, graph=graph, model=model, scores=scores, clock=clock
        )
        first = monitor.tick(
            tenant_id="acme", compiled=compiled, assessment_id="2026-09-29"
        )
        second = monitor.tick(
            tenant_id="acme", compiled=compiled, assessment_id="2026-09-30"
        )
        assert first.state.incident_id != second.state.incident_id

    def test_two_tenants_get_independent_runs(self, kb, scored, tmp_path):
        # The MSSP case: fifty tenants, fifty runs, no races on one thread.
        graph, model, scores = scored
        clock = SimulationClock(START)
        agent = SupplyChainAgent(kb=kb, clock=clock)
        compiled = build_supply_chain_review_graph(
            agent=agent, graph=graph, model=model, scores=scores
        )
        monitor = SupplyChainMonitor(
            agent=agent, graph=graph, model=model, scores=scores, clock=clock
        )
        acme = monitor.tick(
            tenant_id="acme", compiled=compiled, assessment_id="2026-09-29"
        )
        other = monitor.tick(
            tenant_id="globex", compiled=compiled, assessment_id="2026-09-29"
        )
        assert acme.state.incident_id != other.state.incident_id
        assert acme.state.tenant_id == "acme"
        assert other.state.tenant_id == "globex"

    def test_the_default_assessment_id_is_the_date(self, kb, scored):
        graph, model, scores = scored
        clock = SimulationClock(START)
        agent = SupplyChainAgent(kb=kb, clock=clock)
        compiled = build_supply_chain_review_graph(
            agent=agent, graph=graph, model=model, scores=scores
        )
        monitor = SupplyChainMonitor(
            agent=agent, graph=graph, model=model, scores=scores, clock=clock
        )
        result = monitor.tick(tenant_id="acme", compiled=compiled)
        expected = synthesize_vendor_alert(
            tenant_id="acme",
            graph=graph,
            at=clock.now(),
            assessment_id=START.date().isoformat(),
        )
        assert result.state.alert.alert_id == expected.alert_id


class TestAuditTrail:
    def test_the_assessment_is_logged_with_its_top_nodes(self, kb, scored, tmp_path):
        graph, model, scores = scored
        _c, _result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        rows = [
            record
            for record in log.iter_records()
            if record.event_type is AuditEventType.SUPPLY_CHAIN_ASSESSED
        ]
        assert len(rows) == 1
        payload = rows[0].payload
        assert payload["nodes"] == graph.n_nodes
        assert payload["flagged"] >= 1
        assert isinstance(payload["top"], list) and payload["top"]
        assert {"node_id", "kind", "risk", "driver", "intrinsic"} <= set(
            payload["top"][0]
        )
        log.close()

    def test_the_chain_verifies_and_nothing_executed_ungated(
        self, kb, scored, tmp_path
    ):
        graph, model, scores = scored
        _c, _result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        assert verify_no_ungated_execution(log) == ()
        assert log.verify().findings == ()
        log.close()

    def test_the_approval_precedes_the_execution_in_the_chain(
        self, kb, scored, tmp_path
    ):
        graph, model, scores = scored
        _c, _result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        granted = next(
            record.seq
            for record in log.iter_records()
            if record.event_type is AuditEventType.APPROVAL_GRANTED
        )
        executed = next(
            record.seq
            for record in log.iter_records()
            if record.event_type is AuditEventType.ACTION_EXECUTED
        )
        assert granted < executed
        log.close()


class TestTimings:
    def test_timings_come_from_the_run_history(self, kb, scored, tmp_path):
        graph, model, scores = scored
        _c, result, _conn, log, _i, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path
        )
        assessed, acted = supply_chain_timings(result.state)
        assert assessed is not None and assessed >= 0.0
        # The scripted 45s reviewer delay is inside the time to remediation.
        assert acted is not None and acted >= 45.0
        log.close()

    def test_an_unfinished_run_reports_no_remediation_time(self, kb, scored, tmp_path):
        graph, model, scores = scored
        clock = SimulationClock(START)
        agent = SupplyChainAgent(kb=kb, clock=clock)
        compiled = build_supply_chain_review_graph(
            agent=agent, graph=graph, model=model, scores=scores
        )
        monitor = SupplyChainMonitor(
            agent=agent, graph=graph, model=model, scores=scores, clock=clock
        )
        paused = monitor.tick(tenant_id="acme", compiled=compiled)
        assessed, acted = supply_chain_timings(paused.state)
        assert assessed is not None
        assert acted is None


class TestUnattendedTier:
    def test_the_autonomous_tier_needs_no_gate(self, kb, scored, tmp_path):
        # The regression the shared executable_action predicate exists for: at a tier
        # permitting unattended execution the action is never APPROVED, because nobody
        # approves it, and a node looking only for APPROVED failed the run.
        graph, model, scores = scored
        _c, result, connector, log, interrupted, _m, _clock = _drive(
            kb, graph, model, scores, tmp_path, tier=RiskTier.AUTONOMOUS
        )
        assert not interrupted
        assert result.state.status is IncidentStatus.COMPLETED
        assert result.state.visited == (NODE_ASSESS, NODE_REMEDIATE)
        assert len(connector.executed) == 1
        log.close()
