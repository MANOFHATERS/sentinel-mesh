"""The whole agent layer, end to end (PRD F-04, F-05, F-08, Figure 3, Section 9.1).

These tests run real alerts through the real graph with the real models and the
real audit log. Where a claim is about a negative — *"zero actions executed
without a logged approval"* — it is checked by reading the audit log back, not by
asking the objects that enforce the rule whether they enforced it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from sentinel.agents.contain import (
    ContainmentAgent,
    ContainmentError,
    SimulatedConnector,
    approval_queue,
    verify_no_ungated_execution,
)
from sentinel.agents.engine import HostileEngine
from sentinel.agents.investigate import InvestigationAgent, InvestigationError
from sentinel.agents.orchestrator import (
    NODE_APPROVE,
    NODE_CONTAIN,
    NODE_EXECUTE,
    NODE_INVESTIGATE,
    NODE_TRIAGE,
    build_incident_graph,
    incident_timings,
    new_incident,
)
from sentinel.agents.state import HumanDecision, IncidentStatus
from sentinel.agents.triage import BENIGN_FAMILY, TriageAgent, TriageModel
from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.clock import FrozenClock
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import (
    ActionType,
    ApprovalStatus,
    AuditEventType,
    RiskTier,
    TriageDecision,
)
from sentinel.core.untrusted import UntrustedText
from sentinel.kb.retrieve import KnowledgeBase
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.ml.metrics import four_way_split

START = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build()


@pytest.fixture(scope="module")
def fitted():
    alerts = generate_alerts(4000, seed=4242)
    labels = np.array(
        [0 if a.ground_truth_label == BENIGN_FAMILY else 1 for a in alerts], dtype=int
    )
    split = four_way_split(labels, seed=4242)
    model = TriageModel.fit(
        [alerts[i] for i in split.train_benign],
        train_labelled=[alerts[i] for i in split.train_labelled],
        validation=[alerts[i] for i in split.validation],
        seed=4242,
    )
    return model, [alerts[i] for i in split.test]


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(START)


@pytest.fixture
def connector() -> SimulatedConnector:
    return SimulatedConnector()


@pytest.fixture
def graph(fitted, kb: KnowledgeBase, clock: FrozenClock, connector: SimulatedConnector):
    model, _test = fitted
    return build_incident_graph(
        triage=TriageAgent(model=model, clock=clock),
        investigation=InvestigationAgent(kb=kb, clock=clock),
        containment=ContainmentAgent(clock=clock),
        connector=connector,
    )


@pytest.fixture
def audit(tmp_path, clock: FrozenClock) -> HashChainedAuditLog:
    with HashChainedAuditLog(tmp_path / "audit.sqlite", clock=clock) as log:
        yield log


def _run(graph, alerts, clock, audit, *, limit: int = 120):
    """Push ``limit`` alerts through the graph. Returns (results, waiting threads)."""
    results = []
    waiting = []
    for alert in alerts[:limit]:
        clock.advance(1.0)
        result = graph.invoke(
            new_incident(alert, at=clock.now()), clock=clock, audit=audit
        )
        results.append(result)
        if result.interrupted:
            waiting.append(result.state.incident_id)
    return results, waiting


# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_every_incident_reaches_a_defined_state(
        self, graph, fitted, clock, audit
    ) -> None:
        results, waiting = _run(graph, fitted[1], clock, audit)
        for result in results:
            assert result.state.status in (
                IncidentStatus.DISMISSED,
                IncidentStatus.COMPLETED,
                IncidentStatus.AWAITING_APPROVAL,
            )
        assert waiting, "no incident reached the approval gate; F-08 would pass vacuously"

    def test_the_full_path_is_exercised(self, graph, fitted, clock, audit) -> None:
        """Figure 3's longest route: triage → investigate → contain → approve → execute."""
        _results, waiting = _run(graph, fitted[1], clock, audit)
        clock.advance(20.0)
        resumed = graph.resume(
            waiting[0],
            HumanDecision(approver="analyst@acme", approved=True, decided_at=clock.now()),
            clock=clock,
            audit=audit,
        )
        assert resumed.state.visited == (
            NODE_TRIAGE,
            NODE_INVESTIGATE,
            NODE_CONTAIN,
            NODE_APPROVE,
            NODE_EXECUTE,
        )
        assert resumed.state.status is IncidentStatus.COMPLETED

    def test_dismissed_incidents_stop_at_triage(self, graph, fitted, clock, audit) -> None:
        results, _waiting = _run(graph, fitted[1], clock, audit)
        dismissed = [
            r for r in results if r.state.status is IncidentStatus.DISMISSED
        ]
        assert dismissed
        for result in dismissed:
            assert NODE_INVESTIGATE not in result.state.visited
            assert result.state.report is None
            assert result.state.actions == ()

    def test_incident_ids_are_deterministic(self, graph, fitted, clock, audit) -> None:
        alert = fitted[1][0]
        assert (
            new_incident(alert, at=START).incident_id
            == new_incident(alert, at=clock.now()).incident_id
        )

    def test_the_audit_chain_verifies_after_the_whole_run(
        self, graph, fitted, clock, audit
    ) -> None:
        _results, waiting = _run(graph, fitted[1], clock, audit)
        for index, thread in enumerate(waiting):
            clock.advance(10.0)
            graph.resume(
                thread,
                HumanDecision(
                    approver="analyst@acme",
                    approved=index % 2 == 0,
                    decided_at=clock.now(),
                ),
                clock=clock,
                audit=audit,
            )
        audit.verify().raise_if_broken()
        assert audit.count() > len(waiting)


class TestF05Grounding:
    """Every factual claim traces to a retrieved KB chunk or a raw log line."""

    def test_every_claim_cites_resolvable_evidence(
        self, graph, fitted, clock, audit, kb: KnowledgeBase
    ) -> None:
        results, _waiting = _run(graph, fitted[1], clock, audit)
        reports = [r.state.report for r in results if r.state.report is not None]
        assert reports, "no investigations ran"
        for report in reports:
            known = {item.ref for item in report.evidence}
            for statement, refs in report.claims:
                assert refs, f"uncited claim: {statement!r}"
                assert set(refs) <= known
            kb_refs = [ref for ref in known if ref.startswith("kb://")]
            assert kb.resolves_all(kb_refs)

    def test_every_report_makes_at_least_one_claim(
        self, graph, fitted, clock, audit
    ) -> None:
        """A report with no claims satisfies the schema and says nothing."""
        results, _waiting = _run(graph, fitted[1], clock, audit)
        for result in results:
            if result.state.report is not None:
                assert result.state.report.is_grounded

    def test_asserted_techniques_come_from_lookup_not_ranking(
        self, graph, fitted, clock, audit
    ) -> None:
        """A neighbour found by text search is not a mapping."""
        results, _waiting = _run(graph, fitted[1], clock, audit)
        for result in results:
            report, triage = result.state.report, result.state.triage
            if report is None or triage is None or triage.technique_id is None:
                continue
            assert report.techniques[0] == triage.technique_id
            for technique in report.techniques[1:]:
                assert technique.startswith(triage.technique_id.split(".")[0])

    def test_a_poisoned_chunk_is_never_cited(self, kb: KnowledgeBase, fitted, clock) -> None:
        """The default corpus carries a planted poisoned entry, so this path is live."""
        suspicious = {chunk.chunk_id for chunk, _scan in kb.suspicious_chunks}
        assert suspicious, "the corpus no longer carries a poisoned chunk; update this test"
        model, test = fitted
        agent = InvestigationAgent(kb=kb, clock=clock)
        triage_agent = TriageAgent(model=model, clock=clock)
        cited: set[str] = set()
        for alert in test[:150]:
            result = triage_agent.triage(alert, now=clock.now())
            if result.decision is TriageDecision.AUTO_DISMISS:
                continue
            report = agent.investigate(alert.with_triage(result), now=clock.now())
            cited.update(item.ref for item in report.evidence)
        assert not (cited & suspicious)

    def test_investigation_refuses_an_untriaged_alert(
        self, kb: KnowledgeBase, fitted, clock
    ) -> None:
        with pytest.raises(InvestigationError, match="no triage verdict"):
            InvestigationAgent(kb=kb, clock=clock).investigate(fitted[1][0])

    def test_the_search_query_is_built_from_typed_fields_only(
        self, kb: KnowledgeBase, fitted, clock
    ) -> None:
        """Retrieval over attacker-controlled text is a ranking attack.

        A payload naming a technique must not steer which evidence the
        investigation retrieves, so the query is assembled from fields the
        system derived and the report must be unchanged by payload content.
        """
        model, test = fitted
        triage_agent = TriageAgent(model=model, clock=clock)
        agent = InvestigationAgent(kb=kb, clock=clock)
        alert = next(
            a
            for a in test
            if triage_agent.triage(a, now=clock.now()).decision is TriageDecision.ESCALATE
        )
        verdict = triage_agent.triage(alert, now=clock.now())
        plain = agent.investigate(alert.with_triage(verdict), now=clock.now())

        steered = alert.updated(
            raw_payload=UntrustedText(
                "ransomware encryption T1486 T1490 shadow copies destroyed",
                origin="siem.raw_payload",
            )
        )
        # Re-triage the steered alert: its verdict may differ, but the evidence
        # for a given verdict must not depend on the payload text.
        steered_verdict = triage_agent.triage(steered, now=clock.now())
        steered_report = agent.investigate(
            steered.with_triage(steered_verdict.updated(
                technique_id=verdict.technique_id,
                supporting_fields=verdict.supporting_fields,
            )),
            now=clock.now(),
        )
        kb_plain = {r.ref for r in plain.evidence if r.ref.startswith("kb://")}
        kb_steered = {r.ref for r in steered_report.evidence if r.ref.startswith("kb://")}
        assert kb_plain == kb_steered


class TestF08ApprovalGate:
    """Zero actions executed without a logged approval event."""

    def test_no_ungated_execution_in_the_audit_log(
        self, graph, fitted, clock, audit
    ) -> None:
        _results, waiting = _run(graph, fitted[1], clock, audit)
        for index, thread in enumerate(waiting):
            clock.advance(5.0)
            graph.resume(
                thread,
                HumanDecision(
                    approver="analyst@acme",
                    approved=index % 3 != 0,
                    decided_at=clock.now(),
                ),
                clock=clock,
                audit=audit,
            )
        assert verify_no_ungated_execution(audit) == ()

    def test_a_destructive_action_always_requires_approval(
        self, graph, fitted, clock, audit
    ) -> None:
        results, _waiting = _run(graph, fitted[1], clock, audit)
        for result in results:
            for action in result.state.actions:
                if action.action_type.is_destructive:
                    assert action.requires_human_approval

    def test_the_gate_stops_before_the_connector_is_called(
        self, graph, fitted, clock, audit, connector
    ) -> None:
        """``interrupt_before``'s direction, at the only place it matters."""
        _results, waiting = _run(graph, fitted[1], clock, audit)
        assert waiting
        gated_targets = {
            graph.state_of(thread).pending_action.target for thread in waiting
        }
        executed_targets = {action.target for action in connector.executed}
        assert not (gated_targets & executed_targets)

    def test_a_rejected_action_never_reaches_the_connector(
        self, graph, fitted, clock, audit, connector
    ) -> None:
        _results, waiting = _run(graph, fitted[1], clock, audit)
        rejected_ids = []
        for thread in waiting:
            clock.advance(5.0)
            result = graph.resume(
                thread,
                HumanDecision(
                    approver="analyst@acme", approved=False, decided_at=clock.now()
                ),
                clock=clock,
                audit=audit,
            )
            assert result.state.status is IncidentStatus.COMPLETED
            assert NODE_EXECUTE not in result.state.visited
            rejected_ids.extend(
                a.action_id
                for a in result.state.actions
                if a.approval_status is ApprovalStatus.REJECTED
            )
        assert rejected_ids
        assert not ({a.action_id for a in connector.executed} & set(rejected_ids))

    def test_an_approval_is_recorded_with_its_approver(
        self, graph, fitted, clock, audit
    ) -> None:
        _results, waiting = _run(graph, fitted[1], clock, audit)
        clock.advance(5.0)
        result = graph.resume(
            waiting[0],
            HumanDecision(
                approver="dana@acme", approved=True, decided_at=clock.now(), note="ok"
            ),
            clock=clock,
            audit=audit,
        )
        executed = [
            a for a in result.state.actions if a.approval_status is ApprovalStatus.EXECUTED
        ]
        assert executed and executed[0].approved_by == "dana@acme"
        grants = [
            r
            for r in audit.iter_records()
            if r.event_type is AuditEventType.APPROVAL_GRANTED
        ]
        assert any(r.payload["approver"] == "dana@acme" for r in grants)

    def test_the_connector_refuses_an_unapproved_action_on_its_own(
        self, fitted, clock
    ) -> None:
        """Defence in depth: the connector does not trust its caller."""
        from sentinel.core.schemas import ActionRequest, AgentName

        action = ActionRequest.propose(
            alert_id="a",
            tenant_id="acme",
            proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.ISOLATE_HOST,
            target="host-1",
            rationale="test",
            risk_tier=RiskTier.RECOMMEND,
            created_at=START,
        )
        with pytest.raises(GuardrailViolation, match="human approval required"):
            SimulatedConnector().execute(action)

    def test_an_approval_request_precedes_every_gated_action(
        self, graph, fitted, clock, audit
    ) -> None:
        _results, waiting = _run(graph, fitted[1], clock, audit)
        requested = {
            r.subject_id
            for r in audit.iter_records()
            if r.event_type is AuditEventType.APPROVAL_REQUESTED
        }
        for thread in waiting:
            pending = graph.state_of(thread).pending_action
            assert pending is not None
            assert pending.action_id in requested

    def test_the_approval_queue_lists_the_waiting_actions(
        self, graph, fitted, clock, audit
    ) -> None:
        results, waiting = _run(graph, fitted[1], clock, audit)
        actions = [a for r in results for a in r.state.actions]
        assert len(approval_queue(actions)) == len(waiting)

    def test_observe_tier_never_proposes_a_destructive_action(
        self, fitted, kb: KnowledgeBase, clock, connector
    ) -> None:
        """Observe-only means observe only."""
        model, test = fitted
        graph = build_incident_graph(
            triage=TriageAgent(model=model, clock=clock),
            investigation=InvestigationAgent(kb=kb, clock=clock),
            containment=ContainmentAgent(clock=clock, default_tier=RiskTier.OBSERVE),
            connector=connector,
        )
        for alert in test[:60]:
            clock.advance(1.0)
            result = graph.invoke(
                new_incident(alert, at=clock.now(), trust_tier=RiskTier.OBSERVE),
                clock=clock,
            )
            for action in result.state.actions:
                assert not action.action_type.is_destructive

    def test_containment_refuses_an_untriaged_alert(self, fitted, clock) -> None:
        with pytest.raises(ContainmentError, match="no triage verdict"):
            ContainmentAgent(clock=clock).propose(fitted[1][0])


class TestHostileEngineEndToEnd:
    """A compromised model in the live graph changes nothing that matters."""

    def test_no_ungated_execution_with_a_hostile_engine(
        self, fitted, kb: KnowledgeBase, clock, connector, audit
    ) -> None:
        model, test = fitted
        graph = build_incident_graph(
            triage=TriageAgent(model=model, engine=HostileEngine(), clock=clock),
            investigation=InvestigationAgent(kb=kb, engine=HostileEngine(), clock=clock),
            containment=ContainmentAgent(clock=clock),
            connector=connector,
        )
        results, waiting = _run(graph, test, clock, audit, limit=80)
        for index, thread in enumerate(waiting):
            clock.advance(5.0)
            graph.resume(
                thread,
                HumanDecision(
                    approver="analyst@acme",
                    approved=index % 2 == 0,
                    decided_at=clock.now(),
                ),
                clock=clock,
                audit=audit,
            )
        assert verify_no_ungated_execution(audit) == ()
        for result in results:
            if result.state.report is not None:
                known = {item.ref for item in result.state.report.evidence}
                for _statement, refs in result.state.report.claims:
                    assert set(refs) <= known

    def test_a_hostile_engine_cannot_reduce_escalations(
        self, fitted, kb: KnowledgeBase, clock, connector, audit
    ) -> None:
        model, test = fitted
        plain = build_incident_graph(
            triage=TriageAgent(model=model, clock=clock),
            investigation=InvestigationAgent(kb=kb, clock=clock),
            containment=ContainmentAgent(clock=clock),
            connector=SimulatedConnector(),
        )
        hostile = build_incident_graph(
            triage=TriageAgent(model=model, engine=HostileEngine(), clock=clock),
            investigation=InvestigationAgent(kb=kb, clock=clock),
            containment=ContainmentAgent(clock=clock),
            connector=SimulatedConnector(),
        )
        plain_dismissed = sum(
            1 for r in _run(plain, test, clock, audit, limit=80)[0]
            if r.state.status is IncidentStatus.DISMISSED
        )
        hostile_dismissed = sum(
            1 for r in _run(hostile, test, clock, audit, limit=80)[0]
            if r.state.status is IncidentStatus.DISMISSED
        )
        assert hostile_dismissed <= plain_dismissed


class TestSection91Timing:
    def test_mttd_is_inside_the_thirty_second_budget(
        self, graph, fitted, clock, audit
    ) -> None:
        results, _waiting = _run(graph, fitted[1], clock, audit)
        timings = [incident_timings(r.state) for r in results]
        measured = [t.pipeline_detect_seconds for t in timings if t.detected]
        assert measured
        assert max(measured) < 30.0

    def test_mttc_is_inside_the_three_minute_budget(
        self, graph, fitted, clock, audit
    ) -> None:
        """Section 9.1: confirmed threat to approved containment, < 3 minutes.

        Each incident is driven to completion before the next begins, with a
        twelve-second scripted analyst click — the PRD's own construction. That
        matters, because MTTC as defined *includes* the queue: running a hundred
        incidents in parallel against one serial analyst produces a last-in wait
        of several minutes and a "failing" MTTC that is a statement about
        staffing rather than about the software. Measured that way here the
        figure ran to 227s on a 28-deep queue. That is not a defect to tune away
        — it is Section 2.1's thesis showing up as a number — so the budget is
        asserted per incident and the queueing effect is reported separately in
        the evaluation.
        """
        _model, test = fitted
        contained = []
        for alert in test[:60]:
            clock.advance(1.0)
            result = graph.invoke(
                new_incident(alert, at=clock.now()), clock=clock, audit=audit
            )
            if not result.interrupted:
                continue
            clock.advance(12.0)  # the scripted human-approval click
            resumed = graph.resume(
                result.state.incident_id,
                HumanDecision(
                    approver="analyst@acme", approved=True, decided_at=clock.now()
                ),
                clock=clock,
                audit=audit,
            )
            timing = incident_timings(resumed.state)
            if timing.contained:
                contained.append(timing.contain_seconds)
        assert contained, "no incident reached containment"
        assert max(contained) < 180.0

    def test_mttc_grows_with_queue_depth(self, graph, fitted, clock, audit) -> None:
        """The queueing effect, asserted rather than left as an anomaly.

        A batch of incidents parked at the gate and worked through by one analyst
        produces a containment time that grows with position in the queue. The
        system is not slower; the human is one person.
        """
        _results, waiting = _run(graph, fitted[1], clock, audit)
        assert len(waiting) > 5
        waits = []
        for thread in waiting:
            clock.advance(12.0)
            resumed = graph.resume(
                thread,
                HumanDecision(
                    approver="analyst@acme", approved=True, decided_at=clock.now()
                ),
                clock=clock,
                audit=audit,
            )
            timing = incident_timings(resumed.state)
            if timing.contained:
                waits.append(timing.contain_seconds)
        assert waits[-1] > waits[0]

    def test_feed_lag_is_reported_separately_from_detection(
        self, graph, fitted, clock, audit
    ) -> None:
        """On a static corpus the capture timestamp is years old, and that shows.

        The two MTTD definitions are reported side by side precisely so this is
        visible rather than absorbed into the headline number.
        """
        results, _waiting = _run(graph, fitted[1], clock, audit, limit=5)
        timing = incident_timings(results[0].state)
        assert timing.feed_lag_seconds is not None
        assert timing.detect_seconds == pytest.approx(
            timing.feed_lag_seconds + timing.pipeline_detect_seconds, abs=1e-6
        )

    def test_the_replay_service_makes_both_definitions_agree(self) -> None:
        """The PRD's demo path streams alerts, and there the two coincide.

        This is the claim ``incident_timings`` makes in its docstring, so it is
        checked against the real replay service rather than asserted. On a
        streamed feed ``ingested_at`` is the moment the pipeline saw the record,
        so the feed lag is the pipeline's own latency rather than the age of a
        capture — which is what makes ``detect_seconds`` the number to quote in
        a live deployment and ``pipeline_detect_seconds`` the number to quote on
        a static corpus.
        """
        from sentinel.ingest.bus import InMemoryEventBus
        from sentinel.ingest.normalizer import CICIDS2017Normalizer
        from sentinel.ingest.replay import IterableReplaySource, ReplayService, TimeModel
        from sentinel.ml.datasets.synthetic import SyntheticCICGenerator

        service_clock = FrozenClock(START)
        collected: list = []
        service = ReplayService(
            normalizer=CICIDS2017Normalizer(tenant_id="acme", strict=True),
            bus=InMemoryEventBus(clock=service_clock, maxlen=None),
            clock=service_clock,
            time_model=TimeModel.VIRTUAL,
        )
        service.run(
            IterableReplaySource(
                records=list(SyntheticCICGenerator(seed=7).rows(30)), name="synthetic"
            ),
            on_alert=collected.append,
        )
        assert collected
        for alert in collected:
            # Stamped by the service as it emitted the record, not carried in
            # from the capture — so an incident opened immediately after has a
            # feed lag of seconds, not years.
            assert alert.ingested_at >= START
            assert (alert.ingested_at - START).total_seconds() < 60.0
