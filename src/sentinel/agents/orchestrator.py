"""The orchestration graph (PRD F-04, Figure 3) and its MTTD/MTTC instrumentation.

Figure 3 is a state machine with a human interrupt in the middle of it. This
module builds that machine out of the agents and the runtime, and its whole
content is *routing* — which node runs next, and where the human stands.

Shape
-----
::

    triage ──dismiss──────────────────────────────────────────► END (dismissed)
       │
       └─keep─► investigate ─► contain ──nothing to do────────► END (completed)
                                  │
                                  ├─needs approval─► approve ──denied──► END
                                  │                     │
                                  │                  granted
                                  │                     ▼
                                  └─tier permits────► execute ─────────► END

Why approval is a node and not a flag
-------------------------------------
The runtime's ``interrupt_before`` is static per node, and that is a feature
here rather than a limitation worked around. Making the gate its own node means
"this incident is waiting for a human" is a *position in the graph*, visible in
the checkpoint, in ``CompiledGraph.pending()`` and in the audit log, rather than
a boolean somebody has to remember to check. The alternative — one execution
node that sometimes pauses — puts the decision of whether to pause inside the
node that does the acting, which is the one place it must not be.

A rejected action therefore ends the run from the ``approve`` node without ever
reaching ``execute``. The connector is not called, is not passed a rejected
object to be careful with, and is not part of the reasoning about F-08 at all.

MTTD and MTTC
-------------
PRD Section 9.1 asks for mean time to detect (< 30s, ingestion → triage
classification) and mean time to contain (< 3 min, confirmed threat → approved
containment). Both are read from the step history
(:func:`incident_timings`), which is checkpointed state rather than a separate
metrics path — so the number in the report is the number the run recorded, per
Section 9.3's one-pipeline rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sentinel.agents.contain import (
    ContainmentAgent,
    SimulatedConnector,
    executable_action,
    pending_action,
)
from sentinel.agents.investigate import InvestigationAgent
from sentinel.agents.runtime import CompiledGraph, GraphSpec, RunContext
from sentinel.agents.state import END, IncidentState, IncidentStatus, StepOutcome
from sentinel.agents.triage import TriageAgent
from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.errors import SentinelError
from sentinel.core.ids import deterministic_id
from sentinel.core.schemas import (
    ActionRequest,
    AgentName,
    Alert,
    ApprovalStatus,
    AuditEventType,
    RiskTier,
    TriageDecision,
)
from sentinel.core.untrusted import InjectionVerdict

__all__ = [
    "NODE_APPROVE",
    "NODE_CONTAIN",
    "NODE_EXECUTE",
    "NODE_INVESTIGATE",
    "NODE_TRIAGE",
    "IncidentTimings",
    "Orchestrator",
    "OrchestratorError",
    "build_incident_graph",
    "incident_timings",
    "new_incident",
]

NODE_TRIAGE = "triage"
NODE_INVESTIGATE = "investigate"
NODE_CONTAIN = "contain"
NODE_APPROVE = "approve"
NODE_EXECUTE = "execute"


class OrchestratorError(SentinelError):
    """The orchestrator was asked for something incoherent."""


def new_incident(
    alert: Alert, *, at: datetime, trust_tier: RiskTier = RiskTier.RECOMMEND
) -> IncidentState:
    """Open an incident for ``alert``.

    The incident id is derived from the alert id, so replaying the same alert
    produces the same incident rather than a second one — the same idempotence
    discipline :func:`~sentinel.core.ids.deterministic_id` gives the alert
    itself, and what makes "run the demo twice, get identical output" true of
    the agent layer as well as the detector.
    """
    return IncidentState(
        incident_id=deterministic_id("incident", alert.alert_id),
        tenant_id=alert.tenant_id,
        alert=alert,
        trust_tier=trust_tier,
        created_at=at,
        updated_at=at,
    )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def _audit(
    ctx: RunContext,
    event: AuditEventType,
    *,
    actor: AgentName,
    tenant_id: str,
    subject_id: str,
    payload: dict[str, object],
) -> None:
    if ctx.audit is not None:
        ctx.audit.append(
            event,
            actor=actor.value,
            tenant_id=tenant_id,
            subject_id=subject_id,
            payload=payload,
        )


def _triage_node(agent: TriageAgent):
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        if state.alert.triage is not None:
            # Re-entry. Re-triaging would raise, because Alert.with_triage refuses
            # to overwrite a verdict already written to the audit log.
            return state
        result = agent.triage(state.alert, now=ctx.now())
        _audit(
            ctx,
            AuditEventType.TRIAGE_DECIDED,
            actor=AgentName.TRIAGE,
            tenant_id=state.tenant_id,
            subject_id=state.alert.alert_id,
            payload={
                "decision": result.decision.value,
                "severity": result.severity.value,
                "confidence": result.confidence,
                "technique_id": result.technique_id,
                "anomaly_score": result.anomaly_score,
                "model_version": result.model_version,
            },
        )
        if result.injection_verdict is InjectionVerdict.LIKELY_INJECTION:
            _audit(
                ctx,
                AuditEventType.INJECTION_DETECTED,
                actor=AgentName.TRIAGE,
                tenant_id=state.tenant_id,
                subject_id=state.alert.alert_id,
                # The payload digest, never the payload. The audit log is exported
                # to a customer's SIEM; copying the attacker's text into it makes
                # the tamper-evident record a second delivery channel.
                payload={
                    "verdict": result.injection_verdict.value,
                    "payload_sha256": state.alert.raw_payload.digest,
                    "summary": state.alert.injection_scan.summary(),
                },
            )
        return state.with_triage(result, at=ctx.now())

    return node


def _investigate_node(agent: InvestigationAgent):
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        report = agent.investigate(state.alert, now=ctx.now())
        _audit(
            ctx,
            AuditEventType.INVESTIGATION_COMPLETED,
            actor=AgentName.INVESTIGATION,
            tenant_id=state.tenant_id,
            subject_id=state.alert.alert_id,
            payload={
                "report_id": report.report_id,
                "n_claims": len(report.claims),
                "n_evidence": len(report.evidence),
                "techniques": list(report.techniques),
                "grounded": report.is_grounded,
            },
        )
        return state.with_report(report, at=ctx.now())

    return node


def _contain_node(agent: ContainmentAgent):
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        proposal = agent.propose(
            state.alert,
            report=state.report,
            tier=state.trust_tier,
            now=ctx.now(),
            audit_hash_prev=_audit_head(ctx),
        )
        if proposal.action is None:
            _audit(
                ctx,
                AuditEventType.POLICY_UPDATED,
                actor=AgentName.CONTAINMENT,
                tenant_id=state.tenant_id,
                subject_id=state.alert.alert_id,
                payload={
                    "response": proposal.response.value,
                    "tier": proposal.tier.value,
                    "action": None,
                },
            )
            return state
        action = proposal.action
        _audit(
            ctx,
            AuditEventType.ACTION_PROPOSED,
            actor=AgentName.CONTAINMENT,
            tenant_id=state.tenant_id,
            subject_id=action.action_id,
            payload={
                "alert_id": action.alert_id,
                "action_type": action.action_type.value,
                "target": action.target,
                "risk_tier": action.risk_tier.value,
                "requires_human_approval": action.requires_human_approval,
                "response": proposal.response.value,
                "exploratory": proposal.exploratory,
            },
        )
        if action.requires_human_approval:
            _audit(
                ctx,
                AuditEventType.APPROVAL_REQUESTED,
                actor=AgentName.CONTAINMENT,
                tenant_id=state.tenant_id,
                subject_id=action.action_id,
                payload={
                    "action_type": action.action_type.value,
                    "target": action.target,
                    "rationale": action.rationale,
                },
            )
        return state.with_action(action, at=ctx.now())

    return node


def _approve_node():
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        action = _latest_pending(state)
        if action is None:
            raise OrchestratorError(
                "the approval gate was reached with no pending action; this is a "
                "routing bug, and answering it either way would record a human "
                "decision about nothing"
            )
        decision = ctx.require_decision()
        at = ctx.now()
        if decision.approved:
            updated = action.approve(approver=decision.approver, at=at)
            event = AuditEventType.APPROVAL_GRANTED
        else:
            updated = action.reject(approver=decision.approver, at=at)
            event = AuditEventType.APPROVAL_DENIED
        _audit(
            ctx,
            event,
            actor=AgentName.ORCHESTRATOR,
            tenant_id=state.tenant_id,
            subject_id=action.action_id,
            payload={
                "approver": decision.approver,
                "approved": decision.approved,
                "action_type": action.action_type.value,
                "target": action.target,
                "note": decision.note,
            },
        )
        return state.with_action(updated, at=at)

    return node


def _execute_node(connector: SimulatedConnector):
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        action = _executable(state)
        if action is None:
            raise OrchestratorError(
                "the execution node was reached with no executable action; this is a "
                "routing bug"
            )
        at = ctx.now()
        try:
            outcome = connector.execute(action)
        except Exception as exc:  # a connector failure is an outcome, not a crash
            failed = action.mark_failed(reason=f"{type(exc).__name__}: {exc}", at=at)
            _audit(
                ctx,
                AuditEventType.ACTION_FAILED,
                actor=AgentName.CONTAINMENT,
                tenant_id=state.tenant_id,
                subject_id=action.action_id,
                payload={"reason": failed.failure_reason},
            )
            return state.with_action(failed, at=at)

        if not outcome.succeeded:
            failed = action.mark_failed(reason=outcome.detail, at=at)
            _audit(
                ctx,
                AuditEventType.ACTION_FAILED,
                actor=AgentName.CONTAINMENT,
                tenant_id=state.tenant_id,
                subject_id=action.action_id,
                payload={"reason": outcome.detail},
            )
            return state.with_action(failed, at=at)

        executed = action.mark_executed(at=at)
        _audit(
            ctx,
            AuditEventType.ACTION_EXECUTED,
            actor=AgentName.CONTAINMENT,
            tenant_id=state.tenant_id,
            subject_id=action.action_id,
            payload={
                "action_type": executed.action_type.value,
                "target": executed.target,
                # Recorded so verify_no_ungated_execution can read F-08 off the log
                # without consulting the objects that enforce it.
                "requires_human_approval": executed.requires_human_approval,
                "approved_by": executed.approved_by,
                "detail": outcome.detail,
            },
        )
        return state.with_action(executed, at=at)

    return node


def _audit_head(ctx: RunContext) -> str | None:
    if ctx.audit is None:
        return None
    head = ctx.audit.head()
    return None if head is None else head[1]


def _latest_pending(state: IncidentState) -> ActionRequest | None:
    return pending_action(state.actions)


def _executable(state: IncidentState) -> ActionRequest | None:
    return executable_action(state.actions)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def _after_triage(state: IncidentState) -> str:
    triage = state.triage
    if triage is None:  # pragma: no cover - the node always sets one
        return "keep"
    return "dismiss" if triage.decision is TriageDecision.AUTO_DISMISS else "keep"


def _after_contain(state: IncidentState) -> str:
    action = _latest_pending(state)
    if action is None:
        return "none"
    return "gate" if action.requires_human_approval else "execute"


def _after_approve(state: IncidentState) -> str:
    for action in reversed(state.actions):
        if action.approval_status is ApprovalStatus.APPROVED:
            return "approved"
        if action.approval_status is ApprovalStatus.REJECTED:
            return "denied"
    return "denied"  # pragma: no cover - the node always decides


def _dismiss(state: IncidentState, ctx: RunContext) -> IncidentState:
    return state.finished(IncidentStatus.DISMISSED, at=ctx.now())


def build_incident_graph(
    *,
    triage: TriageAgent,
    investigation: InvestigationAgent,
    containment: ContainmentAgent,
    connector: SimulatedConnector | None = None,
    checkpointer=None,
    step_budget: int = 16,
) -> CompiledGraph:
    """Compile PRD Figure 3.

    ``step_budget`` is 16 rather than the runtime default: this graph's longest
    path is five nodes, so anything past a handful is a routing bug and the
    budget should say so early.
    """
    spec = GraphSpec()
    spec.add_node(NODE_TRIAGE, _triage_node(triage))
    spec.add_node("dismiss", _dismiss)
    spec.add_node(NODE_INVESTIGATE, _investigate_node(investigation))
    spec.add_node(NODE_CONTAIN, _contain_node(containment))
    spec.add_node(NODE_APPROVE, _approve_node())
    spec.add_node(NODE_EXECUTE, _execute_node(connector or SimulatedConnector()))

    spec.add_conditional_edges(
        NODE_TRIAGE, _after_triage, {"dismiss": "dismiss", "keep": NODE_INVESTIGATE}
    )
    spec.add_edge("dismiss", END)
    spec.add_edge(NODE_INVESTIGATE, NODE_CONTAIN)
    spec.add_conditional_edges(
        NODE_CONTAIN,
        _after_contain,
        {"none": END, "gate": NODE_APPROVE, "execute": NODE_EXECUTE},
    )
    spec.add_conditional_edges(
        NODE_APPROVE, _after_approve, {"approved": NODE_EXECUTE, "denied": END}
    )
    spec.add_edge(NODE_EXECUTE, END)
    spec.set_entry_point(NODE_TRIAGE)

    return spec.compile(
        checkpointer=checkpointer,
        # The gate, and only the gate. An interrupt anywhere else would be a pause
        # nobody asked a question for.
        interrupt_before=[NODE_APPROVE],
        step_budget=step_budget,
    )


# --------------------------------------------------------------------------- #
# Section 9.1 timing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IncidentTimings:
    """MTTD/MTTC inputs for one incident, read from checkpointed step history."""

    #: Triage end minus ``alert.ingested_at`` — Section 9.1's literal definition.
    #: On a historical replay corpus this is dominated by the age of the capture;
    #: see :func:`incident_timings`.
    detect_seconds: float | None
    #: Triage end minus the moment the incident was opened: time spent *inside*
    #: the pipeline. Equal to ``detect_seconds`` on a live or replayed feed.
    pipeline_detect_seconds: float | None
    #: How stale the alert already was when the pipeline received it.
    feed_lag_seconds: float | None
    contain_seconds: float | None
    approval_wait_seconds: float | None

    @property
    def detected(self) -> bool:
        return self.detect_seconds is not None

    @property
    def contained(self) -> bool:
        return self.contain_seconds is not None


def incident_timings(state: IncidentState) -> IncidentTimings:
    """Derive Section 9.1's timings from the run's own history.

    Section 9.1 defines MTTD as *"from alert ingestion to triage
    classification"*, measured as a *"timestamp delta in the pipeline logs"*.
    Two timestamps can play the role of "ingestion" and they are not always the
    same instant, so both are reported:

    ``detect_seconds``
        From ``alert.ingested_at``. This is the right number on any live feed,
        and on the PRD's own demo path: the replay service
        (:mod:`sentinel.ingest.replay`) stamps ``ingested_at`` with the wall
        clock as it emits each record, so the two definitions coincide there and
        a test asserts it.
    ``pipeline_detect_seconds``
        From the moment the incident was opened. On a *static* corpus — alerts
        built by the offline generator rather than streamed — ``ingested_at``
        carries the original capture timestamp, so ``detect_seconds`` measures
        how old CIC-IDS2017 is (about nine years) rather than how fast triage
        was. Reporting only that number would be absurd; silently substituting
        the other one would be dishonest. Both are here, and
        ``feed_lag_seconds`` is the difference, which on a live feed is the
        genuine ingestion lag and is worth watching in its own right.

    ``contain_seconds`` runs from the triage verdict to the end of execution,
    because Section 9.1 measures *"confirmed threat to approved containment"*
    and the confirmation is the triage verdict. It includes the human's thinking
    time, which is the honest choice: the analyst is inside the loop by design,
    and a containment metric that excluded the part of the loop the PRD put a
    human in would be measuring the software rather than the system.
    ``approval_wait_seconds`` reports that component separately so the two can be
    told apart.
    """
    triage_end = _step_end(state, NODE_TRIAGE)
    if triage_end is None:
        return IncidentTimings(None, None, None, None, None)
    detect = (triage_end - state.alert.ingested_at).total_seconds()
    pipeline = (triage_end - state.created_at).total_seconds()
    lag = (state.created_at - state.alert.ingested_at).total_seconds()

    execute_end = _step_end(state, NODE_EXECUTE)
    contain = None if execute_end is None else (execute_end - triage_end).total_seconds()

    wait = None
    if state.interrupt is not None:
        wait = (state.updated_at - state.interrupt.requested_at).total_seconds()
    else:
        approve_step = next(
            (r for r in state.history if r.node == NODE_APPROVE), None
        )
        contain_step = next(
            (r for r in reversed(state.history) if r.node == NODE_CONTAIN), None
        )
        if approve_step is not None and contain_step is not None:
            wait = (approve_step.started_at - contain_step.ended_at).total_seconds()

    return IncidentTimings(
        detect_seconds=max(0.0, detect),
        pipeline_detect_seconds=max(0.0, pipeline),
        feed_lag_seconds=max(0.0, lag),
        contain_seconds=None if contain is None else max(0.0, contain),
        approval_wait_seconds=None if wait is None else max(0.0, wait),
    )


def _step_end(state: IncidentState, node: str) -> datetime | None:
    for record in reversed(state.history):
        if record.node == node and record.outcome is StepOutcome.OK:
            return record.ended_at
    return None


@dataclass(slots=True)
class Orchestrator:
    """Convenience wrapper: one object that owns the graph and its audit log."""

    graph: CompiledGraph
    audit: HashChainedAuditLog | None = None
    clock: object = field(default=None)

    def start(self, state: IncidentState, **kwargs):
        return self.graph.invoke(state, audit=self.audit, **kwargs)

    def approve(self, thread_id: str, decision, **kwargs):
        return self.graph.resume(thread_id, decision, audit=self.audit, **kwargs)

    def pending(self):
        return self.graph.pending()
