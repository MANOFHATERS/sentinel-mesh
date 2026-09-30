"""JSON views over the workspace: what the dashboard's HTTP API returns.

Pure functions from contracts to plain dictionaries. They live apart from the API
so they can be tested without a web server, and apart from the workspace so the
workspace stays a thin layer over the graphs.

Untrusted text stays text
-------------------------
Three fields here carry attacker-influenced text: ``alert.raw_payload``, every
``Evidence.excerpt`` (a raw log line, a KB chunk, a code excerpt) and a code
finding's ``excerpt``. They are emitted as plain strings **and flagged**
(``"untrusted": true`` on the enclosing object), and the front end renders every
string with ``textContent``, never as markup — see ``static/js/dom.js``. A
dashboard that interpolated an alert payload into HTML would hand the attacker the
analyst's session, which is a worse outcome than the prompt injection Section 5.7
defends the agents against. ``test_dashboard_api.py`` pushes a ``<script>`` payload
through the whole path and asserts it comes back as inert JSON text.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.agents.orchestrator import incident_timings
from sentinel.agents.state import IncidentState, StepRecord
from sentinel.audit.log import AuditRecord, FindingKind, VerificationResult
from sentinel.core.schemas import (
    ActionRequest,
    ApprovalStatus,
    AuditEventType,
    Evidence,
    InvestigationReport,
    TriageResult,
)
from sentinel.dashboard.scenarios import ASSET_INVENTORY, SCENARIOS, ScenarioName
from sentinel.dashboard.workspace import (
    GraphKind,
    ThreadRef,
    Workspace,
    scenario_progress,
)
from sentinel.graph.explain import ExposurePath, NodeExplanation, top_risk_explanations
from sentinel.graph.schema import NodeKind, SupplyChainGraph

__all__ = [
    "action_view",
    "audit_chain_view",
    "audit_view",
    "code_scan_view",
    "evaluation_view",
    "graph_view",
    "incident_detail",
    "incident_summary",
    "incidents_view",
    "node_view",
    "overview",
    "queue_view",
    "scenario_name",
    "scenarios_view",
    "wire_view",
]


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _asset(address: str | None) -> dict[str, Any] | None:
    if address is None:
        return None
    asset = ASSET_INVENTORY.get(address)
    return {
        "address": address,
        "hostname": None if asset is None else asset.hostname,
        "role": None if asset is None else asset.role,
        "protected": False if asset is None else asset.protected,
    }


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #


def evidence_view(item: Evidence) -> dict[str, Any]:
    return {
        "kind": item.kind.value,
        "ref": item.ref,
        "excerpt": item.excerpt.raw,
        "relevance": _finite(item.relevance),
        "untrusted": True,
    }


def action_view(action: ActionRequest) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "action_type": action.action_type.value,
        "destructive": action.action_type.is_destructive,
        "target": action.target,
        "target_asset": _asset(action.target),
        "rationale": action.rationale,
        "proposed_by": action.proposed_by.value,
        "risk_tier": action.risk_tier.value,
        "requires_human_approval": action.requires_human_approval,
        "status": action.approval_status.value,
        "approved_by": action.approved_by,
        "decided_at": _iso(action.decided_at),
        "executed_at": _iso(action.executed_at),
        "failure_reason": action.failure_reason,
        "created_at": _iso(action.created_at),
        "evidence": [evidence_view(item) for item in action.evidence],
    }


def triage_view(result: TriageResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "decision": result.decision.value,
        "severity": result.severity.value,
        "confidence": _finite(result.confidence),
        "technique_id": result.technique_id,
        "rationale": result.rationale,
        "supporting_fields": list(result.supporting_fields),
        "anomaly_score": _finite(result.anomaly_score),
        "injection_verdict": result.injection_verdict.value,
        "model_version": result.model_version,
        "latency_ms": _finite(result.latency_ms),
        "decided_at": _iso(result.decided_at),
    }


def report_view(report: InvestigationReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "report_id": report.report_id,
        "agent": report.agent.value,
        "summary": report.summary,
        "severity": report.severity.value,
        "confidence": _finite(report.confidence),
        "techniques": list(report.techniques),
        "recommended_actions": [a.value for a in report.recommended_actions],
        "claims": [
            {"statement": statement, "refs": list(refs)} for statement, refs in report.claims
        ],
        "evidence": [evidence_view(item) for item in report.evidence],
        "grounded": report.is_grounded,
        "model_version": report.model_version,
        "created_at": _iso(report.created_at),
    }


def step_view(step: StepRecord) -> dict[str, Any]:
    return {
        "node": step.node,
        "outcome": step.outcome.value,
        "started_at": _iso(step.started_at),
        "ended_at": _iso(step.ended_at),
        "duration_ms": _finite(step.duration_ms),
        "detail": step.detail,
    }


def audit_view(record: AuditRecord) -> dict[str, Any]:
    return {
        "seq": record.seq,
        "recorded_at": record.recorded_at,
        "event_type": record.event_type.value,
        "actor": record.actor,
        "subject_id": record.subject_id,
        "payload": record.payload,
        "row_hash": record.row_hash,
        "prev_hash": record.prev_hash,
    }


# --------------------------------------------------------------------------- #
# Incidents
# --------------------------------------------------------------------------- #


def _failed(state: IncidentState) -> list[ActionRequest]:
    return [a for a in state.actions if a.approval_status is ApprovalStatus.FAILED]


def incident_summary(ref: ThreadRef, state: IncidentState) -> dict[str, Any]:
    triage = state.triage
    pending = state.pending_action
    timings = incident_timings(state) if ref.kind == GraphKind.INCIDENT else None
    severity = (
        state.report.severity.value
        if state.report is not None
        else None if triage is None else triage.severity.value
    )
    return {
        "incident_id": state.incident_id,
        "kind": ref.kind,
        "scenario": None if ref.scenario is None else ref.scenario.value,
        "caption": ref.caption,
        "advisory_id": ref.advisory_id,
        "status": state.status.value,
        "severity": severity,
        "decision": None if triage is None else triage.decision.value,
        "confidence": None if triage is None else _finite(triage.confidence),
        "technique_id": None if triage is None else triage.technique_id,
        "injection_verdict": None if triage is None else triage.injection_verdict.value,
        "source": state.alert.source.value,
        "asset": _asset(state.alert.asset_id),
        "src_ip": state.alert.src_ip,
        "dst_ip": state.alert.dst_ip,
        "created_at": _iso(state.created_at),
        "updated_at": _iso(state.updated_at),
        "pending_action": None if pending is None else action_view(pending),
        "actions": [
            {
                "action_id": a.action_id,
                "action_type": a.action_type.value,
                "target": a.target,
                "status": a.approval_status.value,
            }
            for a in state.actions
        ],
        "failed_actions": len(_failed(state)),
        "detect_seconds": None if timings is None else _finite(timings.detect_seconds),
        "contain_seconds": None if timings is None else _finite(timings.contain_seconds),
        "error": state.error,
    }


def incident_detail(workspace: Workspace, thread_id: str) -> dict[str, Any]:
    ref = workspace.ref(thread_id)
    state = workspace.state(thread_id)
    chain_ok, checkpoints, chain_error = workspace.checkpoint_chain_ok(thread_id)
    subjects = {state.alert.alert_id, state.incident_id, *(a.action_id for a in state.actions)}
    records = workspace.records(sorted(subjects))
    action_ids = {a.action_id for a in state.actions}
    alert = state.alert
    scan = alert.injection_scan
    timings = incident_timings(state) if ref.kind == GraphKind.INCIDENT else None
    detail = incident_summary(ref, state)
    detail.update(
        {
            "alert": {
                "alert_id": alert.alert_id,
                "source": alert.source.value,
                "timestamp": _iso(alert.timestamp),
                "ingested_at": _iso(alert.ingested_at),
                "asset": _asset(alert.asset_id),
                "signature": alert.signature,
                "src_ip": alert.src_ip,
                "dst_ip": alert.dst_ip,
                "src_port": alert.src_port,
                "dst_port": alert.dst_port,
                "protocol": alert.protocol,
                "dataset": alert.dataset,
                "raw_payload": alert.raw_payload.raw,
                "untrusted": True,
                "injection": {
                    "verdict": scan.verdict.value,
                    "summary": scan.summary(),
                    "sha256": alert.raw_payload.digest,
                },
                "features": {
                    name: value
                    for name, value in alert.features.items()
                    if not isinstance(value, float) or math.isfinite(value)
                },
            },
            "triage": triage_view(state.triage),
            "report": report_view(state.report),
            "actions_detail": [action_view(a) for a in state.actions],
            "interrupt": None
            if state.interrupt is None
            else {
                "node": state.interrupt.node,
                "reason": state.interrupt.reason,
                "subject_id": state.interrupt.subject_id,
                "requested_at": _iso(state.interrupt.requested_at),
            },
            "history": [step_view(step) for step in state.history],
            "trust_tier": state.trust_tier.value,
            "checkpoints": {"count": checkpoints, "verified": chain_ok, "error": chain_error},
            "timings": None
            if timings is None
            else {
                "detect_seconds": _finite(timings.detect_seconds),
                "pipeline_detect_seconds": _finite(timings.pipeline_detect_seconds),
                "feed_lag_seconds": _finite(timings.feed_lag_seconds),
                "contain_seconds": _finite(timings.contain_seconds),
                "approval_wait_seconds": _finite(timings.approval_wait_seconds),
            },
            "timeline": [audit_view(record) for record in records],
            "wire": _wire_for(workspace, records, action_ids),
            "policy": _policy_for(records),
            "stalled": not state.status.is_terminal and not state.is_waiting,
        }
    )
    return detail


def _policy_for(records: Sequence[AuditRecord]) -> dict[str, Any] | None:
    """The response policy's reasoning for this incident, read from the audit row.

    From the chain rather than recomputed: the dashboard shows what the Containment
    Agent recorded when it decided, which is what an auditor would read.
    """
    for record in records:
        if record.event_type in (
            AuditEventType.ACTION_PROPOSED,
            AuditEventType.POLICY_UPDATED,
        ) and isinstance(record.payload.get("policy"), dict):
            return {**record.payload["policy"], "seq": record.seq}
    return None


def _wire_for(
    workspace: Workspace, records: Sequence[AuditRecord], action_ids: set[str]
) -> dict[str, Any]:
    return {
        "calls": [
            audit_view(r) for r in records if r.event_type is AuditEventType.CONNECTOR_CALLED
        ],
        "refusals": [
            _refusal(r) for r in records if r.event_type is AuditEventType.GUARDRAIL_BLOCKED
        ],
        "executions": [
            _execution(item)
            for item in list(workspace.router.executions)
            if item.action.action_id in action_ids
        ],
    }


def _refusal(record: AuditRecord) -> dict[str, Any]:
    payload = record.payload
    return {
        "seq": record.seq,
        "recorded_at": record.recorded_at,
        "action_id": record.subject_id,
        "actor": record.actor,
        "action_type": payload.get("action_type"),
        "guardrail": payload.get("guardrail"),
        "reason": payload.get("reason"),
    }


def _execution(item) -> dict[str, Any]:
    return {
        "action_id": item.action.action_id,
        "action_type": item.action.action_type.value,
        "target": item.action.target,
        "connector": item.connector,
        "succeeded": item.outcome.succeeded,
        "detail": item.outcome.detail,
        "reference": item.outcome.reference,
        "replayed": item.outcome.replayed,
        "approved_by": item.action.approved_by,
    }


def queue_view(workspace: Workspace) -> list[dict[str, Any]]:
    """Every run waiting on a human, across all three graphs, newest request first.

    Ordered like :func:`~sentinel.agents.contain.approval_queue` (newest first), by
    the moment the gate was reached.
    """
    items = [
        incident_summary(ref, state)
        for ref, state in workspace.states()
        if state.is_waiting and state.pending_action is not None
    ]
    return sorted(
        items,
        key=lambda item: (item["pending_action"]["created_at"], item["incident_id"]),
        reverse=True,
    )


def incidents_view(workspace: Workspace, *, status: str | None = None,
                   kind: str | None = None, limit: int = 200) -> dict[str, Any]:
    rows = [incident_summary(ref, state) for ref, state in workspace.states()]
    rows.reverse()  # newest first
    if status:
        rows = [row for row in rows if row["status"] == status]
    if kind:
        rows = [row for row in rows if row["kind"] == kind]
    return {"total": len(rows), "items": rows[: max(1, min(limit, 1000))]}


# --------------------------------------------------------------------------- #
# Overview and scenarios
# --------------------------------------------------------------------------- #


def overview(workspace: Workspace) -> dict[str, Any]:
    states = workspace.states()
    by_status: dict[str, int] = {}
    ingested = reached_human = 0
    detect: list[float] = []
    contain: list[float] = []
    for ref, state in states:
        by_status[state.status.value] = by_status.get(state.status.value, 0) + 1
        if ref.kind != GraphKind.INCIDENT:
            continue
        ingested += 1
        if state.is_waiting or any(a.requires_human_approval for a in state.actions):
            reached_human += 1
        timings = incident_timings(state)
        if timings.detect_seconds is not None:
            detect.append(timings.detect_seconds)
        if timings.contain_seconds is not None:
            contain.append(timings.contain_seconds)
    chain = workspace.verify_audit()
    ungated = workspace.ungated()
    failed = workspace.failed_actions()
    refusals = workspace.refusal_records()
    return {
        "tenant_id": workspace.tenant_id,
        "incidents": len(states),
        "by_status": by_status,
        "queue": sum(1 for _, state in states if state.is_waiting),
        "stalled": sum(
            1 for _, s in states if not s.status.is_terminal and not s.is_waiting
        ),
        "alerts_ingested": ingested,
        "alerts_reaching_human": reached_human,
        "alert_reduction": None if ingested == 0 else 1.0 - reached_human / ingested,
        "mttd_seconds": None if not detect else float(np.mean(detect)),
        "mttc_seconds": None if not contain else float(np.mean(contain)),
        "audit": _chain_view(chain, fresh=not states),
        "ungated_executions": list(ungated),
        "failed_actions": len(failed),
        "refusals": len(refusals),
        "feed_remaining": workspace.feed_remaining(),
        "scenarios": scenarios_view(workspace),
    }


def _chain_view(result: VerificationResult, *, fresh: bool) -> dict[str, Any]:
    """The chain's state as the dashboard shows it: verified, empty, or broken.

    :meth:`~sentinel.audit.log.HashChainedAuditLog.verify` reports an empty log as a
    finding, deliberately — a log truncated to nothing is the cheapest way to erase
    history. A *fresh* workspace legitimately has no rows, though, and showing it as
    BROKEN teaches analysts to ignore the badge. So "empty" is only reported when
    the workspace's own registry agrees nothing has run (``fresh``); an empty log in
    a workspace that has runs is exactly the truncation the finding exists for.
    """
    only_empty = not result.ok and all(
        finding.kind is FindingKind.EMPTY for finding in result.findings
    )
    status = "verified" if result.ok else ("empty" if only_empty and fresh else "broken")
    return {
        "ok": result.ok,
        "status": status,
        "rows": result.rows_checked,
        "head_seq": result.head_seq,
        "head_hash": result.head_hash,
        "findings": [str(finding) for finding in result.findings[:20]],
        "duration_ms": result.duration_seconds * 1000.0,
        "summary": result.summary(),
    }


def audit_chain_view(workspace: Workspace) -> dict[str, Any]:
    return _chain_view(workspace.verify_audit(), fresh=not workspace.refs())


def scenarios_view(workspace: Workspace) -> list[dict[str, Any]]:
    if workspace.models.mode == "real":
        return []  # scripted stories belong to the synthetic demo
    runs = workspace.scenario_runs()
    items = []
    for name, spec in SCENARIOS.items():
        progress = scenario_progress(workspace, name)
        run = runs.get(name)
        threads = [
            {"incident_id": ref.thread_id, "caption": ref.caption, "kind": ref.kind,
             "status": workspace.state(ref.thread_id).status.value}
            for ref in workspace.refs()
            if ref.scenario is name
        ]
        items.append(
            {
                "name": name.value,
                "title": spec.title,
                "summary": spec.summary,
                "walkthrough": list(spec.walkthrough),
                "agents": list(spec.agents),
                "launched": progress.launched,
                "launched_at": None if run is None else _iso(run.launched_at),
                "launched_by": None if run is None else run.launched_by,
                "complete": progress.complete,
                "total": progress.total,
                "waiting": progress.waiting,
                "running": progress.running,
                "finished": progress.finished,
                "failed_runs": progress.failed_runs,
                "failed_actions": progress.failed_actions,
                "decisions": progress.decisions,
                "threads": threads,
            }
        )
    return items


def scenario_name(value: str) -> ScenarioName:
    return ScenarioName(value)


# --------------------------------------------------------------------------- #
# Wire and guardrails
# --------------------------------------------------------------------------- #


def wire_view(workspace: Workspace) -> dict[str, Any]:
    records = workspace.records()
    sandbox = workspace.sandbox
    calls = [r for r in records if r.event_type is AuditEventType.CONNECTOR_CALLED]
    by_connector: dict[str, int] = {}
    for record in calls:
        name = str(record.payload.get("connector"))
        by_connector[name] = by_connector.get(name, 0) + 1
    failed = [
        {**action_view(action), "incident_id": ref.thread_id, "caption": ref.caption}
        for ref, action in workspace.failed_actions()
    ]
    return {
        "executions": [_execution(item) for item in list(workspace.router.executions)],
        "refusals": [
            _refusal(r) for r in records if r.event_type is AuditEventType.GUARDRAIL_BLOCKED
        ],
        "failed_actions": failed,
        "calls": {"total": len(calls), "by_connector": by_connector,
                  "recent": [audit_view(r) for r in calls[-25:]]},
        "remote": {
            "wazuh": {
                "isolated_hosts": sorted(sandbox.wazuh.isolated_hosts()),
                "blocked_addresses": sorted(sandbox.wazuh.blocked_addresses()),
                "commands": len(sandbox.wazuh.executed),
            },
            "github": {
                "pulls": [
                    {"number": p["number"], "title": p["title"], "draft": p["draft"],
                     "state": p["state"], "merged": p["merged"], "url": p["html_url"],
                     "head": p["head"]["ref"]}
                    for p in sandbox.github.pulls
                ],
                "issues": [
                    {"number": i["number"], "title": i["title"], "state": i["state"],
                     "url": i["html_url"]}
                    for i in sandbox.github.issues
                ],
                "merges": sandbox.github.merges,
            },
            "slack": {
                "messages": len(sandbox.slack.messages),
                "recent": [str(m.get("text", ""))[:500] for m in sandbox.slack.messages[-5:]],
            },
            "scim": {
                "users": [
                    {"userName": user.get("userName"), "active": user.get("active")}
                    for user in sandbox.scim.users.values()
                ],
            },
        },
        "guardrails": {
            "protected_networks": list(workspace.router.targets.protected_networks),
            "protected_hosts": sorted(workspace.router.targets.protected_hosts),
            "blast_radius_per_hour": None
            if workspace.router.limiter is None
            else workspace.router.limiter.max_actions,
            "tenant_id": workspace.router.tenant_id,
            "capabilities": sorted(c.value for c in workspace.router.capabilities),
        },
    }


# --------------------------------------------------------------------------- #
# Supply chain
# --------------------------------------------------------------------------- #


def _path_view(path: ExposurePath) -> dict[str, Any]:
    return {
        "nodes": list(path.nodes),
        "hops": path.hops,
        "source_cves": path.source_cves,
        "source_days_stale": _finite(path.source_days_stale),
        "contribution": _finite(path.contribution),
        "describe": path.describe(),
    }


def node_view(explanation: NodeExplanation, graph: SupplyChainGraph) -> dict[str, Any]:
    node = graph.node(explanation.node_id)
    return {
        "node_id": node.node_id,
        "name": node.name,
        "kind": node.kind.value,
        "risk": _finite(explanation.risk_score),
        "intrinsic": explanation.is_intrinsically_risky,
        "driver": explanation.dominant_driver,
        "own_feature_share": _finite(explanation.own_feature_share),
        "neighbourhood_share": _finite(explanation.neighbourhood_share),
        "features": {
            "cve_exposure_count": node.cve_exposure_count,
            "days_since_last_update": _finite(node.days_since_last_update),
            "sbom_depth": node.sbom_depth,
            "breach_history": node.breach_history,
            "unmaintained": node.is_unmaintained,
        },
        "paths": [_path_view(path) for path in explanation.paths],
        "describe": explanation.describe(),
    }


def _nodes_and_edges(
    graph: SupplyChainGraph, ids: Iterable[str], scores: np.ndarray,
    shares: np.ndarray, sources: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keep = set(ids)
    nodes = []
    for node in graph.nodes:
        if node.node_id not in keep:
            continue
        index = graph.index_of(node.node_id)
        nodes.append(
            {
                "id": node.node_id,
                "name": node.name,
                "kind": node.kind.value,
                "risk": _finite(scores[index]),
                "neighbourhood_share": _finite(shares[index]),
                "intrinsic": node.node_id in sources,
                "cves": node.cve_exposure_count,
            }
        )
    edges = [
        {"source": e.source, "target": e.target, "kind": e.kind.value}
        for e in graph.edges
        if e.source in keep and e.target in keep
    ]
    return nodes, edges


def graph_view(workspace: Workspace, *, scope: str = "top", k: int = 10) -> dict[str, Any]:
    """The supply-chain map's data: nodes, edges, and the paths to highlight.

    ``scope`` is ``"top"`` (the top-``k`` nodes of the whole graph and every node on
    their explanation paths — the dashboard's default review), ``"all"``, or
    ``"advisory:<id>"`` (exactly the scope that advisory's review assessed).
    """
    from sentinel.dashboard.scenarios import risk_sources

    graph = workspace.graph
    scores = workspace.scores
    shares = workspace.shares
    sources = risk_sources(graph)
    advisory = None
    highlight: list[list[str]] = []
    if scope.startswith("advisory:"):
        published = workspace.advisory(scope.split(":", 1)[1])
        ids = list(published.scope)
        advisory = {
            "advisory_id": published.advisory.advisory_id,
            "kind": published.advisory.kind.value,
            "package_id": published.advisory.package_id,
            "title": published.advisory.title,
            "summary": published.advisory.summary,
            "cve_ids": list(published.advisory.cve_ids),
            "published_at": _iso(published.published_at),
        }
        explanations = top_risk_explanations(
            published.subgraph, published.scores, k=10,
            shares=np.array([shares[graph.index_of(n)] for n in published.subgraph.node_ids()]),
            include=(published.advisory.package_id,), max_hops=4,
        )
        for explanation in explanations:
            highlight.extend(list(path.nodes) for path in explanation.paths[:1])
    elif scope == "all":
        ids = graph.node_ids()
        explanations = top_risk_explanations(graph, scores, k=k, shares=shares, max_hops=4)
    elif scope == "top":
        explanations = top_risk_explanations(graph, scores, k=k, shares=shares, max_hops=4)
        wanted: dict[str, None] = {}
        for explanation in explanations:
            wanted[explanation.node_id] = None
            for path in explanation.paths[:2]:
                wanted.update(dict.fromkeys(path.nodes))
        ids = list(wanted)
    else:
        raise ValueError(f"unknown scope {scope!r}")
    if not highlight:
        for explanation in explanations:
            highlight.extend(list(path.nodes) for path in explanation.paths[:1])
    nodes, edges = _nodes_and_edges(graph, ids, scores, shares, sources)
    return {
        "scope": scope,
        "advisory": advisory,
        "nodes": nodes,
        "edges": edges,
        "flagged": [
            {"node_id": e.node_id, "kind": e.kind.value, "risk": _finite(e.risk_score),
             "intrinsic": e.is_intrinsically_risky, "driver": e.dominant_driver,
             "paths": len(e.paths)}
            for e in explanations
        ],
        "highlight": highlight,
        "advisories": [
            {"advisory_id": p.advisory.advisory_id, "title": p.advisory.title,
             "package_id": p.advisory.package_id, "kind": p.advisory.kind.value}
            for p in workspace.advisories()
        ],
        "counts": {
            "nodes": graph.n_nodes,
            "edges": len(graph.edges),
            "by_kind": {kind.value: len(graph.nodes_of_kind(kind)) for kind in NodeKind},
            "risk_sources": len(sources),
        },
    }


# --------------------------------------------------------------------------- #
# Code scan
# --------------------------------------------------------------------------- #


def code_scan_view(workspace: Workspace) -> dict[str, Any] | None:
    found = workspace.code_scan()
    if found is None:
        return None
    ref, assessment = found
    state = workspace.state(ref.thread_id)
    draft = assessment.draft
    diff_digest = None if draft is None else _sha256(draft.combined_diff)
    proposed_digest = None
    for record in workspace.records([a.action_id for a in state.actions]):
        if record.event_type is AuditEventType.ACTION_PROPOSED:
            proposed_digest = record.payload.get("diff_sha256")
    execution = next(
        (
            _execution(item)
            for item in list(workspace.router.executions)
            if any(item.action.action_id == a.action_id for a in state.actions)
        ),
        None,
    )
    return {
        "incident_id": ref.thread_id,
        "status": state.status.value,
        "repository": "acme/billing",
        "files_scanned": assessment.scan.files_scanned,
        "lines_scanned": assessment.scan.lines_scanned,
        "summary": assessment.scan.summary(),
        "findings": [
            {
                "ref": finding.ref,
                "rule_id": finding.rule_id,
                "cwe": finding.cwe,
                "title": finding.title,
                "path": finding.path,
                "line": finding.line,
                "severity": finding.severity.value,
                "confidence": finding.confidence.value,
                "message": finding.message,
                "remediation": finding.remediation,
                "excerpt": finding.excerpt,
                "tainted": finding.is_tainted,
                "untrusted": True,
            }
            for finding in assessment.findings
        ],
        "security_findings": list(assessment.security_findings),
        "draft": None
        if draft is None
        else {
            "branch": draft.branch,
            "title": draft.title,
            "body": draft.body,
            "files": list(draft.files_touched),
            "unpatched_refs": list(draft.unpatched_refs),
            "patches": [
                {"finding_ref": p.finding_ref, "path": p.path, "rule_id": p.rule_id,
                 "diff": p.diff, "checks": list(p.checks), "notes": list(p.notes)}
                for p in draft.patches
            ],
            "diff_sha256": diff_digest,
        },
        "approval_bound_digest": proposed_digest,
        "digest_matches": diff_digest is not None and diff_digest == proposed_digest,
        "report": report_view(state.report),
        "execution": execution,
    }


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Evaluation report (F-12)
# --------------------------------------------------------------------------- #


def _gates(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key.endswith("_pass") and isinstance(item, bool):
                gates.append({"gate": path, "passed": item})
            else:
                gates.extend(_gates(item, path))
    return gates


def _augmentation_summary(section: Any) -> dict[str, Any] | None:
    if not isinstance(section, dict):
        return None
    return {
        "levels": section.get("levels"),
        "raw_peak_overconfidence": section.get("raw_peak_overconfidence"),
        "gated_peak_overconfidence": section.get("gated_peak_overconfidence"),
    }


def evaluation_view(path: Path) -> dict[str, Any]:
    """The F-12 report as the one evaluation pipeline wrote it, plus its gates.

    Read from the artifact, never recomputed: PRD Section 9.3 requires that the
    numbers shown in the dashboard are the numbers the evaluation produced.
    """
    if not path.is_file():
        return {"available": False, "path": str(path), "gates": [], "headline": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test", {}) if isinstance(data, dict) else {}
    agents = data.get("agents", {}) if isinstance(data, dict) else {}
    headline = {
        "roc_auc": test.get("roc_auc"),
        "pr_auc": test.get("pr_auc"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "alert_reduction": test.get("alert_reduction"),
        "alert_reduction_at_soc_base_rate": test.get("alert_reduction_at_soc_base_rate"),
        "per_family_recall": test.get("per_family_recall"),
        "timing": agents.get("timing"),
        "supply_chain_top_k_precision": (data.get("supply_chain") or {}).get("test"),
        "response_policy_regret_ratio": (data.get("response_policy") or {}).get(
            "regret_ratio"
        ),
        "regret_curves": (data.get("response_policy") or {}).get("curves"),
        "augmentation": _augmentation_summary(data.get("augmentation")),
    }
    gates = _gates(data)
    return {
        "available": True,
        "path": str(path),
        "n_alerts": data.get("n_alerts"),
        "dataset": data.get("dataset"),
        "spec_fingerprint": data.get("spec_fingerprint"),
        "gates": gates,
        "passed": sum(1 for gate in gates if gate["passed"]),
        "headline": headline,
        "sections": sorted(key for key, item in data.items() if isinstance(item, dict)),
    }

