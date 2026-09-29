"""The Supply-Chain Agent (PRD F-06's guardrail, Sections 3.4, 5.4, 5.5.3).

Parts 2.3 built the graph, the GraphSAGE risk propagation and the path attribution.
This module is the *agent*: the thing that decides which of 500 scored nodes a human
should be shown, cites why, proposes a remediation, and stops at the Human Approval
Gate before doing anything about it.

F-06's acceptance criterion (top-10 precision ≥ 0.80) is measured by
``scripts/evaluate.py --graph`` and belongs to the model. F-06's *guardrail* is what
this module is for:

    *"Flags are explainable via the specific graph path that drove the score."*

So the report's claims are built from :class:`~sentinel.graph.explain.ExposurePath`
objects, cited as ``graph://path/...`` refs that
:class:`~sentinel.core.schemas.InvestigationReport` validates, and the *model's* own
attribution — how much of the score came from the node's features versus its
neighbourhood — is reported **beside** the structural story rather than blended into
it. When the two disagree, that disagreement is a claim in the report, because a
confident path narration over a model that ignored the graph is the failure mode
:mod:`sentinel.graph.explain` exists to make visible.

The trigger question, which is the actual design problem
-------------------------------------------------------
The other four agents are alert-driven: something happened, an alert arrived, the
graph runs. Supply-chain risk is **continuous** — nothing happened; a dependency
has been unmaintained for four years and that was equally true yesterday. So there
is no alert to trigger on, and PRD Figure 3's graph has nowhere to put this.

Three options were available and the third is implemented:

1.  *Bolt it onto the incident graph as a sixth node.* Rejected: the node would run
    on every alert, re-scoring a 500-node graph per network flow, and its output
    would have nothing to do with the alert that triggered it.
2.  *Run it as a standalone job with its own storage.* Rejected: it would need its
    own approval gate, its own audit log and its own checkpointing, which means
    F-08's guarantee would hold in two implementations instead of one — and the
    second one would be the untested one.
3.  **A scheduled assessment that mints its own alert.**
    :class:`SupplyChainMonitor` runs on a cadence, scores the graph, and for each
    node worth a human's attention calls :func:`synthesize_vendor_alert` to mint an
    ``AlertSource.VENDOR_FEED`` alert. That alert is the subject of a run through
    :func:`build_supply_chain_review_graph`, which reuses
    :class:`~sentinel.agents.state.IncidentState`, the checkpoint chain, the gate and
    the audit log **unchanged**. Only the node set is new.

``AlertSource.VENDOR_FEED`` exists in Part 1's schema because Part 1 anticipated
this; the enum member was not added for it here.

What the remediation can and cannot be
--------------------------------------
For a flagged *package* the remediation is a dependency change, so the proposed
action is ``OPEN_PATCH_PR`` — destructive, therefore gated. The agent does **not**
draft the manifest edit, and that is a stated limitation rather than an oversight:
PRD Section 5.5.3 scopes the sprint's graph as synthetic (``package-0129`` is not a
PyPI project), so generating a version pin would mean inventing a version number and
presenting a fabrication as a patch. The Code-Scan Agent's patches are real because
they are rewrites of real syntax; a manifest patch becomes real with Phase 2's
CycloneDX/SPDX ingestion, and until then the action carries the exposure path and the
reviewer makes the edit.

For a flagged *vendor* or *organisation* there is no patch at any phase — you cannot
upgrade a payroll provider — so the action is ``NOTIFY_ANALYST``, which is not
destructive and not gated. Section 2.5's compliance story is the point of those:
the evidence attached to the notification is what turns *"prove your vendor risk
posture"* into something other than a spreadsheet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Final

import numpy as np

from sentinel.agents.contain import executable_action, pending_action
from sentinel.agents.engine import (
    NullEngine,
    ReasoningEngine,
    parse_narrative_opinion,
    reconcile_claims,
    sanitize_engine_text,
)
from sentinel.agents.prompts import SUPPLY_CHAIN_SYSTEM_PROMPT, AgentPrompt
from sentinel.agents.runtime import CompiledGraph, GraphSpec, RunContext
from sentinel.agents.state import END, IncidentState, IncidentStatus, StepOutcome
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import SentinelError
from sentinel.core.ids import deterministic_id
from sentinel.core.schemas import (
    ActionRequest,
    ActionType,
    AgentName,
    Alert,
    AlertSource,
    ApprovalStatus,
    AuditEventType,
    Evidence,
    EvidenceKind,
    InvestigationReport,
    RiskTier,
    Severity,
    TriageDecision,
    TriageResult,
)
from sentinel.graph.explain import NodeExplanation, top_risk_explanations
from sentinel.graph.schema import NodeKind, SupplyChainGraph
from sentinel.kb.corpus import DocumentKind
from sentinel.kb.retrieve import KnowledgeBase

if TYPE_CHECKING:
    from sentinel.graph.gnn import SupplyChainGNN

__all__ = [
    "DEPENDENCY_TECHNIQUE",
    "NODE_ASSESS",
    "SUPPLY_CHAIN_STEP_BUDGET",
    "SupplyChainAgent",
    "SupplyChainAssessment",
    "SupplyChainError",
    "SupplyChainMonitor",
    "VendorRiskFinding",
    "build_supply_chain_review_graph",
    "new_vendor_risk_incident",
    "supply_chain_timings",
    "synthesize_vendor_alert",
]

SUPPLY_CHAIN_STEP_BUDGET: Final[int] = 8
"""The longest path is three nodes. See :func:`build_supply_chain_review_graph`."""

NODE_ASSESS: Final[str] = "assess"
NODE_APPROVE: Final[str] = "approve"
NODE_REMEDIATE: Final[str] = "remediate"

DEPENDENCY_TECHNIQUE: Final[str] = "T1195.001"
"""ATT&CK *Compromise Software Dependencies and Development Tools*.

Asserted by **identifier lookup**, never by text search, on the same reasoning as
:mod:`sentinel.agents.investigate`: a dependency-risk finding *is* this technique by
definition, so there is nothing to rank and a search would only introduce the chance
of citing a plausible neighbour. The knowledge base links it to the real
dependency-compromise advisories — event-stream, colors/faker, node-ipc,
torchtriton, Codecov — which are the citations that make the risk concrete to
somebody who has not read the DBIR.
"""

#: Severity by how the risk arrived. An *intrinsically* risky package — unmaintained
#: and carrying known CVEs — is a present defect in your own dependency tree. An
#: inherited exposure is real and one step removed, which is exactly the distinction
#: a vCISO needs in order to decide what to do this week.
_SEVERITY_FOR_INTRINSIC: Final[Severity] = Severity.HIGH
_SEVERITY_FOR_INHERITED: Final[Severity] = Severity.MEDIUM


class SupplyChainError(SentinelError):
    """The Supply-Chain Agent could not produce a grounded assessment."""


@dataclass(frozen=True, slots=True)
class VendorRiskFinding:
    """One flagged node, with the path that explains it."""

    explanation: NodeExplanation
    rank: int
    risk_score: float

    @property
    def node_id(self) -> str:
        return self.explanation.node_id

    @property
    def kind(self) -> NodeKind:
        return self.explanation.kind

    @property
    def severity(self) -> Severity:
        return (
            _SEVERITY_FOR_INTRINSIC
            if self.explanation.is_intrinsically_risky
            else _SEVERITY_FOR_INHERITED
        )

    @property
    def action_type(self) -> ActionType:
        """What can actually be done about this node.

        A package can be upgraded, pinned or dropped, so the action is a patch PR and
        the gate applies. A vendor or an organisation cannot be patched at all, so
        the action is a notification carrying the evidence — which is the artifact
        PRD Section 2.5's compliance argument is about.
        """
        return (
            ActionType.OPEN_PATCH_PR
            if self.kind is NodeKind.PACKAGE
            else ActionType.NOTIFY_ANALYST
        )

    @property
    def attribution_disagrees(self) -> bool:
        """True when the graph shows inherited exposure the model did not key on.

        Reported as a claim rather than resolved. The structural walk and the model
        ablation answer different questions (see :mod:`sentinel.graph.explain`), and
        when they point in opposite directions the honest output is to say so: either
        the model is ignoring the graph, or the paths are incidental. Blending them
        into one number would hide whichever is true.
        """
        return (
            bool(self.explanation.paths)
            and self.explanation.dominant_driver == "own_features"
            and self.explanation.total_inherited_contribution > 0.5
        )

    def describe(self) -> str:
        return f"#{self.rank} {self.explanation.describe()}"


@dataclass(frozen=True, slots=True)
class SupplyChainAssessment:
    """One scored graph, as the agent reports it."""

    findings: tuple[VendorRiskFinding, ...]
    report: InvestigationReport
    n_nodes: int
    n_edges: int
    #: Nodes whose structural and model attributions disagree.
    disagreements: tuple[str, ...] = ()

    @property
    def by_kind(self) -> dict[NodeKind, int]:
        counts: dict[NodeKind, int] = {}
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        return counts

    @property
    def explainable(self) -> tuple[VendorRiskFinding, ...]:
        """Findings that carry at least one concrete path or an intrinsic reason.

        F-06's guardrail is that a *flag* is explainable, so this is the set the
        acceptance check counts. A flagged node with neither a path nor an intrinsic
        reason is a score with no story, which is the thing the guardrail forbids.
        """
        return tuple(
            finding
            for finding in self.findings
            if finding.explanation.paths or finding.explanation.is_intrinsically_risky
        )

    def describe(self) -> str:
        lines = [
            f"{len(self.findings)} flagged node(s) of {self.n_nodes} "
            f"({self.n_edges} edges); "
            f"{len(self.explainable)} with a concrete explanation"
        ]
        for finding in self.findings[:5]:
            lines.append(f"  {finding.describe()}")
        for node_id in self.disagreements:
            lines.append(
                f"  ! {node_id}: the graph shows inherited exposure the model did not "
                "key on"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class SupplyChainAgent:
    """Scores a vendor/dependency graph and explains what it flags."""

    kb: KnowledgeBase
    engine: ReasoningEngine = field(default_factory=NullEngine)
    clock: Clock = field(default_factory=SystemClock)
    #: How many nodes reach a human per assessment. Ten, matching F-06's own
    #: top-10 metric — and because a review queue longer than a screen is a review
    #: queue nobody finishes.
    top_k: int = 10
    #: Hops the structural explainer walks. Four, because PRD Section 5.5.3's
    #: headline scenario is a *fourth-order* dependency; the 2-layer GNN cannot see
    #: that far (Part 2 finding 10) but the graph walk can, which is why the
    #: explanation is computed from the graph rather than read out of the model.
    max_hops: int = 4
    version: str = "supply-chain-1"

    def assess(
        self,
        graph: SupplyChainGraph,
        *,
        alert: Alert,
        model: SupplyChainGNN | None = None,
        scores: np.ndarray | None = None,
        now: datetime | None = None,
    ) -> SupplyChainAssessment:
        """Score ``graph``, explain the top ``top_k``, and build a cited report.

        ``model`` is optional and ``scores`` may be supplied instead. Both, because
        the structural half of the explanation does not need a trained model — a
        fresh tenant, or a dashboard rendering before the nightly training run, still
        gets the exposure paths — and because the offline evaluation already has the
        scores in hand and re-running the forward pass would be measuring a different
        thing than it reported.
        """
        created_at = now or self.clock.now()
        if scores is None:
            if model is None:
                raise SupplyChainError(
                    "assess() needs either a fitted model or a score vector; without "
                    "one there is nothing to rank and the top-k would be arbitrary"
                )
            scores = model.risk_scores(graph)
        values = np.asarray(scores, dtype=float).ravel()
        if values.size != graph.n_nodes:
            raise SupplyChainError(
                f"{values.size} scores for {graph.n_nodes} nodes; a misaligned score "
                "vector would attribute one node's risk to another"
            )

        explanations = top_risk_explanations(
            graph, values, model=model, k=self.top_k, max_hops=self.max_hops
        )
        findings = tuple(
            VendorRiskFinding(
                explanation=explanation, rank=index + 1, risk_score=explanation.risk_score
            )
            for index, explanation in enumerate(explanations)
        )

        evidence, technique_refs = self._gather(findings)
        claims = self._claims(findings, evidence, technique_refs)
        summary = self._summary(findings, graph)
        if not isinstance(self.engine, NullEngine):
            claims, summary = self._with_engine(findings, evidence, claims, summary)

        return SupplyChainAssessment(
            findings=findings,
            report=InvestigationReport(
                report_id=deterministic_id("supply-chain", alert.alert_id, self.version),
                alert_ids=(alert.alert_id,),
                tenant_id=alert.tenant_id,
                summary=summary,
                claims=claims,
                evidence=evidence,
                techniques=self._techniques(),
                severity=_worst_severity(findings),
                confidence=_assessment_confidence(findings),
                recommended_actions=tuple(
                    dict.fromkeys(finding.action_type for finding in findings)
                ),
                agent=AgentName.SUPPLY_CHAIN,
                model_version=self.version,
                created_at=created_at,
            ),
            n_nodes=graph.n_nodes,
            n_edges=len(graph.edges),
            disagreements=tuple(
                finding.node_id for finding in findings if finding.attribution_disagrees
            ),
        )

    # --- evidence -------------------------------------------------------------- #

    def _gather(
        self, findings: Sequence[VendorRiskFinding]
    ) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
        """Graph-path evidence from the explainer, plus the technique and its advisories."""
        collected: dict[str, Evidence] = {}
        for finding in findings:
            for item in finding.explanation.as_evidence():
                collected.setdefault(item.ref, item)
            # The model's own attribution is a factual claim about this run, so it is
            # citable like any other. An uncited "the model keyed on the supply chain"
            # is precisely the unexplained black-box output PRD Section 9.2 forbids.
            ref = f"model://{self.version}#{finding.node_id}/attribution"
            collected.setdefault(
                ref,
                Evidence(
                    kind=EvidenceKind.MODEL_OUTPUT,
                    ref=ref,
                    excerpt=(
                        f"risk={finding.risk_score:.4f} "
                        f"own_features={finding.explanation.own_feature_share:.3f} "
                        f"neighbourhood={finding.explanation.neighbourhood_share:.3f} "
                        f"driver={finding.explanation.dominant_driver}"
                    ),
                    relevance=1.0,
                ),
            )

        technique_refs: list[str] = []
        retrieved_at = self.clock.now()
        for chunk in self.kb.chunks_for(DEPENDENCY_TECHNIQUE)[:2]:
            evidence = chunk.as_evidence(
                evidence_kind=EvidenceKind.ATTACK_TECHNIQUE,
                relevance=1.0,
                retrieved_at=retrieved_at,
            )
            collected.setdefault(evidence.ref, evidence)
            technique_refs.append(evidence.ref)
        # The advisories the corpus links to that technique: event-stream, colors,
        # node-ipc, torchtriton, Codecov. Real incidents, reached by the relation
        # graph rather than by ranking, and budgeted so one technique's long link
        # list cannot crowd the report.
        cited = 0
        for related in self.kb.related(DEPENDENCY_TECHNIQUE):
            if cited >= 3:
                break
            chunks = self.kb.chunks_for(related)
            if not chunks or chunks[0].kind == DocumentKind.TECHNIQUE.value:
                continue
            evidence = chunks[0].as_evidence(
                evidence_kind=EvidenceKind.CVE_RECORD,
                relevance=0.8,
                retrieved_at=retrieved_at,
            )
            collected.setdefault(evidence.ref, evidence)
            technique_refs.append(evidence.ref)
            cited += 1
        return tuple(collected.values()), tuple(technique_refs)

    def _claims(
        self,
        findings: Sequence[VendorRiskFinding],
        evidence: tuple[Evidence, ...],
        technique_refs: tuple[str, ...],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        known = {item.ref for item in evidence}
        claims: list[tuple[str, tuple[str, ...]]] = []

        for finding in findings:
            refs = tuple(
                item.ref for item in finding.explanation.as_evidence() if item.ref in known
            )
            attribution_ref = f"model://{self.version}#{finding.node_id}/attribution"
            if attribution_ref in known:
                refs = (*refs, attribution_ref)
            if not refs:
                # No path and no intrinsic reason: F-06's guardrail says a flag must
                # be explainable, so rather than emit an uncited claim the finding is
                # left out of the narrative and counted against `explainable`.
                continue
            reason = (
                "is unmaintained and carries known CVEs, so it is a risk source in "
                "its own right"
                if finding.explanation.is_intrinsically_risky
                else f"inherits exposure over {len(finding.explanation.paths)} path(s)"
            )
            claims.append(
                (
                    f"{finding.node_id} ({finding.kind.value}) ranks #{finding.rank} at "
                    f"risk {finding.risk_score:.3f}: it {reason}. The model attributes "
                    f"{finding.explanation.neighbourhood_share:.0%} of that score to "
                    f"its neighbourhood and "
                    f"{finding.explanation.own_feature_share:.0%} to its own features.",
                    refs,
                )
            )
            for path in finding.explanation.paths[:2]:
                path_refs = tuple(
                    item.ref
                    for item in finding.explanation.as_evidence()
                    if item.ref.startswith(f"graph://path/{finding.node_id}")
                )
                if path_refs:
                    claims.append((path.describe(), path_refs))
            if finding.attribution_disagrees:
                claims.append(
                    (
                        f"For {finding.node_id} the graph walk and the model disagree: "
                        f"{len(finding.explanation.paths)} exposure path(s) contribute "
                        f"{finding.explanation.total_inherited_contribution:.2f} while "
                        "the model scored it mostly on its own features. Treat the "
                        "paths as context, not as the model's reasoning.",
                        refs,
                    )
                )

        if technique_refs:
            claims.append(
                (
                    "Dependency and third-party exposure is MITRE ATT&CK "
                    f"{DEPENDENCY_TECHNIQUE}, and the knowledge base records real "
                    "incidents of it rather than only the technique description.",
                    tuple(ref for ref in technique_refs if ref in known),
                )
            )
        return tuple(claims)

    def _summary(
        self, findings: Sequence[VendorRiskFinding], graph: SupplyChainGraph
    ) -> str:
        if not findings:
            return f"Assessed {graph.n_nodes} nodes. Nothing flagged."
        intrinsic = sum(
            1 for finding in findings if finding.explanation.is_intrinsically_risky
        )
        return (
            f"Assessed {graph.n_nodes} nodes and {len(graph.edges)} relationships; "
            f"{len(findings)} flagged. {intrinsic} carry their own CVEs while "
            f"unmaintained; {len(findings) - intrinsic} inherit exposure through the "
            f"chain. Highest: {findings[0].node_id} at {findings[0].risk_score:.3f}."
        )[:8000]

    def _techniques(self) -> tuple[str, ...]:
        """The technique this agent asserts, if the knowledge base actually has it.

        Guarded rather than hardcoded: a report asserting a technique whose chunks it
        could not cite would be an uncited claim wearing an identifier, and
        :class:`~sentinel.core.schemas.InvestigationReport` would be right to have no
        opinion about it because ``techniques`` is not validated against evidence.
        """
        return (DEPENDENCY_TECHNIQUE,) if self.kb.chunks_for(DEPENDENCY_TECHNIQUE) else ()

    # --- engine ---------------------------------------------------------------- #

    def _with_engine(
        self,
        findings: Sequence[VendorRiskFinding],
        evidence: tuple[Evidence, ...],
        claims: tuple[tuple[str, tuple[str, ...]], ...],
        summary: str,
    ) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], str]:
        """Append narrative the engine can cite. Scores and paths are never read back.

        There is no ``monotone_caution`` analogue to apply here, because there is
        nothing for the engine to move: the risk score comes from the GNN, the paths
        come from the graph walk, the flagged set is the top-k of the scores, and
        ``as_evidence`` is computed from the explanation. The engine contributes
        prose, and prose that cites a ref which does not resolve is dropped whole.
        That is a narrower role than the Triage Agent gives it, and it is the right
        one: an analyst acting on a vendor flag needs the path, and the path is not
        something a language model is in a position to know.
        """
        response = self.engine.respond(self._prompt(findings, evidence))
        opinion = parse_narrative_opinion(response)
        if opinion is None:
            return claims, summary
        kept = reconcile_claims(opinion.claims, resolvable=[item.ref for item in evidence])
        existing = {statement for statement, _refs in claims}
        extended = claims + tuple(
            (statement, refs) for statement, refs in kept if statement not in existing
        )
        if opinion.summary:
            addition = sanitize_engine_text(opinion.summary, engine=response.engine)
            if addition:
                summary = f"{summary} {addition}"[:8000]
        return extended, summary

    def _prompt(
        self, findings: Sequence[VendorRiskFinding], evidence: tuple[Evidence, ...]
    ) -> AgentPrompt:
        catalogue = "\n".join(f"  {item.ref}  [{item.kind.value}]" for item in evidence)
        task = (
            f"{len(findings)} flagged node(s):\n"
            + "\n".join(
                f"  #{finding.rank} {finding.node_id} ({finding.kind.value}) "
                f"risk={finding.risk_score:.3f} "
                f"driver={finding.explanation.dominant_driver}"
                for finding in findings
            )
            + "\n\nEvidence refs you may cite, and no others:\n"
            + catalogue
            + "\n\nExplain the exposure for a vCISO. Every claim cites a ref above."
        )
        prompt = AgentPrompt(
            system=SUPPLY_CHAIN_SYSTEM_PROMPT,
            task=task,
            response_schema='{"summary": "string", "claims": [["statement", ["ref"]]]}',
        )
        for item in evidence:
            if item.kind is EvidenceKind.GRAPH_PATH:
                # Vendor and package names come from third parties, so a path
                # description is attacker-influenced text and goes through the fence.
                prompt = prompt.with_untrusted(
                    item.excerpt, label="graph.path", origin="supply_chain.graph"
                )
        return prompt


def _worst_severity(findings: Sequence[VendorRiskFinding]) -> Severity:
    if not findings:
        return Severity.INFO
    return max((finding.severity for finding in findings), key=lambda s: s.rank)


def _assessment_confidence(findings: Sequence[VendorRiskFinding]) -> float:
    """Confidence = the share of flagged nodes that have a concrete explanation.

    Not the model's score, and not a constant. F-06's guardrail is explainability, so
    the honest confidence in a *flag set* is how much of it can be explained: ten
    flags with ten paths is a report an analyst can act on, and ten flags with two
    paths is a ranking with a story attached to a fifth of it.
    """
    if not findings:
        return 1.0
    explained = sum(
        1
        for finding in findings
        if finding.explanation.paths or finding.explanation.is_intrinsically_risky
    )
    return explained / len(findings)


# --------------------------------------------------------------------------- #
# The alert a continuous assessment mints for itself
# --------------------------------------------------------------------------- #


def synthesize_vendor_alert(
    *,
    tenant_id: str,
    graph: SupplyChainGraph,
    at: datetime,
    assessment_id: str = "scheduled",
) -> Alert:
    """Mint the ``AlertSource.VENDOR_FEED`` alert one assessment is filed under.

    Pre-triaged, for the reason :func:`~sentinel.agents.codescan.synthesize_scan_alert`
    is: everything downstream reads ``alert.triage``, and there is no flow-based
    anomaly score to compute for a dependency graph. The verdict records what is
    known at intake — an assessment was scheduled, it escalates by construction — and
    the *real* severity comes from the assessment, which is what the report carries.

    Deterministic in ``(tenant, assessment_id)``, so a re-run of the same scheduled
    assessment resumes rather than forking. A caller passing a date as
    ``assessment_id`` gets one run per day; passing a constant gets one run, ever,
    which is occasionally what a demo wants.
    """
    identifier = deterministic_id("vendor-alert", tenant_id, assessment_id)
    payload = (
        f'{{"assessment": "{assessment_id}", "nodes": {graph.n_nodes}, '
        f'"edges": {len(graph.edges)}}}'
    )
    alert = Alert(
        alert_id=identifier,
        tenant_id=tenant_id,
        source=AlertSource.VENDOR_FEED,
        timestamp=at,
        ingested_at=at,
        asset_id=f"supply-chain/{tenant_id}",
        signature="supply-chain-assessment-scheduled",
        raw_payload=payload,
        features={"nodes": graph.n_nodes, "edges": len(graph.edges)},
        dataset="supply-chain",
    )
    return alert.with_triage(
        TriageResult(
            severity=Severity.MEDIUM,
            confidence=1.0,
            decision=TriageDecision.ESCALATE,
            rationale=(
                f"Scheduled supply-chain assessment {assessment_id} over "
                f"{graph.n_nodes} nodes. Routed to the Supply-Chain Agent; severity "
                "is set by what the scoring finds."
            ),
            model_version="supply-chain-intake-1",
            latency_ms=0.0,
            decided_at=at,
        )
    )


def new_vendor_risk_incident(
    alert: Alert, *, at: datetime, trust_tier: RiskTier = RiskTier.RECOMMEND
) -> IncidentState:
    """Open a run for a supply-chain assessment."""
    if alert.source is not AlertSource.VENDOR_FEED:
        raise SupplyChainError(
            f"alert {alert.alert_id} is from {alert.source.value}, not vendor_feed; "
            "this graph's nodes assume a dependency graph as the subject"
        )
    return IncidentState(
        incident_id=deterministic_id("supply-chain-incident", alert.alert_id),
        tenant_id=alert.tenant_id,
        alert=alert,
        trust_tier=trust_tier,
        created_at=at,
        updated_at=at,
    )


# --------------------------------------------------------------------------- #
# The graph
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


def _audit_head(ctx: RunContext) -> str | None:
    if ctx.audit is None:
        return None
    head = ctx.audit.head()
    return None if head is None else head[1]


def _assess_node(
    agent: SupplyChainAgent,
    graph: SupplyChainGraph,
    model: SupplyChainGNN | None,
    scores: np.ndarray | None,
):
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        assessment = agent.assess(
            graph, alert=state.alert, model=model, scores=scores, now=ctx.now()
        )
        _audit(
            ctx,
            AuditEventType.SUPPLY_CHAIN_ASSESSED,
            actor=AgentName.SUPPLY_CHAIN,
            tenant_id=state.tenant_id,
            subject_id=state.alert.alert_id,
            payload={
                "nodes": assessment.n_nodes,
                "edges": assessment.n_edges,
                "flagged": len(assessment.findings),
                "explainable": len(assessment.explainable),
                "by_kind": {
                    kind.value: count for kind, count in assessment.by_kind.items()
                },
                "worst_severity": assessment.report.severity.value,
                "attribution_disagreements": list(assessment.disagreements),
                "top": [
                    {
                        "node_id": finding.node_id,
                        "kind": finding.kind.value,
                        "risk": round(finding.risk_score, 6),
                        "driver": finding.explanation.dominant_driver,
                        "intrinsic": finding.explanation.is_intrinsically_risky,
                    }
                    for finding in assessment.findings[:10]
                ],
            },
        )
        state = state.with_report(assessment.report, at=ctx.now())

        # One action per flagged node would mean ten approval requests from one
        # scheduled job, and an analyst facing ten identical-looking prompts approves
        # them as a batch without reading any. So the run proposes the single most
        # severe remediation and the report carries the rest; a dashboard splits them.
        actionable = _most_severe_actionable(assessment)
        if actionable is None:
            _audit(
                ctx,
                AuditEventType.POLICY_UPDATED,
                actor=AgentName.SUPPLY_CHAIN,
                tenant_id=state.tenant_id,
                subject_id=state.alert.alert_id,
                payload={"action": None, "reason": "nothing flagged with an explanation"},
            )
            return state

        action = ActionRequest.propose(
            alert_id=state.alert.alert_id,
            tenant_id=state.tenant_id,
            proposed_by=AgentName.SUPPLY_CHAIN,
            action_type=actionable.action_type,
            target=actionable.node_id,
            rationale=_rationale(actionable, assessment),
            risk_tier=state.trust_tier,
            created_at=ctx.now(),
            evidence=actionable.explanation.as_evidence()[:8],
            **({"audit_hash_prev": head} if (head := _audit_head(ctx)) else {}),
        )
        _audit(
            ctx,
            AuditEventType.ACTION_PROPOSED,
            actor=AgentName.SUPPLY_CHAIN,
            tenant_id=state.tenant_id,
            subject_id=action.action_id,
            payload={
                "action_type": action.action_type.value,
                "target": action.target,
                "requires_human_approval": action.requires_human_approval,
                "node_kind": actionable.kind.value,
                "risk": round(actionable.risk_score, 6),
                "paths": len(actionable.explanation.paths),
            },
        )
        if action.requires_human_approval:
            _audit(
                ctx,
                AuditEventType.APPROVAL_REQUESTED,
                actor=AgentName.SUPPLY_CHAIN,
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


def _most_severe_actionable(
    assessment: SupplyChainAssessment,
) -> VendorRiskFinding | None:
    """The finding worth proposing: explainable first, then severity, then rank.

    Explainable *first*, ahead of severity, and that ordering is F-06's guardrail
    expressed as a preference. An unexplainable node might score higher, and
    proposing a dependency change with no path to show the reviewer is the flag the
    guardrail forbids — so a slightly less severe finding that can be justified wins.
    """
    candidates = assessment.explainable or assessment.findings
    if not candidates:
        return None
    return min(candidates, key=lambda finding: (-finding.severity.rank, finding.rank))


def _rationale(finding: VendorRiskFinding, assessment: SupplyChainAssessment) -> str:
    head = (
        f"{finding.node_id} ({finding.kind.value}) is the highest-severity explainable "
        f"exposure in this assessment: rank #{finding.rank} of {len(assessment.findings)} "
        f"flagged, risk {finding.risk_score:.3f}."
    )
    if finding.explanation.is_intrinsically_risky:
        head += " It is unmaintained and carries known CVEs."
    if finding.explanation.paths:
        head += f" {finding.explanation.paths[0].describe()}"
    if finding.action_type is ActionType.OPEN_PATCH_PR:
        head += (
            " Proposed remediation: a dependency change. The manifest edit is not "
            "drafted here — the sprint graph is synthetic, so a version pin would be "
            "invented rather than derived; Phase 2's SBOM ingestion supplies it."
        )
    else:
        head += (
            " Proposed remediation: notify the analyst with the evidence attached; a "
            "vendor relationship cannot be patched."
        )
    return head[:4000]


def _approve_node():
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        action = pending_action(state.actions)
        if action is None:
            raise SupplyChainError(
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


def _remediate_node(connector):
    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        action = executable_action(state.actions)
        if action is None:
            raise SupplyChainError(
                "the remediation node was reached with no executable action; this is "
                "a routing bug"
            )
        at = ctx.now()
        try:
            outcome = connector.execute(action)
        except Exception as exc:
            failed = action.mark_failed(reason=f"{type(exc).__name__}: {exc}", at=at)
            _audit(
                ctx,
                AuditEventType.ACTION_FAILED,
                actor=AgentName.SUPPLY_CHAIN,
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
                actor=AgentName.SUPPLY_CHAIN,
                tenant_id=state.tenant_id,
                subject_id=action.action_id,
                payload={"reason": outcome.detail},
            )
            return state.with_action(failed, at=at)
        executed = action.mark_executed(at=at)
        _audit(
            ctx,
            AuditEventType.ACTION_EXECUTED,
            actor=AgentName.SUPPLY_CHAIN,
            tenant_id=state.tenant_id,
            subject_id=action.action_id,
            payload={
                "action_type": executed.action_type.value,
                "target": executed.target,
                "requires_human_approval": executed.requires_human_approval,
                "approved_by": executed.approved_by,
                "detail": outcome.detail,
            },
        )
        return state.with_action(executed, at=at)

    return node


def _after_assess(state: IncidentState) -> str:
    action = pending_action(state.actions)
    if action is None:
        return "none"
    return "gate" if action.requires_human_approval else "act"


def _after_approve(state: IncidentState) -> str:
    for action in reversed(state.actions):
        if action.approval_status is ApprovalStatus.APPROVED:
            return "approved"
        if action.approval_status is ApprovalStatus.REJECTED:
            return "denied"
    return "denied"  # pragma: no cover - the node always decides


def _finish(state: IncidentState, ctx: RunContext) -> IncidentState:
    return state.finished(IncidentStatus.COMPLETED, at=ctx.now())


def build_supply_chain_review_graph(
    *,
    agent: SupplyChainAgent,
    graph: SupplyChainGraph,
    model: SupplyChainGNN | None = None,
    scores: np.ndarray | None = None,
    connector=None,
    checkpointer=None,
    step_budget: int = SUPPLY_CHAIN_STEP_BUDGET,
) -> CompiledGraph:
    """Compile the supply-chain review workflow.

    ::

        assess ──nothing explainable─────────────────────► END (completed)
           │
           ├─needs approval─► approve ──denied──────────► END
           │                     │
           │                  granted
           │                     ▼
           └─tier permits────► remediate ───────────────► END

    Named *review* rather than *supply chain* to keep it distinct from
    :class:`~sentinel.graph.schema.SupplyChainGraph`, which is the data structure. One
    of these is a workflow and the other is a dependency graph, and confusing them in
    a call site is the kind of mistake that type checks away only sometimes.

    A ``NOTIFY_ANALYST`` remediation is not destructive, so it routes straight to
    ``remediate`` with no gate — which is correct and worth noticing: the gate exists
    for actions with side effects on a monitored system, and telling a human something
    is not one. A flagged *package* proposes ``OPEN_PATCH_PR``, which is destructive,
    so the same run stops for approval. The route therefore depends on what was found,
    and both paths are exercised by ``tests/integration/test_supplychain_pipeline.py``.
    """
    from sentinel.agents.contain import SimulatedConnector

    spec = GraphSpec()
    spec.add_node(NODE_ASSESS, _assess_node(agent, graph, model, scores))
    spec.add_node(NODE_APPROVE, _approve_node())
    spec.add_node(NODE_REMEDIATE, _remediate_node(connector or SimulatedConnector()))
    spec.add_node("finish", _finish)

    spec.add_conditional_edges(
        NODE_ASSESS,
        _after_assess,
        {"none": "finish", "gate": NODE_APPROVE, "act": NODE_REMEDIATE},
    )
    spec.add_conditional_edges(
        NODE_APPROVE, _after_approve, {"approved": NODE_REMEDIATE, "denied": END}
    )
    spec.add_edge(NODE_REMEDIATE, END)
    spec.add_edge("finish", END)
    spec.set_entry_point(NODE_ASSESS)

    return spec.compile(
        checkpointer=checkpointer,
        interrupt_before=[NODE_APPROVE],
        step_budget=step_budget,
    )


def supply_chain_timings(state: IncidentState) -> tuple[float | None, float | None]:
    """``(assess_seconds, time_to_remediation_seconds)`` from the run's step history."""
    assessed = _step_end(state, NODE_ASSESS)
    if assessed is None:
        return None, None
    acted = _step_end(state, NODE_REMEDIATE)
    return (
        max(0.0, (assessed - state.created_at).total_seconds()),
        None if acted is None else max(0.0, (acted - state.created_at).total_seconds()),
    )


def _step_end(state: IncidentState, node: str) -> datetime | None:
    for record in reversed(state.history):
        if record.node == node and record.outcome is StepOutcome.OK:
            return record.ended_at
    return None


# --------------------------------------------------------------------------- #
# The trigger
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SupplyChainMonitor:
    """The scheduled job that turns continuous scoring into gated runs.

    This is the answer to the trigger question in the module docstring, and it is
    deliberately thin — an assessment per tick, each one a normal run through the
    normal state machine. The cadence is the caller's (a cron entry, an MSSP's nightly
    job); what belongs here is the part that must not be re-invented per deployment:
    each tick gets its own deterministic ``assessment_id``, so re-running a tick
    *resumes* that assessment instead of opening a second one, and an MSSP scoring
    fifty tenants gets fifty independent runs rather than fifty races on one thread.
    """

    agent: SupplyChainAgent
    graph: SupplyChainGraph
    model: SupplyChainGNN | None = None
    scores: np.ndarray | None = None
    clock: Clock = field(default_factory=SystemClock)

    def tick(
        self,
        *,
        tenant_id: str,
        compiled: CompiledGraph,
        audit=None,
        assessment_id: str | None = None,
    ):
        """Run one assessment. Returns the runtime's :class:`RunResult`."""
        now = self.clock.now()
        identifier = assessment_id or now.date().isoformat()
        alert = synthesize_vendor_alert(
            tenant_id=tenant_id,
            graph=self.graph,
            at=now,
            assessment_id=identifier,
        )
        return compiled.invoke(
            new_vendor_risk_incident(alert, at=now), clock=self.clock, audit=audit
        )
