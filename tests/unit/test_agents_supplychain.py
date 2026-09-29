"""The Supply-Chain Agent: F-06's guardrail, and the trigger design.

F-06's *metric* (top-10 precision) belongs to the model and is tested in
``test_graph_gnn.py``. F-06's *guardrail* — "flags are explainable via the specific
graph path that drove the score" — is what this file tests: that every claim cites a
concrete ``graph://path/...`` ref, that the structural and model attributions are
reported separately rather than blended, and that a disagreement between them becomes
a claim instead of being smoothed over.

The second subject is the trigger. Supply-chain risk is continuous, so there is no
alert to fire on, and the design mints one. These tests pin the determinism that makes
re-running a scheduled assessment resume rather than fork.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.agents.engine import HostileEngine, NullEngine, ScriptedEngine
from sentinel.agents.supplychain import (
    DEPENDENCY_TECHNIQUE,
    SupplyChainAgent,
    SupplyChainError,
    VendorRiskFinding,
    new_vendor_risk_incident,
    synthesize_vendor_alert,
)
from sentinel.core.clock import FrozenClock
from sentinel.core.schemas import (
    ActionType,
    AgentName,
    AlertSource,
    EvidenceKind,
    TriageDecision,
)
from sentinel.graph.explain import ExposurePath, NodeExplanation, explain_node
from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
from sentinel.graph.schema import NodeKind
from sentinel.graph.synthetic import SyntheticGraphGenerator
from sentinel.kb.retrieve import KnowledgeBase
from tests.conftest import FIXED_NOW

SEED = 20260928


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build()


@pytest.fixture(scope="module")
def scored():
    """A fitted model and its scores. Module-scoped: fitting is the slow part."""
    graph, truth = SyntheticGraphGenerator(seed=SEED).generate()
    node_ids = graph.node_ids()
    labels = truth.labels(node_ids)
    split = GraphSplit.stratified(labels, seed=SEED)
    model = SupplyChainGNN(random_state=SEED).fit(
        graph, labels, split, exposure=truth.risk_vector(node_ids)
    )
    return graph, truth, model, model.risk_scores(graph)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(FIXED_NOW)


def _alert(graph, at=FIXED_NOW, assessment_id="2026-09-29"):
    return synthesize_vendor_alert(
        tenant_id="acme", graph=graph, at=at, assessment_id=assessment_id
    )


def _explanation(**overrides) -> NodeExplanation:
    defaults = dict(
        node_id="pkg-0001",
        kind=NodeKind.PACKAGE,
        risk_score=0.9,
        paths=(),
        own_feature_share=0.5,
        neighbourhood_share=0.5,
        is_intrinsically_risky=False,
    )
    return NodeExplanation(**{**defaults, **overrides})


def _path(contribution: float = 0.6) -> ExposurePath:
    return ExposurePath(
        nodes=("pkg-0099", "pkg-0001"),
        source_cves=11,
        source_days_stale=1961.0,
        contribution=contribution,
    )


class TestSynthesizedAlert:
    def test_the_alert_is_a_vendor_feed_alert(self, scored):
        graph, *_ = scored
        assert _alert(graph).source is AlertSource.VENDOR_FEED

    def test_it_carries_an_intake_triage_verdict(self, scored):
        graph, *_ = scored
        triage = _alert(graph).triage
        assert triage is not None
        assert triage.decision is TriageDecision.ESCALATE

    def test_the_id_is_deterministic_in_the_assessment(self, scored):
        graph, *_ = scored
        assert _alert(graph).alert_id == _alert(graph).alert_id

    def test_a_different_assessment_id_is_a_different_run(self, scored):
        graph, *_ = scored
        assert _alert(graph, assessment_id="2026-09-30").alert_id != _alert(graph).alert_id

    def test_a_different_tenant_is_a_different_run(self, scored):
        graph, *_ = scored
        other = synthesize_vendor_alert(
            tenant_id="other", graph=graph, at=FIXED_NOW, assessment_id="2026-09-29"
        )
        assert other.alert_id != _alert(graph).alert_id

    def test_an_incident_refuses_a_non_vendor_feed_alert(self, small_alerts):
        with pytest.raises(SupplyChainError, match="not vendor_feed"):
            new_vendor_risk_incident(small_alerts[0], at=FIXED_NOW)

    def test_the_incident_id_is_derived_from_the_alert(self, scored):
        graph, *_ = scored
        alert = _alert(graph)
        assert (
            new_vendor_risk_incident(alert, at=FIXED_NOW).incident_id
            == new_vendor_risk_incident(alert, at=FIXED_NOW).incident_id
        )


class TestAssessment:
    def test_requires_a_model_or_scores(self, kb, clock, scored):
        graph, *_ = scored
        with pytest.raises(SupplyChainError, match="fitted model or a score vector"):
            SupplyChainAgent(kb=kb, clock=clock).assess(graph, alert=_alert(graph))

    def test_a_misaligned_score_vector_is_refused(self, kb, clock, scored):
        graph, *_ = scored
        with pytest.raises(SupplyChainError, match="scores for"):
            SupplyChainAgent(kb=kb, clock=clock).assess(
                graph, alert=_alert(graph), scores=np.zeros(3)
            )

    def test_scores_alone_are_enough_for_the_structural_half(self, kb, clock, scored):
        # A fresh tenant with no trained model still gets exposure paths, which is why
        # the model is optional.
        graph, _truth, _model, scores = scored
        assessment = SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), scores=scores
        )
        assert assessment.findings
        assert any(f.explanation.paths for f in assessment.findings)

    def test_top_k_is_honoured(self, kb, clock, scored):
        graph, _truth, model, scores = scored
        assessment = SupplyChainAgent(kb=kb, clock=clock, top_k=4).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert len(assessment.findings) == 4

    def test_findings_are_ranked_by_score(self, kb, clock, scored):
        graph, _truth, model, scores = scored
        assessment = SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        values = [f.risk_score for f in assessment.findings]
        assert values == sorted(values, reverse=True)
        assert [f.rank for f in assessment.findings] == list(
            range(1, len(assessment.findings) + 1)
        )

    def test_the_report_is_attributed_to_the_supply_chain_agent(self, kb, clock, scored):
        graph, _truth, model, scores = scored
        assessment = SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert assessment.report.agent is AgentName.SUPPLY_CHAIN


class TestF06Guardrail:
    """"Flags are explainable via the specific graph path that drove the score.\""""

    @pytest.fixture
    def assessment(self, kb, clock, scored):
        graph, _truth, model, scores = scored
        return SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )

    def test_every_flagged_node_is_explainable(self, assessment):
        assert len(assessment.explainable) == len(assessment.findings)

    def test_graph_paths_are_cited_as_typed_evidence(self, assessment):
        kinds = {item.kind for item in assessment.report.evidence}
        assert EvidenceKind.GRAPH_PATH in kinds

    def test_every_claim_cites_resolvable_evidence(self, assessment):
        known = {item.ref for item in assessment.report.evidence}
        assert assessment.report.claims
        for _statement, refs in assessment.report.claims:
            assert refs and set(refs) <= known

    def test_a_concrete_path_appears_verbatim_as_a_claim(self, assessment):
        # Not "vendor-017 is high risk" but the chain that makes it so.
        assert any("reaches" in statement and "hop(s)" in statement
                   for statement, _refs in assessment.report.claims)

    def test_the_model_attribution_is_cited_separately(self, assessment):
        kinds = {item.kind for item in assessment.report.evidence}
        assert EvidenceKind.MODEL_OUTPUT in kinds
        assert any("attribution" in item.ref for item in assessment.report.evidence)

    def test_a_disagreement_between_the_two_attributions_becomes_a_claim(
        self, assessment
    ):
        # If the graph shows inherited exposure the model did not key on, the honest
        # output says so rather than presenting the path as the model's reasoning.
        if not assessment.disagreements:
            pytest.skip("this seed produced no attribution disagreement")
        assert any("disagree" in statement for statement, _ in assessment.report.claims)

    def test_confidence_is_the_explainable_share(self, assessment):
        expected = len(assessment.explainable) / len(assessment.findings)
        assert assessment.report.confidence == pytest.approx(expected)

    def test_a_node_with_no_explanation_is_excluded_from_the_narrative(self, kb, clock):
        # Rather than emitting an uncited claim, which the guardrail forbids.
        finding = VendorRiskFinding(explanation=_explanation(), rank=1, risk_score=0.9)
        assert finding.explanation.paths == ()
        assert not finding.explanation.is_intrinsically_risky
        agent = SupplyChainAgent(kb=kb, clock=clock)
        claims = agent._claims((finding,), (), ())
        assert not any(finding.node_id in statement for statement, _ in claims)


class TestTechniqueProvenance:
    def test_the_technique_is_asserted_by_lookup(self, kb, clock, scored):
        graph, _truth, model, scores = scored
        assessment = SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert assessment.report.techniques == (DEPENDENCY_TECHNIQUE,)
        assert kb.chunks_for(DEPENDENCY_TECHNIQUE)

    def test_real_dependency_incidents_are_cited_not_only_the_description(
        self, kb, clock, scored
    ):
        # event-stream, colors/faker, node-ipc, torchtriton, Codecov. Reached through
        # the corpus relation graph, not by ranking.
        graph, _truth, model, scores = scored
        assessment = SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert any(
            item.kind is EvidenceKind.CVE_RECORD for item in assessment.report.evidence
        )

    def test_the_technique_is_omitted_when_the_corpus_lacks_it(self, clock, scored):
        graph, _truth, model, scores = scored

        class Empty:
            chunks: tuple = ()
            scans: tuple = ()

            def chunks_for(self, doc_id):
                return ()

            def related(self, doc_id):
                return ()

            def search(self, *args, **kwargs):
                return ()

        assessment = SupplyChainAgent(kb=Empty(), clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert assessment.report.techniques == ()


class TestRemediationChoice:
    def test_a_package_proposes_a_patch_pr(self):
        finding = VendorRiskFinding(
            explanation=_explanation(kind=NodeKind.PACKAGE), rank=1, risk_score=0.9
        )
        assert finding.action_type is ActionType.OPEN_PATCH_PR
        assert finding.action_type.is_destructive

    @pytest.mark.parametrize("kind", [NodeKind.VENDOR, NodeKind.ORGANIZATION])
    def test_a_vendor_or_organisation_proposes_a_notification(self, kind):
        # You cannot upgrade a payroll provider, so there is no patch at any phase.
        finding = VendorRiskFinding(
            explanation=_explanation(kind=kind), rank=1, risk_score=0.9
        )
        assert finding.action_type is ActionType.NOTIFY_ANALYST
        assert not finding.action_type.is_destructive

    def test_an_intrinsic_risk_is_more_severe_than_an_inherited_one(self):
        intrinsic = VendorRiskFinding(
            explanation=_explanation(is_intrinsically_risky=True), rank=1, risk_score=0.9
        )
        inherited = VendorRiskFinding(
            explanation=_explanation(paths=(_path(),)), rank=2, risk_score=0.8
        )
        assert intrinsic.severity > inherited.severity

    def test_the_most_severe_explainable_finding_is_chosen(self):
        from sentinel.agents.supplychain import (
            SupplyChainAssessment,
            _most_severe_actionable,
        )

        unexplainable = VendorRiskFinding(
            explanation=_explanation(node_id="pkg-a"), rank=1, risk_score=0.99
        )
        explainable = VendorRiskFinding(
            explanation=_explanation(node_id="pkg-b", is_intrinsically_risky=True),
            rank=2,
            risk_score=0.5,
        )
        assessment = SupplyChainAssessment(
            findings=(unexplainable, explainable),
            report=None,  # type: ignore[arg-type]
            n_nodes=2,
            n_edges=1,
        )
        # Explainable wins even at a lower score: proposing a dependency change with
        # no path to show the reviewer is the flag F-06's guardrail forbids.
        chosen = _most_severe_actionable(assessment)
        assert chosen is not None and chosen.node_id == "pkg-b"

    def test_nothing_flagged_means_nothing_proposed(self):
        from sentinel.agents.supplychain import (
            SupplyChainAssessment,
            _most_severe_actionable,
        )

        empty = SupplyChainAssessment(
            findings=(), report=None, n_nodes=0, n_edges=0  # type: ignore[arg-type]
        )
        assert _most_severe_actionable(empty) is None


class TestAttributionDisagreement:
    def test_flagged_when_paths_contribute_but_the_model_ignored_them(self):
        finding = VendorRiskFinding(
            explanation=_explanation(
                paths=(_path(0.8),), own_feature_share=0.95, neighbourhood_share=0.05
            ),
            rank=1,
            risk_score=0.9,
        )
        assert finding.attribution_disagrees

    def test_not_flagged_when_the_model_keyed_on_the_neighbourhood(self):
        finding = VendorRiskFinding(
            explanation=_explanation(
                paths=(_path(0.8),), own_feature_share=0.1, neighbourhood_share=0.9
            ),
            rank=1,
            risk_score=0.9,
        )
        assert not finding.attribution_disagrees

    def test_not_flagged_when_there_are_no_paths(self):
        finding = VendorRiskFinding(
            explanation=_explanation(own_feature_share=1.0, neighbourhood_share=0.0),
            rank=1,
            risk_score=0.9,
        )
        assert not finding.attribution_disagrees


class TestEngine:
    def test_a_hostile_engine_changes_no_score_and_removes_no_finding(
        self, kb, clock, scored
    ):
        graph, _truth, model, scores = scored
        baseline = SupplyChainAgent(kb=kb, clock=clock, engine=NullEngine()).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        hostile = SupplyChainAgent(kb=kb, clock=clock, engine=HostileEngine()).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert [f.node_id for f in baseline.findings] == [
            f.node_id for f in hostile.findings
        ]
        assert [f.risk_score for f in baseline.findings] == [
            f.risk_score for f in hostile.findings
        ]

    def test_an_engine_claim_citing_an_unknown_ref_is_dropped(self, kb, clock, scored):
        graph, _truth, model, scores = scored
        engine = ScriptedEngine(
            [{"summary": "s", "claims": [["invented", ["graph://path/nope#0"]]]}]
        )
        assessment = SupplyChainAgent(kb=kb, clock=clock, engine=engine).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert not any("invented" in s for s, _ in assessment.report.claims)

    def test_an_engine_claim_citing_a_real_ref_is_kept(self, kb, clock, scored):
        graph, _truth, model, scores = scored
        baseline = SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        ref = next(
            item.ref
            for item in baseline.report.evidence
            if item.kind is EvidenceKind.GRAPH_PATH
        )
        engine = ScriptedEngine([{"summary": "s", "claims": [["grounded", [ref]]]}])
        assessment = SupplyChainAgent(kb=kb, clock=clock, engine=engine).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert any("grounded" in s for s, _ in assessment.report.claims)

    def test_graph_paths_reach_the_prompt_only_through_the_fence(
        self, kb, clock, scored
    ):
        graph, _truth, model, scores = scored
        captured = []

        class Recorder:
            name = "recorder"

            def respond(self, prompt):
                captured.append(prompt)
                from sentinel.agents.engine import EngineResponse

                return EngineResponse.decline("recorder", "no opinion")

        SupplyChainAgent(kb=kb, clock=clock, engine=Recorder()).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        assert captured and captured[0].blocks
        assert all(block.label == "graph.path" for block in captured[0].blocks)


class TestDeterminism:
    def test_two_assessments_hash_identically(self, kb, scored):
        graph, _truth, model, scores = scored
        agent = SupplyChainAgent(kb=kb, clock=FrozenClock(FIXED_NOW))
        first = agent.assess(graph, alert=_alert(graph), model=model, scores=scores)
        second = agent.assess(graph, alert=_alert(graph), model=model, scores=scores)
        assert first.report.canonical_hash() == second.report.canonical_hash()

    def test_explain_node_agrees_with_the_agent(self, kb, clock, scored):
        # The agent must not be computing its own explanation; it uses explain.py.
        graph, _truth, model, scores = scored
        assessment = SupplyChainAgent(kb=kb, clock=clock).assess(
            graph, alert=_alert(graph), model=model, scores=scores
        )
        top = assessment.findings[0]
        direct = explain_node(
            graph, top.node_id, risk_score=top.risk_score, max_hops=4
        )
        assert [p.nodes for p in direct.paths] == [
            p.nodes for p in top.explanation.paths
        ]
