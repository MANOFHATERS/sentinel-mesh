"""Path attribution (PRD F-06 guardrail, Section 5.4).

F-06's guardrail is *"flags are explainable via the specific graph path that drove
the score"*, and Section 5.1 gives the reason: *"an analyst should never have to
trust an agent's conclusion without being able to see why."*

Two attributions are produced and they must not be conflated:

*   **Structural** — the concrete exposure paths, computed from the graph. Correct
    whether or not the model is.
*   **Model** — how much of *this model's* score came from the node's own features
    versus its neighbourhood, by ablation.

:class:`TestAttributionHonesty` is the reason they are separate. A confident path
narration attached to a model that is actually ignoring the graph would be a
plausible story rather than an explanation, and that is the failure mode most likely
to survive review.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.schemas import Evidence, EvidenceKind
from sentinel.graph.explain import (
    ExposurePath,
    explain_node,
    top_risk_explanations,
)
from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
from sentinel.graph.schema import (
    EdgeKind,
    GraphError,
    Node,
    NodeKind,
    SupplyChainEdge,
    SupplyChainGraph,
)
from sentinel.graph.synthetic import SyntheticGraphGenerator

SEED = 20260928


@pytest.fixture
def planted() -> SupplyChainGraph:
    """A graph with one deliberately planted exposure chain.

    ``bad`` (11 CVEs, 1,900 days stale) -> ``mid`` -> ``direct`` -> ``vendor`` -> ``org``

    Plus a clean parallel chain, so "found the risky path" is distinguishable from
    "returned every path".
    """
    graph = SupplyChainGraph()
    graph.add_node(
        Node(
            node_id="bad",
            kind=NodeKind.PACKAGE,
            name="abandoned-lib",
            cve_exposure_count=11,
            days_since_last_update=1900.0,
            sbom_depth=4,
        )
    )
    for node_id, depth in (("mid", 3), ("direct", 1), ("clean_mid", 3), ("clean_direct", 1)):
        graph.add_node(
            Node(
                node_id=node_id,
                kind=NodeKind.PACKAGE,
                name=node_id,
                cve_exposure_count=0,
                days_since_last_update=30.0,
                sbom_depth=depth,
            )
        )
    graph.add_node(Node(node_id="vendor", kind=NodeKind.VENDOR, name="vendor"))
    graph.add_node(Node(node_id="org", kind=NodeKind.ORGANIZATION, name="org"))

    graph.add_edge(SupplyChainEdge("bad", "mid", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("mid", "direct", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("direct", "vendor", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("clean_mid", "clean_direct", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("clean_direct", "vendor", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("vendor", "org", EdgeKind.CONTRACTUAL))
    return graph


class TestExposurePath:
    def test_reports_hops_source_and_target(self) -> None:
        path = ExposurePath(
            nodes=("bad", "mid", "direct"),
            source_cves=11,
            source_days_stale=1900.0,
            contribution=0.49,
        )
        assert path.hops == 2
        assert path.source == "bad"
        assert path.target == "direct"

    def test_a_degenerate_path_is_refused(self) -> None:
        with pytest.raises(GraphError, match="source and a target"):
            ExposurePath(
                nodes=("only",), source_cves=1, source_days_stale=1.0, contribution=0.1
            )

    def test_describe_names_the_source_and_the_chain(self) -> None:
        path = ExposurePath(
            nodes=("bad", "mid", "direct"),
            source_cves=11,
            source_days_stale=1900.0,
            contribution=0.49,
        )
        text = path.describe()
        assert "bad" in text and "11 CVEs" in text and "1900 days stale" in text
        assert "bad -> mid -> direct" in text


class TestStructuralAttribution:
    def test_finds_the_planted_path(self, planted) -> None:
        explanation = explain_node(planted, "vendor")
        assert explanation.paths, "no exposure path found for the exposed vendor"
        best = explanation.paths[0]
        assert best.nodes == ("bad", "mid", "direct", "vendor")
        assert best.source_cves == 11

    def test_paths_run_source_first_as_a_causal_chain(self, planted) -> None:
        for path in explain_node(planted, "org").paths:
            assert path.target == "org"
            assert path.source == "bad"

    def test_reaches_the_organisation_four_hops_away(self, planted) -> None:
        """The PRD's fourth-order case, end to end.

        The structural explainer walks the graph, so unlike the 2-layer GNN it has no
        receptive-field limit — which is precisely why both halves exist.
        """
        explanation = explain_node(planted, "org", max_hops=4)
        assert explanation.paths
        assert explanation.paths[0].hops == 4

    def test_the_hop_horizon_is_respected(self, planted) -> None:
        assert explain_node(planted, "org", max_hops=3).paths == ()

    def test_a_clean_node_gets_no_path(self, planted) -> None:
        assert explain_node(planted, "clean_direct").paths == ()

    def test_the_risky_source_is_flagged_as_intrinsic(self, planted) -> None:
        assert explain_node(planted, "bad").is_intrinsically_risky
        assert not explain_node(planted, "mid").is_intrinsically_risky

    def test_contribution_decays_with_distance(self, planted) -> None:
        near = explain_node(planted, "mid").paths[0]
        far = explain_node(planted, "org", max_hops=4).paths[0]
        assert near.contribution > far.contribution

    def test_paths_are_ordered_deterministically(self, planted) -> None:
        """A dashboard whose rows reshuffle between refreshes teaches distrust."""
        first = explain_node(planted, "org", max_hops=4).paths
        second = explain_node(planted, "org", max_hops=4).paths
        assert [p.nodes for p in first] == [p.nodes for p in second]

    def test_independent_paths_are_all_reported(self) -> None:
        """Two risky packages reaching one vendor by disjoint routes.

        A global visited-set implementation would report only whichever it found
        first, losing the "you are exposed through several independent paths"
        finding that justifies a high score.
        """
        graph = SupplyChainGraph()
        for name in ("bad1", "bad2"):
            graph.add_node(
                Node(
                    node_id=name,
                    kind=NodeKind.PACKAGE,
                    name=name,
                    cve_exposure_count=9,
                    days_since_last_update=1500.0,
                    sbom_depth=2,
                )
            )
        for name in ("route1", "route2"):
            graph.add_node(
                Node(node_id=name, kind=NodeKind.PACKAGE, name=name, sbom_depth=1)
            )
        graph.add_node(Node(node_id="v", kind=NodeKind.VENDOR, name="v"))
        graph.add_edge(SupplyChainEdge("bad1", "route1", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("bad2", "route2", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("route1", "v", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("route2", "v", EdgeKind.DEPENDENCY))

        sources = {p.source for p in explain_node(graph, "v").paths}
        assert sources == {"bad1", "bad2"}

    def test_max_paths_truncates_the_report(self) -> None:
        graph = SupplyChainGraph()
        graph.add_node(Node(node_id="v", kind=NodeKind.VENDOR, name="v"))
        for index in range(8):
            graph.add_node(
                Node(
                    node_id=f"bad{index}",
                    kind=NodeKind.PACKAGE,
                    name=f"bad{index}",
                    cve_exposure_count=6,
                    days_since_last_update=1000.0,
                    sbom_depth=1,
                )
            )
            graph.add_edge(SupplyChainEdge(f"bad{index}", "v", EdgeKind.DEPENDENCY))
        assert len(explain_node(graph, "v", max_paths=3).paths) == 3

    def test_works_without_a_model(self, planted) -> None:
        """A fresh tenant, or a dashboard rendering before the nightly training run.

        The graph-walk half does not depend on a fitted model and must stay available.
        """
        explanation = explain_node(planted, "vendor", model=None)
        assert explanation.paths
        assert np.isnan(explanation.risk_score)


class TestAttributionHonesty:
    """The model attribution must reflect the model, not the story."""

    @pytest.fixture(scope="class")
    def fitted(self):
        graph, gt = SyntheticGraphGenerator(seed=SEED).generate()
        ids = graph.node_ids()
        y = gt.labels(ids)
        split = GraphSplit.stratified(y, seed=SEED)
        model = SupplyChainGNN(random_state=SEED).fit(
            graph, y, split, exposure=gt.risk_vector(ids)
        )
        return graph, gt, model, model.risk_scores(graph)

    def test_shares_are_a_partition(self, fitted) -> None:
        graph, _, model, scores = fitted
        for explanation in top_risk_explanations(graph, scores, model=model, k=5):
            assert explanation.own_feature_share >= 0.0
            assert explanation.neighbourhood_share >= 0.0
            assert explanation.own_feature_share + explanation.neighbourhood_share == (
                pytest.approx(1.0)
            )

    def test_ablation_restores_the_model(self, fitted) -> None:
        """Attribution must not leave the model permanently blinded.

        The ablation zeroes every aggregation matrix in place. A missing restore
        would be a silent, model-wide corruption rather than a failed explanation.
        """
        graph, _, model, scores = fitted
        before = model.risk_scores(graph)
        top_risk_explanations(graph, scores, model=model, k=3)
        np.testing.assert_array_equal(model.risk_scores(graph), before)

    def test_the_model_attributes_an_organisation_to_its_supply_chain(
        self, fitted
    ) -> None:
        """An organisation's own features carry no risk signal by construction.

        Its CVE count is its own and its update age is its own; everything that makes
        it risky arrives across edges. So if the model's score for organisations were
        dominated by own-features, the paths shown next to it would be decoration.
        """
        graph, _, model, scores = fitted
        organisations = graph.indices_of_kind(NodeKind.ORGANIZATION)
        shares = [
            explanation.neighbourhood_share
            for explanation in top_risk_explanations(
                graph,
                np.where(
                    np.isin(np.arange(graph.n_nodes), organisations), scores, -np.inf
                ),
                model=model,
                k=5,
            )
        ]
        assert float(np.mean(shares)) > 0.5, (
            f"the model attributes organisations' risk mostly to their own features "
            f"(chain share {np.mean(shares):.2f}); the graph paths shown beside those "
            "scores would not be the reason the model fired"
        )

    def test_dominant_driver_agrees_with_the_shares(self, fitted) -> None:
        graph, _, model, scores = fitted
        for explanation in top_risk_explanations(graph, scores, model=model, k=8):
            expected = (
                "own_features"
                if explanation.own_feature_share >= explanation.neighbourhood_share
                else "supply_chain"
            )
            assert explanation.dominant_driver == expected

    def test_ablation_is_computed_once_for_a_top_k_table(self, fitted) -> None:
        """Ten rows must not cost ten full-graph ablation passes.

        Asserted by counting forward passes rather than by timing, which would be
        flaky on a loaded machine.
        """
        graph, _, model, scores = fitted
        calls = 0
        original = model._forward

        def counted(x):
            nonlocal calls
            calls += 1
            return original(x)

        model._forward = counted  # type: ignore[method-assign]
        try:
            top_risk_explanations(graph, scores, model=model, k=10)
        finally:
            model._forward = original  # type: ignore[method-assign]
        # One ablation = a small constant number of passes, not one per row.
        assert calls <= 6, f"{calls} forward passes for a 10-row table"


class TestTopRiskExplanations:
    @pytest.fixture
    def scored(self, planted):
        # Hand-made scores so the ranking is unambiguous.
        scores = np.zeros(planted.n_nodes)
        scores[planted.index_of("org")] = 0.95
        scores[planted.index_of("vendor")] = 0.90
        scores[planted.index_of("bad")] = 0.85
        return planted, scores

    def test_returns_the_highest_scoring_nodes_in_order(self, scored) -> None:
        graph, scores = scored
        explanations = top_risk_explanations(graph, scores, k=3)
        assert [e.node_id for e in explanations] == ["org", "vendor", "bad"]

    def test_k_is_clamped_to_the_graph(self, scored) -> None:
        graph, scores = scored
        assert len(top_risk_explanations(graph, scores, k=999)) == graph.n_nodes

    def test_mismatched_score_length_is_refused(self, planted) -> None:
        with pytest.raises(GraphError, match="do not match node count"):
            top_risk_explanations(planted, np.zeros(3))

    def test_invalid_k_is_refused(self, scored) -> None:
        graph, scores = scored
        with pytest.raises(GraphError, match="k must be"):
            top_risk_explanations(graph, scores, k=0)


class TestEvidenceRendering:
    """Typed evidence, so the InvestigationReport validator can do its job."""

    def test_paths_become_graph_path_evidence(self, planted) -> None:
        evidence = explain_node(planted, "vendor").as_evidence()
        assert evidence
        assert all(isinstance(item, Evidence) for item in evidence)
        assert all(item.kind is EvidenceKind.GRAPH_PATH for item in evidence)

    def test_intrinsic_risk_gets_its_own_citation(self, planted) -> None:
        refs = [item.ref for item in explain_node(planted, "bad").as_evidence()]
        assert any(ref.endswith("#intrinsic") for ref in refs)

    def test_refs_are_unique_so_citations_resolve(self, planted) -> None:
        """``InvestigationReport`` rejects a claim citing an unknown ref, and two
        pieces of evidence sharing a ref would make one of them unaddressable."""
        refs = [item.ref for item in explain_node(planted, "org", max_hops=4).as_evidence()]
        assert len(refs) == len(set(refs))

    def test_relevance_is_the_paths_share_of_inherited_risk(self, planted) -> None:
        evidence = explain_node(planted, "org", max_hops=4).as_evidence()
        for item in evidence:
            assert 0.0 <= item.relevance <= 1.0

    def test_excerpt_is_readable_by_an_analyst(self, planted) -> None:
        evidence = explain_node(planted, "vendor").as_evidence()
        assert "bad" in evidence[0].excerpt.raw
        assert "CVEs" in evidence[0].excerpt.raw

    def test_a_clean_node_produces_no_evidence(self, planted) -> None:
        assert explain_node(planted, "clean_direct").as_evidence() == ()

    def test_describe_is_a_complete_analyst_summary(self, planted) -> None:
        text = explain_node(planted, "vendor").describe()
        assert "vendor" in text and "bad" in text

    def test_describe_says_so_when_there_is_no_path(self, planted) -> None:
        assert "no exposure path" in explain_node(planted, "clean_direct").describe()
