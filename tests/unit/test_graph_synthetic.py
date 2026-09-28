"""The synthetic graph and its ground truth (PRD Section 7.1).

The ground truth *is* the F-06 benchmark, so these tests are mostly about whether
the benchmark is capable of measuring anything. Three specific failures were found
by measurement during the build and each now has a test:

1.  **A degenerate label.** With 4 organisations and binary reachability, 32/32
    vendors and 4/4 organisations were high-risk — nothing to rank.
2.  **Isolated nodes.** 41 packages had no edge in either direction, which is not a
    thing that exists in an SBOM.
3.  **An incoherent score.** Intrinsically risky packages were labelled high-risk
    while carrying an exposure score of 0.0, so a model regressing on the score
    ranked the genuinely compromised packages last.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sentinel.graph.schema import GraphError, NodeKind
from sentinel.graph.synthetic import (
    DECAY,
    EXPOSURE_HOPS,
    EXPOSURE_THRESHOLD,
    INTRINSIC_CVE_THRESHOLD,
    MAX_SBOM_DEPTH,
    GroundTruth,
    SyntheticGraphGenerator,
    build_demo_graph,
)

SEED = 20260928


@pytest.fixture(scope="module")
def demo():
    return build_demo_graph()


class TestShape:
    def test_is_about_five_hundred_nodes(self, demo) -> None:
        graph, _ = demo
        assert graph.n_nodes == 500  # PRD Section 7.1

    def test_every_node_kind_is_present(self, demo) -> None:
        graph, _ = demo
        for kind in NodeKind:
            assert graph.nodes_of_kind(kind), f"no {kind.value} nodes"

    def test_no_isolated_nodes(self, demo) -> None:
        """A package with no edges is not in anyone's SBOM.

        An early version produced 41 of them, because depth-1 packages were only
        linked when a vendor happened to pick them at random.
        """
        graph, _ = demo
        assert graph.describe()["n_isolated"] == 0

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
    def test_no_isolated_nodes_at_any_seed(self, seed: int) -> None:
        graph, _ = SyntheticGraphGenerator(seed=seed).generate()
        assert graph.describe()["n_isolated"] == 0

    def test_depth_reaches_the_configured_maximum(self, demo) -> None:
        graph, _ = demo
        assert graph.describe()["max_sbom_depth"] == MAX_SBOM_DEPTH

    def test_more_packages_are_shallow_than_deep(self, demo) -> None:
        """32 vendors each ship their own app, so depth 1 is populous.

        Reversing this starved depth 1, forced every vendor onto the same small pool
        of direct dependencies, and made one tainted package contaminate the entire
        graph.
        """
        graph, _ = demo
        by_depth: dict[int, int] = {}
        for node in graph.nodes_of_kind(NodeKind.PACKAGE):
            by_depth[node.sbom_depth] = by_depth.get(node.sbom_depth, 0) + 1
        assert by_depth[1] > by_depth[MAX_SBOM_DEPTH]

    def test_fan_out_is_heavy_tailed(self, demo) -> None:
        """Preferential attachment: a few packages are reused widely.

        Uniform wiring would make no single package matter much, removing the
        phenomenon the product exists to detect.
        """
        graph, _ = demo
        out_degree = graph.adjacency().astype(bool).sum(axis=0)
        assert out_degree.max() >= 3 * max(1.0, float(np.median(out_degree[out_degree > 0])))

    def test_the_graph_is_acyclic(self, demo) -> None:
        graph, _ = demo
        graph.assert_acyclic()

    def test_generation_is_deterministic(self) -> None:
        first, gt_first = SyntheticGraphGenerator(seed=SEED).generate()
        second, gt_second = SyntheticGraphGenerator(seed=SEED).generate()
        np.testing.assert_array_equal(first.adjacency(), second.adjacency())
        np.testing.assert_array_equal(first.feature_matrix(), second.feature_matrix())
        assert gt_first.high_risk == gt_second.high_risk

    def test_a_different_seed_gives_a_different_graph(self) -> None:
        first, _ = SyntheticGraphGenerator(seed=1).generate()
        second, _ = SyntheticGraphGenerator(seed=2).generate()
        assert not np.array_equal(first.adjacency(), second.adjacency())


class TestGroundTruthIsMeasurable:
    """Whether the benchmark can distinguish a good ranking from a bad one."""

    def test_prevalence_leaves_room_to_rank(self, demo) -> None:
        _, gt = demo
        rate = len(gt.high_risk) / 500
        assert 0.04 < rate < 0.35, f"high-risk prevalence {rate:.1%} is degenerate"

    def test_vendors_are_not_all_high_risk(self, demo) -> None:
        """The specific degeneracy that broke the first version (32/32 positive)."""
        graph, gt = demo
        vendors = graph.nodes_of_kind(NodeKind.VENDOR)
        flagged = sum(1 for node in vendors if node.node_id in gt.high_risk)
        assert 0 < flagged < len(vendors), f"{flagged}/{len(vendors)} vendors flagged"

    def test_packages_are_not_all_high_risk(self, demo) -> None:
        graph, gt = demo
        packages = graph.nodes_of_kind(NodeKind.PACKAGE)
        flagged = sum(1 for node in packages if node.node_id in gt.high_risk)
        assert 0 < flagged < len(packages)

    def test_both_risk_sources_are_populated(self, demo) -> None:
        """Intrinsic and inherited must both exist, or the GNN comparison is vacuous."""
        _, gt = demo
        assert len(gt.intrinsic) >= 5
        assert len(gt.inherited) >= 5

    def test_intrinsic_and_inherited_are_disjoint(self, demo) -> None:
        _, gt = demo
        assert not (gt.intrinsic & gt.inherited)

    def test_overlapping_sets_are_refused(self) -> None:
        with pytest.raises(GraphError, match="cannot be both"):
            GroundTruth(
                intrinsic=frozenset({"a"}),
                inherited=frozenset({"a"}),
                exposure_paths={},
            )


class TestRiskScoreCoherence:
    """One continuous quantity, one threshold — not a union of two notions."""

    def test_the_binary_label_is_exactly_a_threshold_on_the_score(self, demo) -> None:
        graph, gt = demo
        ids = graph.node_ids()
        labels = gt.labels(ids)
        risk = gt.risk_vector(ids)
        derived = (risk >= EXPOSURE_THRESHOLD).astype(float)
        np.testing.assert_array_equal(labels, derived)

    def test_intrinsically_risky_packages_score_above_the_threshold(self, demo) -> None:
        """The incoherence that broke regression: intrinsic nodes scored 0.0.

        They were labelled high-risk while carrying no exposure, so a model trained
        to predict the score ranked the genuinely compromised packages *last* — and
        still reported a healthy rank correlation while its top-10 precision fell to
        0.60.
        """
        _, gt = demo
        for node_id in gt.intrinsic:
            assert gt.risk_scores[node_id] >= EXPOSURE_THRESHOLD, (
                f"{node_id} is intrinsically risky but scores "
                f"{gt.risk_scores[node_id]:.3f}"
            )

    def test_intrinsic_is_a_subset_of_high_risk_by_construction(self, demo) -> None:
        _, gt = demo
        assert gt.intrinsic <= gt.high_risk

    def test_scores_are_graded_not_binary(self, demo) -> None:
        """Regression needs an ordering to learn; a two-valued target has none."""
        _, gt = demo
        values = np.array(list(gt.risk_scores.values()))
        assert np.unique(values).size > 20

    def test_risk_vector_is_zero_for_unexposed_nodes(self, demo) -> None:
        graph, gt = demo
        risk = gt.risk_vector(graph.node_ids())
        assert float(risk.min()) == 0.0

    def test_intrinsic_definition_is_a_conjunction(self, demo) -> None:
        """CVEs *and* unmaintained. Either alone is normal and must not qualify.

        The generator deliberately emits maintained-but-CVE-heavy and
        abandoned-but-clean packages, so a single threshold on one column cannot
        solve the problem.
        """
        graph, gt = demo
        packages = graph.nodes_of_kind(NodeKind.PACKAGE)
        cve_heavy_maintained = [
            node
            for node in packages
            if node.cve_exposure_count >= INTRINSIC_CVE_THRESHOLD
            and not node.is_unmaintained
        ]
        abandoned_clean = [
            node
            for node in packages
            if node.is_unmaintained and node.cve_exposure_count < INTRINSIC_CVE_THRESHOLD
        ]
        assert cve_heavy_maintained, "no maintained-but-CVE-heavy decoys generated"
        assert abandoned_clean, "no abandoned-but-clean decoys generated"
        for node in cve_heavy_maintained + abandoned_clean:
            assert node.node_id not in gt.intrinsic


class TestDecayedPropagation:
    def test_exposure_attenuates_with_distance(self, demo) -> None:
        """Nodes further from risk must carry less of it.

        Restricted to nodes exposed to **exactly one** intrinsic source, because a
        node with several exposures accumulates them and its total says nothing about
        any single distance. Comparing group means over singly-exposed nodes isolates
        the decay, which is the thing the name claims to test.
        """
        graph, gt = demo

        # Count how many distinct intrinsic sources reach each node, and at what hop.
        reach: dict[str, list[int]] = {}
        for source in gt.intrinsic:
            for node_id, hops in graph.downstream(source, max_hops=EXPOSURE_HOPS).items():
                reach.setdefault(node_id, []).append(hops)

        by_hop: dict[int, list[float]] = {}
        for node_id, hop_list in reach.items():
            if len(hop_list) != 1 or node_id in gt.intrinsic:
                continue  # multiply-exposed, or carrying its own intrinsic term
            by_hop.setdefault(hop_list[0], []).append(gt.risk_scores[node_id])

        populated = sorted(h for h, values in by_hop.items() if len(values) >= 3)
        assert len(populated) >= 2, (
            f"need singly-exposed nodes at two different distances to compare; got "
            f"{ {h: len(v) for h, v in by_hop.items()} }"
        )
        near, far = populated[0], populated[-1]
        assert float(np.mean(by_hop[near])) > float(np.mean(by_hop[far])), (
            f"risk at {near} hop(s) ({np.mean(by_hop[near]):.4f}) is not greater than "
            f"at {far} hop(s) ({np.mean(by_hop[far]):.4f}); decay is not being applied"
        )

    def test_a_fourth_order_exposure_is_material_but_not_sufficient_alone(self) -> None:
        """The calibration the threshold and the decay jointly imply.

        At ``DECAY=0.7`` a four-hop exposure contributes ``0.7**4 = 0.240``, which is
        24% of the threshold: not decisive on its own, but enough that four or five
        of them flag a node, and enough to tip a node already near the line. That is
        the intended semantics — deep exposure counts and has to accumulate.

        Both bounds are asserted because both have been wrong. At ``DECAY=0.45`` a
        four-hop exposure was worth 0.041, or 14% of the then-threshold, and **no**
        fourth-order exposure was ever labelled — the PRD's headline scenario was
        simply absent from the benchmark. Pushing decay much higher fails the other
        way: it makes distance nearly free and the label collapses toward binary
        reachability.
        """
        single = DECAY**EXPOSURE_HOPS
        assert single < EXPOSURE_THRESHOLD, "a lone distant exposure should not suffice"
        assert single / EXPOSURE_THRESHOLD >= 0.15, (
            f"a fourth-order exposure is worth only {single / EXPOSURE_THRESHOLD:.1%} "
            "of the threshold; it can never influence a label and the PRD's "
            "fourth-order claim would be untestable"
        )
        assert single / EXPOSURE_THRESHOLD <= 0.60, (
            "distance has become nearly free; the label is collapsing toward binary "
            "reachability, which is the degeneracy the decay exists to avoid"
        )

    def test_deep_exposure_actually_occurs_in_the_generated_graph(self, demo) -> None:
        """Arithmetic permitting deep exposure is not the same as it happening.

        Walks the graph directly: are there nodes genuinely three or more hops
        downstream of an intrinsically risky package? Without this, the test above
        could pass on a graph whose trees are all two levels deep.
        """
        graph, gt = demo
        deep_reach: set[str] = set()
        for source in gt.intrinsic:
            for node_id, hops in graph.downstream(
                source, max_hops=EXPOSURE_HOPS
            ).items():
                if hops >= 3:
                    deep_reach.add(node_id)
        assert len(deep_reach) >= 10, (
            f"only {len(deep_reach)} nodes sit 3+ hops downstream of a risky package; "
            "the graph is too shallow to exercise transitive propagation"
        )
        # And at least some of them are flagged, so deep exposure reaches the label.
        assert deep_reach & gt.high_risk

    def test_organisations_saturate_and_that_is_recorded_not_accidental(
        self, demo
    ) -> None:
        """A structural property of the graph, pinned so it cannot drift silently.

        Organisations aggregate 4-12 vendors, each carrying 9-13 direct dependencies,
        so an organisation sits 2-4 hops from most of the package graph and
        accumulates far more exposure than any single package can. **No single global
        threshold both de-saturates organisations and keeps deep exposure
        representable** — raising it to spread the organisations out pushes
        fourth-order contributions below relevance, and lowering it flags every
        organisation.

        This is the PRD's own thesis as a measurement rather than a defect: nearly
        every mid-market company *does* have third-party exposure, so the product
        question is never "which client is exposed" but "which is exposed most, and
        through what". The ranking metrics and the per-kind breakdown in
        ``test_graph_gnn.py`` are what carry the evaluation for organisations.
        """
        graph, gt = demo
        organisations = graph.nodes_of_kind(NodeKind.ORGANIZATION)
        flagged = sum(1 for node in organisations if node.node_id in gt.high_risk)
        assert flagged >= 0.7 * len(organisations), (
            f"only {flagged}/{len(organisations)} organisations flagged — if the "
            "generator has changed so that organisations now discriminate, this test "
            "and the per-kind reporting it justifies should be revisited"
        )
        # Their *scores* must still spread, which is what makes ranking possible.
        scores = np.array([gt.risk_scores.get(n.node_id, 0.0) for n in organisations])
        assert scores.std() > 0.1, "organisation risk scores are flat; nothing to rank"

    def test_exposure_paths_run_source_first(self, demo) -> None:
        """Paths must read as causal chains: risky package first, flagged node last."""
        _, gt = demo
        for node_id, path in gt.exposure_paths.items():
            assert path[-1] == node_id
            assert path[0] in gt.intrinsic

    def test_exposure_paths_are_real_edges(self, demo) -> None:
        """A cited path that is not walkable is worse than no citation."""
        graph, gt = demo
        for path in gt.exposure_paths.values():
            for source, target in itertools.pairwise(path):
                assert target in graph.targets_of(source), (
                    f"path claims {source} -> {target} but no such edge exists"
                )

    def test_paths_respect_the_hop_horizon(self, demo) -> None:
        _, gt = demo
        for path in gt.exposure_paths.values():
            assert len(path) - 1 <= EXPOSURE_HOPS


class TestGeneratorValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_organizations": 0}, "at least one"),
            ({"intrinsic_risk_rate": 0.0}, "intrinsic_risk_rate"),
            ({"intrinsic_risk_rate": 0.9}, "intrinsic_risk_rate"),
            ({"max_depth": 1}, "max_depth"),
            ({"exposure_threshold": 0.0}, "exposure_threshold"),
        ],
    )
    def test_invalid_settings_are_refused(self, kwargs: dict, match: str) -> None:
        with pytest.raises(GraphError, match=match):
            SyntheticGraphGenerator(**kwargs)

    def test_too_few_packages_for_the_depth_is_refused(self) -> None:
        with pytest.raises(GraphError, match="cannot populate"):
            SyntheticGraphGenerator(n_packages=3, max_depth=6).generate()

    def test_summary_is_informative(self, demo) -> None:
        _, gt = demo
        text = gt.summary()
        assert "intrinsic" in text and "inherited" in text

