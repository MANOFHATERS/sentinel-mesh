"""Graph schema: node/edge validity, and above all **edge direction**.

Direction is the load-bearing invariant. Every edge points the way risk flows, so a
dependency edge runs ``dependency -> dependent`` — the reverse of SBOM notation and
the reverse of how most people first draw it. Transposing it produces a model that
still trains, still converges, and confidently answers a different question.

:class:`TestDirectionSemantics` therefore uses an **asymmetric** fixture. A
symmetric test graph would pass with the adjacency transposed, which is the one
thing these tests exist to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.graph.schema import (
    FEATURE_NAMES,
    UNMAINTAINED_DAYS,
    EdgeKind,
    GraphError,
    Node,
    NodeKind,
    SupplyChainEdge,
    SupplyChainGraph,
)


def pkg(node_id: str, **kwargs: object) -> Node:
    return Node(node_id=node_id, kind=NodeKind.PACKAGE, name=node_id, **kwargs)  # type: ignore[arg-type]


def vendor(node_id: str, **kwargs: object) -> Node:
    return Node(node_id=node_id, kind=NodeKind.VENDOR, name=node_id, **kwargs)  # type: ignore[arg-type]


def org(node_id: str, **kwargs: object) -> Node:
    return Node(node_id=node_id, kind=NodeKind.ORGANIZATION, name=node_id, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def chain() -> SupplyChainGraph:
    """An asymmetric chain: ``deep -> mid -> shallow -> v1 -> o1``.

    Deliberately a path with no back-edges and distinct node kinds, so transposing
    the adjacency changes every answer below.
    """
    graph = SupplyChainGraph()
    for node in (pkg("deep"), pkg("mid"), pkg("shallow")):
        graph.add_node(node)
    graph.add_node(vendor("v1"))
    graph.add_node(org("o1"))
    graph.add_edge(SupplyChainEdge("deep", "mid", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("mid", "shallow", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("shallow", "v1", EdgeKind.DEPENDENCY))
    graph.add_edge(SupplyChainEdge("v1", "o1", EdgeKind.CONTRACTUAL))
    return graph


class TestNode:
    def test_feature_vector_follows_the_declared_order(self) -> None:
        node = pkg(
            "p",
            cve_exposure_count=7,
            days_since_last_update=900.0,
            sbom_depth=3,
            breach_history=1,
        )
        np.testing.assert_allclose(node.feature_vector(), [7.0, 900.0, 3.0, 1.0, 1.0])
        assert len(FEATURE_NAMES) == node.feature_vector().size

    def test_unmaintained_is_a_threshold_on_staleness(self) -> None:
        assert pkg("a", days_since_last_update=UNMAINTAINED_DAYS).is_unmaintained
        assert not pkg("b", days_since_last_update=UNMAINTAINED_DAYS - 1).is_unmaintained

    def test_features_are_raw_not_normalised(self) -> None:
        """Scaling here would hide the leak it is supposed to prevent.

        Standardisation must be fitted on training nodes only, which is the model's
        job. A node that normalised itself could only do so against global
        statistics, making a transductive leak invisible at the call site.
        """
        assert pkg("p", days_since_last_update=900.0).feature_vector()[1] == 900.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"cve_exposure_count": -1}, "cve_exposure_count"),
            ({"days_since_last_update": -1.0}, "days_since_last_update"),
            ({"sbom_depth": -1}, "sbom_depth"),
            ({"breach_history": -1}, "breach_history"),
        ],
    )
    def test_negative_features_are_refused(self, kwargs: dict, match: str) -> None:
        with pytest.raises(GraphError, match=match):
            pkg("p", **kwargs)

    @pytest.mark.parametrize("bad", ["", "  padded  "])
    def test_bad_node_ids_are_refused(self, bad: str) -> None:
        with pytest.raises(GraphError, match="node_id"):
            Node(node_id=bad, kind=NodeKind.PACKAGE, name="x")


class TestEdge:
    def test_self_loop_is_refused(self) -> None:
        with pytest.raises(GraphError, match="self-loop"):
            SupplyChainEdge("a", "a", EdgeKind.DEPENDENCY)

    @pytest.mark.parametrize("weight", [0.0, -0.5, 1.5])
    def test_out_of_range_weight_is_refused(self, weight: float) -> None:
        with pytest.raises(GraphError, match="weight"):
            SupplyChainEdge("a", "b", EdgeKind.DEPENDENCY, weight=weight)

    def test_a_dependency_edge_from_an_organisation_is_refused(self) -> None:
        """The transposed-graph bug announcing itself at the edge level.

        Risk never flows from an organisation into a package, so this pairing can
        only mean the graph is being built backwards. Catching it here costs nothing;
        inferring it later from a strange top-10 list costs a day.
        """
        graph = SupplyChainGraph()
        graph.add_node(org("o1"))
        graph.add_node(pkg("p1"))
        with pytest.raises(GraphError, match="not a direction risk flows"):
            graph.add_edge(SupplyChainEdge("o1", "p1", EdgeKind.DEPENDENCY))

    def test_a_contractual_edge_between_packages_is_refused(self) -> None:
        graph = SupplyChainGraph()
        graph.add_node(pkg("p1"))
        graph.add_node(pkg("p2"))
        with pytest.raises(GraphError, match="not a direction risk flows"):
            graph.add_edge(SupplyChainEdge("p1", "p2", EdgeKind.CONTRACTUAL))

    def test_an_edge_to_an_unknown_node_is_refused(self, chain) -> None:
        with pytest.raises(GraphError, match="unknown node"):
            chain.add_edge(SupplyChainEdge("deep", "ghost", EdgeKind.DEPENDENCY))

    def test_duplicate_node_ids_are_refused(self, chain) -> None:
        with pytest.raises(GraphError, match="duplicate node_id"):
            chain.add_node(pkg("deep"))


class TestDirectionSemantics:
    """The adjacency must encode risk flow, not SBOM notation."""

    def test_adjacency_is_indexed_target_source(self, chain) -> None:
        a = chain.adjacency()
        deep, mid = chain.index_of("deep"), chain.index_of("mid")
        # Risk flows deep -> mid, so mid's ROW records deep as an input.
        assert a[mid, deep] == 1.0
        assert a[deep, mid] == 0.0

    def test_aggregation_row_gathers_what_flows_in(self, chain) -> None:
        """``(A @ H)[v]`` must be the mean over v's risk *sources*."""
        a = chain.aggregation_matrix()
        h = np.arange(chain.n_nodes, dtype=float).reshape(-1, 1)
        aggregated = a @ h
        mid = chain.index_of("mid")
        assert aggregated[mid, 0] == pytest.approx(float(chain.index_of("deep")))

    def test_sources_and_targets_are_not_interchangeable(self, chain) -> None:
        assert chain.sources_of("mid") == ["deep"]
        assert chain.targets_of("mid") == ["shallow"]

    def test_downstream_follows_risk_flow_to_the_organisation(self, chain) -> None:
        reached = chain.downstream("deep")
        assert reached == {"mid": 1, "shallow": 2, "v1": 3, "o1": 4}

    def test_upstream_is_the_exposure_ancestry(self, chain) -> None:
        assert chain.upstream("o1") == {"v1": 1, "shallow": 2, "mid": 3, "deep": 4}

    def test_a_leaf_dependency_has_no_sources(self, chain) -> None:
        assert chain.sources_of("deep") == []
        assert chain.downstream("o1") == {}

    def test_max_hops_truncates_at_the_horizon(self, chain) -> None:
        assert chain.downstream("deep", max_hops=2) == {"mid": 1, "shallow": 2}

    def test_negative_max_hops_is_refused(self, chain) -> None:
        with pytest.raises(GraphError, match="max_hops"):
            chain.downstream("deep", max_hops=-1)

    def test_distances_are_shortest_not_first_found(self) -> None:
        """BFS, not DFS. Exposure at 1 hop and at 4 hops are different risks.

        ``a`` reaches ``d`` both directly and via ``b -> c``. A depth-first walk
        would report whichever it stumbled on first.
        """
        graph = SupplyChainGraph()
        for node_id in "abcd":
            graph.add_node(pkg(node_id))
        graph.add_edge(SupplyChainEdge("a", "b", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("b", "c", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("c", "d", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("a", "d", EdgeKind.DEPENDENCY))
        assert graph.downstream("a")["d"] == 1


class TestAggregationMatrix:
    def test_mean_rows_sum_to_one_where_there_are_inputs(self, chain) -> None:
        a = chain.aggregation_matrix(mode="mean")
        sums = a.sum(axis=1)
        for node in chain.nodes:
            index = chain.index_of(node.node_id)
            expected = 1.0 if chain.sources_of(node.node_id) else 0.0
            assert sums[index] == pytest.approx(expected)

    def test_a_node_with_no_inputs_gets_a_zero_row_not_a_nan(self, chain) -> None:
        """Division by a zero degree would poison every downstream gradient."""
        a = chain.aggregation_matrix()
        assert np.all(np.isfinite(a))
        assert np.all(a[chain.index_of("deep")] == 0.0)

    def test_sum_mode_preserves_the_count(self) -> None:
        """The difference that motivates having both modes.

        Two vendors, one with a single risky dependency and one with four. Mean
        reports them identically; sum does not.
        """
        graph = SupplyChainGraph()
        graph.add_node(vendor("one"))
        graph.add_node(vendor("four"))
        graph.add_node(pkg("p0"))
        graph.add_edge(SupplyChainEdge("p0", "one", EdgeKind.DEPENDENCY))
        for i in range(1, 5):
            graph.add_node(pkg(f"q{i}"))
            graph.add_edge(SupplyChainEdge(f"q{i}", "four", EdgeKind.DEPENDENCY))

        h = np.ones((graph.n_nodes, 1))
        mean = graph.aggregation_matrix(mode="mean") @ h
        summed = graph.aggregation_matrix(mode="sum") @ h
        one, four = graph.index_of("one"), graph.index_of("four")

        assert mean[one, 0] == pytest.approx(mean[four, 0]), "mean should collapse them"
        assert summed[four, 0] == pytest.approx(4.0 * summed[one, 0])

    def test_self_loops_are_off_by_default(self, chain) -> None:
        """GraphSAGE keeps ``W_self`` separate; folding a self-loop in would blend a
        node's own risk into the signal meant to represent *inherited* risk, and the
        explainer could no longer separate the two."""
        assert np.all(np.diag(chain.aggregation_matrix()) == 0.0)
        assert np.any(np.diag(chain.aggregation_matrix(add_self_loops=True)) > 0.0)

    def test_an_unknown_mode_is_refused(self, chain) -> None:
        with pytest.raises(GraphError, match="aggregation mode"):
            chain.aggregation_matrix(mode="max")

    def test_edge_weights_are_honoured(self) -> None:
        graph = SupplyChainGraph()
        graph.add_node(vendor("v"))
        graph.add_node(org("o"))
        graph.add_edge(SupplyChainEdge("v", "o", EdgeKind.API, weight=0.5))
        assert graph.adjacency()[graph.index_of("o"), graph.index_of("v")] == 0.5


class TestStructure:
    def test_feature_matrix_is_one_row_per_node_in_order(self, chain) -> None:
        matrix = chain.feature_matrix()
        assert matrix.shape == (chain.n_nodes, len(FEATURE_NAMES))
        for node in chain.nodes:
            np.testing.assert_allclose(
                matrix[chain.index_of(node.node_id)], node.feature_vector()
            )

    def test_empty_graph_feature_matrix_is_refused(self) -> None:
        with pytest.raises(GraphError, match="empty graph"):
            SupplyChainGraph().feature_matrix()

    def test_acyclic_check_passes_on_a_chain(self, chain) -> None:
        chain.assert_acyclic()

    def test_acyclic_check_detects_a_cycle(self) -> None:
        graph = SupplyChainGraph()
        for node_id in "abc":
            graph.add_node(pkg(node_id))
        graph.add_edge(SupplyChainEdge("a", "b", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("b", "c", EdgeKind.DEPENDENCY))
        graph.add_edge(SupplyChainEdge("c", "a", EdgeKind.DEPENDENCY))
        with pytest.raises(GraphError, match="cycle"):
            graph.assert_acyclic()

    def test_unknown_node_lookup_is_refused(self, chain) -> None:
        with pytest.raises(GraphError, match="unknown node_id"):
            chain.index_of("nope")

    def test_indices_of_kind_selects_correctly(self, chain) -> None:
        indices = chain.indices_of_kind(NodeKind.PACKAGE)
        assert {chain.nodes[i].node_id for i in indices} == {"deep", "mid", "shallow"}

    def test_describe_reports_shape(self, chain) -> None:
        description = chain.describe()
        assert description["n_nodes"] == 5
        assert description["n_edges"] == 4
        assert description["nodes_by_kind"]["package"] == 3
        assert description["edges_by_kind"]["dependency"] == 3
        assert description["n_isolated"] == 0

    def test_subgraph_keeps_only_internal_edges(self, chain) -> None:
        sub = chain.subgraph(["mid", "shallow", "v1"])
        assert set(sub.node_ids()) == {"mid", "shallow", "v1"}
        # The deep->mid edge had one endpoint outside the selection and must be gone.
        assert len(sub.edges) == 2

    def test_subgraph_with_unknown_ids_is_refused(self, chain) -> None:
        with pytest.raises(GraphError, match="unknown node ids"):
            chain.subgraph(["mid", "ghost"])

    def test_from_parts_round_trips(self, chain) -> None:
        rebuilt = SupplyChainGraph.from_parts(chain.nodes, chain.edges)
        np.testing.assert_allclose(rebuilt.adjacency(), chain.adjacency())
