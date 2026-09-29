"""Scoped explanation (Part 5): precomputed ``shares`` and forced ``include``.

An advisory-driven review explains a *subgraph* — what one package reaches — but
the GNN is full-graph and cannot be re-run on it, and organisations saturate the
ranking so the advisory's own package can fall outside the top k. These are the two
extensions that make the scoped review possible, and the property they must keep is
that scoping changes which nodes are shown, never what any of them scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.graph.explain import neighbourhood_shares, top_risk_explanations
from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
from sentinel.graph.schema import GraphError
from sentinel.graph.synthetic import SyntheticGraphGenerator


@pytest.fixture(scope="module")
def fitted():
    graph, truth = SyntheticGraphGenerator(seed=20260928).generate()
    ids = graph.node_ids()
    labels = truth.labels(ids)
    model = SupplyChainGNN(random_state=20260928).fit(
        graph, labels, GraphSplit.stratified(labels, seed=20260928),
        exposure=truth.risk_vector(ids),
    )
    return graph, model, model.risk_scores(graph)


def test_precomputed_shares_equal_the_model_s_own(fitted):
    graph, model, scores = fitted
    direct = top_risk_explanations(graph, scores, model=model, k=10)
    shares = neighbourhood_shares(model, graph)
    handed = top_risk_explanations(graph, scores, k=10, shares=shares)
    assert [e.node_id for e in direct] == [e.node_id for e in handed]
    for a, b in zip(direct, handed, strict=True):
        assert a.neighbourhood_share == pytest.approx(b.neighbourhood_share)
        assert a.dominant_driver == b.dominant_driver


def test_a_full_graph_model_cannot_score_a_subgraph(fitted):
    graph, model, _scores = fitted
    sub = graph.subgraph(graph.node_ids()[:7])
    with pytest.raises(GraphError, match="500 nodes"):
        model.risk_scores(sub)


def test_restricted_shares_let_a_subgraph_be_explained(fitted):
    graph, model, scores = fitted
    ids = graph.node_ids()[100:160]
    sub = graph.subgraph(ids)
    rows = [graph.index_of(n) for n in sub.node_ids()]
    shares = neighbourhood_shares(model, graph)
    explained = top_risk_explanations(sub, scores[rows], k=5, shares=shares[rows])
    for item in explained:
        assert item.risk_score == pytest.approx(scores[graph.index_of(item.node_id)])
        assert item.neighbourhood_share == pytest.approx(shares[graph.index_of(item.node_id)])


@pytest.mark.parametrize("bad", [np.full(3, 0.5), np.full(500, 1.5), np.full(500, np.nan)])
def test_misaligned_or_invalid_shares_are_refused(fitted, bad):
    graph, _model, scores = fitted
    with pytest.raises(GraphError):
        top_risk_explanations(graph, scores, k=5, shares=bad)


def test_include_appends_nodes_outside_the_top_k(fitted):
    graph, _model, scores = fitted
    order = np.argsort(-scores, kind="stable")
    low = graph.nodes[int(order[-1])].node_id
    mid = graph.nodes[int(order[200])].node_id
    explained = top_risk_explanations(graph, scores, k=5, include=(low, mid))
    ids = [e.node_id for e in explained]
    assert ids[:5] == [graph.nodes[int(i)].node_id for i in order[:5]]
    assert ids[5:] == [mid, low], "extras follow the top k, in score order"


def test_include_does_not_duplicate_a_top_node(fitted):
    graph, _model, scores = fitted
    top = graph.nodes[int(np.argmax(scores))].node_id
    explained = top_risk_explanations(graph, scores, k=5, include=(top,))
    assert [e.node_id for e in explained].count(top) == 1
    assert len(explained) == 5


def test_parallel_edges_are_one_path_not_two(fitted):
    # Part 5 finding: 67 vendor->organisation pairs carry both an API and a
    # contractual edge, and each was walked separately, so the same node sequence
    # was reported twice and its contribution counted twice.
    from collections import Counter

    from sentinel.graph.explain import explain_node

    graph, _model, _scores = fitted
    pairs = Counter((e.source, e.target) for e in graph.edges)
    doubled = [pair for pair, count in pairs.items() if count > 1]
    assert doubled, "the synthetic graph has parallel edges to exercise"
    for _vendor, org in doubled[:10]:
        paths = explain_node(graph, org, max_paths=500).paths
        sequences = [p.nodes for p in paths]
        assert len(sequences) == len(set(sequences)), org
