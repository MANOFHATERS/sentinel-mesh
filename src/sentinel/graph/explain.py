"""Why a node scored high: the specific path that drove it (PRD F-06, Section 5.4).

F-06's guardrail is not "produce a score", it is:

    *"Flags are explainable via the specific graph path that drove the score."*

That is a hard requirement rather than a nicety, and PRD Section 5.1 says why:
*"an analyst should never have to trust an agent's conclusion without being able to
see why."* A supply-chain risk score with no path attached is unactionable — the
vCISO reading it cannot tell whether to chase a vendor, pin a dependency, or ignore
it.

Two attributions, because they answer different questions
---------------------------------------------------------
:func:`explain_node` produces both, and they are deliberately not the same number.

**Structural attribution** walks the graph and reports the concrete exposure paths:
*"vendor-017 is high-risk because pkg-0231 (11 CVEs, 1,847 days stale) reaches it
via pkg-0088 -> pkg-0012."* This is ground-truth-shaped, model-independent, and it
is what goes in a report to a human. It is computed from the graph, so it is
correct whether or not the model is.

**Model attribution** reports how much of *this model's* score came from the node's
own features versus from its neighbourhood, by re-running the forward pass with the
neighbour branch ablated. This is what makes the model auditable rather than
merely accompanied by a plausible story. The distinction matters: if the structural
explanation says "inherited from four risky dependencies" while the model
attribution says the score was 95% own-features, then the story being shown to the
analyst is not the reason the model fired, and one of the two is wrong.

Keeping them separate is the point. A single blended "explanation" would let a
confident-sounding path narration paper over a model that is actually ignoring the
graph — which is exactly the failure
``test_graph_explain.py::TestAttributionHonesty`` exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from sentinel.core.schemas import Evidence, EvidenceKind
from sentinel.graph.schema import GraphError, Node, NodeKind, SupplyChainGraph

if TYPE_CHECKING:
    from sentinel.graph.gnn import SupplyChainGNN

__all__ = [
    "ExposurePath",
    "NodeExplanation",
    "explain_node",
    "top_risk_explanations",
]

DTYPE: Final = np.float64


@dataclass(frozen=True, slots=True)
class ExposurePath:
    """One concrete route by which risk reaches a node.

    ``nodes`` runs source-first: ``(risky_package, ..., explained_node)``, the
    direction risk flows, so it reads as a causal chain rather than needing to be
    reversed in the reader's head.
    """

    nodes: tuple[str, ...]
    source_cves: int
    source_days_stale: float
    contribution: float

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise GraphError(
                f"an exposure path needs a source and a target, got {self.nodes}"
            )

    @property
    def hops(self) -> int:
        return len(self.nodes) - 1

    @property
    def source(self) -> str:
        return self.nodes[0]

    @property
    def target(self) -> str:
        return self.nodes[-1]

    def describe(self) -> str:
        chain = " -> ".join(self.nodes)
        return (
            f"{self.source} ({self.source_cves} CVEs, "
            f"{self.source_days_stale:.0f} days stale) reaches {self.target} "
            f"in {self.hops} hop(s) via {chain}; contribution {self.contribution:.3f}"
        )


@dataclass(frozen=True, slots=True)
class NodeExplanation:
    """Everything an analyst needs to act on one flagged node."""

    node_id: str
    kind: NodeKind
    risk_score: float
    paths: tuple[ExposurePath, ...]
    own_feature_share: float
    neighbourhood_share: float
    is_intrinsically_risky: bool

    @property
    def dominant_driver(self) -> str:
        """``"own_features"`` or ``"supply_chain"`` — what the *model* keyed on."""
        return (
            "own_features"
            if self.own_feature_share >= self.neighbourhood_share
            else "supply_chain"
        )

    @property
    def total_inherited_contribution(self) -> float:
        return float(sum(path.contribution for path in self.paths))

    def describe(self) -> str:
        lines = [
            f"{self.node_id} ({self.kind.value}) risk={self.risk_score:.4f} "
            f"driver={self.dominant_driver} "
            f"(own {self.own_feature_share:.0%} / chain {self.neighbourhood_share:.0%})"
        ]
        if self.is_intrinsically_risky:
            lines.append("  intrinsically risky: unmaintained with known CVEs")
        if not self.paths:
            lines.append("  no exposure path within the search horizon")
        for path in self.paths:
            lines.append(f"  {path.describe()}")
        return "\n".join(lines)

    def as_evidence(self) -> tuple[Evidence, ...]:
        """Render as :class:`~sentinel.core.schemas.Evidence` for an agent report.

        The Supply-Chain Agent's findings flow into an
        :class:`~sentinel.core.schemas.InvestigationReport`, whose validator rejects
        any claim citing a ref that does not resolve. Emitting typed evidence here —
        rather than a formatted string the agent layer would have to re-parse — is
        what makes that validator able to do its job, and it is why
        :class:`~sentinel.core.schemas.EvidenceKind` already has a ``GRAPH_PATH``
        member: Part 1 anticipated this shape.
        """
        items: list[Evidence] = []
        if self.is_intrinsically_risky:
            items.append(
                Evidence(
                    kind=EvidenceKind.GRAPH_PATH,
                    ref=f"graph://node/{self.node_id}#intrinsic",
                    excerpt=(
                        f"{self.node_id} is unmaintained and carries known CVEs, so it "
                        "is a risk source in its own right, not only a conduit."
                    ),
                    relevance=1.0,
                )
            )
        for index, path in enumerate(self.paths):
            items.append(
                Evidence(
                    kind=EvidenceKind.GRAPH_PATH,
                    ref=f"graph://path/{self.node_id}#{index}",
                    excerpt=path.describe(),
                    # Relevance is the path's share of this node's inherited risk, so
                    # an analyst reading a sorted evidence list sees the path worth
                    # chasing first. Clamped into [0, 1] because Confidence requires it.
                    relevance=min(
                        1.0,
                        max(
                            0.0,
                            path.contribution / max(self.total_inherited_contribution, 1e-12),
                        ),
                    ),
                )
            )
        return tuple(items)


# --------------------------------------------------------------------------- #
# Structural attribution
# --------------------------------------------------------------------------- #


def _risk_sources(graph: SupplyChainGraph, cve_threshold: int) -> set[str]:
    """Packages that are risk sources in their own right."""
    return {
        node.node_id
        for node in graph.nodes
        if node.kind is NodeKind.PACKAGE
        and node.cve_exposure_count >= cve_threshold
        and node.is_unmaintained
    }


def _enumerate_paths(
    graph: SupplyChainGraph,
    target: str,
    sources: set[str],
    max_hops: int,
) -> list[tuple[str, ...]]:
    """All simple paths of length <= ``max_hops`` from any source into ``target``.

    Searched **backwards** from the target rather than forwards from every source.
    Forward search from each of ~25 sources explores most of the graph and then
    discards everything that does not happen to end at this node; backward search
    from one target touches only its ``max_hops``-hop ancestry, which is a few dozen
    nodes. For a 500-node graph either terminates, but the explainer is called once
    per row of a top-k table in a dashboard request, so the constant matters.

    ``visited`` tracking is per-path, not global: two different sources legitimately
    reach the same node through overlapping routes, and a global visited set would
    silently report only the first one found — losing exactly the "you are exposed
    through several independent paths" finding that justifies a high score.
    """
    found: list[tuple[str, ...]] = []
    # Each frontier entry is a path written target-last, built up backwards.
    frontier: list[tuple[str, ...]] = [(target,)]
    for _ in range(max_hops):
        next_frontier: list[tuple[str, ...]] = []
        for path in frontier:
            for predecessor in graph.sources_of(path[0]):
                if predecessor in path:
                    continue  # no cycles, and no revisiting within this path
                extended = (predecessor, *path)
                if predecessor in sources:
                    found.append(extended)
                    # Do not extend past a source: the path "risky package reaches
                    # you" is the finding. Continuing through it would enumerate
                    # that source's own ancestry, which is a different node's
                    # explanation.
                    continue
                next_frontier.append(extended)
        frontier = next_frontier
        if not frontier:
            break
    return found


# --------------------------------------------------------------------------- #
# Model attribution
# --------------------------------------------------------------------------- #


def _ablation_shares(model: SupplyChainGNN, graph: SupplyChainGraph) -> np.ndarray:
    """Per-node share of the score attributable to the neighbourhood, in ``[0, 1]``.

    Computed by ablation: run the forward pass normally, then again with every
    ``SageLayer``'s aggregation matrix zeroed so each node sees only itself, and
    compare. The share is

        ``|full - self_only| / (|full - self_only| + |self_only - baseline|)``

    where ``baseline`` is the score of a node with mean features and no neighbours.

    Ablation rather than gradients, for a reason worth stating. A gradient
    ``d score / d neighbour_features`` is a *local* sensitivity — it answers "if this
    neighbour got slightly worse, how much would the score move" — whereas an
    analyst is asking the counterfactual "would this node be flagged at all if it
    had no supply chain". Those diverge badly at saturation: a node whose score is
    pinned near 1.0 by its neighbourhood has a near-zero gradient with respect to
    it, and a gradient attribution would report the neighbourhood as irrelevant
    precisely when it is the whole story.

    The matrices are restored in a ``finally``, so an exception mid-ablation cannot
    leave the model permanently blinded — which would be a silent, model-wide
    corruption rather than a failed explanation.
    """
    from sentinel.graph.gnn import SageLayer

    x = model._standardize(graph.feature_matrix())
    full = model.logits(graph)

    sage_layers = [layer for layer in model._layers if isinstance(layer, SageLayer)]
    if not sage_layers:
        return np.zeros(graph.n_nodes, dtype=DTYPE)

    originals = [layer.aggregation for layer in sage_layers]
    try:
        for layer in sage_layers:
            layer.aggregation = np.zeros_like(layer.aggregation)
        self_only = model._forward(x)
        # A node with average features and no neighbours: the reference point for
        # "how much did this node's own features move it away from typical".
        neutral = np.zeros_like(x)
        baseline = float(np.mean(model._forward(neutral)))
    finally:
        for layer, original in zip(sage_layers, originals, strict=True):
            layer.aggregation = original

    from_neighbours = np.abs(full - self_only)
    from_self = np.abs(self_only - baseline)
    total = from_neighbours + from_self
    return np.divide(
        from_neighbours,
        total,
        out=np.full_like(from_neighbours, 0.5),  # perfectly ambiguous when both are 0
        where=total > 1e-12,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def explain_node(
    graph: SupplyChainGraph,
    node_id: str,
    *,
    model: SupplyChainGNN | None = None,
    risk_score: float | None = None,
    max_hops: int = 4,
    max_paths: int = 5,
    cve_threshold: int = 4,
    decay: float = 0.7,
    severity_normaliser: float = 12.0,
) -> NodeExplanation:
    """Explain one node: its exposure paths and what the model keyed on.

    ``model`` is optional. Without it the structural explanation is still produced —
    which is deliberate, because the graph-walk half of the answer does not depend
    on a trained model and should remain available when one has not been fitted (a
    fresh tenant, or a dashboard rendering before the nightly training run).
    """
    node = graph.node(node_id)
    sources = _risk_sources(graph, cve_threshold)

    raw_paths = _enumerate_paths(graph, node_id, sources, max_hops)
    scored: list[ExposurePath] = []
    for path in raw_paths:
        source_node: Node = graph.node(path[0])
        severity = min(1.0, source_node.cve_exposure_count / severity_normaliser)
        hops = len(path) - 1
        scored.append(
            ExposurePath(
                nodes=path,
                source_cves=source_node.cve_exposure_count,
                source_days_stale=source_node.days_since_last_update,
                contribution=severity * (decay ** hops),
            )
        )
    # Strongest contribution first, then shortest, then by id — a total order, so
    # two runs present the same paths in the same sequence. A dashboard whose rows
    # reshuffle between refreshes teaches analysts to distrust it.
    scored.sort(key=lambda p: (-p.contribution, p.hops, p.nodes))

    own_share = 0.5
    neighbourhood_share = 0.5
    score = risk_score if risk_score is not None else float("nan")
    if model is not None:
        shares = _ablation_shares(model, graph)
        index = graph.index_of(node_id)
        neighbourhood_share = float(shares[index])
        own_share = 1.0 - neighbourhood_share
        if risk_score is None:
            score = float(model.risk_scores(graph)[index])

    return NodeExplanation(
        node_id=node_id,
        kind=node.kind,
        risk_score=score,
        paths=tuple(scored[:max_paths]),
        own_feature_share=own_share,
        neighbourhood_share=neighbourhood_share,
        is_intrinsically_risky=node_id in sources,
    )


def top_risk_explanations(
    graph: SupplyChainGraph,
    scores: np.ndarray,
    *,
    model: SupplyChainGNN | None = None,
    k: int = 10,
    **kwargs: Any,
) -> list[NodeExplanation]:
    """Explain the ``k`` highest-scoring nodes — the dashboard's review queue.

    Ablation shares are computed once for the whole graph and passed through, rather
    than recomputed per node: the ablation is a full forward pass, and doing it ten
    times for ten rows of one table would triple the cost of rendering the view for
    no additional information.
    """
    values = np.asarray(scores, dtype=DTYPE).ravel()
    if values.size != graph.n_nodes:
        raise GraphError(
            f"scores ({values.size}) do not match node count ({graph.n_nodes})"
        )
    if k < 1:
        raise GraphError("k must be >= 1")

    shares = (
        _ablation_shares(model, graph) if model is not None else np.full(values.size, 0.5)
    )
    ranking = np.argsort(-values, kind="stable")[: min(k, values.size)]

    explanations: list[NodeExplanation] = []
    for index in ranking:
        explanation = explain_node(
            graph,
            graph.nodes[index].node_id,
            model=None,  # shares supplied below; avoid a second ablation pass
            risk_score=float(values[index]),
            **kwargs,
        )
        neighbourhood = float(shares[index])
        explanations.append(
            NodeExplanation(
                node_id=explanation.node_id,
                kind=explanation.kind,
                risk_score=explanation.risk_score,
                paths=explanation.paths,
                own_feature_share=1.0 - neighbourhood,
                neighbourhood_share=neighbourhood,
                is_intrinsically_risky=explanation.is_intrinsically_risky,
            )
        )
    return explanations
