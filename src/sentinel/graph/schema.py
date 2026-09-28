"""The supply-chain risk graph (PRD F-06, Section 5.5.3).

    *"The vendor/dependency graph is built with organizations, vendors, and
    open-source packages as nodes and contractual, API, and dependency
    relationships as edges. Node features include CVE exposure count,
    days-since-last-update, SBOM depth, and public breach history."*

This module defines the graph itself. The risk propagation model is in
:mod:`sentinel.graph.gnn` and the path attribution F-06 requires is in
:mod:`sentinel.graph.explain`.

Edge direction is the whole ballgame
------------------------------------
Every edge here points **in the direction risk flows**: from the thing that
carries the risk to the thing that inherits it. If package ``left-pad`` has a
critical CVE, risk flows *from* ``left-pad`` *to* everything that depends on it,
and onward to the vendor shipping that code and the organisation buying from that
vendor. So a dependency edge runs ``dependency -> dependent``, which is the
**reverse** of how an SBOM or a ``package.json`` is written and the reverse of how
most people first draw it.

This is stated three times — here, in :class:`EdgeKind`, and in
:meth:`SupplyChainGraph.aggregation_matrix` — because reversing it is the single
most damaging mistake available in this file, and it is silent. A model trained on
the transposed graph still trains, still converges, still produces a confident
top-10 list. It just answers a different question: "which packages are exposed to
risky *organisations*", which is meaningless. ``test_graph.py`` pins the direction
with an asymmetric three-node fixture, because a symmetric test graph would pass
either way.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import numpy as np

from sentinel.core.errors import SentinelError

__all__ = [
    "FEATURE_NAMES",
    "EdgeKind",
    "GraphError",
    "Node",
    "NodeKind",
    "SupplyChainEdge",
    "SupplyChainGraph",
]

DTYPE: Final = np.float64


class GraphError(SentinelError):
    """The graph is malformed, or a query does not make sense against it."""


class NodeKind(StrEnum):
    """PRD Section 5.5.3: organizations, vendors, and open-source packages."""

    ORGANIZATION = "organization"
    VENDOR = "vendor"
    PACKAGE = "package"


class EdgeKind(StrEnum):
    """Relationship types, all oriented in the direction risk flows.

    ``DEPENDENCY``
        ``package -> (package | vendor)``. The dependency is the source; whatever
        consumes it is the target. Reverse of SBOM notation — see the module
        docstring.
    ``API``
        ``vendor -> (vendor | organization)``. A live integration: the provider is
        the source, the caller is the target.
    ``CONTRACTUAL``
        ``vendor -> organization``. A commercial relationship, which carries risk
        even with no technical integration (a breached payroll provider holds your
        employee data whether or not you call its API).
    """

    DEPENDENCY = "dependency"
    API = "api"
    CONTRACTUAL = "contractual"


#: Node feature order, fixed. PRD Section 5.5.3 names the first four; the fifth is
#: derived (see Node.feature_vector) and is what makes an *unmaintained* dependency
#: distinguishable from a merely old one.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "cve_exposure_count",
    "days_since_last_update",
    "sbom_depth",
    "breach_history",
    "is_unmaintained",
)

#: Days without a release after which a package is treated as unmaintained. Two
#: years is the convention most SCA tools settle on, and the distinction matters:
#: a stable, finished library and an abandoned one look identical on
#: days-since-update alone, but only one of them will ship a patch.
UNMAINTAINED_DAYS: Final = 730.0


@dataclass(frozen=True, slots=True)
class Node:
    """One organisation, vendor or package, with the PRD's four risk features."""

    node_id: str
    kind: NodeKind
    name: str
    cve_exposure_count: int = 0
    days_since_last_update: float = 0.0
    sbom_depth: int = 0
    breach_history: int = 0

    def __post_init__(self) -> None:
        if not self.node_id or self.node_id != self.node_id.strip():
            raise GraphError(f"node_id must be non-empty and unpadded, got {self.node_id!r}")
        if self.cve_exposure_count < 0:
            raise GraphError(f"{self.node_id}: cve_exposure_count cannot be negative")
        if self.days_since_last_update < 0:
            raise GraphError(f"{self.node_id}: days_since_last_update cannot be negative")
        if self.sbom_depth < 0:
            raise GraphError(f"{self.node_id}: sbom_depth cannot be negative")
        if self.breach_history < 0:
            raise GraphError(f"{self.node_id}: breach_history cannot be negative")

    @property
    def is_unmaintained(self) -> bool:
        return self.days_since_last_update >= UNMAINTAINED_DAYS

    def feature_vector(self) -> np.ndarray:
        """Raw features in :data:`FEATURE_NAMES` order.

        Deliberately *raw* — no scaling here. Standardisation belongs to the model
        and must be fitted on training nodes only, exactly as
        :class:`~sentinel.ml.featurestore.AlertVectorizer` does for alerts. A
        node that normalised itself would make the leak invisible.
        """
        return np.array(
            [
                float(self.cve_exposure_count),
                float(self.days_since_last_update),
                float(self.sbom_depth),
                float(self.breach_history),
                1.0 if self.is_unmaintained else 0.0,
            ],
            dtype=DTYPE,
        )


@dataclass(frozen=True, slots=True)
class SupplyChainEdge:
    """A directed edge, oriented so risk flows ``source -> target``."""

    source: str
    target: str
    kind: EdgeKind
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise GraphError(f"self-loop on {self.source!r}: a node cannot expose itself")
        if not 0.0 < self.weight <= 1.0:
            raise GraphError(
                f"edge {self.source}->{self.target} weight must be in (0, 1], got "
                f"{self.weight}"
            )


#: Which ``(source kind, target kind)`` pairs each edge kind permits. Enforced
#: rather than documented: a DEPENDENCY edge from an organisation to a package is
#: the transposed-graph bug announcing itself, and it is much cheaper to catch
#: here than to infer from a strange top-10 list three days later.
_ALLOWED: Final[Mapping[EdgeKind, frozenset[tuple[NodeKind, NodeKind]]]] = {
    EdgeKind.DEPENDENCY: frozenset(
        {
            (NodeKind.PACKAGE, NodeKind.PACKAGE),
            (NodeKind.PACKAGE, NodeKind.VENDOR),
            (NodeKind.PACKAGE, NodeKind.ORGANIZATION),
        }
    ),
    EdgeKind.API: frozenset(
        {
            (NodeKind.VENDOR, NodeKind.VENDOR),
            (NodeKind.VENDOR, NodeKind.ORGANIZATION),
        }
    ),
    EdgeKind.CONTRACTUAL: frozenset({(NodeKind.VENDOR, NodeKind.ORGANIZATION)}),
}


@dataclass
class SupplyChainGraph:
    """Nodes, directed edges, and the adjacency structure the GNN consumes."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[SupplyChainEdge] = field(default_factory=list)

    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {}
        for node in self.nodes:
            self._register(node)
        for edge in self.edges:
            self._validate_edge(edge)

    # --- construction -------------------------------------------------------

    def _register(self, node: Node) -> None:
        if node.node_id in self._index:
            raise GraphError(f"duplicate node_id {node.node_id!r}")
        self._index[node.node_id] = len(self._index)

    def add_node(self, node: Node) -> SupplyChainGraph:
        self._register(node)
        self.nodes.append(node)
        return self

    def add_edge(self, edge: SupplyChainEdge) -> SupplyChainGraph:
        self._validate_edge(edge)
        self.edges.append(edge)
        return self

    def _validate_edge(self, edge: SupplyChainEdge) -> None:
        for endpoint in (edge.source, edge.target):
            if endpoint not in self._index:
                raise GraphError(
                    f"edge {edge.source}->{edge.target} references unknown node "
                    f"{endpoint!r}"
                )
        pair = (self.node(edge.source).kind, self.node(edge.target).kind)
        if pair not in _ALLOWED[edge.kind]:
            raise GraphError(
                f"{edge.kind.value} edge {edge.source}->{edge.target} connects "
                f"{pair[0].value}->{pair[1].value}, which is not a direction risk "
                f"flows. Edges point from the thing carrying risk to the thing "
                f"inheriting it; a dependency edge runs dependency->dependent, the "
                f"reverse of SBOM notation."
            )

    # --- lookups ------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def index_of(self, node_id: str) -> int:
        try:
            return self._index[node_id]
        except KeyError:
            raise GraphError(f"unknown node_id {node_id!r}") from None

    def node(self, node_id: str) -> Node:
        return self.nodes[self.index_of(node_id)]

    def node_ids(self) -> list[str]:
        return [node.node_id for node in self.nodes]

    def nodes_of_kind(self, kind: NodeKind) -> list[Node]:
        return [node for node in self.nodes if node.kind is kind]

    def indices_of_kind(self, kind: NodeKind) -> np.ndarray:
        return np.array(
            [i for i, node in enumerate(self.nodes) if node.kind is kind], dtype=int
        )

    # --- matrices -----------------------------------------------------------

    def feature_matrix(self) -> np.ndarray:
        """``(n_nodes, len(FEATURE_NAMES))`` of raw, unscaled node features."""
        if not self.nodes:
            raise GraphError("cannot build a feature matrix for an empty graph")
        return np.vstack([node.feature_vector() for node in self.nodes])

    def adjacency(self) -> np.ndarray:
        """``A[target, source] = weight``, so row ``v`` holds what flows *into* ``v``.

        Indexed ``[target, source]`` rather than ``[source, target]`` so that
        ``A @ H`` is the aggregation over each node's risk sources — which is what
        every message-passing layer wants, and which makes the transpose a
        deliberate act rather than something that happens by accident in an
        einsum.
        """
        matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=DTYPE)
        for edge in self.edges:
            matrix[self.index_of(edge.target), self.index_of(edge.source)] += edge.weight
        return matrix

    def aggregation_matrix(
        self, *, add_self_loops: bool = False, mode: str = "mean"
    ) -> np.ndarray:
        """Aggregation operator for message passing.

        ``mode="mean"`` row-normalises (the GraphSAGE mean aggregator the PRD
        names); ``mode="sum"`` leaves the weighted adjacency unnormalised.

        There is a tempting theoretical argument for ``sum`` here, and it did not
        survive measurement. The argument: supply-chain exposure **accumulates**, so
        a vendor depending on six separately-abandoned libraries is in worse shape
        than one depending on a single one — and a mean aggregator divides the count
        away, making "one risky dependency out of one" and "six risky out of six"
        produce an identical message. That reasoning is sound as far as it goes, and
        the ground truth in :mod:`sentinel.graph.synthetic` *is* an additive sum.

        Measured over 8 seeds on F-06's top-10 precision:

        ====================  ======  ======
        objective             sum     mean
        ====================  ======  ======
        classification        0.800   0.788
        regression            0.750   0.825
        ====================  ======  ======

        ``sum`` wins narrowly for classification and **loses** for regression, which
        is the objective that ships. The likely reason is that degree varies widely
        in a preferential-attachment graph (in-degree 1 to 12 here), so an
        unnormalised sum feeds the next layer activations whose scale tracks degree
        rather than risk, and the network spends capacity undoing that. The mean
        keeps the input scale stable and the *own-node* term — ``W_self``, which the
        mean does not touch — still carries the magnitude information.

        So ``mean`` is the default, matching the PRD, on evidence rather than on the
        argument above. ``sum`` is kept, tested and comparable, because the argument
        is not wrong in principle and a different graph shape could flip it back.

        ``(A_norm @ H)[v]`` is the weighted mean of the representations of every
        node whose risk flows into ``v``. A node with no incoming edges — a leaf
        dependency, the bottom of the SBOM — gets an all-zero row rather than a
        division by zero, which is correct: it inherits nothing from anyone, and
        its own features still reach the layer through the self-transform.

        ``add_self_loops`` is available but **off by default**, and that is a real
        modelling decision rather than a default that happened. GraphSAGE keeps a
        separate ``W_self`` for a node's own representation, so folding a self-loop
        into the neighbour mean would blend a node's own risk into the signal that
        is supposed to represent *inherited* risk. Keeping them separate is what
        makes the explainer able to say "this organisation's score comes from its
        vendors, not from itself" — which is exactly the sentence F-06's guardrail
        requires.
        """
        if mode not in ("mean", "sum"):
            raise GraphError(f"aggregation mode must be 'mean' or 'sum', got {mode!r}")
        matrix = self.adjacency()
        if add_self_loops:
            matrix = matrix + np.eye(self.n_nodes, dtype=DTYPE)
        if mode == "sum":
            return matrix
        degree = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, degree, out=np.zeros_like(matrix), where=degree > 0)

    # --- traversal ----------------------------------------------------------

    def sources_of(self, node_id: str) -> list[str]:
        """Nodes whose risk flows directly into ``node_id`` (its dependencies)."""
        index = self.index_of(node_id)
        return [
            edge.source for edge in self.edges if self.index_of(edge.target) == index
        ]

    def targets_of(self, node_id: str) -> list[str]:
        """Nodes that inherit risk from ``node_id`` (its dependents)."""
        index = self.index_of(node_id)
        return [
            edge.target for edge in self.edges if self.index_of(edge.source) == index
        ]

    def downstream(self, node_id: str, *, max_hops: int | None = None) -> dict[str, int]:
        """Everything reachable *following* risk flow, mapped to its hop distance.

        Breadth-first, so each node maps to its **shortest** hop count. That
        matters for the ground-truth construction in
        :mod:`sentinel.graph.synthetic`: exposure through one hop and exposure
        through five are not the same risk, and a depth-first walk would report
        whichever path it happened to find first.
        """
        return self._reach(node_id, max_hops, forward=True)

    def upstream(self, node_id: str, *, max_hops: int | None = None) -> dict[str, int]:
        """Everything this node is exposed *to*, mapped to hop distance."""
        return self._reach(node_id, max_hops, forward=False)

    def _reach(self, node_id: str, max_hops: int | None, *, forward: bool) -> dict[str, int]:
        self.index_of(node_id)  # validates
        if max_hops is not None and max_hops < 0:
            raise GraphError("max_hops must be non-negative")
        step = self.targets_of if forward else self.sources_of
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        seen = {node_id}
        while queue:
            current, hops = queue.popleft()
            if max_hops is not None and hops >= max_hops:
                continue
            for neighbour in step(current):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                distances[neighbour] = hops + 1
                queue.append((neighbour, hops + 1))
        return distances

    # --- diagnostics --------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """Shape statistics, for asserting the generator matches real SBOMs."""
        in_degree = self.adjacency().astype(bool).sum(axis=1)
        out_degree = self.adjacency().astype(bool).sum(axis=0)
        return {
            "n_nodes": self.n_nodes,
            "n_edges": len(self.edges),
            "nodes_by_kind": {
                kind.value: len(self.nodes_of_kind(kind)) for kind in NodeKind
            },
            "edges_by_kind": {
                kind.value: sum(1 for e in self.edges if e.kind is kind) for kind in EdgeKind
            },
            "max_in_degree": int(in_degree.max()) if self.n_nodes else 0,
            "max_out_degree": int(out_degree.max()) if self.n_nodes else 0,
            "mean_in_degree": float(in_degree.mean()) if self.n_nodes else 0.0,
            "max_sbom_depth": max((n.sbom_depth for n in self.nodes), default=0),
            "n_isolated": int(((in_degree == 0) & (out_degree == 0)).sum()),
        }

    def assert_acyclic(self) -> None:
        """Kahn's algorithm. A cycle would make risk propagation non-terminating.

        Real dependency graphs are overwhelmingly acyclic (npm and PyPI both permit
        cycles in principle and they are rare and pathological in practice). The
        GNN itself is finite-depth and would tolerate a cycle, but the ground-truth
        construction and the path explainer both assume reachability terminates, so
        the assumption is checked rather than hoped for.
        """
        remaining = self.adjacency().astype(bool).sum(axis=1)
        ready = deque(int(i) for i in np.flatnonzero(remaining == 0))
        visited = 0
        adjacency = self.adjacency().astype(bool)
        while ready:
            current = ready.popleft()
            visited += 1
            for target in np.flatnonzero(adjacency[:, current]):
                remaining[target] -= 1
                if remaining[target] == 0:
                    ready.append(int(target))
        if visited != self.n_nodes:
            stuck = [self.nodes[i].node_id for i in np.flatnonzero(remaining > 0)][:10]
            raise GraphError(
                f"graph has a cycle: {self.n_nodes - visited} node(s) never reach "
                f"in-degree zero (e.g. {stuck})"
            )

    @classmethod
    def from_parts(
        cls, nodes: Iterable[Node], edges: Iterable[SupplyChainEdge]
    ) -> SupplyChainGraph:
        graph = cls()
        for node in nodes:
            graph.add_node(node)
        for edge in edges:
            graph.add_edge(edge)
        return graph

    def subgraph(self, node_ids: Sequence[str]) -> SupplyChainGraph:
        """Induced subgraph, for rendering one organisation's neighbourhood."""
        keep = set(node_ids)
        missing = keep - set(self._index)
        if missing:
            raise GraphError(f"unknown node ids in subgraph request: {sorted(missing)}")
        return SupplyChainGraph.from_parts(
            [node for node in self.nodes if node.node_id in keep],
            [
                edge
                for edge in self.edges
                if edge.source in keep and edge.target in keep
            ],
        )
