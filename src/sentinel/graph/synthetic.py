"""A synthetic vendor/dependency graph with hand-built ground truth (PRD Section 7.1).

    *"~500 nodes generated to match real-world SBOM depth/fan-out distributions;
    not real company data."*

Acceptance criterion F-06 is *"top-10 flagged nodes match >= 80% of the synthetic
ground-truth high-risk set"*, so the ground truth is the benchmark. That makes how
it is constructed the most important decision in this module, and there is one way
to get it wrong that would invalidate the entire GNN.

The trap: a ground truth a GNN is not needed for
------------------------------------------------
The easy construction is "a node is high-risk if it has many CVEs and is stale".
That label is a function of the node's **own features**, so logistic regression on
four columns scores a perfect top-10, the GNN also scores a perfect top-10, F-06
passes, and the graph contributed nothing. The PRD's actual claim would be
untested:

    *"a company's exposure through a fourth-order dependency — the kind that a
    static vendor questionnaire never reaches — surfaces as a scored, explainable
    path rather than an invisible risk."*

So risk here has two distinct sources, and they are kept structurally separate:

**Intrinsic risk** — a package with real CVEs that is unmaintained. Visible in the
node's own features. A features-only model finds these.

**Inherited risk** — an organisation or vendor is high-risk *because* it is
transitively exposed to an intrinsically risky package, within
:data:`EXPOSURE_HOPS` hops. Here is the part that matters: **an organisation's own
feature vector carries no trace of this.** Its CVE count is its own, its update age
is its own. The only way to know it is exposed is to walk the graph.

That makes the two populations separable in evaluation, and
``test_graph_gnn.py::TestGnnVersusFeaturesOnly`` holds the GNN to beating a
features-only baseline **on the inherited-risk nodes specifically**. If the GNN
ever stops beating it there, the graph has become decoration and the test says so.

Shape realism
-------------
Fan-out follows preferential attachment, because real dependency use is heavy-
tailed: a handful of packages (``lodash``, ``requests``) are depended on by
thousands while the median package is depended on by one or two. Uniform random
wiring would produce a graph where no single package matters much, which quietly
removes the phenomenon the product exists to detect. Depth runs to
:data:`MAX_SBOM_DEPTH`, matching the 4-8 transitive levels typical of a real
``node_modules`` or resolved Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from sentinel.graph.schema import (
    UNMAINTAINED_DAYS,
    EdgeKind,
    GraphError,
    Node,
    NodeKind,
    SupplyChainEdge,
    SupplyChainGraph,
)

__all__ = [
    "DECAY",
    "EXPOSURE_HOPS",
    "EXPOSURE_THRESHOLD",
    "INTRINSIC_BASE",
    "INTRINSIC_CVE_THRESHOLD",
    "MAX_SBOM_DEPTH",
    "SEVERITY_NORMALISER",
    "GroundTruth",
    "SyntheticGraphGenerator",
    "build_demo_graph",
    "label_ground_truth",
]

#: Transitive depth of the package tree. PRD Section 5.5.3 speaks of fourth-order
#: dependencies; 6 gives headroom past that so "4 hops" is an interior case rather
#: than the boundary, which would make the 2-layer-GNN experiment untestable.
MAX_SBOM_DEPTH: Final = 6

#: How far intrinsic risk is considered to propagate when labelling ground truth.
#: 4 is chosen to match the PRD's "fourth-order dependency" language exactly, which
#: also makes it strictly deeper than a 2-layer GNN's receptive field — see
#: :mod:`sentinel.graph.gnn` for what that measurement showed.
EXPOSURE_HOPS: Final = 4

#: A package is intrinsically risky when it has at least this many CVEs *and* is
#: unmaintained. The conjunction is the point: CVEs in a maintained package are
#: normal and get patched, and an unmaintained package with no known CVEs is not
#: yet a problem. It is the combination that a questionnaire never catches.
INTRINSIC_CVE_THRESHOLD: Final = 4

#: Per-hop attenuation of inherited risk.
#:
#: Calibrated against the PRD's own claim rather than picked for looking
#: reasonable. Section 5.5.3 promises that *"a company's exposure through a
#: fourth-order dependency ... surfaces as a scored, explainable path rather than
#: an invisible risk"*. For that to be a property anything can be tested against,
#: a fourth-order exposure has to be able to cross the label threshold.
#:
#: At the first value tried, 0.45, it could not: ``0.45**4 = 0.041`` against a
#: threshold of 0.30, so **no** 4-hop exposure was labelled — the measured path
#: lengths came out ``{1: 23, 2: 5, 3: 1}`` and the headline scenario was simply
#: absent from the benchmark. At 0.7, ``0.7**4 = 0.24``: one distant risky
#: dependency is not enough on its own, two are. That is the intended semantics —
#: deep exposure counts, but it has to accumulate — and it keeps the fourth-order
#: case present and rankable.
DECAY: Final = 0.7

#: Risk a package carries purely for being intrinsically compromised, before any
#: inherited exposure is added. Set to :data:`EXPOSURE_THRESHOLD` so that an
#: intrinsically risky package is *always* above the label threshold on its own
#: account — which makes ``intrinsic`` a subset of the high-risk set by
#: construction, and keeps one threshold on one quantity as the whole definition.
INTRINSIC_BASE: Final = 1.0

#: CVE count at which a package's severity contribution saturates at 1.0. Twelve
#: known CVEs in an abandoned package is already a decisive signal; distinguishing
#: 12 from 30 adds nothing and would let one extreme node dominate every score.
SEVERITY_NORMALISER: Final = 12.0

#: Accumulated, decayed exposure at which a node is labelled inherited-risk.
#:
#: Calibrated by sweep (see ``docs/BUILD_PLAN.md``) to put ~11% of the graph in the
#: high-risk set — 54/500 at the default shape, of which 29 are inherited — so
#: top-k precision measures ranking rather than class prevalence.
#:
#: **Organisations saturate, and no single threshold fixes it.** At every threshold
#: that leaves vendors and packages discriminative, 18-20 of the 20 organisations
#: are high-risk, because each buys from 4-12 vendors carrying 9-13 direct
#: dependencies each — so an organisation sits 2-4 hops from most of the package
#: graph and accumulates far more exposure than any package can.
#:
#: This constant is therefore pulled in two directions at once, and the trade is
#: worth stating plainly because it constrains what F-06 can measure:
#:
#: * **Raising it** spreads the organisations out, but pushes a fourth-order
#:   contribution (``DECAY**4 = 0.24``) below the point where it can influence any
#:   label — which deletes the PRD's headline scenario from the benchmark.
#: * **Lowering it** keeps deep exposure meaningful and flags every organisation.
#:
#: 1.0 sits where deep exposure is still worth 24% of the threshold (material,
#: accumulating) while vendors run ~15% positive and packages ~6%. Organisations
#: stay saturated by design, which is the PRD's own thesis as a measurement:
#: essentially every mid-market company *has* third-party exposure, so the useful
#: question is never "which client is exposed" but "which is exposed *most*, and
#: through what". The binary label does its discriminative work on vendors and
#: packages; organisations are evaluated by **ranking** against
#: :attr:`GroundTruth.risk_scores`, and the F-06 test reports precision per node
#: kind so this cannot hide inside an aggregate.
EXPOSURE_THRESHOLD: Final = 1.0


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Which nodes are high-risk, and why — the F-06 benchmark.

    ``intrinsic`` and ``inherited`` are disjoint by construction, and the split is
    what lets evaluation ask the only question that matters about a GNN: does the
    graph structure buy anything a feature model could not get for free?
    """

    intrinsic: frozenset[str]
    inherited: frozenset[str]
    exposure_paths: dict[str, tuple[str, ...]]
    #: Total risk per node: its own intrinsic term plus distance-decayed inherited
    #: exposure, including nodes below the label threshold. This is the *graded*
    #: target — the binary label is exactly ``risk_scores >= threshold`` — so a
    #: model can be trained and scored on ordering rather than only on a boolean.
    risk_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = self.intrinsic & self.inherited
        if overlap:
            raise GraphError(
                f"a node cannot be both intrinsically and inherently risky: "
                f"{sorted(overlap)[:5]}"
            )

    @property
    def high_risk(self) -> frozenset[str]:
        return self.intrinsic | self.inherited

    def labels(self, node_ids: list[str]) -> np.ndarray:
        """Binary label vector aligned to ``node_ids``."""
        high_risk = self.high_risk
        return np.array([1.0 if nid in high_risk else 0.0 for nid in node_ids])

    def risk_vector(self, node_ids: list[str]) -> np.ndarray:
        """Graded risk aligned to ``node_ids``; 0.0 for nodes with no exposure."""
        return np.array([self.risk_scores.get(nid, 0.0) for nid in node_ids])

    def summary(self) -> str:
        return (
            f"{len(self.high_risk)} high-risk nodes: {len(self.intrinsic)} intrinsic "
            f"(visible in node features), {len(self.inherited)} inherited "
            f"(visible only through the graph)"
        )


class SyntheticGraphGenerator:
    """Generates a ~500-node graph plus its ground truth, deterministically."""

    def __init__(
        self,
        *,
        # An MSSP-shaped book of business rather than a single tenant. PRD Section
        # 3.3 puts the channel ICP at "MSSPs serving 15-150 mid-market clients",
        # and Section 2.4 cites an average of 286 vendors per organisation, so a
        # 20-client / 60-vendor graph is the small end of the real thing.
        #
        # It is also the only shape in which organisations are *measurable*: with
        # the first draft's 4 organisations, all 4 landed in the high-risk set at
        # every threshold that kept the rest of the graph discriminative, so the
        # node kind the product is sold on contributed no signal to F-06 at all.
        n_organizations: int = 20,
        n_vendors: int = 60,
        n_packages: int = 420,
        seed: int = 20260928,
        intrinsic_risk_rate: float = 0.05,
        max_depth: int = MAX_SBOM_DEPTH,
        exposure_hops: int = EXPOSURE_HOPS,
        exposure_threshold: float = EXPOSURE_THRESHOLD,
    ) -> None:
        if n_organizations < 1 or n_vendors < 1 or n_packages < 1:
            raise GraphError("need at least one of each node kind")
        if not 0.0 < intrinsic_risk_rate < 0.5:
            raise GraphError("intrinsic_risk_rate must be in (0, 0.5)")
        if max_depth < 2:
            raise GraphError("max_depth must be >= 2 for a transitive graph")
        self.n_organizations = n_organizations
        self.n_vendors = n_vendors
        self.n_packages = n_packages
        self.seed = seed
        self.intrinsic_risk_rate = intrinsic_risk_rate
        self.max_depth = max_depth
        self.exposure_hops = exposure_hops
        if exposure_threshold <= 0.0:
            raise GraphError("exposure_threshold must be positive")
        self.exposure_threshold = exposure_threshold

    # --- generation ---------------------------------------------------------

    def generate(self) -> tuple[SupplyChainGraph, GroundTruth]:
        rng = np.random.default_rng(self.seed)
        graph = SupplyChainGraph()

        # Which packages are intrinsically risky is decided *before* features are
        # drawn, so the features can be drawn consistently with the label rather
        # than the label inferred from a threshold that may or may not have been
        # crossed. Deciding it the other way round makes the risky population's
        # size depend on sampling noise.
        n_risky = max(1, round(self.n_packages * self.intrinsic_risk_rate))
        risky_positions = set(
            rng.choice(self.n_packages, size=n_risky, replace=False).tolist()
        )

        depths = self._assign_depths(rng)
        packages = [
            self._make_package(index, depths[index], index in risky_positions, rng)
            for index in range(self.n_packages)
        ]
        for package in packages:
            graph.add_node(package)

        vendors = [self._make_vendor(index, rng) for index in range(self.n_vendors)]
        for vendor in vendors:
            graph.add_node(vendor)

        organizations = [
            self._make_organization(index, rng) for index in range(self.n_organizations)
        ]
        for organization in organizations:
            graph.add_node(organization)

        self._wire_packages(graph, packages, depths, rng)
        self._wire_vendors(graph, packages, vendors, depths, rng)
        self._wire_organizations(graph, vendors, organizations, rng)

        graph.assert_acyclic()
        return graph, self._label(graph)

    # --- nodes --------------------------------------------------------------

    def _assign_depths(self, rng: np.random.Generator) -> list[int]:
        """Depth per package: **many shallow, few deep**.

        This is the opposite of the intuition for a single application, where the
        transitive set widens with depth, and it is the correct shape for a graph
        spanning 32 vendors. Each vendor ships its own application with its own
        direct dependencies, so depth 1 is populous and vendor-specific; the deep
        levels hold the small set of universally-reused libraries (the ``lodash``
        and ``requests`` tier) that many trees converge onto.

        Measured why it matters: with the weights running the other way, only ~18
        packages sat at depth 1, all 32 vendors drew their dependencies from that
        one pool, and a single tainted package put **every vendor and every
        organisation** in the high-risk set. A benchmark where 32/32 vendors are
        positive cannot measure ranking quality at all.
        """
        levels = np.arange(1, self.max_depth + 1)
        weights = 1.0 / (1.5 ** levels)
        weights = weights / weights.sum()
        counts = rng.multinomial(self.n_packages, weights)
        # Every level must be populated or the tree has a hole and deep exposure
        # paths cannot exist at all.
        while np.any(counts == 0):
            empty = int(np.argmin(counts))
            fullest = int(np.argmax(counts))
            if counts[fullest] <= 1:
                raise GraphError(
                    f"{self.n_packages} packages cannot populate {self.max_depth} depth "
                    "levels; increase n_packages or reduce max_depth"
                )
            counts[empty] += 1
            counts[fullest] -= 1
        return [
            int(level) for level, count in zip(levels, counts, strict=True) for _ in range(count)
        ]

    def _make_package(
        self, index: int, depth: int, risky: bool, rng: np.random.Generator
    ) -> Node:
        if risky:
            # Risky: real CVEs AND abandoned. Both conditions, so the conjunction
            # is what identifies it (see INTRINSIC_CVE_THRESHOLD).
            cves = int(rng.integers(INTRINSIC_CVE_THRESHOLD, 18))
            days = float(rng.uniform(UNMAINTAINED_DAYS, 2600.0))
        else:
            # Benign packages span the full range on each axis *individually*,
            # including some with many CVEs (actively maintained, patched fast) and
            # some long-abandoned but clean. That overlap is essential: if risky
            # packages were the only ones with high CVE counts, a single threshold
            # on one column would solve the problem and no model would be needed.
            cves = int(rng.poisson(1.2))
            if rng.random() < 0.12:
                cves = int(rng.integers(INTRINSIC_CVE_THRESHOLD, 14))  # maintained but CVE-heavy
                days = float(rng.uniform(0.0, 120.0))
            elif rng.random() < 0.18:
                days = float(rng.uniform(UNMAINTAINED_DAYS, 2200.0))  # abandoned but clean
            else:
                days = float(rng.uniform(0.0, 700.0))
        return Node(
            node_id=f"pkg-{index:04d}",
            kind=NodeKind.PACKAGE,
            name=f"package-{index:04d}",
            cve_exposure_count=cves,
            days_since_last_update=days,
            sbom_depth=depth,
            breach_history=0,  # packages do not have breach history; vendors do
        )

    def _make_vendor(self, index: int, rng: np.random.Generator) -> Node:
        return Node(
            node_id=f"vendor-{index:03d}",
            kind=NodeKind.VENDOR,
            name=f"vendor-{index:03d}",
            cve_exposure_count=int(rng.poisson(1.0)),
            days_since_last_update=float(rng.uniform(0.0, 400.0)),
            sbom_depth=0,
            breach_history=int(rng.random() < 0.15),
        )

    def _make_organization(self, index: int, rng: np.random.Generator) -> Node:
        return Node(
            node_id=f"org-{index:02d}",
            kind=NodeKind.ORGANIZATION,
            name=f"organization-{index:02d}",
            cve_exposure_count=int(rng.poisson(0.5)),
            days_since_last_update=float(rng.uniform(0.0, 90.0)),
            sbom_depth=0,
            breach_history=int(rng.random() < 0.10),
        )

    # --- edges --------------------------------------------------------------

    def _wire_packages(
        self,
        graph: SupplyChainGraph,
        packages: list[Node],
        depths: list[int],
        rng: np.random.Generator,
    ) -> None:
        """Each package at depth ``d`` feeds one or more at depth ``d-1``.

        Targets are chosen by **preferential attachment** — probability
        proportional to a shallower package's existing in-degree plus one — which
        reproduces the heavy-tailed reuse real ecosystems show. Edges run
        ``deeper -> shallower``, i.e. dependency to dependent, so risk flows the
        way the schema defines.
        """
        by_depth: dict[int, list[int]] = {}
        for position, depth in enumerate(depths):
            by_depth.setdefault(depth, []).append(position)

        in_degree = np.ones(len(packages), dtype=DTYPE_INT)
        for depth in range(self.max_depth, 1, -1):
            consumers = by_depth.get(depth - 1, [])
            if not consumers:
                continue
            consumer_array = np.array(consumers)
            for position in by_depth.get(depth, []):
                # 1-3 dependents: most packages are used once or twice, a few widely.
                n_targets = min(len(consumers), int(rng.integers(1, 4)))
                weights = in_degree[consumer_array].astype(DTYPE)
                probabilities = weights / weights.sum()
                chosen = rng.choice(
                    consumer_array, size=n_targets, replace=False, p=probabilities
                )
                for target in np.atleast_1d(chosen):
                    graph.add_edge(
                        SupplyChainEdge(
                            source=packages[position].node_id,
                            target=packages[int(target)].node_id,
                            kind=EdgeKind.DEPENDENCY,
                        )
                    )
                    in_degree[int(target)] += 1

    def _wire_vendors(
        self,
        graph: SupplyChainGraph,
        packages: list[Node],
        vendors: list[Node],
        depths: list[int],
        rng: np.random.Generator,
    ) -> None:
        """Depth-1 packages are the ones vendors actually ship.

        Every depth-1 package is assigned to at least one vendor before any vendor
        gets extra picks. Without that guarantee the generator produced **41
        isolated nodes** — packages with no edge in either direction — which is not
        a thing that exists: a package is in the graph because something depends on
        it. Isolated nodes also dilute the F-06 benchmark, since they can never be
        inherited-risk and can never be ranked for a reason.
        """
        direct = [i for i, depth in enumerate(depths) if depth == 1]
        if not direct:
            raise GraphError("no depth-1 packages; vendors would have no dependencies")

        def link(package_position: int, vendor: Node) -> None:
            graph.add_edge(
                SupplyChainEdge(
                    source=packages[package_position].node_id,
                    target=vendor.node_id,
                    kind=EdgeKind.DEPENDENCY,
                )
            )

        # Pass 1: cover every depth-1 package, round-robin over a shuffled order so
        # the assignment is even and seed-stable.
        assigned: dict[int, set[int]] = {i: set() for i in range(len(vendors))}
        order = rng.permutation(direct)
        for position, package_position in enumerate(order):
            vendor_index = position % len(vendors)
            link(int(package_position), vendors[vendor_index])
            assigned[vendor_index].add(int(package_position))

        # Pass 2: extra shared dependencies, so vendors overlap the way real ones do
        # (everybody ends up depending on the same handful of utility libraries).
        for vendor_index, vendor in enumerate(vendors):
            n_extra = int(rng.integers(1, 5))
            candidates = [p for p in direct if p not in assigned[vendor_index]]
            if not candidates:
                continue
            for package_position in rng.choice(
                candidates, size=min(n_extra, len(candidates)), replace=False
            ):
                link(int(package_position), vendor)
                assigned[vendor_index].add(int(package_position))

    def _wire_organizations(
        self,
        graph: SupplyChainGraph,
        vendors: list[Node],
        organizations: list[Node],
        rng: np.random.Generator,
    ) -> None:
        """Vendors connect to organisations contractually, some also via live API.

        Both kinds are generated because they carry different risk: an API
        integration is a technical path, a contract is a data-custody path, and
        PRD Section 5.5.3 names both.
        """
        for organization in organizations:
            n_vendors = min(len(vendors), int(rng.integers(4, 13)))
            for position in rng.choice(len(vendors), size=n_vendors, replace=False):
                graph.add_edge(
                    SupplyChainEdge(
                        source=vendors[int(position)].node_id,
                        target=organization.node_id,
                        kind=EdgeKind.CONTRACTUAL,
                    )
                )
                if rng.random() < 0.45:
                    graph.add_edge(
                        SupplyChainEdge(
                            source=vendors[int(position)].node_id,
                            target=organization.node_id,
                            kind=EdgeKind.API,
                            weight=0.8,
                        )
                    )

    # --- ground truth -------------------------------------------------------

    def _label(self, graph: SupplyChainGraph) -> GroundTruth:
        return label_ground_truth(
            graph, exposure_hops=self.exposure_hops, exposure_threshold=self.exposure_threshold
        )


def label_ground_truth(
    graph: SupplyChainGraph,
    *,
    exposure_hops: int = EXPOSURE_HOPS,
    exposure_threshold: float = EXPOSURE_THRESHOLD,
) -> GroundTruth:
    """Intrinsic risk, plus inherited risk as **decayed accumulated exposure**.
    
        The first version of this used binary reachability — exposed to any risky
        package within 4 hops means high-risk — and it produced a useless
        benchmark: 32/32 vendors and 4/4 organisations positive, nothing to rank.
    
        Distance-decayed accumulation fixes that and is the more defensible model
        anyway. Two things a binary rule cannot express, and both are how
        practitioners actually reason:
    
        *   **Distance attenuates risk.** A CVE in a direct dependency is a
            different problem from the same CVE five levels down behind two
            abstraction layers. ``DECAY ** hops`` says so quantitatively.
        *   **Exposures accumulate.** A vendor pulling in six separately-abandoned
            libraries is in worse shape than one pulling in a single one, and a
            reachability predicate scores them identically.
    
        The resulting label is a threshold on a *graded* quantity, which is also
        what makes it a fair target for a model that outputs a score: the ordering
        it has to learn exists in the ground truth rather than being an artefact of
        where a boolean happened to cut.
        """
    # Public so the same rule labels a graph built from real data: intrinsic risk is a
    # package with enough CVEs that is also stale; inherited risk is distance-decayed
    # exposure to those. The rule is a definition, so the labels are as real as the
    # features and the structure they are computed from.

    intrinsic = {
        node.node_id
        for node in graph.nodes
        if node.kind is NodeKind.PACKAGE
        and node.cve_exposure_count >= INTRINSIC_CVE_THRESHOLD
        and node.is_unmaintained
    }
    if not intrinsic:
        raise GraphError("no intrinsically risky packages were generated")

    def severity_of(node_id: str) -> float:
        return min(
            1.0, graph.node(node_id).cve_exposure_count / SEVERITY_NORMALISER
        )

    # One continuous risk quantity for every node, not two incompatible ones.
    #
    # The first version defined high-risk as the *union* of "is intrinsically
    # risky" and "accumulated exposure >= threshold", and kept only the second
    # as a score. That was incoherent, and it showed up as a measurement: an
    # intrinsically risky package was labelled high-risk while carrying an
    # exposure score of 0.0, because the propagation loop skipped intrinsic
    # nodes. A model regressing on the score therefore ranked the genuinely
    # compromised packages *last* and still scored a respectable rank
    # correlation (0.61) while its top-10 precision collapsed to 0.60.
    #
    # Intrinsic risk is now the node's own term in the same sum. INTRINSIC_BASE
    # puts any intrinsically risky package above the threshold on its own
    # account, so ``intrinsic`` is a subset of ``high_risk`` by construction
    # rather than by a union, and one threshold on one quantity defines the label.
    risk: dict[str, float] = {
        node_id: INTRINSIC_BASE + severity_of(node_id) for node_id in intrinsic
    }
    closest: dict[str, tuple[str, int]] = {}
    for source in sorted(intrinsic):
        severity = severity_of(source)
        for node_id, hops in graph.downstream(
            source, max_hops=exposure_hops
        ).items():
            risk[node_id] = risk.get(node_id, 0.0) + severity * (DECAY ** hops)
            # Keep the nearest, most severe source for the explainer to be
            # checked against. Ties broken by severity, then id, for determinism.
            previous = closest.get(node_id)
            if previous is None or (hops, -severity, source) < (
                previous[1],
                -severity_of(previous[0]),
                previous[0],
            ):
                closest[node_id] = (source, hops)

    inherited_paths: dict[str, tuple[str, ...]] = {}
    for node_id, score in risk.items():
        if node_id in intrinsic or score < exposure_threshold:
            continue
        source, hops = closest[node_id]
        path = _shortest_path(graph, source, node_id, hops)
        if path is not None:
            inherited_paths[node_id] = path

    return GroundTruth(
        intrinsic=frozenset(intrinsic),
        inherited=frozenset(inherited_paths),
        exposure_paths=inherited_paths,
        risk_scores=dict(risk),
    )


def _shortest_path(
graph: SupplyChainGraph, source: str, target: str, hops: int
) -> tuple[str, ...] | None:
    """Reconstruct one shortest ``source -> target`` path of length ``hops``."""
    frontier: list[tuple[str, ...]] = [(source,)]
    for _ in range(hops):
        next_frontier: list[tuple[str, ...]] = []
        for path in frontier:
            for neighbour in graph.targets_of(path[-1]):
                if neighbour in path:
                    continue
                extended = (*path, neighbour)
                if neighbour == target:
                    return extended
                next_frontier.append(extended)
        frontier = next_frontier
        if not frontier:
            return None
    return None


DTYPE = np.float64
DTYPE_INT = np.int64


def build_demo_graph(
    *, seed: int = 20260928, n_packages: int = 420
) -> tuple[SupplyChainGraph, GroundTruth]:
    """The ~500-node demo graph from PRD Section 7.1."""
    return SyntheticGraphGenerator(seed=seed, n_packages=n_packages).generate()
