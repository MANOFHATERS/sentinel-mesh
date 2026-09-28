"""Layer 3 — the supply-chain risk graph (PRD F-06, Sections 5.5.3 and 7.1).

The PRD's strategic differentiator. Section 2.4 is the argument for it: third-party
involvement in confirmed breaches went 15% -> 30% -> 48% across three DBIR
editions, and none of the five agentic-SOC competitors lead with it.

    ``schema``     nodes, directed edges, adjacency (direction = risk flow)
    ``synthetic``  ~500-node graph with intrinsic/inherited ground truth
    ``gnn``        GraphSAGE risk propagation
    ``explain``    the specific path that drove a score (F-06's guardrail)
"""

from __future__ import annotations

from sentinel.graph.schema import (
    EdgeKind,
    GraphError,
    Node,
    NodeKind,
    SupplyChainEdge,
    SupplyChainGraph,
)
from sentinel.graph.synthetic import GroundTruth, SyntheticGraphGenerator, build_demo_graph

__all__ = [
    "EdgeKind",
    "GraphError",
    "GroundTruth",
    "Node",
    "NodeKind",
    "SupplyChainEdge",
    "SupplyChainGraph",
    "SyntheticGraphGenerator",
    "build_demo_graph",
]
