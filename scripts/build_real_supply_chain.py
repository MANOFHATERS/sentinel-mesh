"""Build the real supply-chain snapshot the REAL DATA workspace shows.

    python scripts/build_real_supply_chain.py

Reads six real public projects' dependency lockfiles from GitHub, asks deps.dev (Google Open Source
Insights) for each pinned package's release date and declared dependencies, and asks OSV.dev for
every known vulnerability affecting each pinned version. No account or key is needed. Every response
is cached in ``data/real/supply-chain-cache.json``, so the first run takes several minutes and a
re-run is offline and instant. The result is ``data/real/supply-chain.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sentinel.real.supplychain import (
    CACHE_PATH,
    SNAPSHOT_PATH,
    Fetcher,
    build_snapshot,
    graph_from_snapshot,
)


def main() -> int:
    started = time.time()
    fetcher = Fetcher(CACHE_PATH, pause=0.02)
    snapshot = build_snapshot(fetcher, progress=lambda m: print(f"  {m}", flush=True))
    Path(SNAPSHOT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SNAPSHOT_PATH).write_text(json.dumps(snapshot), encoding="utf-8")
    graph, facts = graph_from_snapshot(snapshot)
    risky = sum(1 for n in graph.nodes if n.cve_exposure_count >= 4)
    print(
        f"wrote {SNAPSHOT_PATH}: {graph.n_nodes} nodes, {len(graph.edges)} edges, "
        f"{risky} packages with 4+ known vulnerabilities, "
        f"{facts['cycle_edges_cut']} cyclic edge(s) cut, {time.time() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
