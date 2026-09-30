"""A real software supply chain, from real lockfiles and real vulnerability records.

The synthetic supply-chain graph is generated. This one is built from what actually exists:

*   **Organisations** are six real public Python projects, and their **packages** are the exact
    versions pinned in each project's own dependency lockfile on GitHub.
*   **Edges** come from each pinned package's declared requirements (deps.dev, no key needed):
    an edge runs *dependency -> dependent*, the direction risk flows, restricted to packages that
    are pinned in the same project. A pinned package that nothing else in that project requires is
    a direct dependency of the project.
*   **Features** are real: the number of distinct known vulnerabilities affecting the pinned
    version (OSV.dev, deduplicated across GHSA / CVE / PYSEC aliases), the age of that version
    (deps.dev publish date), the depth below the project, and whether OSV lists the version as
    malware (``MAL-``).
*   **Ground truth** is the same rule the synthetic benchmark uses
    (:func:`sentinel.graph.synthetic.label_ground_truth`): a package with at least four known
    vulnerabilities whose version is stale is intrinsically risky, and risk propagates, attenuated
    by distance, to whatever depends on it. The rule is a definition, so the labels are exactly as
    real as the vulnerability data and the dependency structure they are computed from.

One deliberate difference from the synthetic graph: "days since last update" here is the **age of
the version in use**. A project runs a pinned version, and a maintained package pinned three years
ago is exactly the exposure a lockfile creates, so that is what the feature measures.

Everything network-facing is cached to disk, so building is slow once and rebuilding is offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from sentinel.core.errors import SentinelError
from sentinel.dashboard.scenarios import Advisory, AdvisoryKind
from sentinel.graph.schema import (
    EdgeKind,
    Node,
    NodeKind,
    SupplyChainEdge,
    SupplyChainGraph,
)

__all__ = [
    "PROJECTS",
    "Project",
    "RealAdvisory",
    "SupplyChainError",
    "build_snapshot",
    "graph_from_snapshot",
    "load_snapshot",
    "parse_requirements",
    "real_advisories",
]

SNAPSHOT_PATH: Final[Path] = Path("data/real/supply-chain.json")
CACHE_PATH: Final[Path] = Path("data/real/supply-chain-cache.json")
DEPS_DEV: Final[str] = "https://api.deps.dev/v3/systems/pypi/packages"
OSV: Final[str] = "https://api.osv.dev/v1"


class SupplyChainError(SentinelError):
    """The supply-chain snapshot is missing, malformed, or cannot be built."""


@dataclass(frozen=True, slots=True)
class Project:
    repo: str
    lockfile: str
    #: A project whose code the workspace also scans, so one real project appears in both the
    #: code scan and the supply chain.
    primary: bool = False

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/HEAD/{self.lockfile}"


PROJECTS: Final[tuple[Project, ...]] = (
    Project("adeyosemanputra/pygoat", "requirements.txt", primary=True),
    Project("anxolerd/dvpwa", "requirements.txt"),
    Project("ansible/awx", "requirements/requirements.txt"),
    Project("apache/superset", "requirements/base.txt"),
    Project("Netflix/lemur", "requirements.txt"),
    Project("netbox-community/netbox", "requirements.txt"),
)

_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)(?:\[[^\]]*\])?\s*==\s*([A-Za-z0-9_.\-+!]+)")


def normalise(name: str) -> str:
    """PEP 503 name normalisation, so ``Django`` and ``django`` are one package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> dict[str, str]:
    """``{normalised name: version}`` for every ``name==version`` line in a lockfile."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        match = _PIN.match(line.split("#", 1)[0])
        if match:
            pins[normalise(match.group(1))] = match.group(2)
    return pins


# --------------------------------------------------------------------------- #
# Fetching, cached
# --------------------------------------------------------------------------- #


class Fetcher:
    """GET / POST JSON with a disk cache, retries with backoff, and a polite pause."""

    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        *,
        client: httpx.Client | None = None,
        pause: float = 0.0,
    ) -> None:
        self._path = cache_path
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)
        self._pause = pause
        self._cache: dict[str, Any] = {}
        if cache_path.is_file():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self._dirty = 0

    def _key(self, method: str, url: str, payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True) if payload is not None else ""
        return f"{method} {url} {hashlib.sha1(blob.encode()).hexdigest() if blob else ''}".strip()

    def request(self, method: str, url: str, payload: Any = None) -> Any:
        key = self._key(method, url, payload)
        if key in self._cache:
            return self._cache[key]
        for attempt in range(4):
            try:
                response = self._client.request(method, url, json=payload)
            except httpx.HTTPError:
                response = None
            if response is not None and response.status_code == 200:
                data = response.json()
                break
            if response is not None and response.status_code == 404:
                data = None
                break
            time.sleep(1.5 * (attempt + 1))
        else:
            raise SupplyChainError(f"{method} {url} kept failing")
        self._cache[key] = data
        self._dirty += 1
        if self._dirty >= 50:
            self.save()
        if self._pause:
            time.sleep(self._pause)
        return data

    def get(self, url: str) -> Any:
        return self.request("GET", url)

    def post(self, url: str, payload: Any) -> Any:
        return self.request("POST", url, payload)

    def text(self, url: str) -> str:
        response = self._client.get(url)
        if response.status_code != 200:
            raise SupplyChainError(f"{url} answered {response.status_code}")
        return response.text

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self._path)
        self._dirty = 0


# --------------------------------------------------------------------------- #
# The snapshot
# --------------------------------------------------------------------------- #


def _pid(name: str, version: str) -> str:
    return f"pypi:{name}@{version}"


def build_snapshot(
    fetcher: Fetcher,
    projects: tuple[Project, ...] = PROJECTS,
    *,
    progress: Callable[[str], None] = lambda _msg: None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch every lockfile, every pinned package's metadata, and every vulnerability record."""
    generated = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    locks: dict[str, dict[str, str]] = {}
    for project in projects:
        locks[project.repo] = parse_requirements(fetcher.text(project.url))
        progress(f"{project.repo}: {len(locks[project.repo])} pinned packages")
    unique = sorted({(n, v) for pins in locks.values() for n, v in pins.items()})

    packages: dict[str, dict[str, Any]] = {}
    for done, (name, version) in enumerate(unique, start=1):
        base = f"{DEPS_DEV}/{name}/versions/{version}"
        meta = fetcher.get(base) or {}
        reqs = fetcher.get(base + ":requirements") or {}
        deps = [
            normalise(d["projectName"])
            for d in (reqs.get("pypi", {}).get("dependencies", []))
            if "extra ==" not in d.get("environmentMarker", "")
        ]
        packages[_pid(name, version)] = {
            "name": name,
            "version": version,
            "published_at": meta.get("publishedAt"),
            "requires": sorted(set(deps)),
            "advisory_keys": [k["id"] for k in meta.get("advisoryKeys", [])],
        }
        if done % 50 == 0:
            progress(f"package metadata {done}/{len(unique)}")
    fetcher.save()

    # OSV: which vulnerability records affect each pinned version, then the records themselves.
    queries = [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in unique]
    affected: dict[str, list[str]] = {}
    for start in range(0, len(queries), 100):
        chunk = queries[start : start + 100]
        result = fetcher.post(f"{OSV}/querybatch", {"queries": chunk})
        for (name, version), item in zip(
            unique[start : start + 100], result["results"], strict=True
        ):
            affected[_pid(name, version)] = sorted({v["id"] for v in item.get("vulns", [])})
    progress(f"OSV: {sum(len(v) for v in affected.values())} advisory matches")
    vulns: dict[str, dict[str, Any]] = {}
    ids = sorted({i for v in affected.values() for i in v})
    for done, vuln_id in enumerate(ids, start=1):
        record = fetcher.get(f"{OSV}/vulns/{vuln_id}") or {}
        vulns[vuln_id] = _compact(record)
        if done % 100 == 0:
            progress(f"vulnerability records {done}/{len(ids)}")
    fetcher.save()
    for pid, listed in affected.items():
        packages[pid]["vulns"] = listed

    return {
        "generated_at": generated,
        "sources": {
            "lockfiles": "raw.githubusercontent.com (each project's own dependency lockfile)",
            "dependencies_and_release_dates": "api.deps.dev (Google Open Source Insights)",
            "vulnerabilities": "api.osv.dev (OSV.dev, includes GitHub advisories and PyPA)",
        },
        "projects": [
            {
                "repo": p.repo,
                "lockfile": p.lockfile,
                "primary": p.primary,
                "pins": {n: v for n, v in sorted(locks[p.repo].items())},
            }
            for p in projects
        ],
        "packages": packages,
        "vulns": vulns,
    }


def _compact(record: Mapping[str, Any]) -> dict[str, Any]:
    fixed: list[str] = []
    for affected in record.get("affected", []):
        for rng in affected.get("ranges", []):
            fixed += [e["fixed"] for e in rng.get("events", []) if "fixed" in e]
    specific = record.get("database_specific", {}) or {}
    return {
        "aliases": sorted(record.get("aliases", [])),
        "summary": record.get("summary") or (record.get("details") or "")[:200],
        "published": record.get("published"),
        "withdrawn": bool(record.get("withdrawn")),
        "severity": specific.get("severity"),
        "cwes": specific.get("cwe_ids", []),
        "fixed": sorted(set(fixed)),
    }


def load_snapshot(path: Path | str = SNAPSHOT_PATH) -> dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        raise SupplyChainError(f"{file} not found; run scripts/build_real_supply_chain.py")
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SupplyChainError(f"{file} is not readable: {exc}") from exc


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #


def issue_clusters(vulns: Mapping[str, Mapping[str, Any]], ids: list[str]) -> list[list[str]]:
    """Group advisory records that describe the same issue (they share an alias or an id)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for vuln_id in ids:
        for alias in vulns[vuln_id]["aliases"]:
            parent[find(alias)] = find(vuln_id)
        find(vuln_id)
    groups: dict[str, list[str]] = defaultdict(list)
    for vuln_id in ids:
        groups[find(vuln_id)].append(vuln_id)
    return [sorted(g) for g in groups.values()]


def canonical(vulns: Mapping[str, Mapping[str, Any]], cluster: list[str]) -> str:
    """A CVE id if any record in the cluster has one, else the first record id."""
    for vuln_id in cluster:
        for alias in [vuln_id, *vulns[vuln_id]["aliases"]]:
            if alias.startswith("CVE-"):
                return alias
    return cluster[0]


def graph_from_snapshot(snapshot: Mapping[str, Any]) -> tuple[SupplyChainGraph, dict[str, Any]]:
    """The supply-chain graph and a small facts dict (issues per package, projects, cycles cut)."""
    today = datetime.fromisoformat(snapshot["generated_at"])
    vulns = snapshot["vulns"]
    packages = snapshot["packages"]
    nodes: dict[str, Node] = {}
    issues: dict[str, list[dict[str, Any]]] = {}

    for pid, info in packages.items():
        live = [v for v in info.get("vulns", []) if not vulns[v]["withdrawn"]]
        malware = [v for v in live if v.startswith("MAL-")]
        cluster_ids = issue_clusters(vulns, [v for v in live if not v.startswith("MAL-")])
        issues[pid] = [
            {
                "id": canonical(vulns, c),
                "records": c,
                "severity": next((vulns[r]["severity"] for r in c if vulns[r]["severity"]), None),
                "summary": next((vulns[r]["summary"] for r in c if vulns[r]["summary"]), ""),
                "fixed": sorted({f for r in c for f in vulns[r]["fixed"]}),
                "cwes": sorted({w for r in c for w in vulns[r]["cwes"]}),
            }
            for c in cluster_ids
        ]
        published = info.get("published_at")
        age = (
            max(0.0, (today - datetime.fromisoformat(published.replace("Z", "+00:00"))).days)
            if published
            else 0.0
        )
        nodes[pid] = Node(
            node_id=pid,
            kind=NodeKind.PACKAGE,
            name=f"{info['name']} {info['version']}",
            cve_exposure_count=len(cluster_ids),
            days_since_last_update=float(age),
            sbom_depth=0,  # set below, once the edges are known
            breach_history=1 if malware else 0,
        )

    edges: dict[tuple[str, str], EdgeKind] = {}
    org_edges: set[tuple[str, str]] = set()
    orgs: list[Node] = []
    for project in snapshot["projects"]:
        pins: dict[str, str] = project["pins"]
        org_id = f"org:{project['repo']}"
        orgs.append(Node(org_id, NodeKind.ORGANIZATION, project["repo"], 0, 0.0, 0, 0))
        required_by_someone: set[str] = set()
        for name, version in pins.items():
            pid = _pid(name, version)
            for dep in packages[pid]["requires"]:
                if dep in pins and dep != name:
                    edges[(_pid(dep, pins[dep]), pid)] = EdgeKind.DEPENDENCY
                    required_by_someone.add(dep)
        for name, version in pins.items():
            if name not in required_by_someone:
                org_edges.add((_pid(name, version), org_id))

    graph = SupplyChainGraph()
    for node in [*nodes.values(), *orgs]:
        graph.add_node(node)
    cut = 0
    kept: list[tuple[str, str]] = []
    for edge in sorted(edges):
        # Real dependency graphs occasionally contain cycles (a plugin and its host). The graph
        # model is acyclic, so an edge that would close a loop is dropped and counted.
        if _reaches(kept, edge[1], edge[0]):
            cut += 1
            continue
        kept.append(edge)
    for source, target in kept:
        graph.add_edge(SupplyChainEdge(source, target, EdgeKind.DEPENDENCY))
    for source, target in sorted(org_edges):
        graph.add_edge(SupplyChainEdge(source, target, EdgeKind.DEPENDENCY))

    depth = _depths(graph, [o.node_id for o in orgs])
    graph = SupplyChainGraph.from_parts(
        [
            Node(
                n.node_id,
                n.kind,
                n.name,
                n.cve_exposure_count,
                n.days_since_last_update,
                depth.get(n.node_id, 0) if n.kind is NodeKind.PACKAGE else 0,
                n.breach_history,
            )
            for n in graph.nodes
        ],
        graph.edges,
    )
    graph.assert_acyclic()
    facts = {
        "issues": issues,
        "projects": [p["repo"] for p in snapshot["projects"]],
        "primary": next((p["repo"] for p in snapshot["projects"] if p.get("primary")), None),
        "cycle_edges_cut": cut,
        "generated_at": snapshot["generated_at"],
        "sources": snapshot["sources"],
    }
    return graph, facts


def _reaches(edges: list[tuple[str, str]], start: str, goal: str) -> bool:
    """Is ``goal`` reachable from ``start`` along ``edges``? (small graphs; built incrementally)"""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _depths(graph: SupplyChainGraph, org_ids: list[str]) -> dict[str, int]:
    """Hops from the nearest organisation, walking edges backwards (dependent -> dependency)."""
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target].append(edge.source)
    depth: dict[str, int] = {}
    queue = deque((org, 0) for org in org_ids)
    while queue:
        node, d = queue.popleft()
        for source in incoming.get(node, ()):
            if source not in depth:
                depth[source] = d + 1
                queue.append((source, d + 1))
    return depth


# --------------------------------------------------------------------------- #
# Real advisories
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RealAdvisory(Advisory):
    """A real OSV advisory. Its vulnerabilities are already in the graph's features, so applying
    it changes nothing: the advisory only *scopes* the review to what that package reaches."""

    def __post_init__(self) -> None:
        return None

    def apply(self, graph: SupplyChainGraph) -> SupplyChainGraph:
        return graph


_SEVERITY_ORDER: Final[dict[str, int]] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MODERATE": 2,
    "MEDIUM": 2,
    "LOW": 1,
}


def real_advisories(
    graph: SupplyChainGraph, facts: Mapping[str, Any], *, limit: int = 6
) -> dict[str, RealAdvisory]:
    """One real advisory per package that most needs reviewing: the worst issue on packages that
    carry at least four known issues, most-depended-on first. Deterministic."""
    dependents = {
        n.node_id: len(graph.downstream(n.node_id, max_hops=4))
        for n in graph.nodes
        if n.kind is NodeKind.PACKAGE and n.cve_exposure_count >= 4
    }
    ranked = sorted(dependents, key=lambda pid: (-dependents[pid], pid))[:limit]
    out: dict[str, RealAdvisory] = {}
    for pid in ranked:
        issues = facts["issues"][pid]
        worst = max(issues, key=lambda i: (_SEVERITY_ORDER.get(i["severity"] or "", 0), i["id"]))
        cves = tuple(i["id"] for i in issues if i["id"].startswith("CVE-"))[:8]
        node = graph.node(pid)
        out[worst["id"]] = RealAdvisory(
            advisory_id=worst["id"],
            kind=AdvisoryKind.CVE,
            package_id=pid,
            title=f"{node.name}: {worst['summary'] or worst['id']}",
            summary=(
                f"{len(issues)} known vulnerabilities affect the pinned version {node.name}"
                f"; the most severe is {worst['id']} ({worst['severity'] or 'unrated'})."
                + (f" Fixed in {', '.join(worst['fixed'])}." if worst["fixed"] else "")
            ),
            cve_ids=cves,
        )
    return out
