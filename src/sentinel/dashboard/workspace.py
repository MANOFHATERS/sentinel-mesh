"""The live mesh behind the dashboard: one tenant, three graphs, one audit chain.

Part 3 built three checkpointed graphs and Part 4 put real connectors behind them;
until now the only thing that drove them was ``scripts/evaluate.py``, which runs a
feed, answers every approval with a scripted ``HumanDecision``, and exits. F-10
needs the opposite shape: a process that *stays up*, holds runs that are waiting
for a person, and resumes each one when that person answers from a browser.

:class:`Workspace` is that process state, and it deliberately adds no new way of
doing anything. Every method is a thin, locked call into an existing function:

==========================  ===================================================
Dashboard need              Existing function it calls
==========================  ===================================================
approval queue              ``CompiledGraph.pending()`` over the shared store
approve / reject            ``CompiledGraph.resume(thread_id, HumanDecision)``
crash recovery              ``CompiledGraph.recover(thread_id)``
investigation timeline      ``HashChainedAuditLog.iter_records()``
what happened on the wire   ``connector_called`` / ``guardrail_blocked`` rows,
                            ``ConnectorRouter.executions``
supply-chain map            ``top_risk_explanations``, ``explain_node``
code-scan view              ``CodeScanAgent.assess`` (deterministic in its input)
F-08 read-back              ``verify_no_ungated_execution``
==========================  ===================================================

Durability
----------
Runs survive a restart, because F-04's *"resume with full context intact"* has to
hold for a dashboard that is redeployed while an incident waits on a human. Four
things persist under ``workdir``: the audit chain, the checkpoint store, the
execution journal (exactly-once across a crash, Part 4), and a small registry of
*which graph owns which thread*. The registry is the one piece the Part 3 runtime
did not need — a single-graph caller always knows its graph — and without it a
restarted dashboard could list a waiting run but not resume it.

A published advisory changes the dependency graph a review run was built over, so
the registry also records advisories in publication order; reopening re-applies
them to the base graph and rebuilds each review graph exactly. Model training is
seeded, so the rebuilt graph scores identically, and
``test_dashboard_workspace.py`` asserts a restarted workspace resumes a waiting
review to the same outcome.

Concurrency
-----------
One re-entrant lock serialises every mutation and every read that spans more than
one store. Runs are short (tens of milliseconds, dominated by loopback HTTP), and a
dashboard serving a handful of analysts gains nothing from parallel resumes but
would inherit every race between "read the pending action" and "resume it". The
stale-decision check in :meth:`Workspace.decide` closes the remaining gap between
what an analyst *saw* and what they are approving.
"""

from __future__ import annotations

import random
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np

from sentinel.agents.checkpoint import SqliteCheckpointer, verify_chain
from sentinel.agents.codescan import (
    CodeScanAgent,
    CodeScanAssessment,
    build_code_scan_graph,
    new_code_scan_incident,
    synthesize_scan_alert,
)
from sentinel.agents.contain import ContainmentAgent, verify_no_ungated_execution
from sentinel.agents.investigate import InvestigationAgent
from sentinel.agents.orchestrator import build_incident_graph, new_incident
from sentinel.agents.runtime import CompiledGraph, RunResult
from sentinel.agents.state import HumanDecision, IncidentState, IncidentStatus
from sentinel.agents.supplychain import (
    SupplyChainAgent,
    build_supply_chain_review_graph,
    new_vendor_risk_incident,
    synthesize_vendor_alert,
)
from sentinel.agents.triage import TriageAgent, TriageModel
from sentinel.audit.log import AuditRecord, HashChainedAuditLog, VerificationResult
from sentinel.connectors.journal import BlastRadiusLimiter, SqliteJournal
from sentinel.connectors.router import ConnectorRouter
from sentinel.connectors.sandbox import Sandbox
from sentinel.connectors.targets import TargetPolicy
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import SentinelError
from sentinel.core.ids import deterministic_id
from sentinel.core.schemas import Alert, ApprovalStatus, AuditEventType
from sentinel.dashboard.lab import (
    DiffusionStudy,
    PolicyTraining,
    ServingPolicy,
    graph_evaluation,
    train_response_policy,
)
from sentinel.dashboard.scenarios import (
    ASSET_INVENTORY,
    PROTECTED_NETWORKS,
    SCENARIOS,
    Advisory,
    AdvisoryKind,
    ScenarioName,
    cve_advisory,
    exposure_scope,
    malicious_advisory,
    pick_story_flow,
    script_alert,
    select_cve_package,
    select_malicious_package,
)
from sentinel.graph.explain import neighbourhood_shares
from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
from sentinel.graph.schema import SupplyChainGraph
from sentinel.graph.synthetic import SyntheticGraphGenerator
from sentinel.kb.retrieve import KnowledgeBase
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.ml.metrics import four_way_split
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.seeded import FIXTURE_DIR

__all__ = [
    "DEFAULT_ALERTS",
    "DEFAULT_SEED",
    "Conflict",
    "GraphKind",
    "MeshModels",
    "NotFound",
    "ScenarioRun",
    "ThreadRef",
    "Workspace",
    "WorkspaceError",
]

DEFAULT_SEED: Final[int] = 20260928
#: Corpus size for the dashboard's models. Smaller than the evaluation's 20,000 so a
#: dashboard starts in seconds; large enough that the held-out split carries
#: escalated flows of every family the scenarios need (asserted at build time).
DEFAULT_ALERTS: Final[int] = 12_000
#: The demo repository the Code-Scan graph pushes to, matching the sandbox.
REPOSITORY: Final[str] = "acme/billing"
#: Blast-radius ceiling, as ``router_from_env`` configures it (Part 4).
BLAST_RADIUS: Final[int] = 25


class WorkspaceError(SentinelError):
    """The request is incoherent for the workspace's current state."""


class NotFound(WorkspaceError):
    """No such thread, scenario, advisory or node *for this tenant*."""


class Conflict(WorkspaceError):
    """The request raced another decision, or repeats a completed one."""


class GraphKind:
    INCIDENT = "incident"
    CODE_SCAN = "code_scan"
    SUPPLY_CHAIN = "supply_chain"


# --------------------------------------------------------------------------- #
# Models: expensive, tenant-agnostic, built once per process
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MeshModels:
    """Everything trained or loaded once and shared by every tenant's workspace.

    Sharing is safe because nothing here is mutated after :meth:`build`: the triage
    model, knowledge base and GNN are fitted objects, the base graph is never
    modified (an advisory produces a copy), and the flow pools are tuples of frozen
    alerts. An MSSP console with fifty tenants trains once, not fifty times.
    """

    seed: int
    triage_model: TriageModel
    kb: KnowledgeBase
    snapshot: RepoSnapshot
    graph: SupplyChainGraph
    gnn: SupplyChainGNN
    base_scores: np.ndarray
    #: ``(scenario, step index) -> flow``, assigned once so launch order is irrelevant.
    story_flows: dict[tuple[ScenarioName, int], Alert]
    #: Held-out flows not used by any scenario, in stream order: the background feed.
    feed: tuple[Alert, ...]
    cve_package: str
    malicious_package: str
    #: Part 5.1: the Section 5.5.4 response policy, trained at start-up and served
    #: greedily to every workspace's Containment Agent.
    policy: ServingPolicy
    policy_training: PolicyTraining
    #: F-06 on this process's own test split, next to a features-only baseline.
    graph_evaluation: dict[str, Any]
    #: Section 5.5.5, run on a background thread once :meth:`start_background` is
    #: called (the dashboard does; tests that do not need it never pay for it).
    diffusion: DiffusionStudy
    #: ``"synthetic"`` (the scripted demo, the default) or ``"real"``: the triage model was
    #: trained on a real public capture, the knowledge base is MITRE's real ATT&CK catalogue
    #: and the feed is real held-out flows. See :meth:`build_real`.
    mode: str = "synthetic"
    #: The real-data evaluation report that :meth:`build_real` produced, when ``mode == "real"``.
    real_report: dict[str, Any] | None = None
    #: The repository the code-scan graph reviews and opens draft pull requests against.
    repository: str = "acme/billing"
    #: The real supply chain's facts (issues per package, provenance), when one was loaded.
    real_supply: dict[str, Any] | None = None
    #: Real OSV advisories the real workspace can review, by advisory id.
    real_advisories: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, *, seed: int = DEFAULT_SEED, n_alerts: int = DEFAULT_ALERTS) -> MeshModels:
        alerts = generate_alerts(n_alerts, seed=seed)
        labels = np.array([0 if a.ground_truth_label == "benign" else 1 for a in alerts])
        split = four_way_split(labels, seed=seed)
        model = TriageModel.fit(
            [alerts[i] for i in split.train_benign],
            train_labelled=[alerts[i] for i in split.train_labelled],
            validation=[alerts[i] for i in split.validation],
            seed=seed,
        )
        held_out = [alerts[i] for i in split.test]

        # Eligibility is decided by the real agent, on flows it has never seen, *as
        # scripted*: scenario 3's payload note is part of what triage judges.
        agent = TriageAgent(model=model)
        story: dict[tuple[ScenarioName, int], Alert] = {}
        used: set[str] = set()
        for name, spec in SCENARIOS.items():
            for index, flow in enumerate(spec.flows):
                story[(name, index)] = pick_story_flow(
                    held_out, flow, agent=agent, used=used, scenario=name, index=index
                )
                used.add(story[(name, index)].alert_id)
        feed = tuple(alert for alert in held_out if alert.alert_id not in used)

        graph, truth = SyntheticGraphGenerator(seed=seed).generate()
        node_ids = graph.node_ids()
        node_labels = truth.labels(node_ids)
        graph_split = GraphSplit.stratified(node_labels, seed=seed)
        gnn = SupplyChainGNN(random_state=seed).fit(
            graph,
            node_labels,
            graph_split,
            exposure=truth.risk_vector(node_ids),
        )
        bandit, policy_training = train_response_policy(seed=seed)
        cve_package = select_cve_package(graph)
        # Neither advisory's package may sit inside the other's reach, so the two
        # reviews stay about their own advisory whichever is launched first.
        cve_reach = set(graph.downstream(cve_package, max_hops=4))
        malicious_package = select_malicious_package(
            graph, exclude={cve_package, *cve_reach}, must_not_reach={cve_package}
        )
        return cls(
            seed=seed,
            triage_model=model,
            kb=KnowledgeBase.build(),
            snapshot=RepoSnapshot.from_dir(FIXTURE_DIR),
            graph=graph,
            gnn=gnn,
            base_scores=gnn.risk_scores(graph),
            story_flows=story,
            feed=feed,
            cve_package=cve_package,
            malicious_package=malicious_package,
            policy=ServingPolicy(bandit),
            policy_training=policy_training,
            graph_evaluation=graph_evaluation(graph, truth, gnn, graph_split),
            diffusion=DiffusionStudy(seed=seed, n_alerts=n_alerts),
        )

    @classmethod
    def build_real(
        cls,
        base: MeshModels,
        *,
        dataset_path: str | Path,
        attack_path: str | Path,
        seed: int = DEFAULT_SEED,
        limit: int = 20_000,
        supply_chain_path: str | Path | None = None,
        repo_cache_dir: str | Path | None = None,
    ) -> MeshModels:
        """The real-data variant: real triage model, real ATT&CK, real held-out flows.

        Built *from* ``base`` so nothing the real workspace does not use is retrained. What is
        replaced is exactly what the audience sees as data: the triage model (fitted on the real
        training split only), the knowledge base (MITRE's published catalogue) and the feed (the
        real test split, in file order). There are no scripted scenarios, and the supply-chain
        graph, GNN and repository fixture are shared but unreachable: the API refuses those
        routes for a real workspace, because there is no real graph to show (PRD Phase 2).
        """
        from dataclasses import replace

        from sentinel.real.attack import build_attack_kb
        from sentinel.real.network import train_real

        model, test, report = train_real(dataset_path, limit=limit, seed=seed)
        feed = list(test)
        if report.get("has_addresses"):
            # Real capture times: stream the flows in the order they happened, bursts and all.
            feed.sort(key=lambda a: a.timestamp)
        else:
            # A file without times is stored in stretches of one kind of traffic; streamed in
            # file order the first minutes of the feed would be all benign or all attack. A
            # seeded shuffle interleaves them the way a mixed feed arrives, reproducibly.
            random.Random(seed).shuffle(feed)
        extra: dict[str, Any] = {}
        if supply_chain_path is not None and Path(supply_chain_path).is_file():
            extra.update(cls._real_supply_chain(Path(supply_chain_path), seed=seed))
        if repo_cache_dir is not None and extra.get("real_supply"):
            from sentinel.real.repos import load_repo_snapshot

            primary = extra["real_supply"]["primary"]
            if primary:
                try:
                    extra["snapshot"] = load_repo_snapshot(primary, repo_cache_dir)
                    extra["repository"] = primary
                except Exception:  # offline and not cached: the workspace simply has no code scan
                    pass
        return replace(
            base,
            seed=seed,
            triage_model=model,
            kb=build_attack_kb(attack_path),
            story_flows={},
            feed=tuple(feed),
            mode="real",
            real_report=report,
            **extra,
        )

    @staticmethod
    def _real_supply_chain(path: Path, *, seed: int) -> dict[str, Any]:
        """The real graph, its rule-defined ground truth, and a GNN trained on it."""
        from sentinel.graph.synthetic import label_ground_truth
        from sentinel.real.supplychain import graph_from_snapshot, load_snapshot, real_advisories

        graph, facts = graph_from_snapshot(load_snapshot(path))
        truth = label_ground_truth(graph)
        node_ids = graph.node_ids()
        labels = truth.labels(node_ids)
        split = GraphSplit.stratified(labels, seed=seed)
        gnn = SupplyChainGNN(random_state=seed).fit(
            graph, labels, split, exposure=truth.risk_vector(node_ids)
        )
        from sentinel.real.grounding import load_exploitation
        from sentinel.real.supplychain import annotate_exploitation

        exploitation = load_exploitation(path.parent)
        if exploitation is not None:
            facts["exploited_in_the_wild"] = annotate_exploitation(facts, exploitation)
            facts["epss_date"] = exploitation.epss_date
        advisories = real_advisories(graph, facts)
        facts = {
            **facts,
            "truth": {
                "intrinsic": len(truth.intrinsic),
                "inherited": len(truth.inherited),
                "nodes": graph.n_nodes,
                "edges": len(graph.edges),
            },
        }
        return {
            "graph": graph,
            "gnn": gnn,
            "base_scores": gnn.risk_scores(graph),
            "graph_evaluation": graph_evaluation(graph, truth, gnn, split),
            "real_supply": facts,
            "real_advisories": advisories,
            "cve_package": next(iter(advisories.values())).package_id
            if advisories
            else graph.node_ids()[0],
        }

    def start_background(self) -> MeshModels:
        """Start the slow studies (diffusion) that the Models page reports on."""
        self.diffusion.start()
        return self

    def advisory(self, kind: AdvisoryKind) -> Advisory:
        if kind is AdvisoryKind.CVE:
            return cve_advisory(self.cve_package)
        return malicious_advisory(self.malicious_package)


# --------------------------------------------------------------------------- #
# Registry: which graph owns which thread
# --------------------------------------------------------------------------- #

_REGISTRY_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    scenario    TEXT,
    caption     TEXT NOT NULL,
    advisory_id TEXT,
    position    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS advisories (
    advisory_id  TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    published_at TEXT NOT NULL,
    position     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS scenarios (
    name        TEXT PRIMARY KEY,
    launched_at TEXT NOT NULL,
    launched_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def workspace_features(models: MeshModels) -> dict[str, bool]:
    """Which pages have data behind them for this workspace.

    A synthetic workspace has everything. A real one has a page only if real data backs it: the
    supply-chain map needs a real supply-chain snapshot, the code scan needs a real repository.
    The API refuses a page that fails this test, and the sidebar does not offer it.
    """
    real = models.mode == "real"
    return {
        "scenarios": True,
        "supply_chain": not real or models.real_supply is not None,
        "code_scan": not real or models.repository != "acme/billing",
        "models": True,
        "kb": real,
    }


def _tag_value(tag: ScenarioName | str | None) -> str | None:
    """A scenario tag as stored: a scripted scenario's value, or a real scenario's id."""
    return None if tag is None else (tag.value if isinstance(tag, ScenarioName) else str(tag))


def _parse_tag(text: str | None) -> ScenarioName | str | None:
    if text is None:
        return None
    try:
        return ScenarioName(text)
    except ValueError:
        return text


@dataclass(frozen=True, slots=True)
class ThreadRef:
    thread_id: str
    kind: str
    scenario: ScenarioName | str | None
    caption: str
    advisory_id: str | None
    position: int


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    name: ScenarioName | str
    launched_at: datetime
    launched_by: str


@dataclass(slots=True)
class _PublishedAdvisory:
    advisory: Advisory
    published_at: datetime
    scope: tuple[str, ...]
    subgraph: SupplyChainGraph
    scores: np.ndarray
    compiled: CompiledGraph


# --------------------------------------------------------------------------- #
# The workspace
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Stores:
    audit: HashChainedAuditLog
    checkpoints: SqliteCheckpointer
    journal: SqliteJournal
    registry: sqlite3.Connection


class Workspace:
    """One tenant's live mesh. Every public method takes the workspace lock."""

    def __init__(
        self,
        models: MeshModels,
        *,
        tenant_id: str,
        workdir: str | Path,
        clock: Clock | None = None,
        connector_wrapper: Callable[[ConnectorRouter], Any] | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("a workspace serves exactly one tenant; name it")
        self.models = models
        self.tenant_id = tenant_id
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.clock: Clock = clock or SystemClock()
        self._lock = threading.RLock()
        self._closed = False

        registry = sqlite3.connect(
            str(self.workdir / "registry.sqlite"), check_same_thread=False, isolation_level=None
        )
        registry.executescript(_REGISTRY_DDL)
        self._stores = _Stores(
            audit=HashChainedAuditLog(self.workdir / "audit.sqlite", clock=self.clock),
            checkpoints=SqliteCheckpointer(self.workdir / "checkpoints.sqlite"),
            journal=SqliteJournal(self.workdir / "journal.sqlite"),
            registry=registry,
        )
        hosts = [asset.address for asset in ASSET_INVENTORY.values() if asset.hostname[0] != "("]
        hosts += [alert.asset_id for alert in models.feed]
        owner, _, repo = models.repository.partition("/")
        self.sandbox = Sandbox(
            repo_files={item.path: item.text for item in models.snapshot.files},
            owner=owner,
            repo=repo,
            hosts=hosts,
            clock=self.clock,
        ).start()
        self.router = self.sandbox.router(
            tenant_id=tenant_id,
            audit=self._stores.audit,
            journal=self._stores.journal,
            limiter=BlastRadiusLimiter(
                max_actions=BLAST_RADIUS, window=timedelta(hours=1), clock=self.clock
            ),
            targets=TargetPolicy(
                protected_networks=PROTECTED_NETWORKS,
                protected_hosts=frozenset(
                    asset.hostname for asset in ASSET_INVENTORY.values() if asset.protected
                ),
            ),
        )
        self._connector = (
            connector_wrapper(self.router) if connector_wrapper is not None else self.router
        )

        self.incident_graph = build_incident_graph(
            triage=TriageAgent(model=models.triage_model, clock=self.clock),
            investigation=InvestigationAgent(kb=models.kb, clock=self.clock),
            containment=ContainmentAgent(clock=self.clock, policy=models.policy, triage_floor=True),
            connector=self._connector,
            checkpointer=self._stores.checkpoints,
        )
        self.code_agent = CodeScanAgent(kb=models.kb, clock=self.clock)
        self.code_graph = build_code_scan_graph(
            agent=self.code_agent,
            snapshot=models.snapshot,
            connector=self._connector,
            checkpointer=self._stores.checkpoints,
        )
        self.supply_agent = SupplyChainAgent(kb=models.kb, clock=self.clock)
        self.graph: SupplyChainGraph = models.graph
        self.scores: np.ndarray = models.base_scores
        self.shares: np.ndarray = neighbourhood_shares(models.gnn, models.graph)
        self._advisories: dict[str, _PublishedAdvisory] = {}
        self._threads: dict[str, ThreadRef] = {}
        self._scenarios: dict[ScenarioName | str, ScenarioRun] = {}
        self._code_scan: CodeScanAssessment | None = None
        self._restore()

    # --- lifecycle ------------------------------------------------------------ #

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.sandbox.stop()
            self._stores.journal.close()
            self._stores.checkpoints.close()
            self._stores.audit.close()
            self._stores.registry.close()

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock:
            if self._closed:
                raise WorkspaceError("this workspace is closed")
            yield

    @property
    def audit(self) -> HashChainedAuditLog:
        return self._stores.audit

    # --- restore -------------------------------------------------------------- #

    def _restore(self) -> None:
        db = self._stores.registry
        for advisory_id, kind, published_at in db.execute(
            "SELECT advisory_id, kind, published_at FROM advisories ORDER BY position"
        ):
            advisory = self.models.real_advisories.get(advisory_id) or self.models.advisory(
                AdvisoryKind(kind)
            )
            if advisory.advisory_id != advisory_id:
                raise WorkspaceError(
                    f"registry names advisory {advisory_id} but the models derive "
                    f"{advisory.advisory_id}; the workspace was written by different models"
                )
            self._apply_advisory(advisory, datetime.fromisoformat(published_at))
        for row in db.execute(
            "SELECT thread_id, kind, scenario, caption, advisory_id, position FROM threads"
        ):
            ref = ThreadRef(
                thread_id=row[0],
                kind=row[1],
                scenario=_parse_tag(row[2]),
                caption=row[3],
                advisory_id=row[4],
                position=row[5],
            )
            self._threads[ref.thread_id] = ref
            if ref.kind == GraphKind.CODE_SCAN:
                self._code_scan = self._assess_code(self._state(ref).alert)
        for name, launched_at, launched_by in db.execute(
            "SELECT name, launched_at, launched_by FROM scenarios"
        ):
            scenario = _parse_tag(name)
            self._scenarios[scenario] = ScenarioRun(
                scenario, datetime.fromisoformat(launched_at), launched_by
            )

    def _register(self, ref: ThreadRef) -> None:
        self._stores.registry.execute(
            "INSERT INTO threads(thread_id, kind, scenario, caption, advisory_id, position) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                ref.thread_id,
                ref.kind,
                _tag_value(ref.scenario),
                ref.caption,
                ref.advisory_id,
                ref.position,
            ),
        )
        self._threads[ref.thread_id] = ref

    def _meta(self, key: str, default: str) -> str:
        row = self._stores.registry.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return default if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self._stores.registry.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # --- graph lookup ---------------------------------------------------------- #

    def _graph_for(self, ref: ThreadRef) -> CompiledGraph:
        if ref.kind == GraphKind.INCIDENT:
            return self.incident_graph
        if ref.kind == GraphKind.CODE_SCAN:
            return self.code_graph
        if ref.kind == GraphKind.SUPPLY_CHAIN and ref.advisory_id in self._advisories:
            return self._advisories[ref.advisory_id].compiled
        raise WorkspaceError(f"thread {ref.thread_id} has no graph (kind {ref.kind})")

    def _ref(self, thread_id: str) -> ThreadRef:
        ref = self._threads.get(thread_id)
        if ref is None:
            # Deliberately the same error for "exists for another tenant" and "does
            # not exist": a 403 would confirm the id is real.
            raise NotFound(f"no incident {thread_id!r}")
        return ref

    def _state(self, ref: ThreadRef) -> IncidentState:
        state = self._graph_for(ref).state_of(ref.thread_id)
        if state is None:
            raise NotFound(f"incident {ref.thread_id!r} has no checkpoint")
        if state.tenant_id != self.tenant_id:  # pragma: no cover - defence in depth
            raise NotFound(f"no incident {ref.thread_id!r}")
        return state

    def refs(self) -> tuple[ThreadRef, ...]:
        with self._locked():
            return tuple(sorted(self._threads.values(), key=lambda r: r.position))

    def ref(self, thread_id: str) -> ThreadRef:
        with self._locked():
            return self._ref(thread_id)

    def state(self, thread_id: str) -> IncidentState:
        with self._locked():
            return self._state(self._ref(thread_id))

    def states(self) -> list[tuple[ThreadRef, IncidentState]]:
        with self._locked():
            return [(ref, self._state(ref)) for ref in self.refs()]

    def checkpoint_chain_ok(self, thread_id: str) -> tuple[bool, int, str | None]:
        """``(verified, checkpoint_count, error)`` for one thread's checkpoint chain."""
        with self._locked():
            self._ref(thread_id)
            history = self._stores.checkpoints.history(thread_id)
            try:
                verify_chain(history)
            except SentinelError as exc:
                return False, len(history), str(exc)
            return True, len(history), None

    # --- running --------------------------------------------------------------- #

    def _next_position(self) -> int:
        return len(self._threads)

    def _start(
        self,
        graph: CompiledGraph,
        state: IncidentState,
        *,
        kind: str,
        caption: str,
        scenario: ScenarioName | str | None,
        advisory_id: str | None = None,
    ) -> RunResult:
        if state.incident_id in self._threads:
            raise Conflict(f"incident {state.incident_id} already exists")
        # Registered before the run so a crash inside it still leaves a thread the
        # dashboard can find and recover.
        self._register(
            ThreadRef(
                thread_id=state.incident_id,
                kind=kind,
                scenario=scenario,
                caption=caption[:300],
                advisory_id=advisory_id,
                position=self._next_position(),
            )
        )
        return graph.invoke(state, clock=self.clock, audit=self._stores.audit)

    def launch(self, name: ScenarioName, *, launched_by: str) -> tuple[str, ...]:
        """Start a scenario. Returns the thread ids it opened, in story order."""
        if self.models.mode == "real":
            raise WorkspaceError(
                "scenarios are scripted stories; this workspace runs on real data and has none"
            )
        with self._locked():
            if name in self._scenarios:
                raise Conflict(f"scenario {name.value} has already been launched")
            spec = SCENARIOS[name]
            now = self.clock.now()
            run = ScenarioRun(name, now, launched_by)
            self._stores.registry.execute(
                "INSERT INTO scenarios(name, launched_at, launched_by) VALUES (?, ?, ?)",
                (name.value, now.isoformat(), launched_by),
            )
            self._scenarios[name] = run
            opened: list[str] = []

            if spec.advisory is not None:
                opened.append(
                    self._publish_and_review(self.models.advisory(spec.advisory), scenario=name)
                )
            for index, flow in enumerate(spec.flows):
                alert = script_alert(
                    self.models.story_flows[(name, index)],
                    flow,
                    tenant_id=self.tenant_id,
                    scenario=name.value,
                    index=index,
                    launched_at=now,
                )
                result = self._start(
                    self.incident_graph,
                    new_incident(alert, at=now),
                    kind=GraphKind.INCIDENT,
                    caption=flow.caption,
                    scenario=name,
                )
                opened.append(result.state.incident_id)
            if spec.code_scan:
                opened.append(self._scan_repository(scenario=name))
            return tuple(opened)

    def real_scenario(self, scenario_id: str):
        from sentinel.real.scenarios import real_scenarios

        for spec in real_scenarios(self.models):
            if spec.id == scenario_id:
                return spec
        raise NotFound(f"no real scenario {scenario_id!r}")

    def launch_real(self, scenario_id: str, *, launched_by: str) -> tuple[str, ...]:
        """Launch one real case (see :mod:`sentinel.real.scenarios`). Returns the thread ids."""
        from sentinel.real.scenarios import CAMPAIGN_FLOWS

        if self.models.mode != "real":
            raise WorkspaceError("this workspace has no real scenarios")
        spec = self.real_scenario(scenario_id)
        with self._locked():
            now = self.clock.now()
            opened: list[str] = []
            if spec.kind == "campaign":
                pool = [a for a in self.models.feed if a.ground_truth_label == spec.subject]
                key = f"campaign:{spec.subject}"
                cursor = int(self._meta(key, "0"))
                batch = pool[cursor : cursor + CAMPAIGN_FLOWS]
                if not batch:
                    raise Conflict(f"every held-out {spec.subject} flow has been replayed")
                for offset, source in enumerate(batch):
                    number = cursor + offset + 1
                    alert = source.updated(
                        tenant_id=self.tenant_id,
                        alert_id=deterministic_id(
                            "real-campaign", self.tenant_id, spec.subject, number
                        ),
                        ingested_at=max(now, source.timestamp),
                    )
                    result = self._start(
                        self.incident_graph,
                        new_incident(alert, at=now),
                        kind=GraphKind.INCIDENT,
                        caption=f"{spec.title} #{number}",
                        scenario=spec.id,
                    )
                    opened.append(result.state.incident_id)
                self._set_meta(key, str(cursor + len(batch)))
            elif spec.kind == "advisory":
                advisory = self.models.real_advisories[spec.subject]
                opened.append(self._publish_and_review(advisory, scenario=spec.id))
            else:
                if self._code_scan is not None:
                    raise Conflict(f"{self.models.repository} has already been scanned")
                opened.append(self._scan_repository(scenario=spec.id))
            self._stores.registry.execute(
                "INSERT OR REPLACE INTO scenarios(name, launched_at, launched_by) VALUES (?, ?, ?)",
                (spec.id, now.isoformat(), launched_by),
            )
            self._scenarios[spec.id] = ScenarioRun(spec.id, now, launched_by)
            return tuple(opened)

    def _publish_and_review(
        self, advisory: Advisory, *, scenario: ScenarioName | str | None
    ) -> str:
        if advisory.advisory_id in self._advisories:
            raise Conflict(f"advisory {advisory.advisory_id} is already published")
        now = self.clock.now()
        published = self._apply_advisory(advisory, now)
        self._stores.registry.execute(
            "INSERT INTO advisories(advisory_id, kind, published_at, position) VALUES (?, ?, ?, ?)",
            (advisory.advisory_id, advisory.kind.value, now.isoformat(), len(self._advisories) - 1),
        )
        alert = synthesize_vendor_alert(
            tenant_id=self.tenant_id,
            graph=published.subgraph,
            at=now,
            assessment_id=advisory.advisory_id,
        )
        result = self._start(
            published.compiled,
            new_vendor_risk_incident(alert, at=now),
            kind=GraphKind.SUPPLY_CHAIN,
            caption=f"{advisory.advisory_id}: {advisory.title}",
            scenario=scenario,
            advisory_id=advisory.advisory_id,
        )
        return result.state.incident_id

    def _apply_advisory(self, advisory: Advisory, published_at: datetime) -> _PublishedAdvisory:
        self.graph = advisory.apply(self.graph)
        self.scores = self.models.gnn.risk_scores(self.graph)
        self.shares = neighbourhood_shares(self.models.gnn, self.graph)
        scope = tuple(exposure_scope(self.graph, advisory.package_id))
        subgraph = self.graph.subgraph(scope)
        # Scores and attributions come from the *whole* graph and are then restricted,
        # so scoping the review changes which nodes are shown, never what any of them
        # scores or what the model keyed on. (The GNN is full-graph and could not be
        # re-run on the subgraph anyway; see neighbourhood_shares.)
        rows = [self.graph.index_of(node) for node in subgraph.node_ids()]
        scores = self.scores[rows]
        compiled = build_supply_chain_review_graph(
            agent=self.supply_agent,
            graph=subgraph,
            scores=scores,
            shares=self.shares[rows],
            include=(advisory.package_id,),
            connector=self._connector,
            checkpointer=self._stores.checkpoints,
        )
        published = _PublishedAdvisory(advisory, published_at, scope, subgraph, scores, compiled)
        self._advisories[advisory.advisory_id] = published
        return published

    def _assess_code(self, alert: Alert) -> CodeScanAssessment:
        return self.code_agent.assess(self.models.snapshot, alert=alert, now=alert.ingested_at)

    def _scan_repository(self, *, scenario: ScenarioName | str | None) -> str:
        now = self.clock.now()
        alert = synthesize_scan_alert(
            self.models.snapshot,
            tenant_id=self.tenant_id,
            repository=self.models.repository,
            at=now,
            commit="HEAD",
        )
        self._code_scan = self._assess_code(alert)
        result = self._start(
            self.code_graph,
            new_code_scan_incident(alert, at=now),
            kind=GraphKind.CODE_SCAN,
            caption=f"Code scan of {self.models.repository}",
            scenario=scenario,
        )
        return result.state.incident_id

    def replay(self, count: int) -> dict[str, int]:
        """Push the next ``count`` background flows through the incident graph.

        Each is re-stamped with the wall-clock ingestion time, which is what the
        replay service does (:mod:`sentinel.ingest.replay`), so MTTD on the
        dashboard is the pipeline's latency and not the age of the capture.
        """
        if not 1 <= count <= 500:
            raise WorkspaceError("replay between 1 and 500 flows at a time")
        with self._locked():
            cursor = int(self._meta("feed_cursor", "0"))
            batch = self.models.feed[cursor : cursor + count]
            if not batch:
                raise Conflict("the background feed is exhausted")
            outcome = {"ingested": 0, "dismissed": 0, "handled": 0, "gated": 0, "failed": 0}
            for offset, source in enumerate(batch):
                now = self.clock.now()
                alert = source.updated(
                    tenant_id=self.tenant_id,
                    alert_id=deterministic_id("feed-alert", self.tenant_id, cursor + offset),
                    ingested_at=max(now, source.timestamp),
                )
                result = self._start(
                    self.incident_graph,
                    new_incident(alert, at=now),
                    kind=GraphKind.INCIDENT,
                    caption=f"Background flow #{cursor + offset + 1}",
                    scenario=None,
                )
                outcome["ingested"] += 1
                status = result.state.status
                if status is IncidentStatus.DISMISSED:
                    outcome["dismissed"] += 1
                elif status is IncidentStatus.AWAITING_APPROVAL:
                    outcome["gated"] += 1
                elif status is IncidentStatus.FAILED:
                    outcome["failed"] += 1
                else:
                    outcome["handled"] += 1
            self._set_meta("feed_cursor", str(cursor + len(batch)))
            return outcome

    def feed_remaining(self) -> int:
        with self._locked():
            return len(self.models.feed) - int(self._meta("feed_cursor", "0"))

    # --- deciding --------------------------------------------------------------- #

    def decide(
        self,
        thread_id: str,
        *,
        action_id: str,
        approved: bool,
        approver: str,
        note: str = "",
    ) -> IncidentState:
        """Answer the gate for ``thread_id``.

        ``action_id`` is the action the analyst was shown. If the run is now waiting
        on a different action — or on nothing — the decision is refused rather than
        applied to whatever happens to be pending: an approval is for the thing that
        was reviewed, which is the same rule the code-scan graph applies to a diff.
        """
        with self._locked():
            ref = self._ref(thread_id)
            graph = self._graph_for(ref)
            state = self._state(ref)
            if not state.is_waiting or state.interrupt is None:
                raise Conflict(
                    f"incident {thread_id} is {state.status.value}; there is no pending "
                    "decision (it may already have been answered)"
                )
            if state.interrupt.subject_id != action_id:
                raise Conflict(
                    "the pending action is not the one you reviewed; reload before deciding"
                )
            decision = HumanDecision(
                approver=approver,
                approved=approved,
                decided_at=self.clock.now(),
                note=note,
            )
            return graph.resume(
                thread_id, decision, clock=self.clock, audit=self._stores.audit
            ).state

    def recover(self, thread_id: str) -> IncidentState:
        """Continue a run whose process died mid-node (Part 4's ``recover``)."""
        with self._locked():
            ref = self._ref(thread_id)
            state = self._state(ref)
            if state.status.is_terminal or state.is_waiting:
                raise Conflict(
                    f"incident {thread_id} is {state.status.value}; only a stalled "
                    "running incident can be recovered"
                )
            return (
                self._graph_for(ref)
                .recover(thread_id, clock=self.clock, audit=self._stores.audit)
                .state
            )

    # --- reading ---------------------------------------------------------------- #

    def scenario_runs(self) -> dict[ScenarioName | str, ScenarioRun]:
        with self._locked():
            return dict(self._scenarios)

    def advisories(self) -> list[_PublishedAdvisory]:
        with self._locked():
            return sorted(self._advisories.values(), key=lambda p: p.published_at)

    def advisory(self, advisory_id: str) -> _PublishedAdvisory:
        with self._locked():
            published = self._advisories.get(advisory_id)
            if published is None:
                raise NotFound(f"no published advisory {advisory_id!r}")
            return published

    def code_scan(self) -> tuple[ThreadRef, CodeScanAssessment] | None:
        with self._locked():
            if self._code_scan is None:
                return None
            ref = next(r for r in self._threads.values() if r.kind == GraphKind.CODE_SCAN)
            return ref, self._code_scan

    def records(self, subjects: Sequence[str] | None = None) -> list[AuditRecord]:
        """Audit rows for this tenant, optionally restricted to ``subjects``."""
        with self._locked():
            wanted = None if subjects is None else set(subjects)
            return [
                record
                for record in self._stores.audit.iter_records()
                if record.tenant_id == self.tenant_id
                and (wanted is None or record.subject_id in wanted)
            ]

    def verify_audit(self) -> VerificationResult:
        with self._locked():
            return self._stores.audit.verify()

    def ungated(self) -> tuple[str, ...]:
        with self._locked():
            return verify_no_ungated_execution(self._stores.audit, tenant_id=self.tenant_id)

    def explain(self, node_id: str, *, advisory_id: str | None = None):
        """``explain_node`` on the current graph, or inside one advisory's scope.

        The scope matters more than it looks. On the whole graph an organisation is
        reached by many risk sources, most of them nearer than four hops, so the
        advisory's fourth-order path is crowded out of the top five — correctly, as
        a ranking, and uselessly for an analyst reviewing *that* advisory. Inside the
        advisory's scope the only source is its package, so the path is the answer.
        Scores and model attributions are the whole graph's either way.
        """
        from dataclasses import replace

        from sentinel.graph.explain import explain_node

        with self._locked():
            graph = self.graph
            if advisory_id is not None:
                published = self._advisories.get(advisory_id)
                if published is None:
                    raise NotFound(f"no published advisory {advisory_id!r}")
                graph = published.subgraph
            if node_id not in set(graph.node_ids()):
                raise NotFound(f"no node {node_id!r}")
            index = self.graph.index_of(node_id)
            explanation = explain_node(graph, node_id, risk_score=float(self.scores[index]))
            share = float(self.shares[index])
            return replace(explanation, own_feature_share=1.0 - share, neighbourhood_share=share)

    def failed_actions(self) -> list[tuple[ThreadRef, Any]]:
        with self._locked():
            return [
                (ref, action)
                for ref, state in self.states()
                for action in state.actions
                if action.approval_status is ApprovalStatus.FAILED
            ]

    def refusal_records(self) -> list[AuditRecord]:
        with self._locked():
            return [
                record
                for record in self.records()
                if record.event_type is AuditEventType.GUARDRAIL_BLOCKED
            ]


def scenario_threads(workspace: Workspace, name: ScenarioName | str) -> list[ThreadRef]:
    return [ref for ref in workspace.refs() if _tag_value(ref.scenario) == _tag_value(name)]


@dataclass(frozen=True, slots=True)
class ScenarioProgress:
    """Where one scenario stands, derived from its threads' checkpointed states."""

    name: ScenarioName | str
    launched: bool
    total: int
    waiting: int
    running: int
    finished: int
    failed_runs: int
    failed_actions: int
    decisions: int
    steps: tuple[str, ...] = field(default=())

    @property
    def complete(self) -> bool:
        """F-10's bar: launched, and every run reached a terminal state without crashing.

        A FAILED *action* inside a COMPLETED run still completes the scenario — the
        guardrail refusal in scenario 1 is a designed outcome and the run ends
        cleanly. A FAILED *run* (a node raised) does not.
        """
        return (
            self.launched
            and self.total > 0
            and self.finished == self.total
            and self.failed_runs == 0
        )


def scenario_progress(workspace: Workspace, name: ScenarioName | str) -> ScenarioProgress:
    launched = any(_tag_value(k) == _tag_value(name) for k in workspace.scenario_runs())
    refs = scenario_threads(workspace, name)
    waiting = running = finished = failed_runs = failed_actions = decisions = 0
    for ref in refs:
        state = workspace.state(ref.thread_id)
        if state.is_waiting:
            waiting += 1
        elif state.status.is_terminal:
            finished += 1
            failed_runs += state.status is IncidentStatus.FAILED
        else:
            running += 1
        for action in state.actions:
            failed_actions += action.approval_status is ApprovalStatus.FAILED
            decisions += action.requires_human_approval and action.approved_by is not None
    return ScenarioProgress(
        name=name,
        launched=launched,
        total=len(refs),
        waiting=waiting,
        running=running,
        finished=finished,
        failed_runs=failed_runs,
        failed_actions=failed_actions,
        decisions=decisions,
    )
