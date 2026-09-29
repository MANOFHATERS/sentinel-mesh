"""The orchestration state machine (PRD F-04, Figure 3).

What this is, and why it is not LangGraph
-----------------------------------------
PRD Section 5.6 names LangGraph, and LangGraph is a good library. This module is
a deliberate substitution, for the same reason :mod:`sentinel.ml.nn` is a
hand-written engine instead of PyTorch, and the reason is worth stating plainly
because "we wrote our own framework" is usually the wrong answer:

*   **F-04's acceptance criterion is this module's semantics.** *"Any node can
    pause for human input and resume with full context intact"* is not a property
    of the agents; it is a property of the interrupt/checkpoint/resume loop. Using
    a library for it would mean the project's central orchestration claim is
    tested by mocking the library that implements it. Owning ~350 lines means
    ``test_runtime.py`` can assert the claim directly — and does, by resuming at
    *every* node in the graph and checking the finished state is hash-identical to
    an uninterrupted run.
*   **The checkpoint must be a contract, not a pickle.** Section 5.7's
    tamper-evidence requirement reaches the orchestrator: a resumable state
    carries approval status, so a checkpoint store that round-trips through
    msgpack without a verifiable hash is an approval-forgery path. The state here
    is a frozen pydantic contract that canonicalizes, hashes and revalidates on
    load, which is a different storage contract than a general workflow engine
    offers.
*   **Untrusted text must survive the round trip as untrusted.**
    ``Alert.raw_payload`` is an :class:`~sentinel.core.untrusted.UntrustedText`.
    Any checkpointer that serializes it to ``str`` and restores it as ``str`` has
    silently deleted the type that prevents it being interpolated into a prompt.
    :class:`~sentinel.agents.state.IncidentState` restores it through the schema's
    own validator, so the property is preserved by construction.

The API is deliberately shaped like LangGraph's — ``add_node``, ``add_edge``,
``add_conditional_edges``, ``set_entry_point``, ``compile(interrupt_before=...)``,
``invoke``/``resume`` — so the substitution stays a substitution. Porting to
LangGraph is a rewrite of this file and nothing else, which is the property that
makes the deviation defensible rather than merely convenient.

Execution model
---------------
A run is a loop over ``state.cursor``:

1.  If the cursor is :data:`~sentinel.agents.state.END`, the run is finished.
2.  If the cursor names a node in ``interrupt_before`` and no
    :class:`~sentinel.agents.state.HumanDecision` is pending, the run checkpoints
    and returns ``AWAITING_APPROVAL``. **The node has not run.** This is the
    direction that matters: interrupting *after* the containment node has already
    proposed-and-executed is not a gate.
3.  Otherwise the node runs, its step is recorded, the router picks the next
    cursor, and a checkpoint is written.

Failure is a state, not an exception. A node that raises produces a terminal
``FAILED`` checkpoint carrying the error, because a run that vanishes mid-flight
leaves a proposed action with no record of why nothing happened to it. The
exception type is preserved in the message; the traceback is not persisted,
since checkpoints are read back by a dashboard and a traceback is an information
leak with no operational use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from sentinel.agents.checkpoint import Checkpoint, Checkpointer, InMemoryCheckpointer
from sentinel.agents.state import (
    END,
    HumanDecision,
    IncidentState,
    IncidentStatus,
    Interrupt,
    StepOutcome,
    StepRecord,
)
from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import SentinelError
from sentinel.core.schemas import GENESIS_HASH

__all__ = [
    "DEFAULT_STEP_BUDGET",
    "CompiledGraph",
    "GraphSpec",
    "Node",
    "OrchestrationError",
    "RunContext",
    "RunResult",
]

DEFAULT_STEP_BUDGET: Final[int] = 64
"""Maximum node executions in one ``invoke``/``resume`` call.

A budget rather than a cycle detector. The PRD's graph is a DAG plus one
re-entry (investigation can send an incident back for re-triage), so "a node ran
twice" is legal and "a node ran sixty-four times" is a routing bug. A budget
converts an infinite loop into a ``FAILED`` state with an explanation instead of
a hung worker holding an un-actioned containment proposal.
"""


class OrchestrationError(SentinelError):
    """The graph is malformed, or a run was asked for something incoherent."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """Everything a node may use that is not the state.

    Nodes take ``(state, ctx)`` and return a new state. They do not take a
    ``**kwargs`` bag: a node's dependencies are visible here, so a node that
    reaches for a network client or a global model has to add a field and be
    reviewed for it.
    """

    clock: Clock
    thread_id: str
    audit: HashChainedAuditLog | None = None
    decision: HumanDecision | None = None
    #: Free-form, node-specific configuration. Read-only by convention and by
    #: type: a node that needs to persist something puts it in the state, where
    #: it will be checkpointed, rather than here, where it will not.
    extras: Mapping[str, object] = field(default_factory=dict)

    def now(self) -> datetime:
        return self.clock.now()

    def require_decision(self) -> HumanDecision:
        """The human decision this node was resumed with, or a refusal.

        Called by the approval gate. Raising here rather than defaulting to
        "denied" is deliberate: a gate that silently treats a missing decision as
        a denial is indistinguishable from a gate that was never reached, and
        F-08 needs the difference to be loud.
        """
        if self.decision is None:
            raise OrchestrationError(
                "this node requires a human decision but the run was not resumed with "
                "one; it should have been reached through interrupt_before"
            )
        return self.decision

    def with_decision(self, decision: HumanDecision | None) -> RunContext:
        return RunContext(
            clock=self.clock,
            thread_id=self.thread_id,
            audit=self.audit,
            decision=decision,
            extras=self.extras,
        )


Node = Callable[[IncidentState, RunContext], IncidentState]
"""A node: pure in the state, effectful only through ``ctx``."""

Router = Callable[[IncidentState], str]
"""Picks the next node's *key*, which ``add_conditional_edges`` maps to a node."""


@dataclass(slots=True)
class GraphSpec:
    """A graph under construction. :meth:`compile` validates it once, at build time."""

    _nodes: dict[str, Node] = field(default_factory=dict)
    _edges: dict[str, str] = field(default_factory=dict)
    _conditional: dict[str, tuple[Router, dict[str, str]]] = field(default_factory=dict)
    _entry: str | None = None

    def add_node(self, name: str, node: Node) -> GraphSpec:
        if not name or name == END:
            raise OrchestrationError(f"invalid node name {name!r}")
        if name in self._nodes:
            raise OrchestrationError(f"node {name!r} is already defined")
        self._nodes[name] = node
        return self

    def add_edge(self, source: str, target: str) -> GraphSpec:
        self._reject_double_wiring(source)
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self, source: str, router: Router, mapping: Mapping[str, str]
    ) -> GraphSpec:
        """Route from ``source`` by key. ``mapping`` is exhaustive by construction.

        The mapping is required rather than allowing the router to return a node
        name directly. A router that returns node names couples every routing
        decision to the graph's current shape, so renaming a node silently
        reroutes an incident instead of failing at compile time.
        """
        self._reject_double_wiring(source)
        if not mapping:
            raise OrchestrationError(f"conditional edges from {source!r} map nothing")
        self._conditional[source] = (router, dict(mapping))
        return self

    def _reject_double_wiring(self, source: str) -> None:
        if source in self._edges or source in self._conditional:
            raise OrchestrationError(
                f"node {source!r} already has outgoing edges; a node with two routing "
                "rules has an order-dependent one"
            )

    def set_entry_point(self, name: str) -> GraphSpec:
        self._entry = name
        return self

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    def compile(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        interrupt_before: Iterable[str] = (),
        step_budget: int = DEFAULT_STEP_BUDGET,
    ) -> CompiledGraph:
        """Validate the graph and freeze it. Every structural error surfaces here."""
        if self._entry is None:
            raise OrchestrationError("graph has no entry point")
        if self._entry not in self._nodes:
            raise OrchestrationError(f"entry point {self._entry!r} is not a node")

        targets: list[tuple[str, str]] = list(self._edges.items())
        for source, (_router, mapping) in self._conditional.items():
            targets.extend((source, target) for target in mapping.values())
        for source, target in targets:
            if source not in self._nodes:
                raise OrchestrationError(f"edge from unknown node {source!r}")
            if target != END and target not in self._nodes:
                raise OrchestrationError(f"edge {source!r} -> unknown node {target!r}")

        dangling = sorted(
            name
            for name in self._nodes
            if name not in self._edges and name not in self._conditional
        )
        if dangling:
            raise OrchestrationError(
                f"nodes with no outgoing edge: {dangling}. Every node must route "
                f"somewhere, even if that is {END!r} — a node that falls off the end "
                "leaves an incident in RUNNING forever"
            )

        reachable = self._reachable_from(self._entry)
        unreachable = sorted(set(self._nodes) - reachable)
        if unreachable:
            raise OrchestrationError(
                f"unreachable nodes: {unreachable}. An unreachable agent is a guardrail "
                "that looks wired up and never runs"
            )

        interrupts = frozenset(interrupt_before)
        unknown = sorted(interrupts - set(self._nodes))
        if unknown:
            raise OrchestrationError(f"interrupt_before names unknown nodes: {unknown}")
        if step_budget < 1:
            raise OrchestrationError("step_budget must be at least 1")

        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional={k: (r, dict(m)) for k, (r, m) in self._conditional.items()},
            entry=self._entry,
            interrupt_before=interrupts,
            checkpointer=checkpointer if checkpointer is not None else InMemoryCheckpointer(),
            step_budget=step_budget,
        )

    def _reachable_from(self, start: str) -> set[str]:
        seen: set[str] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            if current in seen or current == END:
                continue
            seen.add(current)
            if current in self._edges:
                frontier.append(self._edges[current])
            elif current in self._conditional:
                frontier.extend(self._conditional[current][1].values())
        return seen


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of one ``invoke``/``resume`` call."""

    state: IncidentState
    checkpoints: tuple[Checkpoint, ...]
    steps_executed: int

    @property
    def interrupted(self) -> bool:
        return self.state.is_waiting

    @property
    def final(self) -> bool:
        return self.state.status.is_terminal


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    """A validated graph. Immutable, so one compiled graph serves every incident."""

    nodes: Mapping[str, Node]
    edges: Mapping[str, str]
    conditional: Mapping[str, tuple[Router, Mapping[str, str]]]
    entry: str
    interrupt_before: frozenset[str]
    checkpointer: Checkpointer
    step_budget: int

    # --- running -------------------------------------------------------------- #

    def invoke(
        self,
        state: IncidentState,
        *,
        clock: Clock | None = None,
        audit: HashChainedAuditLog | None = None,
        thread_id: str | None = None,
        extras: Mapping[str, object] | None = None,
    ) -> RunResult:
        """Start a run at the entry point and execute until END or an interrupt."""
        if state.status is not IncidentStatus.RUNNING:
            raise OrchestrationError(
                f"cannot start a run from status {state.status.value}; invoke() is for "
                "new incidents and resume() is for waiting ones"
            )
        thread = thread_id or state.incident_id
        if self.checkpointer.latest(thread) is not None:
            raise OrchestrationError(
                f"thread {thread!r} already has checkpoints; starting it again would "
                "fork the incident's history. Use resume()."
            )
        ctx = RunContext(
            clock=clock or SystemClock(),
            thread_id=thread,
            audit=audit,
            extras=dict(extras or {}),
        )
        started = state.routed_to(self.entry, at=ctx.now())
        return self._drive(started, ctx, step=0)

    def resume(
        self,
        thread_id: str,
        decision: HumanDecision,
        *,
        clock: Clock | None = None,
        audit: HashChainedAuditLog | None = None,
        extras: Mapping[str, object] | None = None,
    ) -> RunResult:
        """Continue a waiting run, supplying the human's answer.

        The state comes from the checkpoint store, never from the caller. A
        ``resume(state, decision)`` signature would let a caller resume from a
        state they constructed — which is a complete bypass of every invariant
        the checkpoint chain exists to enforce.
        """
        head = self.checkpointer.latest(thread_id)
        if head is None:
            raise OrchestrationError(f"no checkpoint for thread {thread_id!r}")
        state = head.state
        if not state.is_waiting:
            raise OrchestrationError(
                f"thread {thread_id!r} is {state.status.value}, not waiting for a decision"
            )
        ctx = RunContext(
            clock=clock or SystemClock(),
            thread_id=thread_id,
            audit=audit,
            decision=decision,
            extras=dict(extras or {}),
        )
        assert state.interrupt is not None  # guaranteed by IncidentState._consistency
        resumed = state.resumed(cursor=state.interrupt.node, at=ctx.now())
        return self._drive(resumed, ctx, step=head.step + 1)

    # --- the loop ------------------------------------------------------------- #

    def _drive(self, state: IncidentState, ctx: RunContext, *, step: int) -> RunResult:
        written: list[Checkpoint] = []
        executed = 0
        decision = ctx.decision

        while True:
            if state.cursor == END:
                if not state.status.is_terminal:
                    state = state.finished(IncidentStatus.COMPLETED, at=ctx.now())
                written.append(self._save(state, ctx, step=step))
                return RunResult(state=state, checkpoints=tuple(written), steps_executed=executed)

            node_name = state.cursor
            if node_name in self.interrupt_before and decision is None:
                waiting = state.waiting_on(
                    Interrupt(
                        node=node_name,
                        reason=_interrupt_reason(state, node_name),
                        subject_id=_interrupt_subject(state),
                        requested_at=ctx.now(),
                    )
                )
                written.append(self._save(waiting, ctx, step=step))
                return RunResult(
                    state=waiting, checkpoints=tuple(written), steps_executed=executed
                )

            if executed >= self.step_budget:
                failed = state.finished(
                    IncidentStatus.FAILED,
                    at=ctx.now(),
                    error=(
                        f"step budget of {self.step_budget} exhausted at node "
                        f"{node_name!r}; the route is cycling"
                    ),
                )
                written.append(self._save(failed, ctx, step=step))
                return RunResult(state=failed, checkpoints=tuple(written), steps_executed=executed)

            node = self.nodes[node_name]
            node_ctx = ctx.with_decision(decision)
            # A decision is consumed by exactly one node. Leaving it on the context
            # would let a later gate in the same run reuse an approval that was
            # granted for a different action.
            decision = None
            started_at = ctx.now()

            try:
                produced = node(state, node_ctx)
            except Exception as exc:
                ended_at = ctx.now()
                recorded = state.with_step(
                    StepRecord(
                        node=node_name,
                        started_at=started_at,
                        ended_at=ended_at,
                        outcome=StepOutcome.ERROR,
                        duration_ms=_elapsed_ms(started_at, ended_at),
                        detail=f"{type(exc).__name__}: {exc}"[:2000],
                    )
                )
                failed = recorded.finished(
                    IncidentStatus.FAILED,
                    at=ended_at,
                    error=f"node {node_name!r} raised {type(exc).__name__}: {exc}"[:2000],
                )
                written.append(self._save(failed, ctx, step=step))
                return RunResult(state=failed, checkpoints=tuple(written), steps_executed=executed)

            executed += 1
            ended_at = ctx.now()
            if produced.incident_id != state.incident_id:
                raise OrchestrationError(
                    f"node {node_name!r} returned a state for a different incident"
                )

            if produced.is_waiting:
                # A node may raise its own interrupt (the approval gate does not,
                # but a future human-in-the-loop node might). Record the step and
                # stop; the cursor stays on whatever the node set.
                recorded = produced.with_step(
                    StepRecord(
                        node=node_name,
                        started_at=started_at,
                        ended_at=ended_at,
                        outcome=StepOutcome.INTERRUPTED,
                        duration_ms=_elapsed_ms(started_at, ended_at),
                        detail=produced.interrupt.reason if produced.interrupt else "",
                    )
                )
                written.append(self._save(recorded, ctx, step=step))
                return RunResult(
                    state=recorded, checkpoints=tuple(written), steps_executed=executed
                )

            recorded = produced.with_step(
                StepRecord(
                    node=node_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    outcome=StepOutcome.OK,
                    duration_ms=_elapsed_ms(started_at, ended_at),
                )
            )
            if recorded.status.is_terminal:
                # A node that finishes the run (a dismissal, say) gets one terminal
                # checkpoint, not two: falling through to the loop top would write a
                # second identical one and make the history lie about how many steps
                # the incident took.
                written.append(self._save(recorded, ctx, step=step))
                return RunResult(
                    state=recorded, checkpoints=tuple(written), steps_executed=executed
                )
            state = recorded.routed_to(self._next(node_name, recorded), at=ended_at)
            written.append(self._save(state, ctx, step=step))
            step += 1

    def _next(self, node_name: str, state: IncidentState) -> str:
        if node_name in self.edges:
            return self.edges[node_name]
        router, mapping = self.conditional[node_name]
        key = router(state)
        if key not in mapping:
            raise OrchestrationError(
                f"router for {node_name!r} returned {key!r}, which is not one of "
                f"{sorted(mapping)}"
            )
        return mapping[key]

    def _save(self, state: IncidentState, ctx: RunContext, *, step: int) -> Checkpoint:
        head = self.checkpointer.latest(ctx.thread_id)
        checkpoint = Checkpoint.of(
            thread_id=ctx.thread_id,
            step=step,
            node=state.cursor,
            state=state,
            parent_hash=GENESIS_HASH if head is None else head.link_hash,
            audit_head=None if ctx.audit is None else ctx.audit.head(),
        )
        self.checkpointer.put(checkpoint)
        return checkpoint

    # --- inspection ----------------------------------------------------------- #

    def state_of(self, thread_id: str) -> IncidentState | None:
        head = self.checkpointer.latest(thread_id)
        return None if head is None else head.state

    def pending(self) -> tuple[IncidentState, ...]:
        """Every run currently waiting for a human. The approval queue's data source."""
        waiting: list[IncidentState] = []
        for thread in self.checkpointer.threads():
            head = self.checkpointer.latest(thread)
            if head is not None and head.state.is_waiting:
                waiting.append(head.state)
        return tuple(waiting)


def _elapsed_ms(start: datetime, end: datetime) -> float:
    """Elapsed milliseconds, clamped at zero.

    Clamped rather than validated because a :class:`~sentinel.core.clock.Clock`
    is an injected dependency and a frozen clock legitimately returns the same
    instant twice. A negative duration means the clock went backwards, which is
    the wall-clock adjustment ``StepRecord.duration_ms`` exists to survive; the
    orchestrator is not the right place to fail a run over it.
    """
    return max(0.0, (end - start).total_seconds() * 1000.0)


def _interrupt_reason(state: IncidentState, node: str) -> str:
    action = state.actions[-1] if state.actions else None
    if action is None:
        return f"node {node!r} requires human input before it may run"
    return (
        f"{action.action_type.value} on {action.target} requires approval "
        f"(trust tier {action.risk_tier.value}): {action.rationale}"
    )[:1000]


def _interrupt_subject(state: IncidentState) -> str:
    """What the question is about: the newest action, else the alert.

    The newest action rather than the newest *pending* one: a gate reached with
    no pending action is a routing bug, and pointing the interrupt at an older
    already-decided action would hide it behind a plausible-looking question.
    """
    return state.actions[-1].action_id if state.actions else state.alert.alert_id
