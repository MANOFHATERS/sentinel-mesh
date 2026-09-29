"""The orchestration runtime (PRD F-04).

The headline criterion — *"any node can pause for human input and resume with
full context intact"* — is tested by construction rather than by example:
``TestResumeIsFaithful`` parametrises over **every** node in a graph, interrupts
there, resumes, and asserts the finished state is hash-identical to a run that
was never interrupted. A test that checks one hand-picked interrupt point proves
that one point works.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.agents.checkpoint import (
    Checkpoint,
    CheckpointError,
    InMemoryCheckpointer,
    SqliteCheckpointer,
)
from sentinel.agents.runtime import (
    CompiledGraph,
    GraphSpec,
    OrchestrationError,
    RunContext,
)
from sentinel.agents.state import (
    END,
    HumanDecision,
    IncidentState,
    IncidentStatus,
    Interrupt,
    StepOutcome,
)
from sentinel.core.clock import FrozenClock
from sentinel.core.schemas import GENESIS_HASH, Alert, RiskTier, Severity

FIXED_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures: a toy graph whose nodes leave a visible trace in the state
# --------------------------------------------------------------------------- #


def _marker_node(name: str):
    """A node that records its own name as an evidence ref.

    Evidence rather than history: the runtime writes history itself, so a node
    that only appears there proves nothing about the node having run.
    """

    def node(state: IncidentState, ctx: RunContext) -> IncidentState:
        from sentinel.core.schemas import Evidence, EvidenceKind

        return state.with_evidence(
            (
                Evidence(
                    kind=EvidenceKind.MODEL_OUTPUT,
                    ref=f"node:{name}",
                    excerpt=f"{name} ran",
                    relevance=1.0,
                ),
            ),
            at=ctx.now(),
        )

    return node


@pytest.fixture
def state(alert: Alert) -> IncidentState:
    return IncidentState(
        incident_id="incident-1",
        tenant_id=alert.tenant_id,
        alert=alert,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


def _linear_spec() -> GraphSpec:
    spec = GraphSpec()
    for name in ("alpha", "beta", "gamma"):
        spec.add_node(name, _marker_node(name))
    spec.add_edge("alpha", "beta")
    spec.add_edge("beta", "gamma")
    spec.add_edge("gamma", END)
    spec.set_entry_point("alpha")
    return spec


@pytest.fixture
def linear() -> CompiledGraph:
    return _linear_spec().compile(checkpointer=InMemoryCheckpointer())


def _decision(*, approved: bool = True) -> HumanDecision:
    return HumanDecision(approver="analyst@acme", approved=approved, decided_at=FIXED_NOW)


# --------------------------------------------------------------------------- #
# Graph construction is validated once, loudly
# --------------------------------------------------------------------------- #


class TestGraphValidation:
    def test_entry_point_is_required(self) -> None:
        spec = GraphSpec().add_node("a", _marker_node("a")).add_edge("a", END)
        with pytest.raises(OrchestrationError, match="no entry point"):
            spec.compile()

    def test_entry_point_must_be_a_node(self) -> None:
        spec = GraphSpec().add_node("a", _marker_node("a")).add_edge("a", END)
        spec.set_entry_point("nope")
        with pytest.raises(OrchestrationError, match="not a node"):
            spec.compile()

    def test_edge_to_unknown_node_is_rejected(self) -> None:
        spec = GraphSpec().add_node("a", _marker_node("a")).add_edge("a", "ghost")
        spec.set_entry_point("a")
        with pytest.raises(OrchestrationError, match="unknown node 'ghost'"):
            spec.compile()

    def test_node_with_no_outgoing_edge_is_rejected(self) -> None:
        spec = GraphSpec()
        spec.add_node("a", _marker_node("a")).add_node("b", _marker_node("b"))
        spec.add_edge("a", "b").set_entry_point("a")
        with pytest.raises(OrchestrationError, match="no outgoing edge"):
            spec.compile()

    def test_unreachable_node_is_rejected(self) -> None:
        """An orphaned guardrail node is the failure this catches."""
        spec = GraphSpec()
        spec.add_node("a", _marker_node("a")).add_node("orphan", _marker_node("orphan"))
        spec.add_edge("a", END).add_edge("orphan", END).set_entry_point("a")
        with pytest.raises(OrchestrationError, match="unreachable nodes"):
            spec.compile()

    def test_a_node_cannot_have_two_routing_rules(self) -> None:
        spec = GraphSpec().add_node("a", _marker_node("a")).add_edge("a", END)
        with pytest.raises(OrchestrationError, match="already has outgoing edges"):
            spec.add_conditional_edges("a", lambda _s: "x", {"x": END})

    def test_duplicate_node_names_are_rejected(self) -> None:
        spec = GraphSpec().add_node("a", _marker_node("a"))
        with pytest.raises(OrchestrationError, match="already defined"):
            spec.add_node("a", _marker_node("a"))

    def test_end_is_not_a_legal_node_name(self) -> None:
        with pytest.raises(OrchestrationError, match="invalid node name"):
            GraphSpec().add_node(END, _marker_node("x"))

    def test_interrupt_before_must_name_real_nodes(self) -> None:
        with pytest.raises(OrchestrationError, match="unknown nodes"):
            _linear_spec().compile(interrupt_before=["ghost"])

    def test_conditional_edges_need_a_mapping(self) -> None:
        spec = GraphSpec().add_node("a", _marker_node("a"))
        with pytest.raises(OrchestrationError, match="map nothing"):
            spec.add_conditional_edges("a", lambda _s: "x", {})

    def test_step_budget_must_be_positive(self) -> None:
        with pytest.raises(OrchestrationError, match="step_budget"):
            _linear_spec().compile(step_budget=0)


# --------------------------------------------------------------------------- #
# Straight-line execution
# --------------------------------------------------------------------------- #


class TestExecution:
    def test_runs_every_node_in_order(self, linear: CompiledGraph, state: IncidentState) -> None:
        result = linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert result.state.visited == ("alpha", "beta", "gamma")
        assert [e.ref for e in result.state.evidence] == [
            "node:alpha",
            "node:beta",
            "node:gamma",
        ]

    def test_finishes_completed_at_end(self, linear: CompiledGraph, state: IncidentState) -> None:
        result = linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert result.state.status is IncidentStatus.COMPLETED
        assert result.state.cursor == END
        assert result.final and not result.interrupted

    def test_every_step_is_checkpointed(
        self, linear: CompiledGraph, state: IncidentState
    ) -> None:
        result = linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        history = linear.checkpointer.history("incident-1")
        # one per node, plus the terminal checkpoint
        assert len(history) == 4
        assert [cp.step for cp in history] == [0, 1, 2, 3]
        assert history[-1].state == result.state

    def test_the_input_state_is_not_mutated(
        self, linear: CompiledGraph, state: IncidentState
    ) -> None:
        before = state.canonical_hash()
        linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert state.canonical_hash() == before

    def test_conditional_routing_picks_the_mapped_node(self, state: IncidentState) -> None:
        spec = GraphSpec()
        for name in ("start", "left", "right"):
            spec.add_node(name, _marker_node(name))
        spec.add_conditional_edges(
            "start", lambda _s: "go_right", {"go_left": "left", "go_right": "right"}
        )
        spec.add_edge("left", END).add_edge("right", END).set_entry_point("start")
        result = spec.compile().invoke(state, clock=FrozenClock(FIXED_NOW))
        assert result.state.visited == ("start", "right")

    def test_a_router_returning_an_unmapped_key_is_an_error(self, state: IncidentState) -> None:
        spec = GraphSpec()
        spec.add_node("start", _marker_node("start"))
        spec.add_conditional_edges("start", lambda _s: "sideways", {"down": END})
        spec.set_entry_point("start")
        graph = spec.compile()
        with pytest.raises(OrchestrationError, match="not one of"):
            graph.invoke(state, clock=FrozenClock(FIXED_NOW))

    def test_steps_record_their_node_and_outcome(
        self, linear: CompiledGraph, state: IncidentState
    ) -> None:
        result = linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert all(step.outcome is StepOutcome.OK for step in result.state.history)
        assert result.state.steps_for("beta")[0].node == "beta"

    def test_durations_come_from_the_injected_clock(self, state: IncidentState) -> None:
        clock = FrozenClock(FIXED_NOW)

        def slow(s: IncidentState, ctx: RunContext) -> IncidentState:
            clock.advance(2.5)
            return s

        spec = GraphSpec().add_node("slow", slow).add_edge("slow", END).set_entry_point("slow")
        result = spec.compile().invoke(state, clock=clock)
        assert result.state.history[0].duration_ms == pytest.approx(2500.0)


# --------------------------------------------------------------------------- #
# F-04: interrupt and resume
# --------------------------------------------------------------------------- #


class TestInterrupt:
    def test_the_interrupted_node_has_not_run(self, state: IncidentState) -> None:
        """The direction of ``interrupt_before`` is the whole guarantee.

        Pausing *after* the containment node has proposed and executed is not a
        gate, so this asserts the node left no trace at all.
        """
        graph = _linear_spec().compile(interrupt_before=["beta"])
        result = graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert result.interrupted
        assert result.state.visited == ("alpha",)
        assert [e.ref for e in result.state.evidence] == ["node:alpha"]

    def test_interrupt_records_what_it_waits_for(self, state: IncidentState) -> None:
        graph = _linear_spec().compile(interrupt_before=["beta"])
        result = graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert result.state.interrupt is not None
        assert result.state.interrupt.node == "beta"
        assert result.state.interrupt.subject_id == state.alert.alert_id
        assert result.state.status is IncidentStatus.AWAITING_APPROVAL

    def test_pending_lists_the_waiting_run(self, state: IncidentState) -> None:
        graph = _linear_spec().compile(interrupt_before=["beta"])
        graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert [s.incident_id for s in graph.pending()] == ["incident-1"]

    def test_resume_clears_the_interrupt_and_continues(self, state: IncidentState) -> None:
        graph = _linear_spec().compile(interrupt_before=["beta"])
        graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        result = graph.resume("incident-1", _decision(), clock=FrozenClock(FIXED_NOW))
        assert result.state.interrupt is None
        assert result.state.visited == ("alpha", "beta", "gamma")
        assert result.state.status is IncidentStatus.COMPLETED

    def test_resume_on_an_unknown_thread_is_refused(self, linear: CompiledGraph) -> None:
        with pytest.raises(OrchestrationError, match="no checkpoint"):
            linear.resume("ghost", _decision())

    def test_resume_on_a_running_thread_is_refused(
        self, linear: CompiledGraph, state: IncidentState
    ) -> None:
        linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        with pytest.raises(OrchestrationError, match="not waiting"):
            linear.resume("incident-1", _decision())

    def test_invoking_an_existing_thread_twice_is_refused(
        self, linear: CompiledGraph, state: IncidentState
    ) -> None:
        """Re-invoking would fork the incident's history behind the checkpoint chain."""
        linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        with pytest.raises(OrchestrationError, match="already has checkpoints"):
            linear.invoke(state, clock=FrozenClock(FIXED_NOW))

    def test_invoke_refuses_a_non_running_state(self, state: IncidentState) -> None:
        waiting = state.waiting_on(
            Interrupt(node="beta", reason="x", subject_id="a", requested_at=FIXED_NOW)
        )
        graph = _linear_spec().compile()
        with pytest.raises(OrchestrationError, match="cannot start a run"):
            graph.invoke(waiting, clock=FrozenClock(FIXED_NOW))

    def test_the_decision_reaches_only_the_gated_node(self, state: IncidentState) -> None:
        """One approval authorises one action, not every gate in the run."""
        seen: dict[str, HumanDecision | None] = {}

        def watcher(name: str):
            def node(s: IncidentState, ctx: RunContext) -> IncidentState:
                seen[name] = ctx.decision
                return s

            return node

        spec = GraphSpec()
        spec.add_node("alpha", watcher("alpha")).add_node("beta", watcher("beta"))
        spec.add_edge("alpha", "beta").add_edge("beta", END).set_entry_point("alpha")
        graph = spec.compile(interrupt_before=["alpha"])
        graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        graph.resume("incident-1", _decision(), clock=FrozenClock(FIXED_NOW))
        assert seen["alpha"] is not None
        assert seen["beta"] is None

    def test_a_second_gate_in_one_run_interrupts_again(self, state: IncidentState) -> None:
        graph = _linear_spec().compile(interrupt_before=["alpha", "gamma"])
        first = graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert first.state.interrupt is not None and first.state.interrupt.node == "alpha"
        second = graph.resume("incident-1", _decision(), clock=FrozenClock(FIXED_NOW))
        assert second.interrupted
        assert second.state.interrupt is not None
        assert second.state.interrupt.node == "gamma"
        third = graph.resume("incident-1", _decision(), clock=FrozenClock(FIXED_NOW))
        assert third.state.status is IncidentStatus.COMPLETED


class TestResumeIsFaithful:
    """F-04: *resume with full context intact*, proven at every interrupt point."""

    @pytest.mark.parametrize("gate", ["alpha", "beta", "gamma"])
    def test_interrupting_anywhere_yields_the_same_final_state(
        self, state: IncidentState, gate: str
    ) -> None:
        uninterrupted = _linear_spec().compile().invoke(state, clock=FrozenClock(FIXED_NOW))

        graph = _linear_spec().compile(interrupt_before=[gate])
        paused = graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        assert paused.interrupted
        resumed = graph.resume("incident-1", _decision(), clock=FrozenClock(FIXED_NOW))

        assert resumed.state.canonical_hash() == uninterrupted.state.canonical_hash()

    @pytest.mark.parametrize("gate", ["alpha", "beta", "gamma"])
    def test_resume_survives_a_new_process(
        self, state: IncidentState, tmp_path, gate: str
    ) -> None:
        """The only honest test of "intact": a fresh object graph over a real file.

        The compiled graph, the checkpointer and every in-memory object are
        rebuilt from scratch between pause and resume, so nothing but the bytes on
        disk carries the run forward.
        """
        path = tmp_path / "checkpoints.sqlite"
        expected = _linear_spec().compile().invoke(state, clock=FrozenClock(FIXED_NOW))

        with SqliteCheckpointer(path) as store:
            paused = _linear_spec().compile(
                checkpointer=store, interrupt_before=[gate]
            ).invoke(state, clock=FrozenClock(FIXED_NOW))
            assert paused.interrupted

        with SqliteCheckpointer(path) as reopened:
            graph = _linear_spec().compile(checkpointer=reopened, interrupt_before=[gate])
            resumed = graph.resume("incident-1", _decision(), clock=FrozenClock(FIXED_NOW))

        assert resumed.state.canonical_hash() == expected.state.canonical_hash()

    def test_untrusted_payload_survives_the_round_trip_as_untrusted(
        self, state: IncidentState, tmp_path
    ) -> None:
        """A checkpointer that degrades ``UntrustedText`` to ``str`` deletes a control.

        The type is what stops the payload being interpolated into a prompt, so
        losing it in storage would be a silent removal of the prompt-injection
        defence rather than a serialization detail.
        """
        from sentinel.core.untrusted import UntrustedText

        path = tmp_path / "checkpoints.sqlite"
        with SqliteCheckpointer(path) as store:
            _linear_spec().compile(checkpointer=store, interrupt_before=["beta"]).invoke(
                state, clock=FrozenClock(FIXED_NOW)
            )
        with SqliteCheckpointer(path) as reopened:
            restored = reopened.latest("incident-1")
        assert restored is not None
        payload = restored.state.alert.raw_payload
        assert isinstance(payload, UntrustedText)
        assert payload.raw == state.alert.raw_payload.raw
        assert "untrusted" in str(payload)


class _Crash(BaseException):
    """A process death: not an ``Exception``, so no node or runtime handler catches it."""


def _crashing_spec(crash_at: str, fired: dict[str, int]) -> GraphSpec:
    """The linear graph, except ``crash_at`` kills the process the first time it runs."""
    spec = GraphSpec()
    for name in ("alpha", "beta", "gamma"):
        inner = _marker_node(name)

        def node(state, ctx, _name=name, _inner=inner):
            fired[_name] = fired.get(_name, 0) + 1
            if _name == crash_at and fired[_name] == 1:
                raise _Crash
            return _inner(state, ctx)

        spec.add_node(name, node)
    spec.add_edge("alpha", "beta")
    spec.add_edge("beta", "gamma")
    spec.add_edge("gamma", END)
    spec.set_entry_point("alpha")
    return spec


class TestRecoverAfterACrash:
    """Part 4: ``resume`` continues a pause; ``recover`` continues a process death."""

    def test_a_crash_in_the_entry_node_leaves_nothing_to_recover_so_invoke_again(
        self, state: IncidentState
    ) -> None:
        # No checkpoint precedes the first node, so the thread has no history to
        # fork and invoke() is the recovery; recover() says so rather than guessing.
        fired: dict[str, int] = {}
        graph = _crashing_spec("alpha", fired).compile()
        with pytest.raises(_Crash):
            graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        with pytest.raises(OrchestrationError, match="no checkpoint"):
            graph.recover("incident-1")
        assert graph.invoke(state, clock=FrozenClock(FIXED_NOW)).state.status is (
            IncidentStatus.COMPLETED
        )

    @pytest.mark.parametrize("crash_at", ["beta", "gamma"])
    def test_a_crash_anywhere_recovers_to_the_uninterrupted_result(
        self, state: IncidentState, tmp_path, crash_at: str
    ) -> None:
        expected = _linear_spec().compile().invoke(state, clock=FrozenClock(FIXED_NOW))
        path = tmp_path / "checkpoints.sqlite"
        fired: dict[str, int] = {}
        with SqliteCheckpointer(path) as store, pytest.raises(_Crash):
            _crashing_spec(crash_at, fired).compile(checkpointer=store).invoke(
                state, clock=FrozenClock(FIXED_NOW)
            )
        # A fresh process image: new graph, new store, only the file carries over.
        with SqliteCheckpointer(path) as reopened:
            graph = _crashing_spec(crash_at, fired).compile(checkpointer=reopened)
            head = graph.state_of("incident-1")
            assert head is not None and head.status is IncidentStatus.RUNNING
            assert head.cursor == crash_at
            with pytest.raises(OrchestrationError, match="not waiting"):
                graph.resume("incident-1", _decision(), clock=FrozenClock(FIXED_NOW))
            recovered = graph.recover("incident-1", clock=FrozenClock(FIXED_NOW))
        assert recovered.state.status is IncidentStatus.COMPLETED
        assert [e.ref for e in recovered.state.evidence] == [
            e.ref for e in expected.state.evidence
        ]
        # The crashed node ran twice (once to die, once to finish); the others once.
        assert fired[crash_at] == 2
        assert all(n == 1 for name, n in fired.items() if name != crash_at)

    def test_recover_stops_at_a_gate_it_has_not_passed(self, state: IncidentState) -> None:
        fired: dict[str, int] = {}
        graph = _crashing_spec("beta", fired).compile(interrupt_before=["gamma"])
        with pytest.raises(_Crash):
            graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        recovered = graph.recover("incident-1", clock=FrozenClock(FIXED_NOW))
        assert recovered.interrupted and recovered.state.interrupt.node == "gamma"

    def test_recover_refuses_a_waiting_run(self, state: IncidentState) -> None:
        graph = _linear_spec().compile(interrupt_before=["beta"])
        graph.invoke(state, clock=FrozenClock(FIXED_NOW))
        with pytest.raises(OrchestrationError, match="use resume"):
            graph.recover("incident-1", clock=FrozenClock(FIXED_NOW))

    def test_recover_refuses_a_finished_run(self, state: IncidentState, linear) -> None:
        linear.invoke(state, clock=FrozenClock(FIXED_NOW))
        with pytest.raises(OrchestrationError, match="already finished"):
            linear.recover("incident-1", clock=FrozenClock(FIXED_NOW))

    def test_recover_refuses_an_unknown_thread(self, linear) -> None:
        with pytest.raises(OrchestrationError, match="no checkpoint"):
            linear.recover("nope")


# --------------------------------------------------------------------------- #
# Failure is a state, not an exception
# --------------------------------------------------------------------------- #


class TestFailure:
    def test_a_raising_node_produces_a_failed_checkpoint(self, state: IncidentState) -> None:
        def explode(_s: IncidentState, _ctx: RunContext) -> IncidentState:
            raise RuntimeError("detector unreachable")

        spec = GraphSpec().add_node("boom", explode).add_edge("boom", END)
        spec.set_entry_point("boom")
        graph = spec.compile()
        result = graph.invoke(state, clock=FrozenClock(FIXED_NOW))

        assert result.state.status is IncidentStatus.FAILED
        assert result.state.error is not None
        assert "detector unreachable" in result.state.error
        assert graph.checkpointer.latest("incident-1") is not None

    def test_the_failing_step_is_recorded(self, state: IncidentState) -> None:
        def explode(_s: IncidentState, _ctx: RunContext) -> IncidentState:
            raise ValueError("bad feature vector")

        spec = GraphSpec().add_node("boom", explode).add_edge("boom", END)
        spec.set_entry_point("boom")
        result = spec.compile().invoke(state, clock=FrozenClock(FIXED_NOW))
        step = result.state.history[-1]
        assert step.node == "boom"
        assert step.outcome is StepOutcome.ERROR
        assert "ValueError" in step.detail

    def test_work_done_before_the_failure_is_preserved(self, state: IncidentState) -> None:
        """A failed run keeps its evidence: the analyst still needs to see it."""

        def explode(_s: IncidentState, _ctx: RunContext) -> IncidentState:
            raise RuntimeError("nope")

        spec = GraphSpec()
        spec.add_node("alpha", _marker_node("alpha")).add_node("boom", explode)
        spec.add_edge("alpha", "boom").add_edge("boom", END).set_entry_point("alpha")
        result = spec.compile().invoke(state, clock=FrozenClock(FIXED_NOW))
        assert [e.ref for e in result.state.evidence] == ["node:alpha"]
        assert result.state.status is IncidentStatus.FAILED

    def test_a_cycling_route_exhausts_the_budget_rather_than_hanging(
        self, state: IncidentState
    ) -> None:
        spec = GraphSpec()
        spec.add_node("a", lambda s, _c: s).add_node("b", lambda s, _c: s)
        spec.add_edge("a", "b").add_edge("b", "a").set_entry_point("a")
        result = spec.compile(step_budget=8).invoke(state, clock=FrozenClock(FIXED_NOW))
        assert result.state.status is IncidentStatus.FAILED
        assert "cycling" in (result.state.error or "")
        assert result.steps_executed == 8

    def test_a_node_returning_someone_elses_incident_is_refused(
        self, state: IncidentState, alert: Alert
    ) -> None:
        def swap(_s: IncidentState, _ctx: RunContext) -> IncidentState:
            return IncidentState(
                incident_id="incident-OTHER",
                tenant_id=alert.tenant_id,
                alert=alert,
                created_at=FIXED_NOW,
                updated_at=FIXED_NOW,
            )

        spec = GraphSpec().add_node("swap", swap).add_edge("swap", END).set_entry_point("swap")
        with pytest.raises(OrchestrationError, match="different incident"):
            spec.compile().invoke(state, clock=FrozenClock(FIXED_NOW))


# --------------------------------------------------------------------------- #
# RunContext
# --------------------------------------------------------------------------- #


class TestRunContext:
    def test_require_decision_refuses_to_default_to_denied(self) -> None:
        """A missing decision must be loud, not quietly safe.

        A gate that treats "no decision" as "denied" is indistinguishable from a
        gate that was never reached, and F-08 needs those to differ.
        """
        ctx = RunContext(clock=FrozenClock(FIXED_NOW), thread_id="t")
        with pytest.raises(OrchestrationError, match="requires a human decision"):
            ctx.require_decision()

    def test_require_decision_returns_the_decision(self) -> None:
        decision = _decision(approved=False)
        ctx = RunContext(clock=FrozenClock(FIXED_NOW), thread_id="t", decision=decision)
        assert ctx.require_decision() is decision

    def test_extras_are_carried_through_with_decision(self) -> None:
        ctx = RunContext(clock=FrozenClock(FIXED_NOW), thread_id="t", extras={"tier": "x"})
        assert ctx.with_decision(_decision()).extras == {"tier": "x"}


# --------------------------------------------------------------------------- #
# Checkpoints as tamper-evidence
# --------------------------------------------------------------------------- #


class TestCheckpointIntegrity:
    def test_a_checkpoint_hash_covers_its_state(self, state: IncidentState) -> None:
        checkpoint = Checkpoint.of(thread_id="t", step=0, node="alpha", state=state)
        checkpoint.verify()
        assert checkpoint.state_hash == state.canonical_hash()

    def test_an_edited_state_fails_verification(self, state: IncidentState) -> None:
        checkpoint = Checkpoint.of(thread_id="t", step=0, node="alpha", state=state)
        forged = checkpoint.updated(state=state.updated(trust_tier=RiskTier.AUTONOMOUS))
        with pytest.raises(CheckpointError, match="does not verify"):
            forged.verify()

    def test_an_edited_row_in_sqlite_is_refused_on_read(
        self, state: IncidentState, tmp_path
    ) -> None:
        """The attack this defends against: promoting a trust tier in the store.

        A run resumed from an edited checkpoint would propose actions at a tier the
        customer never granted, and every schema invariant would still hold — which
        is why the defence has to be at the storage boundary.
        """
        import json
        import sqlite3

        path = tmp_path / "cp.sqlite"
        with SqliteCheckpointer(path) as store:
            store.put(Checkpoint.of(thread_id="t", step=0, node="alpha", state=state))

        conn = sqlite3.connect(path)
        row = conn.execute("SELECT state_json FROM checkpoints").fetchone()
        payload = json.loads(row[0])
        payload["trust_tier"] = "autonomous"
        conn.execute(
            "UPDATE checkpoints SET state_json = ?",
            (json.dumps(payload, separators=(",", ":"), sort_keys=True),),
        )
        conn.commit()
        conn.close()

        with SqliteCheckpointer(path) as reopened, pytest.raises(
            CheckpointError, match="does not verify"
        ):
            reopened.latest("t")

    def test_a_deleted_intermediate_checkpoint_breaks_the_chain(
        self, state: IncidentState, tmp_path
    ) -> None:
        """Deleting the "approval requested" checkpoint is the cheapest cover-up."""
        import sqlite3

        path = tmp_path / "cp.sqlite"
        with SqliteCheckpointer(path) as store:
            graph = _linear_spec().compile(checkpointer=store)
            graph.invoke(state, clock=FrozenClock(FIXED_NOW))

        conn = sqlite3.connect(path)
        conn.execute("DELETE FROM checkpoints WHERE step = 1")
        conn.commit()
        conn.close()

        with SqliteCheckpointer(path) as reopened, pytest.raises(CheckpointError, match="jumps"):
            reopened.history("incident-1")

    def test_checkpoints_are_append_only(self, state: IncidentState) -> None:
        store = InMemoryCheckpointer()
        store.put(Checkpoint.of(thread_id="t", step=0, node="a", state=state))
        with pytest.raises(CheckpointError, match="already exists"):
            store.put(Checkpoint.of(thread_id="t", step=0, node="a", state=state))

    def test_a_checkpoint_that_does_not_link_to_the_head_is_refused(
        self, state: IncidentState
    ) -> None:
        store = InMemoryCheckpointer()
        store.put(Checkpoint.of(thread_id="t", step=0, node="a", state=state))
        orphan = Checkpoint.of(
            thread_id="t", step=1, node="b", state=state, parent_hash=GENESIS_HASH
        )
        with pytest.raises(CheckpointError, match="does not link"):
            store.put(orphan)

    def test_audit_head_is_recorded_and_restored(
        self, state: IncidentState, tmp_path, audit_log
    ) -> None:
        from sentinel.core.schemas import AuditEventType

        audit_log.append(
            AuditEventType.ALERT_INGESTED,
            actor="test",
            tenant_id=state.tenant_id,
            subject_id=state.alert.alert_id,
        )
        path = tmp_path / "cp.sqlite"
        with SqliteCheckpointer(path) as store:
            graph = _linear_spec().compile(checkpointer=store)
            graph.invoke(state, clock=FrozenClock(FIXED_NOW), audit=audit_log)
        with SqliteCheckpointer(path) as reopened:
            head = reopened.latest("incident-1")
            assert head is not None
            assert head.audit_head == audit_log.head()
            reopened.history("incident-1")  # chain still verifies with audit heads stored

    def test_threads_are_listed_in_insertion_order(self, state: IncidentState) -> None:
        store = InMemoryCheckpointer()
        for name in ("t2", "t1", "t3"):
            store.put(Checkpoint.of(thread_id=name, step=0, node="a", state=state))
        assert store.threads() == ("t2", "t1", "t3")


# --------------------------------------------------------------------------- #
# State contract
# --------------------------------------------------------------------------- #


class TestIncidentStateContract:
    def test_tenant_mismatch_is_rejected(self, alert: Alert) -> None:
        with pytest.raises(ValueError, match="cross-tenant"):
            IncidentState(
                incident_id="i",
                tenant_id="other-tenant",
                alert=alert,
                created_at=FIXED_NOW,
                updated_at=FIXED_NOW,
            )

    def test_waiting_without_an_interrupt_is_rejected(self, state: IncidentState) -> None:
        with pytest.raises(ValueError, match="must agree"):
            state.updated(status=IncidentStatus.AWAITING_APPROVAL)

    def test_a_terminal_state_cannot_keep_a_live_cursor(self, state: IncidentState) -> None:
        with pytest.raises(ValueError, match="terminal"):
            state.updated(status=IncidentStatus.COMPLETED, cursor="alpha")

    def test_a_failed_state_must_say_why(self, state: IncidentState) -> None:
        with pytest.raises(ValueError, match="must record why"):
            state.updated(status=IncidentStatus.FAILED, cursor=END)

    def test_triage_is_read_from_the_alert(self, state: IncidentState, triage_result) -> None:
        updated = state.with_triage(triage_result, at=FIXED_NOW)
        assert updated.triage == triage_result
        assert updated.alert.triage == triage_result

    def test_evidence_merge_is_first_writer_wins(self, state: IncidentState) -> None:
        from sentinel.core.schemas import Evidence, EvidenceKind

        first = Evidence(
            kind=EvidenceKind.KB_CHUNK, ref="chunk-1", excerpt="original", relevance=0.9
        )
        second = Evidence(
            kind=EvidenceKind.KB_CHUNK, ref="chunk-1", excerpt="rewritten", relevance=0.1
        )
        merged = state.with_evidence((first,), at=FIXED_NOW).with_evidence(
            (second,), at=FIXED_NOW
        )
        assert len(merged.evidence) == 1
        assert merged.evidence[0].relevance == pytest.approx(0.9)

    def test_an_action_for_another_alert_is_rejected(
        self, state: IncidentState, destructive_action_kwargs
    ) -> None:
        from sentinel.core.schemas import ActionRequest

        action = ActionRequest.propose(**destructive_action_kwargs)
        with pytest.raises(ValueError, match="belongs to alert"):
            state.with_action(action, at=FIXED_NOW)

    def test_with_action_replaces_by_id(
        self, state: IncidentState, destructive_action_kwargs
    ) -> None:
        from sentinel.core.schemas import ActionRequest, ApprovalStatus

        kwargs = {**destructive_action_kwargs, "alert_id": state.alert.alert_id}
        action = ActionRequest.propose(**kwargs)
        approved = action.approve(approver="analyst@acme", at=FIXED_NOW)
        updated = state.with_action(action, at=FIXED_NOW).with_action(approved, at=FIXED_NOW)
        assert len(updated.actions) == 1
        assert updated.actions[0].approval_status is ApprovalStatus.APPROVED

    def test_a_report_for_another_alert_is_rejected(self, state: IncidentState) -> None:
        from sentinel.core.schemas import InvestigationReport

        report = InvestigationReport(
            report_id="r1",
            alert_ids=("some-other-alert",),
            tenant_id=state.tenant_id,
            summary="unrelated",
            severity=Severity.LOW,
            confidence=0.5,
            model_version="test",
            created_at=FIXED_NOW,
        )
        with pytest.raises(ValueError, match="does not cover alert"):
            state.with_report(report, at=FIXED_NOW)

    def test_resumed_on_a_running_state_is_refused(self, state: IncidentState) -> None:
        with pytest.raises(ValueError, match="not waiting"):
            state.resumed(cursor="alpha", at=FIXED_NOW)

    def test_finished_refuses_a_non_terminal_status(self, state: IncidentState) -> None:
        with pytest.raises(ValueError, match="not a terminal status"):
            state.finished(IncidentStatus.RUNNING, at=FIXED_NOW)
