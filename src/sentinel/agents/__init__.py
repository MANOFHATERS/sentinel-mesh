"""Layer 4 — agent orchestration (PRD Section 5.2, Figure 3).

The orchestration graphs and the five agents that run on them. What lives where:

===================================  ==============================================
:mod:`~sentinel.agents.state`        the checkpointed ``IncidentState``
:mod:`~sentinel.agents.checkpoint`   verifiable, chained checkpoint storage
:mod:`~sentinel.agents.runtime`      the state machine: routing, interrupt, resume
:mod:`~sentinel.agents.prompts`      Appendix A prompts and the untrusted fence
:mod:`~sentinel.agents.engine`       the reasoning seam, and monotone caution
:mod:`~sentinel.agents.triage`       Triage Agent (F-02)
:mod:`~sentinel.agents.investigate`  Investigation Agent (F-05)
:mod:`~sentinel.agents.contain`      Containment Agent + Human Approval Gate (F-08)
:mod:`~sentinel.agents.codescan`     Code-Scan / Patch Agent (F-07)
:mod:`~sentinel.agents.supplychain`  Supply-Chain Agent (F-06's guardrail)
:mod:`~sentinel.agents.orchestrator` PRD Figure 3, and the Section 9.1 timings
===================================  ==============================================

Three invariants hold across the whole layer, and all three are enforced by code
rather than by prompt:

1.  **Monotone caution.** A reasoning engine's opinion is merged into a
    deterministic verdict upward only (:func:`~sentinel.agents.engine.monotone_caution`,
    and :func:`~sentinel.agents.codescan.reconcile_findings` for code findings).
    A compromised model can raise a false alarm; it cannot suppress a real one, and
    it cannot write a line of a patch — every diff is an AST rewrite.
2.  **Untrusted stays untrusted.** Attacker-influenced text reaches a prompt only
    through :meth:`~sentinel.agents.prompts.AgentPrompt.with_untrusted`, and an
    engine's own output is typed
    :class:`~sentinel.core.untrusted.UntrustedText` because it is a function of
    that input. Source lines and vendor names are attacker-influenced text like any
    alert payload, and go through the same door.
3.  **One state machine, three graphs.** An incident starts with an alert, a code
    scan with a commit, and a supply-chain assessment with a schedule, so they are
    three graphs — but they share :class:`~sentinel.agents.state.IncidentState`, the
    checkpoint chain, the Human Approval Gate and the audit log. F-08's guarantee is
    implemented once and therefore tested once.
"""

from sentinel.agents.state import END, HumanDecision, IncidentState, IncidentStatus

__all__ = ["END", "HumanDecision", "IncidentState", "IncidentStatus"]
