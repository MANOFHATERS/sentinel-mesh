"""Layer 4 — agent orchestration (PRD Section 5.2, Figure 3).

The orchestration graph and the five agents that run on it. What lives where:

===========================  ====================================================
:mod:`~sentinel.agents.state`        the checkpointed ``IncidentState``
:mod:`~sentinel.agents.checkpoint`   verifiable, chained checkpoint storage
:mod:`~sentinel.agents.runtime`      the state machine: routing, interrupt, resume
:mod:`~sentinel.agents.prompts`      Appendix A prompts and the untrusted fence
:mod:`~sentinel.agents.engine`       the reasoning seam, and monotone caution
:mod:`~sentinel.agents.triage`       Triage Agent (F-02)
===========================  ====================================================

Two invariants hold across the whole layer, and both are enforced by code rather
than by prompt:

1.  **Monotone caution.** A reasoning engine's opinion is merged into a
    deterministic verdict upward only (:func:`~sentinel.agents.engine.monotone_caution`).
    A compromised model can raise a false alarm; it cannot suppress a real one.
2.  **Untrusted stays untrusted.** Attacker-influenced text reaches a prompt only
    through :meth:`~sentinel.agents.prompts.AgentPrompt.with_untrusted`, and an
    engine's own output is typed
    :class:`~sentinel.core.untrusted.UntrustedText` because it is a function of
    that input.
"""

from sentinel.agents.state import END, HumanDecision, IncidentState, IncidentStatus

__all__ = ["END", "HumanDecision", "IncidentState", "IncidentStatus"]
