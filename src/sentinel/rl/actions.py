"""The response-policy action space and its trust-tier mask (PRD F-09, Section 5.7).

The action space
----------------
PRD Section 5.5.4 names four actions: ``{auto-contain, escalate-to-human, monitor,
dismiss}``. They are deliberately *not* :class:`~sentinel.core.schemas.ActionType`,
which enumerates concrete side effects (isolate a host, block an address). The
policy decides *how much autonomy to exercise*; which concrete action implements
``auto_contain`` is the Containment Agent's problem and depends on the alert. Keeping
the two apart is what lets the same policy govern an endpoint isolation and an
account disable without learning a separate arm for each.

Why the mask is structural
--------------------------
PRD Section 5.5.4 describes the Phase 3 upgrade as *"a constrained PPO agent with an
explicit action-masking layer so any action outside the current trust tier is
structurally unreachable, not just discouraged by a soft reward penalty"*. That
sentence is a statement about the *bandit's* deficiency, and it is worth taking
seriously now rather than in Phase 3, because the fix is cheap and the failure it
prevents is severe.

A reward penalty is a preference. It says: choosing this is costly, and the policy
will learn to avoid it. But "learn" means "after enough examples", and the examples
are the violations. Worse, a penalty is a scalar competing with other scalars, so a
sufficiently large modelled benefit outranks it — a policy that has learned
containment usually works will eventually auto-contain at a tier that forbids it, and
will be *right* to, by its own objective.

So :class:`ActionMask` removes forbidden arms from the sampling set entirely.
:meth:`ActionMask.allowed` is the only path by which the bandit learns which arms
exist for a decision, and a masked arm is not scored, not sampled and not updated.
The policy cannot prefer what it cannot see.

This is defence in depth, not the only defence.
:meth:`~sentinel.core.schemas.ActionRequest.propose` already derives
``requires_human_approval`` from the tier, so an ungated destructive request is
unconstructible regardless of what any policy asks for. The mask is the layer that
stops the request being *made*; the schema is the layer that stops it being
*honoured*. Part 1 built the second. This is the first.

Tier semantics
--------------
PRD Section 5.7 defines the ladder as *"observe-only -> recommend ->
auto-act-with-notify -> fully autonomous"*. Mapping that onto the four actions
requires two judgements, stated here because they are judgements:

*   ``dismiss`` needs ``recommend``. Closing an alert is a decision that destroys
    information: nobody looks at a dismissed alert again. An observe-only deployment
    is one where the customer has not yet agreed the system's judgement is sound, and
    silently closing alerts is the one thing that cannot be audited after the fact,
    because there is no artifact left to audit.
*   ``auto_contain`` needs ``auto_with_notify``, which is exactly
    :attr:`~sentinel.core.schemas.RiskTier.permits_unattended_execution`. Reusing
    that property rather than restating the tier list means the two cannot drift
    apart, which is the whole reason Part 1 put it on the enum.

``monitor`` and ``escalate`` are available at every tier. Both are what an
observe-only system does: watch, and tell a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import ActionType, RiskTier, TriageDecision

__all__ = [
    "ALL_ACTIONS",
    "ActionMask",
    "PolicyError",
    "ResponseAction",
]


class PolicyError(SentinelError):
    """The response policy was asked for something it must refuse."""


class ResponseAction(StrEnum):
    """The four policy decisions of PRD Section 5.5.4.

    Ordered from most to least intervention, which is the order the mask opens them
    up in and the order a reader should hold them in.
    """

    AUTO_CONTAIN = "auto_contain"
    ESCALATE = "escalate"
    MONITOR = "monitor"
    DISMISS = "dismiss"

    @property
    def minimum_tier(self) -> RiskTier:
        """The lowest trust tier at which this action is permitted."""
        return _MINIMUM_TIER[self]

    @property
    def intervenes(self) -> bool:
        """True when choosing this changes a monitored system's state."""
        return self is ResponseAction.AUTO_CONTAIN

    @property
    def triage_decision(self) -> TriageDecision:
        """How this decision is recorded on a :class:`~sentinel.core.schemas.Alert`.

        ``auto_contain`` maps to ``escalate`` rather than to a decision of its own:
        containing a host *is* treating the alert as real, and Part 1's
        :class:`~sentinel.core.schemas.TriageResult` deliberately has no
        "act autonomously" decision, because every action still passes the approval
        gate at the tiers where the schema can be made to hold.
        """
        return _TRIAGE_DECISION[self]

    @property
    def implied_action_types(self) -> tuple[ActionType, ...]:
        """The concrete action types this decision may be realised as.

        Advisory, not authoritative: the Containment Agent picks one based on the
        alert, and whatever it picks still goes through
        :meth:`~sentinel.core.schemas.ActionRequest.propose`. The mapping is here so a
        test can assert that every action type reachable from ``auto_contain`` is
        classified destructive, and that nothing reachable from the other three is.
        """
        return _IMPLIED_ACTION_TYPES[self]


ALL_ACTIONS: Final[tuple[ResponseAction, ...]] = tuple(ResponseAction)

_MINIMUM_TIER: Final[dict[ResponseAction, RiskTier]] = {
    # Watching and telling a human are what an observe-only deployment does.
    ResponseAction.MONITOR: RiskTier.OBSERVE,
    ResponseAction.ESCALATE: RiskTier.OBSERVE,
    # Closing an alert destroys the artifact that would let anyone check the call.
    ResponseAction.DISMISS: RiskTier.RECOMMEND,
    # Acting without a human is exactly what the top two tiers mean.
    ResponseAction.AUTO_CONTAIN: RiskTier.AUTO_WITH_NOTIFY,
}

_TRIAGE_DECISION: Final[dict[ResponseAction, TriageDecision]] = {
    ResponseAction.AUTO_CONTAIN: TriageDecision.ESCALATE,
    ResponseAction.ESCALATE: TriageDecision.ESCALATE,
    ResponseAction.MONITOR: TriageDecision.MONITOR,
    ResponseAction.DISMISS: TriageDecision.AUTO_DISMISS,
}

_IMPLIED_ACTION_TYPES: Final[dict[ResponseAction, tuple[ActionType, ...]]] = {
    ResponseAction.AUTO_CONTAIN: (
        ActionType.ISOLATE_HOST,
        ActionType.BLOCK_IP,
        ActionType.DISABLE_ACCOUNT,
        ActionType.KILL_PROCESS,
        ActionType.QUARANTINE_FILE,
    ),
    ResponseAction.ESCALATE: (ActionType.NOTIFY_ANALYST,),
    ResponseAction.MONITOR: (ActionType.ENRICH_ONLY,),
    ResponseAction.DISMISS: (),
}


@dataclass(frozen=True, slots=True)
class ActionMask:
    """Which actions a trust tier permits. The bandit's only view of its arm set.

    Cached per tier: there are four, so every possible mask is built once at import.
    That matters more than it looks — the mask is consulted on every decision, and a
    mask constructed per call is a mask that can be constructed *wrongly* per call.

    ``frozen``, and not merely by convention. The masks are shared singletons, so a
    caller able to assign to one would not be breaking its own copy, it would be
    silently re-permitting a forbidden action for every subsequent decision in the
    process. ``__slots__`` alone does not prevent that — it restricts which attribute
    names exist, not whether they can be rebound — which is what
    ``test_rl_actions.py::test_mask_is_immutable`` was written to find out and did.
    """

    _tier: RiskTier
    _allowed: tuple[ResponseAction, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self._tier, RiskTier):
            raise PolicyError(f"tier must be a RiskTier, got {type(self._tier).__name__}")
        allowed = tuple(
            action
            for action in ALL_ACTIONS
            if self._tier.rank >= action.minimum_tier.rank
        )
        if not allowed:  # pragma: no cover - unreachable while OBSERVE permits two
            raise PolicyError(f"tier {self._tier.value!r} permits no actions at all")
        object.__setattr__(self, "_allowed", allowed)

    @classmethod
    def for_tier(cls, tier: RiskTier) -> ActionMask:
        """The cached mask for ``tier``."""
        if tier not in _MASKS:
            raise PolicyError(f"no mask for tier {tier!r}")
        return _MASKS[tier]

    @property
    def tier(self) -> RiskTier:
        return self._tier

    @property
    def allowed(self) -> tuple[ResponseAction, ...]:
        """Permitted actions, in :class:`ResponseAction` declaration order."""
        return self._allowed

    @property
    def forbidden(self) -> tuple[ResponseAction, ...]:
        return tuple(a for a in ALL_ACTIONS if a not in self._allowed)

    @property
    def indices(self) -> tuple[int, ...]:
        """Arm indices the bandit may sample. The mask's actual interface."""
        return tuple(ALL_ACTIONS.index(action) for action in self._allowed)

    def permits(self, action: ResponseAction) -> bool:
        return action in self._allowed

    def require(self, action: ResponseAction) -> ResponseAction:
        """Return ``action``, or raise if this tier forbids it.

        The assertion point for code that did not go through
        :meth:`~sentinel.rl.bandit.LinearThompsonBandit.select`, which is any code
        path that could reintroduce the failure the mask exists to prevent.
        """
        if not self.permits(action):
            raise PolicyError(
                f"action {action.value!r} requires tier "
                f"{action.minimum_tier.value!r} or above; current tier is "
                f"{self._tier.value!r}"
            )
        return action

    def __len__(self) -> int:
        return len(self._allowed)

    def __contains__(self, action: object) -> bool:
        return action in self._allowed

    def __repr__(self) -> str:
        names = "+".join(a.value for a in self._allowed)
        return f"<ActionMask {self._tier.value}: {names}>"


_MASKS: Final[dict[RiskTier, ActionMask]] = {tier: ActionMask(tier) for tier in RiskTier}
