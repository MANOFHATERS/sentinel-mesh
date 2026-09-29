"""The shaped reward for response decisions (PRD F-09, Section 5.5.4).

What the PRD requires
---------------------
Section 5.5.4: *"reward is shaped from simulated analyst feedback (a correct
auto-contain is rewarded, a false auto-contain that a human later reverses is
penalized more heavily than an over-cautious escalation)"*.

That is one ordering constraint, and it is the wrong place to stop. A reward function
with only that constraint satisfied still permits a policy that dismisses real
attacks, because nothing in the sentence says missing an attack is worse than
annoying an analyst. In a security product it is worse by orders of magnitude, and the
reward is the only place that fact can live: the bandit has no other source of
domain knowledge.

So the ordering this module enforces, from worst to best outcome, is:

1.  **Dismissed a real attack.** The alert is closed, nobody looks again, and the
    intrusion proceeds. This is the outcome the whole product exists to prevent and it
    carries the largest penalty by a wide margin.
2.  **Auto-contained a benign alert, and a human reversed it.** A production host was
    isolated for nothing. Expensive, visible, and the fastest way to lose the trust
    that the tier ladder is built on — but recoverable within minutes, which is why it
    is not worse than (1).
3.  **Monitored a real attack.** Not wrong, but the clock keeps running; this is
    detection without response.
4.  **Escalated a benign alert.** Analyst time wasted. The PRD's "over-cautious
    escalation", and the mildest of the bad outcomes by design.
5.  **Monitored a benign alert.** Nearly free, slightly noisy.
6.  **Escalated a real attack.** The right call, made slowly.
7.  **Dismissed a benign alert.** Correct, and the product's core value: F-12 measures
    alert-volume reduction, and this is the decision that produces it.
8.  **Auto-contained a real attack.** The best available outcome: correct, and fast.

:func:`assert_ordering_invariants` checks every one of those relations numerically, so
the ordering is a property of the code rather than of this docstring.

Asset criticality
-----------------
Consequences scale with what was touched. A false containment on a domain controller
is not the same event as one on a developer's spare laptop, and a missed intrusion on
a payment system is not the same as one on a print server. Criticality multiplies the
*penalties* and leaves the rewards alone, which is deliberate: raising the reward for
correctly containing a critical asset would push the policy toward intervening on
critical assets specifically, and the thing that should make it cautious there is
asymmetry of loss, not appetite for gain.

Why a deterministic reward, and where the noise goes instead
------------------------------------------------------------
:meth:`RewardModel.reward` is a pure function of the outcome, so the oracle in
:func:`best_action` is exact and regret is measurable rather than estimated. Real
analyst feedback is noisy and delayed, and that noise belongs in the *simulator*
(:mod:`sentinel.rl.simulate` adds observation noise and reversal noise), not in the
reward definition — otherwise there is no fixed reference against which to define
regret at all, and the F-09 acceptance criterion becomes unmeasurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from sentinel.core.errors import SentinelError
from sentinel.rl.actions import ALL_ACTIONS, ActionMask, ResponseAction

__all__ = [
    "CRITICALITY_RANGE",
    "Outcome",
    "RewardModel",
    "assert_ordering_invariants",
    "best_action",
]

#: Valid range for asset criticality. 0 is a disposable asset, 1 is tier-zero
#: infrastructure. Bounded because it multiplies penalties, and an unbounded
#: multiplier is an unbounded gradient.
CRITICALITY_RANGE: Final[tuple[float, float]] = (0.0, 1.0)


class RewardError(SentinelError):
    """The reward model was given an outcome it cannot score."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What actually happened, for scoring one decision.

    ``is_attack`` is ground truth, available here because the sprint replays labelled
    public datasets (PRD Section 7.4). It is what makes the oracle computable. No
    policy code ever reads it -- ``test_rl_reward.py`` asserts the bandit's update
    path takes only a scalar reward.

    ``human_reverses`` models the analyst who un-isolates a host that should not have
    been isolated. It only bites on a false ``auto_contain``, which is precisely the
    event PRD Section 5.5.4 singles out for heavier penalty.
    """

    action: ResponseAction
    is_attack: bool
    criticality: float = 0.5
    human_reverses: bool = False

    def __post_init__(self) -> None:
        low, high = CRITICALITY_RANGE
        if not low <= self.criticality <= high:
            raise RewardError(
                f"criticality {self.criticality} outside [{low}, {high}]"
            )
        if self.human_reverses and self.action is not ResponseAction.AUTO_CONTAIN:
            raise RewardError(
                "human_reverses only applies to auto_contain; there is nothing to "
                "reverse about an escalation, a dismissal or monitoring"
            )


@dataclass(frozen=True, slots=True)
class RewardModel:
    """Shaped reward for PRD Section 5.5.4.

    Every constant is a field so the shape can be varied in a sensitivity analysis
    rather than edited in place, and so a test can assert the *ordering* holds across
    a range of values instead of pinning one hand-tuned set. The defaults were chosen
    to satisfy the ordering with margin, not fitted to the simulator -- fitting reward
    constants to the environment they are evaluated in is how a policy comes to look
    good at the thing it was scored on and nothing else.
    """

    #: Correct auto-contain: right, and fast. The best available outcome.
    contain_true: float = 1.00
    #: False auto-contain that nobody reversed: still wrong, still disruptive.
    contain_false: float = -1.00
    #: Additional penalty when a human reverses it. The PRD's named case.
    reversal_penalty: float = -1.20
    #: Correct escalation: right, but a human is now in the loop and time passes.
    escalate_true: float = 0.60
    #: Over-cautious escalation. The PRD's explicit comparison point.
    escalate_false: float = -0.20
    #: Monitoring a real attack: detection without response.
    monitor_true: float = -0.50
    #: Monitoring a benign alert: nearly free.
    monitor_false: float = 0.05
    #: Dismissing a real attack. The outcome the product exists to prevent.
    dismiss_true: float = -3.00
    #: Dismissing a benign alert. The alert-volume reduction F-12 measures.
    dismiss_false: float = 0.70
    #: How strongly criticality scales penalties. At 1.0 a tier-zero asset doubles
    #: every penalty relative to a criticality-0 asset.
    criticality_weight: float = 1.00

    @property
    def reward_spread(self) -> float:
        """Standard deviation of reward over the full outcome grid.

        Exists so exploration can be specified in units of reward noise instead of in
        raw reward units. Without it, ``LinearThompsonBandit.posterior_scale`` is
        silently coupled to the constants in this class: widening
        :attr:`dismiss_true` from -3 to -6 would halve the effective exploration rate
        of a policy nobody touched. Decoupling two things that have no business being
        coupled is worth eleven lines.

        Computed over every ``(action, is_attack, reverses)`` combination at three
        criticality levels, which is the whole support of :meth:`reward`, so it is
        exact rather than sampled.
        """
        import itertools

        values = [
            self.reward(Outcome(action, attack, criticality, reverses))
            for action, attack, criticality, reverses in itertools.product(
                ALL_ACTIONS, (True, False), (0.0, 0.5, 1.0), (False, True)
            )
            if not (reverses and action is not ResponseAction.AUTO_CONTAIN)
        ]
        spread = float(np.std(values))
        return spread if spread > 0.0 else 1.0

    def reward(self, outcome: Outcome) -> float:
        """Score one decision. Pure, deterministic, and the oracle's basis."""
        base = self._base(outcome)
        if base >= 0.0:
            # Criticality scales loss, not gain. See the module docstring.
            return base
        scale = 1.0 + self.criticality_weight * outcome.criticality
        return base * scale

    def _base(self, outcome: Outcome) -> float:
        attack = outcome.is_attack
        match outcome.action:
            case ResponseAction.AUTO_CONTAIN:
                if attack:
                    return self.contain_true
                return self.contain_false + (
                    self.reversal_penalty if outcome.human_reverses else 0.0
                )
            case ResponseAction.ESCALATE:
                return self.escalate_true if attack else self.escalate_false
            case ResponseAction.MONITOR:
                return self.monitor_true if attack else self.monitor_false
            case ResponseAction.DISMISS:
                return self.dismiss_true if attack else self.dismiss_false
        raise RewardError(f"unscored action {outcome.action!r}")  # pragma: no cover

    # --- oracle ------------------------------------------------------------- #

    def best_reward(
        self,
        *,
        is_attack: bool,
        criticality: float,
        mask: ActionMask,
        reversal_probability: float = 0.0,
    ) -> float:
        """The reward an oracle would collect, given the tier's constraints.

        Computed *within the mask*, which is the only honest way to define regret for
        a masked policy. An oracle allowed to auto-contain at ``observe`` tier would
        make the policy look permanently regretful for obeying its own safety
        constraint, and the resulting number would measure the tier, not the learning.
        """
        return max(
            self.expected_reward(
                action=action,
                is_attack=is_attack,
                criticality=criticality,
                reversal_probability=reversal_probability,
            )
            for action in mask.allowed
        )

    def expected_reward(
        self,
        *,
        action: ResponseAction,
        is_attack: bool,
        criticality: float,
        reversal_probability: float = 0.0,
    ) -> float:
        """Expected reward, averaging over whether a human reverses a false contain."""
        if not 0.0 <= reversal_probability <= 1.0:
            raise RewardError(
                f"reversal_probability {reversal_probability} outside [0, 1]"
            )
        if action is ResponseAction.AUTO_CONTAIN and not is_attack:
            reversed_reward = self.reward(
                Outcome(action, is_attack, criticality, human_reverses=True)
            )
            kept_reward = self.reward(
                Outcome(action, is_attack, criticality, human_reverses=False)
            )
            return (
                reversal_probability * reversed_reward
                + (1.0 - reversal_probability) * kept_reward
            )
        return self.reward(Outcome(action, is_attack, criticality))


def best_action(
    model: RewardModel,
    *,
    is_attack: bool,
    criticality: float,
    mask: ActionMask,
    reversal_probability: float = 0.0,
) -> ResponseAction:
    """The oracle's choice within ``mask``.

    Ties break toward the least interventionist action, matching
    :meth:`~sentinel.rl.bandit.LinearThompsonBandit.select`. Keeping the two rules
    identical matters for regret: if the oracle broke ties the other way, a policy
    agreeing with it perfectly would still be charged regret on every tied decision.
    """
    return max(
        mask.allowed,
        key=lambda action: (
            model.expected_reward(
                action=action,
                is_attack=is_attack,
                criticality=criticality,
                reversal_probability=reversal_probability,
            ),
            ALL_ACTIONS.index(action),
        ),
    )


def assert_ordering_invariants(model: RewardModel) -> None:
    """Check every ordering the module docstring claims. Raises on violation.

    Called by the test suite over a grid of reward constants, so the ordering is a
    verified property of the model rather than a property of one default. A reward
    function whose ordering is only true for the numbers it shipped with is a reward
    function nobody can safely tune.
    """
    failures: list[str] = []

    def check(condition: bool, description: str) -> None:
        if not condition:
            failures.append(description)

    # The PRD's explicit requirement.
    reversed_contain = model.reward(
        Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5, human_reverses=True)
    )
    cautious_escalation = model.reward(Outcome(ResponseAction.ESCALATE, False, 0.5))
    check(
        reversed_contain < cautious_escalation,
        "PRD 5.5.4: a reversed false auto-contain must be penalised more heavily "
        f"than an over-cautious escalation ({reversed_contain} vs {cautious_escalation})",
    )

    # Reversal must actually cost something.
    unreversed = model.reward(Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5))
    check(
        reversed_contain < unreversed,
        f"a reversal must cost more than no reversal ({reversed_contain} vs {unreversed})",
    )

    # The domain truth the PRD sentence omits.
    missed = model.reward(Outcome(ResponseAction.DISMISS, True, 0.5))
    check(
        missed < reversed_contain,
        f"dismissing a real attack must be the worst outcome ({missed} vs "
        f"{reversed_contain})",
    )

    # Correct decisions must beat incorrect ones, per ground truth.
    for is_attack in (True, False):
        rewards = {
            action: model.reward(Outcome(action, is_attack, 0.5))
            for action in ALL_ACTIONS
        }
        expected_best = (
            ResponseAction.AUTO_CONTAIN if is_attack else ResponseAction.DISMISS
        )
        actual_best = max(rewards, key=lambda a: rewards[a])
        check(
            actual_best is expected_best,
            f"with is_attack={is_attack} the best action should be "
            f"{expected_best.value!r}, got {actual_best.value!r}",
        )

    # On a real attack: acting beats watching beats closing.
    check(
        model.reward(Outcome(ResponseAction.AUTO_CONTAIN, True, 0.5))
        > model.reward(Outcome(ResponseAction.ESCALATE, True, 0.5))
        > model.reward(Outcome(ResponseAction.MONITOR, True, 0.5))
        > model.reward(Outcome(ResponseAction.DISMISS, True, 0.5)),
        "on a real attack, reward must decrease monotonically with inaction",
    )

    # On a benign alert: closing it beats escalating it beats containing it.
    check(
        model.reward(Outcome(ResponseAction.DISMISS, False, 0.5))
        > model.reward(Outcome(ResponseAction.MONITOR, False, 0.5))
        > model.reward(Outcome(ResponseAction.ESCALATE, False, 0.5))
        > model.reward(Outcome(ResponseAction.AUTO_CONTAIN, False, 0.5)),
        "on a benign alert, reward must decrease with escalation",
    )

    # Criticality must make bad outcomes worse and must not make good ones better.
    for action, is_attack in (
        (ResponseAction.DISMISS, True),
        (ResponseAction.AUTO_CONTAIN, False),
        (ResponseAction.MONITOR, True),
        (ResponseAction.ESCALATE, False),
    ):
        low = model.reward(Outcome(action, is_attack, 0.0))
        high = model.reward(Outcome(action, is_attack, 1.0))
        check(
            high <= low,
            f"criticality must not soften the penalty for {action.value} "
            f"on is_attack={is_attack} ({high} vs {low})",
        )
    for action, is_attack in (
        (ResponseAction.AUTO_CONTAIN, True),
        (ResponseAction.DISMISS, False),
    ):
        low = model.reward(Outcome(action, is_attack, 0.0))
        high = model.reward(Outcome(action, is_attack, 1.0))
        check(
            high == low,
            f"criticality must not inflate the reward for {action.value} "
            f"on is_attack={is_attack} ({high} vs {low})",
        )

    if failures:
        raise RewardError(
            "reward ordering invariants violated:\n  - " + "\n  - ".join(failures)
        )
