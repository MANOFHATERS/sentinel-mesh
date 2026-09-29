"""Layer 3 — the RL response-policy layer (PRD F-09, Section 5.5.4).

    ``actions``   the four policy decisions, and the structural trust-tier mask
    ``reward``    the shaped reward, with its ordering enforced in code
    ``bandit``    linear contextual Thompson sampling, hand-written on numpy
    ``simulate``  episodes from real alerts, and the F-09 regret replay

The mask is the part worth reading first. PRD Section 5.5.4 defers action masking to
Phase 3's PPO upgrade, describing it as making forbidden actions *"structurally
unreachable, not just discouraged by a soft reward penalty"*. That property is cheap
enough to have now, and a reward penalty is the wrong tool for it: a penalty is a
preference a sufficiently confident policy will eventually outvote.
"""

from __future__ import annotations

from sentinel.rl.actions import ALL_ACTIONS, ActionMask, PolicyError, ResponseAction
from sentinel.rl.bandit import ArmPosterior, BanditError, Decision, LinearThompsonBandit
from sentinel.rl.reward import Outcome, RewardModel, assert_ordering_invariants, best_action
from sentinel.rl.simulate import (
    N_FEATURES,
    REPORTING_SEEDS,
    TUNING_SEEDS,
    Episode,
    ReplayResult,
    aggregate,
    assert_f09_gates,
    build_episodes,
    replay,
    replay_many_seeds,
)

__all__ = [
    "ALL_ACTIONS",
    "N_FEATURES",
    "REPORTING_SEEDS",
    "TUNING_SEEDS",
    "ActionMask",
    "ArmPosterior",
    "BanditError",
    "Decision",
    "Episode",
    "LinearThompsonBandit",
    "Outcome",
    "PolicyError",
    "ReplayResult",
    "ResponseAction",
    "RewardModel",
    "aggregate",
    "assert_f09_gates",
    "assert_ordering_invariants",
    "best_action",
    "build_episodes",
    "replay",
    "replay_many_seeds",
]
