"""Linear contextual Thompson sampling (PRD F-09, Section 5.5.4).

The algorithm and why this one
------------------------------
PRD Section 5.5.4 specifies *"a contextual bandit for the sprint: state is the
concatenated alert/investigation embedding plus asset criticality; actions are
{auto-contain, escalate-to-human, monitor, dismiss} ... Thompson sampling gives a
working, statistically grounded policy in the available time."*

"Contextual" rules out the obvious implementation. A Beta-Bernoulli Thompson sampler
per arm is four lines and completely wrong here: it learns one success rate per
action and cannot represent "contain when the anomaly score is high and the asset is
critical", which is the entire decision. So each arm carries a Bayesian linear model
of reward against context, and sampling is from the posterior over that model's
coefficients — linear Thompson sampling, in the standard conjugate form.

Per arm ``a``, with design matrix rows ``x`` and observed rewards ``r``:

    ``A_a = lambda I + sum x x^T``,  ``b_a = sum r x``,  ``mu_a = A_a^-1 b_a``

and the posterior over coefficients is ``N(mu_a, v^2 A_a^-1)``. A decision draws one
``theta_a`` per allowed arm and plays ``argmax_a x . theta_a``. Exploration is
therefore a property of posterior width: an arm with little data has a wide posterior
and gets tried; as ``A_a`` accumulates, the draws concentrate on ``mu_a`` and the
policy exploits. There is no epsilon to schedule and no temperature to decay, which
is the practical reason to prefer Thompson sampling over epsilon-greedy here — a
decaying epsilon is one more thing to tune on data you should not be tuning on.

Numerical choices that are not incidental
-----------------------------------------
**Sherman-Morrison, and a guard against trusting it.** ``A_a^-1`` is maintained
incrementally by rank-1 update rather than re-inverted each step. That is the standard
move and it is also numerically fragile: the update accumulates error, and a matrix
that drifts out of symmetry or positive-definiteness produces a Cholesky failure or,
worse, a silently skewed posterior. So the inverse is re-symmetrised on every update,
and :meth:`LinearThompsonBandit.max_inverse_drift` exposes the distance from a direct
solve so the test suite can assert the fast path *is* the slow path
(``test_rl_bandit.py::TestShermanMorrisonMatchesDirectSolve``).

**Sampling via Cholesky of the inverse, with a jitter ladder.** A draw from
``N(mu, v^2 A^-1)`` needs a factor of ``A^-1``. ``numpy.linalg.cholesky`` fails on a
matrix that is positive definite in theory and marginally indefinite in floating
point, which for a bandit means an exception several thousand episodes into a run.
The jitter ladder adds increasing multiples of the identity until the factorisation
succeeds and records that it happened, so degradation is visible rather than fatal.

**One posterior per arm, not one shared model.** Arms do not share coefficients
because the reward surfaces genuinely differ in shape: the value of ``dismiss`` falls
with the anomaly score while the value of ``auto_contain`` rises with it, and a shared
model would have to represent both through interaction terms the context does not
contain.

Masked arms are not updated, and that has a cost
------------------------------------------------
:class:`~sentinel.rl.actions.ActionMask` removes forbidden arms from the sampling set,
so at ``observe`` tier the ``auto_contain`` posterior never moves. This is correct —
learning from an action you did not take is off-policy estimation, and doing it
casually is how a policy acquires confident beliefs about consequences it has never
observed — but it means a tier promotion hands the policy a cold arm with a wide
prior, which it will then explore on live traffic.

That is a real limitation and it is not hidden: :attr:`ArmPosterior.observations`
reports it per arm, :meth:`LinearThompsonBandit.cold_arms` names them, and the
documented mitigation is the tier ladder itself — PRD Section 5.7 promotes a tier only
after *"a configurable number of correctly-approved recommendations"*, so by the time
``auto_contain`` becomes reachable the analyst has already approved a run of
containment recommendations. Phase 3's constrained PPO is where off-policy correction
belongs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import RiskTier
from sentinel.rl.actions import ALL_ACTIONS, ActionMask, ResponseAction

__all__ = [
    "DEFAULT_POSTERIOR_SCALE",
    "DEFAULT_PRIOR_PRECISION",
    "DEFAULT_PRIOR_REWARD",
    "ArmPosterior",
    "BanditError",
    "Decision",
    "LinearThompsonBandit",
]

DTYPE: Final = np.float64

#: Ridge term ``lambda`` on the prior precision. Larger means a tighter prior, so a
#: single early observation moves the posterior mean less -- which is exactly the
#: failure the tuning seeds showed at ``lambda = 1``, where one unlucky false
#: containment was enough to retire the arm for the rest of the run. 4.0 was selected
#: on :data:`~sentinel.rl.simulate.TUNING_SEEDS`.
DEFAULT_PRIOR_PRECISION: Final[float] = 4.0

#: Posterior width multiplier ``v``. This is the exploration knob: 0 makes the policy
#: greedy on the posterior mean, large values make it explore indefinitely. It scales
#: with reward noise, which is why :attr:`LinearThompsonBandit.reward_scale` exists:
#: this number is denominated in reward standard deviations, so it does not silently
#: change meaning when the reward constants do.
#:
#: 0.10 is deliberately narrow, and the narrowness is affordable only because
#: :data:`DEFAULT_PRIOR_REWARD` supplies the early exploration instead. Measured on the
#: tuning seeds, widening this to 0.25 costs 31% more total regret and widening it to
#: 1.0 costs roughly double, because sustained exploration in this action space means
#: sustained isolation of production hosts.
DEFAULT_POSTERIOR_SCALE: Final[float] = 0.10

#: Prior expected reward per arm, in reward standard deviations. Optimistic, and the
#: reason is a measured failure rather than a preference.
#:
#: With a zero-mean prior and enough exploration to be safe, the policy *bifurcated*
#: across tuning seeds: on two of five it locked onto ``auto_contain`` and reached
#: 92-95% optimal actions, and on the other three it locked onto ``escalate`` and sat
#: at 65-69%, having played the containment arm once or never. The mean total regret
#: looked fine and was hiding two different policies.
#:
#: The obvious remedy -- widen :data:`DEFAULT_POSTERIOR_SCALE` -- fixes the bimodality
#: and doubles total regret, because it keeps exploring forever. In this action space
#: "keep exploring" means "keep isolating production hosts to see what happens", which
#: is the wrong thing to buy a metric with.
#:
#: Optimistic initialisation separates the two. An untried arm looks good, so it is
#: tried; the data then pushes it down permanently. Exploration happens early, where
#: it is cheap, rather than throughout. This is the standard optimism-under-uncertainty
#: argument and it is the right shape for the problem: the cost of exploring is not
#: constant over time, because trust is being earned.
DEFAULT_PRIOR_REWARD: Final[float] = 1.0

#: Jitter multiples tried in order when a Cholesky factorisation fails.
_JITTER_LADDER: Final[tuple[float, ...]] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)


class BanditError(SentinelError):
    """The policy was configured or driven incorrectly."""


@dataclass(slots=True)
class ArmPosterior:
    """Bayesian linear regression posterior for one arm.

    ``precision`` is ``A``, ``inverse`` is ``A^-1`` maintained by rank-1 update, and
    ``moment`` is ``b``. ``mean`` is derived rather than stored, because storing a
    value that must agree with two others is an invariant waiting to be violated.
    """

    n_features: int
    prior_precision: float = DEFAULT_PRIOR_PRECISION
    #: Prior expected reward, applied through the bias feature. Positive values are
    #: *optimistic*: an arm with no data looks good until data says otherwise, so every
    #: arm is tried early and then abandoned on its merits. See
    #: :data:`DEFAULT_PRIOR_REWARD` for why this is here rather than a wider posterior.
    prior_reward: float = 0.0
    precision: npt.NDArray[np.float64] = field(default=None, repr=False)  # type: ignore[assignment]
    inverse: npt.NDArray[np.float64] = field(default=None, repr=False)  # type: ignore[assignment]
    moment: npt.NDArray[np.float64] = field(default=None, repr=False)  # type: ignore[assignment]
    observations: int = 0
    reward_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise BanditError("n_features must be at least 1")
        if self.prior_precision <= 0.0:
            raise BanditError(
                f"prior_precision must be positive, got {self.prior_precision}; a "
                "zero or negative ridge leaves A singular until enough data arrives"
            )
        identity = np.eye(self.n_features, dtype=DTYPE)
        self.precision = self.prior_precision * identity
        self.inverse = identity / self.prior_precision
        self.moment = np.zeros(self.n_features, dtype=DTYPE)
        # ``mean = A^-1 b``, and at construction ``A^-1 = I / lambda``, so seeding
        # ``b[0] = lambda * prior_reward`` puts exactly ``prior_reward`` on the bias
        # coefficient. Expressed this way rather than by assigning to ``mean`` because
        # ``mean`` is derived: the prior has to enter through the same channel as the
        # data, or the first update would silently discard it.
        self.moment[0] = self.prior_precision * self.prior_reward

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        """Posterior mean ``A^-1 b``."""
        return self.inverse @ self.moment

    @property
    def mean_reward(self) -> float:
        """Average observed reward for this arm. Diagnostics only, never a policy input."""
        return self.reward_sum / self.observations if self.observations else 0.0

    def update(self, context: npt.NDArray[np.float64], reward: float) -> None:
        """Absorb one observation by rank-1 update.

        Sherman-Morrison: ``(A + x x^T)^-1 = A^-1 - (A^-1 x)(x^T A^-1) / (1 + x^T A^-1 x)``.
        The denominator is ``1 + x^T A^-1 x`` with ``A^-1`` positive definite, so it is
        bounded below by 1 and cannot vanish -- which is why this is safe here and is
        not safe in general.
        """
        if context.shape != (self.n_features,):
            raise BanditError(
                f"context has shape {context.shape}, expected ({self.n_features},)"
            )
        if not np.all(np.isfinite(context)):
            raise BanditError("context contains non-finite values")
        if not np.isfinite(reward):
            raise BanditError(f"reward must be finite, got {reward}")

        self.precision += np.outer(context, context)
        projected = self.inverse @ context
        denominator = 1.0 + float(context @ projected)
        self.inverse -= np.outer(projected, projected) / denominator
        # Re-symmetrise: the two triangles are equal in exact arithmetic and drift
        # apart in floating point, and an asymmetric "covariance" makes the Cholesky
        # below either fail or lie.
        self.inverse = 0.5 * (self.inverse + self.inverse.T)
        self.moment += reward * context
        self.observations += 1
        self.reward_sum += float(reward)

    def predict(self, context: npt.NDArray[np.float64]) -> float:
        """Posterior-mean reward estimate. The greedy value, with no exploration."""
        return float(context @ self.mean)

    def uncertainty(self, context: npt.NDArray[np.float64]) -> float:
        """Posterior standard deviation of the estimate at ``context``.

        ``sqrt(x^T A^-1 x)``. Clipped at zero before the square root because a
        marginally negative quadratic form is a rounding artefact, not a signal, and
        ``sqrt`` of it is a NaN that would propagate into every later decision.
        """
        quadratic = float(context @ (self.inverse @ context))
        return float(np.sqrt(max(quadratic, 0.0)))

    def direct_inverse(self) -> npt.NDArray[np.float64]:
        """``A^-1`` computed from scratch. The reference for the incremental path."""
        return np.linalg.inv(self.precision)


@dataclass(frozen=True, slots=True)
class Decision:
    """One policy decision, with everything needed to audit it.

    The sampled values are kept because a decision an analyst cannot account for is
    not acceptable in this system, and "the policy explored" is a legitimate
    explanation only if it can be distinguished from "the policy was confident".
    :attr:`was_exploratory` makes that distinction available.
    """

    action: ResponseAction
    tier: RiskTier
    #: Posterior sample per *allowed* action; forbidden arms are absent, not zeroed.
    sampled_values: dict[ResponseAction, float]
    #: Posterior-mean estimate per allowed action, before exploration noise.
    expected_values: dict[ResponseAction, float]
    #: Posterior standard deviation per allowed action at this context.
    uncertainties: dict[ResponseAction, float]

    @property
    def greedy_action(self) -> ResponseAction:
        """What the posterior mean alone would have chosen."""
        # Same tie-break as :meth:`LinearThompsonBandit.select`, so
        # ``was_exploratory`` compares like with like. A greedy action computed under a
        # different tie rule would report exploration that did not happen.
        return max(
            self.expected_values,
            key=lambda a: (self.expected_values[a], ALL_ACTIONS.index(a)),
        )

    @property
    def was_exploratory(self) -> bool:
        """True when sampling overrode the greedy choice."""
        return self.action is not self.greedy_action

    @property
    def confidence(self) -> float:
        """Margin between the chosen arm's sample and the runner-up, in ``[0, 1]``.

        A squashed margin rather than a probability: it is monotone in how clearly the
        arm won and is comparable across decisions, which is what a triage confidence
        needs to be. It is not calibrated and is not presented as such.
        """
        if len(self.sampled_values) < 2:
            return 1.0
        ordered = sorted(self.sampled_values.values(), reverse=True)
        margin = ordered[0] - ordered[1]
        return float(np.tanh(max(margin, 0.0)))


@dataclass(slots=True)
class LinearThompsonBandit:
    """Contextual Thompson sampling with a structural trust-tier action mask."""

    n_features: int
    prior_precision: float = DEFAULT_PRIOR_PRECISION
    posterior_scale: float = DEFAULT_POSTERIOR_SCALE
    #: Rewards are divided by this before the posterior update, so
    #: :attr:`posterior_scale` is denominated in reward standard deviations rather
    #: than in raw reward units. Pass
    #: :attr:`~sentinel.rl.reward.RewardModel.reward_spread`. Leaving it at 1.0 is
    #: valid and means "my rewards are already unit-scaled"; it is not a no-op to be
    #: ignored, because a mis-scaled posterior is an over-confident one.
    reward_scale: float = 1.0
    prior_reward: float = DEFAULT_PRIOR_REWARD
    seed: int = 0
    arms: dict[ResponseAction, ArmPosterior] = field(default_factory=dict, repr=False)
    jitter_events: int = 0
    _rng: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise BanditError("n_features must be at least 1")
        if self.posterior_scale < 0.0:
            raise BanditError(
                f"posterior_scale must be non-negative, got {self.posterior_scale}"
            )
        if self.reward_scale <= 0.0:
            raise BanditError(
                f"reward_scale must be positive, got {self.reward_scale}"
            )
        if not self.arms:
            self.arms = {
                action: ArmPosterior(
                    n_features=self.n_features,
                    prior_precision=self.prior_precision,
                    prior_reward=self.prior_reward,
                )
                for action in ALL_ACTIONS
            }
        missing = [a.value for a in ALL_ACTIONS if a not in self.arms]
        if missing:
            raise BanditError(f"no posterior for arms {missing}")
        self._rng = np.random.default_rng(self.seed)

    # --- selection ------------------------------------------------------------ #

    def select(
        self,
        context: npt.NDArray[np.float64],
        *,
        tier: RiskTier,
        greedy: bool = False,
    ) -> Decision:
        """Choose an action for ``context``, restricted to what ``tier`` permits.

        ``greedy`` suppresses sampling and takes the posterior mean. Not an
        exploration schedule -- it is for evaluation, where the question is what the
        policy has *learned* rather than what it would try next.
        """
        vector = self._validate(context)
        mask = ActionMask.for_tier(tier)

        sampled: dict[ResponseAction, float] = {}
        expected: dict[ResponseAction, float] = {}
        uncertainty: dict[ResponseAction, float] = {}
        for action in mask.allowed:
            arm = self.arms[action]
            expected[action] = arm.predict(vector)
            uncertainty[action] = arm.uncertainty(vector)
            if greedy or self.posterior_scale == 0.0:
                sampled[action] = expected[action]
            else:
                theta = self._sample_theta(arm)
                sampled[action] = float(vector @ theta)

        # Ties break toward the *least* interventionist action, and this is not a
        # cosmetic detail. ``ALL_ACTIONS`` runs most- to least-interventionist, so the
        # larger index is the more cautious choice, and it has to be preferred because
        # the optimistic prior makes the very first decision for any context an exact
        # four-way tie. With the comparison the other way round, the policy's opening
        # move on a fresh deployment at a permissive tier is to contain a host --
        # which is the worst possible cold-start default and is what
        # ``test_ties_prefer_the_less_interventionist_action`` was written to catch.
        action = max(sampled, key=lambda a: (sampled[a], ALL_ACTIONS.index(a)))

        # The mask already excluded everything else; this asserts the outcome rather
        # than trusting the loop above, because this is the invariant that matters.
        mask.require(action)
        return Decision(
            action=action,
            tier=tier,
            sampled_values=sampled,
            expected_values=expected,
            uncertainties=uncertainty,
        )

    def _sample_theta(self, arm: ArmPosterior) -> npt.NDArray[np.float64]:
        """Draw ``theta ~ N(mu, v^2 A^-1)`` via a Cholesky factor of ``A^-1``."""
        covariance = (self.posterior_scale**2) * arm.inverse
        factor = self._cholesky(covariance)
        standard = self._rng.standard_normal(self.n_features)
        return arm.mean + factor @ standard

    def _cholesky(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Cholesky factor, adding the least jitter that makes it succeed."""
        symmetric = 0.5 * (matrix + matrix.T)
        base = float(np.trace(symmetric)) / self.n_features
        for index, jitter in enumerate(_JITTER_LADDER):
            try:
                candidate = symmetric + jitter * max(base, 1.0) * np.eye(
                    self.n_features, dtype=DTYPE
                )
                factor = np.linalg.cholesky(candidate)
            except np.linalg.LinAlgError:
                continue
            if index > 0:
                self.jitter_events += 1
            return factor
        raise BanditError(
            "posterior covariance is not positive definite even with maximum jitter; "
            "the incremental inverse has diverged and the policy must be rebuilt"
        )

    # --- learning ------------------------------------------------------------- #

    def update(
        self,
        context: npt.NDArray[np.float64],
        action: ResponseAction,
        reward: float,
        *,
        tier: RiskTier | None = None,
    ) -> None:
        """Absorb the reward for an action that was actually taken.

        ``tier`` is optional and, when given, is *checked*: updating an arm the tier
        forbids means the arm was played, which is the violation the mask exists to
        prevent. Learning from it would also quietly launder the violation into the
        posterior, so it is refused rather than logged.
        """
        vector = self._validate(context)
        if action not in self.arms:
            raise BanditError(f"unknown action {action!r}")
        if tier is not None:
            ActionMask.for_tier(tier).require(action)
        self.arms[action].update(vector, reward / self.reward_scale)

    def _validate(self, context: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        vector = np.ascontiguousarray(context, dtype=DTYPE).ravel()
        if vector.shape != (self.n_features,):
            raise BanditError(
                f"context has {vector.shape[0]} features, expected {self.n_features}"
            )
        if not np.all(np.isfinite(vector)):
            raise BanditError("context contains non-finite values")
        return vector

    # --- introspection -------------------------------------------------------- #

    @property
    def total_observations(self) -> int:
        return sum(arm.observations for arm in self.arms.values())

    def cold_arms(self, *, minimum: int = 1) -> tuple[ResponseAction, ...]:
        """Arms with fewer than ``minimum`` observations.

        Reported because the mask guarantees they exist: at ``observe`` tier the
        ``auto_contain`` posterior is untouched by construction, so a tier promotion
        hands the policy an arm it has never played. Naming them is the difference
        between a known limitation and a surprise.
        """
        return tuple(
            action
            for action in ALL_ACTIONS
            if self.arms[action].observations < minimum
        )

    def max_inverse_drift(self) -> float:
        """Largest deviation of any incremental ``A^-1`` from a direct solve.

        The audit on the Sherman-Morrison fast path. Asserted small by
        ``test_rl_bandit.py``; if it grows, the incremental update is no longer the
        thing it claims to be.
        """
        if not self.arms:
            return 0.0
        return max(
            float(np.max(np.abs(arm.inverse - arm.direct_inverse())))
            for arm in self.arms.values()
        )

    def snapshot(self) -> dict[str, object]:
        """Policy state for the audit log and the evaluation report."""
        return {
            "n_features": self.n_features,
            "prior_precision": self.prior_precision,
            "posterior_scale": self.posterior_scale,
            "reward_scale": self.reward_scale,
            "prior_reward": self.prior_reward,
            "seed": self.seed,
            "total_observations": self.total_observations,
            "jitter_events": self.jitter_events,
            "arms": {
                action.value: {
                    "observations": arm.observations,
                    "mean_reward": arm.mean_reward,
                    "coefficient_norm": float(np.linalg.norm(arm.mean)),
                }
                for action, arm in self.arms.items()
            },
        }
