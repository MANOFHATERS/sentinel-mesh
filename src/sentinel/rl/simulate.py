"""Episode simulation and the F-09 regret replay (PRD F-09, Section 5.5.4).

F-09's acceptance criterion is *"simulated regret decreases measurably over a
200-episode replay"*. Making that claim mean something requires three things this
module provides and one it deliberately refuses.

**Episodes come from real alerts.** The context is built from alerts produced by the
same pipeline F-03 is measured on — synthetic CIC-IDS2017-shaped rows through the
actual :class:`~sentinel.ingest.normalizer.CICIDS2017Normalizer` and the actual
:class:`~sentinel.ml.featurestore.AlertVectorizer`. A bespoke toy environment would
let the bandit learn a decision boundary that exists nowhere else in the system, and
the regret curve would measure the simulator.

**The context is what the agents would actually have.** PRD Section 5.5.4 says
*"state is the concatenated alert/investigation embedding plus asset criticality"*.
The concrete features are listed on :class:`ContextSpec`. The important omission:
``Alert.ground_truth_label`` is never in the context. It reaches
:class:`~sentinel.rl.reward.Outcome` to compute the reward, exactly as an analyst's
later verdict would, and ``test_rl_simulate.py`` asserts no context feature is a
function of it.

**Regret is measured against an oracle inside the same mask.** An oracle allowed to
auto-contain at ``observe`` tier would make the policy look permanently regretful for
honouring its own safety constraint, and the number would then describe the tier
rather than the learning.

What it refuses: a single seed. :data:`REPORTING_SEEDS` and :data:`TUNING_SEEDS` are
disjoint and fixed here, in advance, because Part 2 already paid for this lesson — a
27-configuration grid scored 0.825 on its tuning seeds and 0.740 held out. Every
hyper-parameter in :mod:`sentinel.rl.bandit` was chosen on :data:`TUNING_SEEDS`; every
number in ``docs/BUILD_PLAN.md`` and the F-09 gate come from :data:`REPORTING_SEEDS`.

Why the noise lives here
------------------------
:meth:`~sentinel.rl.reward.RewardModel.reward` is deterministic so the oracle is exact.
Real feedback is not: analysts reverse some containments and not others, and the
anomaly score the policy sees is an estimate. Both are modelled here —
``reversal_probability`` and ``score_noise`` — which keeps the reward a fixed
reference while still making the learning problem non-trivial. A bandit on a
noiseless problem converges in a handful of episodes and proves nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import Alert, RiskTier, Severity
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.rl.actions import ALL_ACTIONS, ActionMask, ResponseAction
from sentinel.rl.bandit import LinearThompsonBandit
from sentinel.rl.reward import Outcome, RewardModel

__all__ = [
    "GATE_MEAN_SUBLINEARITY",
    "GATE_OPTIMAL_ACTION_RATE",
    "GATE_REGRET_RATIO",
    "GATE_WORST_SUBLINEARITY",
    "N_FEATURES",
    "REPORTING_SEEDS",
    "TUNING_SEEDS",
    "ContextSpec",
    "Episode",
    "ReplayResult",
    "SimulationError",
    "aggregate",
    "assert_f09_gates",
    "asset_criticality",
    "build_context",
    "build_episodes",
    "replay",
    "replay_many_seeds",
]

DTYPE: Final = np.float64

#: Seeds used to choose hyper-parameters. Never reported on.
TUNING_SEEDS: Final[tuple[int, ...]] = (11, 23, 37, 41, 53)

#: Seeds every reported F-09 number comes from. Disjoint from :data:`TUNING_SEEDS`,
#: fixed before any measurement was taken, and never used to select anything.
REPORTING_SEEDS: Final[tuple[int, ...]] = (20260929, 20260930, 20261001, 20261002, 20261003)


class SimulationError(SentinelError):
    """The simulation was configured incorrectly."""


@dataclass(frozen=True, slots=True)
class ContextSpec:
    """The policy's observation. Documented because omissions matter as much as content.

    Present:

    *   ``bias`` — a constant 1.0, so each arm can learn an intercept. Without it a
        linear model is forced through the origin and cannot represent "dismiss is
        usually right", which is the single most common correct answer.
    *   ``anomaly_score`` — the F-03 detector's output for this alert, the main signal.
    *   ``anomaly_score_squared`` — one interaction term. The reward surface is not
        linear in the score: the value of ``escalate`` peaks in the *middle* of the
        range, where the evidence is genuinely ambiguous, and a purely linear model
        cannot express a peak. This is the cheapest possible fix and it is why
        ``escalate`` is learnable at all.
    *   ``severity`` — the triage severity rank, scaled to ``[0, 1]``.
    *   ``criticality`` — the asset criticality the PRD names explicitly.
    *   ``criticality_x_score`` — their product. Containment on a critical asset when
        the score is high is a different decision from either factor alone, and this
        is the term that carries it.
    *   ``flow_burst`` / ``port_rarity`` — two enrichment features from the feature
        store, standing in for the fuller alert embedding Phase 2 would supply.

    Absent, deliberately: ``ground_truth_label`` and anything derived from it. It is
    the reward's input, never the policy's.
    """

    names: tuple[str, ...] = (
        "bias",
        "anomaly_score",
        "anomaly_score_squared",
        "severity",
        "criticality",
        "criticality_x_score",
        "flow_burst",
        "port_rarity",
    )

    def __len__(self) -> int:
        return len(self.names)


CONTEXT_SPEC: Final[ContextSpec] = ContextSpec()
N_FEATURES: Final[int] = len(CONTEXT_SPEC)


@dataclass(frozen=True, slots=True)
class Episode:
    """One decision problem: a context, the truth behind it, and its tier."""

    context: npt.NDArray[np.float64]
    is_attack: bool
    criticality: float
    tier: RiskTier
    alert_id: str
    #: The clean anomaly score, before observation noise. Diagnostics only; the policy
    #: sees the noisy value inside ``context``.
    true_score: float = 0.0

    def __post_init__(self) -> None:
        if self.context.shape != (N_FEATURES,):
            raise SimulationError(
                f"context has shape {self.context.shape}, expected ({N_FEATURES},)"
            )
        if not np.all(np.isfinite(self.context)):
            raise SimulationError("context contains non-finite values")


def _criticality_for(asset_id: str) -> float:
    """Asset criticality, stable per asset and reproducible across processes.

    A property of the asset, so it is derived from the asset id rather than drawn per
    episode. Re-drawing it would turn the ``criticality`` feature into pure noise and
    make ``criticality_x_score`` unlearnable, which would present as the bandit
    failing to learn rather than as a broken simulator.

    BLAKE2b, not :func:`hash`: string hashing is randomised per process, so a
    :func:`hash`-derived criticality would differ between runs and the seeded replay
    would not be reproducible -- the same trap :mod:`sentinel.kb.text` documents.
    """
    digest = int.from_bytes(
        hashlib.blake2b(asset_id.encode("utf-8"), digest_size=4).digest(), "big"
    )
    # Most assets are ordinary; a few are disposable and a few are tier zero.
    return float(np.clip(0.15 + 0.7 * ((digest % 10_000) / 10_000.0), 0.0, 1.0))


def build_episodes(
    *,
    n: int,
    seed: int,
    tier: RiskTier = RiskTier.AUTO_WITH_NOTIFY,
    score_noise: float = 0.12,
    detector_skill: float = 0.80,
) -> tuple[Episode, ...]:
    """Build ``n`` episodes from alerts produced by the real ingestion pipeline.

    ``detector_skill`` is how well the simulated anomaly score separates attacks from
    benign traffic, in ``[0, 1]``. It is a parameter rather than a constant because the
    policy's job depends on it: a perfect detector makes the decision trivial, and a
    useless one makes every policy equivalent. The default is deliberately below the
    detector's measured F-03 performance, so the bandit is not handed an easier problem
    than the rest of the system faces.
    """
    if n < 1:
        raise SimulationError("n must be at least 1")
    if not 0.0 <= score_noise <= 1.0:
        raise SimulationError(f"score_noise {score_noise} outside [0, 1]")
    if not 0.0 <= detector_skill <= 1.0:
        raise SimulationError(f"detector_skill {detector_skill} outside [0, 1]")

    alerts = generate_alerts(n, seed=seed)
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []

    for alert in alerts[:n]:
        is_attack = alert.ground_truth_label not in (None, "benign")
        criticality = _criticality_for(alert.asset_id)

        # A separated-Beta score: attacks concentrate high, benign low, with the
        # overlap controlled by ``detector_skill``. Beta rather than a clipped normal
        # because a score is bounded in [0, 1] and clipping piles mass on the
        # endpoints, which a squared feature then exaggerates.
        sharpness = 1.0 + 8.0 * detector_skill
        if is_attack:
            true_score = float(rng.beta(sharpness, 1.0 + (1.0 - detector_skill) * 4.0))
        else:
            true_score = float(rng.beta(1.0 + (1.0 - detector_skill) * 4.0, sharpness))

        observed = float(
            np.clip(true_score + rng.normal(0.0, score_noise), 0.0, 1.0)
        )
        context = build_context(alert, anomaly_score=observed, criticality=criticality)
        episodes.append(
            Episode(
                context=context,
                is_attack=is_attack,
                criticality=criticality,
                tier=tier,
                alert_id=alert.alert_id,
                true_score=true_score,
            )
        )
    return tuple(episodes)


def build_context(
    alert: Alert, *, anomaly_score: float, criticality: float | None = None
) -> npt.NDArray[np.float64]:
    """The policy's observation vector, in one place.

    Both the offline replay (:func:`build_episodes`) and the live Containment
    Agent call this. That is not tidiness — it is the same discipline PRD Section
    7.3 applies to the feature store, for the same reason. A bandit whose
    posteriors were learned over one arrangement of eight numbers and is then
    served a different arrangement does not fail; it returns confident nonsense,
    because a linear model has no way to notice that ``severity`` and
    ``criticality`` swapped places. There is therefore exactly one construction
    of this vector, and :data:`CONTEXT_SPEC` names its entries in order.

    ``criticality`` defaults to :func:`_criticality_for`, which derives it from
    the asset id, so the live path and the replay agree on what a given asset is
    worth without a shared inventory service.
    """
    if not 0.0 <= anomaly_score <= 1.0:
        raise SimulationError(
            f"anomaly_score {anomaly_score} outside [0, 1]; the policy's posteriors "
            "were learned over a bounded score and an unbounded one silently "
            "rescales every arm"
        )
    weight = _criticality_for(alert.asset_id) if criticality is None else float(criticality)
    if not 0.0 <= weight <= 1.0:
        raise SimulationError(f"criticality {weight} outside [0, 1]")
    vector = np.asarray(
        [
            1.0,
            anomaly_score,
            anomaly_score * anomaly_score,
            _severity_rank(alert),
            weight,
            weight * anomaly_score,
            _feature(alert, "src_flow_count_window", default=0.0, scale=20.0),
            _feature(alert, "dst_port_rarity", default=0.5, scale=1.0),
        ],
        dtype=DTYPE,
    )
    if vector.shape != (N_FEATURES,):  # pragma: no cover - guarded by CONTEXT_SPEC
        raise SimulationError(
            f"context has shape {vector.shape}, but CONTEXT_SPEC names {N_FEATURES} "
            "features; the two have drifted apart"
        )
    return vector


def asset_criticality(asset_id: str) -> float:
    """Public name for the asset-criticality derivation (PRD Section 5.5.4)."""
    return _criticality_for(asset_id)


def _severity_rank(alert: Alert) -> float:
    if alert.triage is not None:
        return alert.triage.severity.rank / float(Severity.CRITICAL.rank)
    return 0.5


def _feature(alert: Alert, name: str, *, default: float, scale: float) -> float:
    raw = alert.features.get(name, default)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return float(np.clip(value / scale, 0.0, 1.0)) if scale else value


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Outcome of one replay: per-episode regret and everything needed to judge it."""

    seed: int
    n_episodes: int
    tier: RiskTier
    #: Per-episode instantaneous regret, ``oracle_reward - policy_reward``.
    regret: npt.NDArray[np.float64] = field(repr=False)
    #: Per-episode reward the policy actually collected.
    rewards: npt.NDArray[np.float64] = field(repr=False)
    #: Per-episode reward the masked oracle would have collected.
    oracle_rewards: npt.NDArray[np.float64] = field(repr=False)
    action_counts: dict[ResponseAction, int] = field(default_factory=dict)
    #: Actions the policy chose that the tier forbade. Must be zero, always.
    violations: int = 0
    exploratory: int = 0
    max_inverse_drift: float = 0.0
    jitter_events: int = 0

    @property
    def cumulative_regret(self) -> npt.NDArray[np.float64]:
        return np.cumsum(self.regret)

    @property
    def total_regret(self) -> float:
        return float(self.regret.sum())

    @property
    def mean_reward(self) -> float:
        return float(self.rewards.mean())

    def window_regret(self, start: float, end: float) -> float:
        """Mean regret over a fractional window of the replay, e.g. ``(0.0, 0.25)``."""
        if not 0.0 <= start < end <= 1.0:
            raise SimulationError(f"invalid window ({start}, {end})")
        lo = int(start * self.n_episodes)
        hi = int(end * self.n_episodes)
        if hi <= lo:
            raise SimulationError("window covers no episodes")
        return float(self.regret[lo:hi].mean())

    @property
    def regret_reduction(self) -> float:
        """Fractional fall in mean regret from the first quarter to the last.

        The F-09 number. Defined as a *fraction* rather than a difference so it is
        comparable across seeds whose initial regret differs, and reported as the
        mean over :data:`REPORTING_SEEDS` rather than from a single run, because a
        single 200-episode bandit trace is noisy enough that one seed can show almost
        anything.
        """
        first = self.window_regret(0.0, 0.25)
        last = self.window_regret(0.75, 1.0)
        if first <= 0.0:
            return 0.0 if last > 0.0 else 1.0
        return float((first - last) / first)

    @property
    def sublinearity(self) -> float:
        """How far below a no-learning projection the total regret came in.

        A policy that never improves accrues its first-quarter regret rate for the
        whole replay, giving ``first_quarter_mean * n_episodes``. This reports the
        fraction of that projection the policy avoided: 0.0 means it never improved,
        1.0 means it stopped accruing regret entirely, negative means it got worse.

        **This is the F-09 statistic**, in preference to
        :attr:`regret_reduction`, and the reason is a measurement rather than taste.
        The quartile ratio is unstable precisely when the policy does well: a policy
        that converges inside the first fifty episodes has almost identical regret in
        its first and last quarters, so ordinary fluctuation can make the ratio
        negative on a run that learned perfectly well. Sublinearity integrates over
        the whole replay instead of comparing two small windows, and on the tuning
        seeds it is positive for every seed where the quartile ratio is not.

        Both are reported. Only this one is gated.
        """
        first = self.window_regret(0.0, 0.25)
        if first <= 0.0:
            return 1.0
        projected = first * self.n_episodes
        return float((projected - self.total_regret) / projected)

    @property
    def optimal_action_rate(self) -> float:
        """Fraction of episodes where the policy matched the masked oracle."""
        return float(np.mean(self.regret <= 1e-12))

    def summary(self) -> str:
        counts = ", ".join(
            f"{a.value}={self.action_counts.get(a, 0)}" for a in ALL_ACTIONS
        )
        return (
            f"seed={self.seed} tier={self.tier.value} n={self.n_episodes} "
            f"total_regret={self.total_regret:.2f} "
            f"first_q={self.window_regret(0.0, 0.25):.3f} "
            f"last_q={self.window_regret(0.75, 1.0):.3f} "
            f"reduction={self.regret_reduction:+.1%} "
            f"sublinear={self.sublinearity:+.1%} "
            f"optimal={self.optimal_action_rate:.1%} violations={self.violations} "
            f"[{counts}]"
        )


def replay(
    *,
    episodes: Sequence[Episode],
    bandit: LinearThompsonBandit | None = None,
    model: RewardModel | None = None,
    seed: int = 0,
    reversal_probability: float = 0.7,
    random_policy: bool = False,
    greedy: bool = False,
) -> ReplayResult:
    """Run one policy over ``episodes`` and measure regret against a masked oracle.

    ``random_policy`` replaces Thompson sampling with a uniform draw over the allowed
    arms. That baseline is not decoration: a regret curve that falls is only evidence
    of learning if a policy that learns nothing produces one that does not, and on a
    non-stationary-looking trace it is easy to mistake the environment for the policy.
    """
    if not episodes:
        raise SimulationError("no episodes to replay")
    if not 0.0 <= reversal_probability <= 1.0:
        raise SimulationError(
            f"reversal_probability {reversal_probability} outside [0, 1]"
        )

    reward_model = model if model is not None else RewardModel()
    policy = (
        bandit
        if bandit is not None
        else LinearThompsonBandit(
            n_features=N_FEATURES, seed=seed, reward_scale=reward_model.reward_spread
        )
    )
    rng = np.random.default_rng(seed ^ 0x5EED)

    regret = np.zeros(len(episodes), dtype=DTYPE)
    rewards = np.zeros(len(episodes), dtype=DTYPE)
    oracle_rewards = np.zeros(len(episodes), dtype=DTYPE)
    counts: dict[ResponseAction, int] = dict.fromkeys(ALL_ACTIONS, 0)
    violations = 0
    exploratory = 0

    for index, episode in enumerate(episodes):
        mask = ActionMask.for_tier(episode.tier)

        if random_policy:
            action = ALL_ACTIONS[
                mask.indices[int(rng.integers(len(mask.indices)))]
            ]
            decision = None
        else:
            decision = policy.select(episode.context, tier=episode.tier, greedy=greedy)
            action = decision.action
            exploratory += int(decision.was_exploratory)

        if not mask.permits(action):  # pragma: no cover - the mask makes this dead
            violations += 1

        # The reversal is drawn per episode, so the *same* false containment is
        # sometimes caught and sometimes not. That is what makes this a bandit problem
        # rather than a lookup: the policy has to learn from a noisy signal.
        reverses = (
            action is ResponseAction.AUTO_CONTAIN
            and not episode.is_attack
            and bool(rng.random() < reversal_probability)
        )
        realised = reward_model.reward(
            Outcome(
                action=action,
                is_attack=episode.is_attack,
                criticality=episode.criticality,
                human_reverses=reverses,
            )
        )
        oracle = reward_model.best_reward(
            is_attack=episode.is_attack,
            criticality=episode.criticality,
            mask=mask,
            reversal_probability=reversal_probability,
        )

        rewards[index] = realised
        oracle_rewards[index] = oracle
        # Expected-reward regret, not realised-reward regret. Scoring against the
        # realised draw would credit the policy for a lucky un-reversed false
        # containment, and regret would then include the environment's coin flip.
        expected = reward_model.expected_reward(
            action=action,
            is_attack=episode.is_attack,
            criticality=episode.criticality,
            reversal_probability=reversal_probability,
        )
        regret[index] = max(oracle - expected, 0.0)
        counts[action] += 1

        if not random_policy:
            policy.update(episode.context, action, realised, tier=episode.tier)

    return ReplayResult(
        seed=seed,
        n_episodes=len(episodes),
        tier=episodes[0].tier,
        regret=regret,
        rewards=rewards,
        oracle_rewards=oracle_rewards,
        action_counts=counts,
        violations=violations,
        exploratory=exploratory,
        max_inverse_drift=0.0 if random_policy else policy.max_inverse_drift(),
        jitter_events=0 if random_policy else policy.jitter_events,
    )


def replay_many_seeds(
    *,
    seeds: Sequence[int] = REPORTING_SEEDS,
    n_episodes: int = 200,
    tier: RiskTier = RiskTier.AUTO_WITH_NOTIFY,
    model: RewardModel | None = None,
    random_policy: bool = False,
    reversal_probability: float = 0.7,
    prior_precision: float | None = None,
    posterior_scale: float | None = None,
    prior_reward: float | None = None,
) -> tuple[ReplayResult, ...]:
    """Replay across several seeds. The only entry point any reported number uses."""
    if not seeds:
        raise SimulationError("no seeds given")
    results: list[ReplayResult] = []
    for seed in seeds:
        episodes = build_episodes(n=n_episodes, seed=seed, tier=tier)
        kwargs: dict[str, float] = {}
        if prior_precision is not None:
            kwargs["prior_precision"] = prior_precision
        if posterior_scale is not None:
            kwargs["posterior_scale"] = posterior_scale
        if prior_reward is not None:
            kwargs["prior_reward"] = prior_reward
        bandit = LinearThompsonBandit(
            n_features=N_FEATURES,
            seed=seed,
            reward_scale=(model if model is not None else RewardModel()).reward_spread,
            **kwargs,  # type: ignore[arg-type]
        )
        results.append(
            replay(
                episodes=episodes,
                bandit=bandit,
                model=model,
                seed=seed,
                random_policy=random_policy,
                reversal_probability=reversal_probability,
            )
        )
    return tuple(results)


#: Mean sublinearity gate. Measured 0.422 on :data:`REPORTING_SEEDS`.
GATE_MEAN_SUBLINEARITY: Final[float] = 0.20

#: Per-seed sublinearity gate. Measured worst case 0.231. Gated per seed as well as on
#: the mean, because a mean over five seeds can be carried by two good ones and the
#: claim being made is that the policy learns, not that it sometimes learns.
GATE_WORST_SUBLINEARITY: Final[float] = 0.10

#: Total regret as a fraction of a non-learning policy's. Measured 0.201.
#: This is the gate that makes the sublinearity numbers mean something: a policy can be
#: sublinear and still be bad, and "improves on itself" is only interesting alongside
#: "beats not learning at all".
GATE_REGRET_RATIO: Final[float] = 0.40

#: Fraction of episodes matching the masked oracle. Measured 0.600, random 0.242.
GATE_OPTIMAL_ACTION_RATE: Final[float] = 0.45


def assert_f09_gates(
    results: Sequence[ReplayResult], baseline: Sequence[ReplayResult]
) -> dict[str, float]:
    """Raise unless the replay clears the F-09 acceptance criteria.

    F-09 reads *"simulated regret decreases measurably over a 200-episode replay"*.
    Four checks, because that sentence alone is satisfiable by a policy nobody would
    ship: regret must fall (sublinearity, on the mean and on every seed), the policy
    must beat not learning at all (the regret ratio), it must actually agree with the
    oracle often (the optimal-action rate), and it must never once have chosen an
    action its trust tier forbade.

    The violation count is checked against zero with no tolerance. It is not a quality
    metric that can be traded against the others -- a single violation means the mask
    failed, and a mask that fails occasionally is not a safety control.
    """
    if not results or not baseline:
        raise SimulationError("both a policy run and a baseline run are required")

    measured = aggregate(results)
    reference = aggregate(baseline)
    ratio = (
        measured["mean_total_regret"] / reference["mean_total_regret"]
        if reference["mean_total_regret"] > 0.0
        else float("inf")
    )

    failures: list[str] = []
    if measured["total_violations"] > 0.0:
        failures.append(
            f"{int(measured['total_violations'])} trust-tier violation(s); the action "
            "mask is not a safety control if it holds only most of the time"
        )
    if measured["mean_sublinearity"] < GATE_MEAN_SUBLINEARITY:
        failures.append(
            f"mean sublinearity {measured['mean_sublinearity']:.3f} < "
            f"gate {GATE_MEAN_SUBLINEARITY:.2f}"
        )
    if measured["worst_sublinearity"] < GATE_WORST_SUBLINEARITY:
        failures.append(
            f"worst-seed sublinearity {measured['worst_sublinearity']:.3f} < "
            f"gate {GATE_WORST_SUBLINEARITY:.2f}"
        )
    if ratio > GATE_REGRET_RATIO:
        failures.append(
            f"total regret is {ratio:.3f} of the non-learning baseline, "
            f"gate {GATE_REGRET_RATIO:.2f}"
        )
    if measured["mean_optimal_action_rate"] < GATE_OPTIMAL_ACTION_RATE:
        failures.append(
            f"optimal-action rate {measured['mean_optimal_action_rate']:.3f} < "
            f"gate {GATE_OPTIMAL_ACTION_RATE:.2f}"
        )
    if failures:
        raise SimulationError("F-09 gates failed: " + "; ".join(failures))

    return {**measured, "regret_ratio": ratio}


def aggregate(results: Sequence[ReplayResult]) -> dict[str, float]:
    """Mean statistics across replays, which is what gets reported."""
    if not results:
        raise SimulationError("no results to aggregate")
    return {
        "seeds": float(len(results)),
        "mean_total_regret": float(np.mean([r.total_regret for r in results])),
        "mean_first_quarter_regret": float(
            np.mean([r.window_regret(0.0, 0.25) for r in results])
        ),
        "mean_last_quarter_regret": float(
            np.mean([r.window_regret(0.75, 1.0) for r in results])
        ),
        "mean_regret_reduction": float(np.mean([r.regret_reduction for r in results])),
        "worst_regret_reduction": float(np.min([r.regret_reduction for r in results])),
        "mean_sublinearity": float(np.mean([r.sublinearity for r in results])),
        "worst_sublinearity": float(np.min([r.sublinearity for r in results])),
        "mean_optimal_action_rate": float(
            np.mean([r.optimal_action_rate for r in results])
        ),
        "worst_optimal_action_rate": float(
            np.min([r.optimal_action_rate for r in results])
        ),
        "mean_reward": float(np.mean([r.mean_reward for r in results])),
        "total_violations": float(sum(r.violations for r in results)),
        "max_inverse_drift": float(np.max([r.max_inverse_drift for r in results])),
    }
