"""The model lab: what the dashboard trains at start-up, and how each model did.

Part 5.1. Two things the PRD asks for were measured offline but absent from the
live system, and this module closes both:

*   **The response policy in the live flow** (PRD Section 5.5.4). F-09's bandit was
    built and its regret measured by ``scripts/evaluate.py --policy``, but the
    dashboard's Containment Agent ran with no policy at all and simply followed
    triage. :func:`train_response_policy` fits the bandit at start-up on the same
    simulated replay F-09 uses, and :class:`ServingPolicy` hands it to the agent.
*   **Seeing the models learn.** Every model here already recorded its training
    curve — the autoencoder's loss, the GNN's loss, the bandit's regret, the
    diffusion model's loss — and nothing showed them. :func:`model_report` collects
    them for the dashboard's *Models* page.

These are *live* numbers from this process's own training run. They are not the
acceptance numbers: those come from the one evaluation pipeline (PRD Section 9.3)
and the dashboard shows them, unchanged, on its Evaluation page. The two are kept
apart on purpose — a demo that recomputed its headline metrics in a second place
would be the "separate version for the report" Section 9.3 forbids.

Serving is greedy
-----------------
The offline replay explores (Thompson sampling), because exploration is how a
bandit learns. The dashboard serves the posterior mean instead: an analyst's
incident is not the place to try an arm to see what happens, and a greedy policy
makes the same incident get the same recommendation twice, which a demo — and an
audit — both need. The policy therefore does not learn online from dashboard
decisions; that loop is Phase 3 work, and saying so is better than a feedback path
that quietly trains on whatever a demo operator clicked.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from sentinel.core.schemas import RiskTier
from sentinel.rl.actions import ALL_ACTIONS
from sentinel.rl.bandit import Decision, LinearThompsonBandit
from sentinel.rl.reward import RewardModel
from sentinel.rl.simulate import N_FEATURES, build_episodes, replay

__all__ = [
    "POLICY_EPISODES",
    "DiffusionStudy",
    "PolicyTraining",
    "ServingPolicy",
    "graph_evaluation",
    "model_report",
    "train_response_policy",
]

#: Episodes the start-up policy trains on. The F-09 replay uses 200 per seed to
#: measure a learning curve; a policy that will actually serve incidents gets more.
#: Measured on this corpus: optimal-action rate 0.63 at 200 episodes, 0.74 at 1,500,
#: in under a second.
POLICY_EPISODES: Final[int] = 1500

#: Scarcity levels for the diffusion study, matching ``evaluate.py --augment``:
#: the rare-family row cap per family, ``None`` meaning all the data.
DIFFUSION_LEVELS: Final[tuple[int | None, ...]] = (None, 200, 60, 25, 12)
RARE_FAMILIES: Final[tuple[str, ...]] = ("web_attack", "botnet", "infiltration", "brute_force")


def _downsample(values: np.ndarray | list[float], points: int = 200) -> list[float]:
    """At most ``points`` evenly spaced values, always keeping the last one."""
    array = np.asarray(values, dtype=float)
    if array.size <= points:
        return [float(v) for v in array]
    index = np.unique(np.linspace(0, array.size - 1, points).round().astype(int))
    return [float(array[i]) for i in index]


# --------------------------------------------------------------------------- #
# The response policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PolicyTraining:
    """How the start-up policy learned, next to a policy that learns nothing."""

    episodes: int
    seed: int
    tier: str
    policy_cumulative_regret: list[float]
    baseline_cumulative_regret: list[float]
    policy_total_regret: float
    baseline_total_regret: float
    optimal_action_rate: float
    regret_reduction: float
    action_counts: dict[str, int]
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "seed": self.seed,
            "tier": self.tier,
            "curve": {
                "episode": _downsample(np.arange(1, self.episodes + 1)),
                "policy": _downsample(self.policy_cumulative_regret),
                "no_learning": _downsample(self.baseline_cumulative_regret),
            },
            "policy_total_regret": self.policy_total_regret,
            "no_learning_total_regret": self.baseline_total_regret,
            "regret_ratio": self.policy_total_regret / max(self.baseline_total_regret, 1e-9),
            "optimal_action_rate": self.optimal_action_rate,
            "regret_reduction": self.regret_reduction,
            "action_counts": self.action_counts,
            "seconds": self.seconds,
        }


def train_response_policy(
    *, seed: int, episodes: int = POLICY_EPISODES
) -> tuple[LinearThompsonBandit, PolicyTraining]:
    """Fit the Section 5.5.4 bandit on the F-09 simulator, and record its curve.

    Trained at ``auto_with_notify`` so every arm is learned: the tier a customer is
    on is applied when the policy *serves* (by the action mask), and a policy that
    had never seen the containment arm could not be promoted to a tier that allows
    it without retraining.
    """
    started = time.perf_counter()
    reward = RewardModel()
    bandit = LinearThompsonBandit(
        n_features=N_FEATURES, seed=seed, reward_scale=reward.reward_spread
    )
    plan = build_episodes(n=episodes, seed=seed, tier=RiskTier.AUTO_WITH_NOTIFY)
    learned = replay(episodes=plan, bandit=bandit, model=reward, seed=seed)
    baseline = replay(episodes=plan, model=reward, seed=seed, random_policy=True)
    training = PolicyTraining(
        episodes=episodes,
        seed=seed,
        tier=RiskTier.AUTO_WITH_NOTIFY.value,
        policy_cumulative_regret=[float(v) for v in learned.cumulative_regret],
        baseline_cumulative_regret=[float(v) for v in baseline.cumulative_regret],
        policy_total_regret=learned.total_regret,
        baseline_total_regret=baseline.total_regret,
        optimal_action_rate=learned.optimal_action_rate,
        regret_reduction=learned.regret_reduction,
        action_counts={a.value: int(learned.action_counts[a]) for a in ALL_ACTIONS},
        seconds=time.perf_counter() - started,
    )
    return bandit, training


@dataclass(slots=True)
class ServingPolicy:
    """The fitted bandit, served greedily (see the module docstring).

    Shaped like :class:`~sentinel.rl.bandit.LinearThompsonBandit` where the
    Containment Agent calls it — ``select(context, tier=...)`` — so it drops into
    the agent's ``policy`` slot with no change to the agent.
    """

    bandit: LinearThompsonBandit
    decisions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def select(self, context: np.ndarray, *, tier: RiskTier, greedy: bool = True) -> Decision:
        del greedy  # always greedy when serving; the argument exists for the protocol
        with self._lock:
            self.decisions += 1
            return self.bandit.select(context, tier=tier, greedy=True)


# --------------------------------------------------------------------------- #
# The supply-chain GNN, measured on its own test split
# --------------------------------------------------------------------------- #


def graph_evaluation(graph, truth, gnn, split, *, k: int = 10) -> dict[str, Any]:
    """F-06's top-k precision on the test split, next to a features-only baseline.

    The baseline is the number that says whether the graph is load-bearing, and it
    is computed the way ``evaluate.py --graph`` computes it: a logistic regression
    on the four PRD node features, standardised on the training split.
    """
    from sklearn.linear_model import LogisticRegression

    from sentinel.graph.gnn import top_k_precision

    labels = truth.labels(graph.node_ids())
    scores = gnn.risk_scores(graph)
    raw = graph.feature_matrix()
    mean = raw[split.train].mean(axis=0)
    deviation = raw[split.train].std(axis=0)
    scaled = (raw - mean) / np.where(deviation > 1e-12, deviation, 1.0)
    baseline = LogisticRegression(max_iter=2000, class_weight="balanced")
    baseline.fit(scaled[split.train], labels[split.train])
    baseline_scores = baseline.predict_proba(scaled)[:, 1]
    test = split.test
    return {
        "k": k,
        "test_nodes": len(test),
        "gnn_top_k_precision": top_k_precision(scores[test], labels[test], k),
        "features_only_top_k_precision": top_k_precision(baseline_scores[test], labels[test], k),
        "target": 0.80,
        "note": (
            "One seed. F-06 is asserted as the mean over 10 seeds by the evaluation "
            "pipeline, because top-10 precision moves 0.1 per node on a 150-node split."
        ),
    }


# --------------------------------------------------------------------------- #
# Diffusion augmentation, studied in the background
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DiffusionStudy:
    """The Section 5.5.5 study, run on a background thread after start-up.

    About six seconds per scarcity level on a laptop, five levels, so it runs off the
    request path: the Models page shows it as *training* and then fills in. The
    result is reported whatever its sign — on this corpus augmentation is roughly
    neutral with full data and *hurts* rare-family recall when data is scarce, which
    is what Part 2 measured too. The generator's case rests on the calibration probe,
    reported by ``evaluate.py --augment`` and shown on the Evaluation page.
    """

    seed: int
    n_alerts: int = 12_000
    levels: tuple[int | None, ...] = DIFFUSION_LEVELS
    epochs: int = 250
    status: str = "pending"
    error: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    loss_curve: list[float] = field(default_factory=list)
    seconds: float = 0.0
    _thread: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> DiffusionStudy:
        with self._lock:
            if self._thread is not None:
                return self
            self._thread = threading.Thread(target=self.run, name="diffusion-study", daemon=True)
            self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self) -> None:
        from sentinel.ml.classify import FamilyClassifier
        from sentinel.ml.datasets.synthetic import generate_alerts
        from sentinel.ml.diffusion import TabularDiffusion
        from sentinel.ml.featurestore import AlertVectorizer
        from sentinel.ml.robustness import augment_training_set

        started = time.perf_counter()
        with self._lock:
            self.status = "running"
        try:
            alerts = generate_alerts(self.n_alerts, seed=self.seed)
            vectorizer = AlertVectorizer().fit(alerts)
            matrix = vectorizer.transform(alerts)
            families = np.asarray([a.ground_truth_label or "benign" for a in alerts])
            order = np.random.default_rng(7).permutation(len(families))
            cut = int(0.6 * len(order))
            x_train, f_train = matrix[order[:cut]], families[order[:cut]]
            x_test, f_test = matrix[order[cut:]], list(families[order[cut:]])

            def recall(model) -> dict[str, float]:
                per_family = model.recall_by_family(x_test, f_test)
                return {name: float(per_family.get(name, 0.0)) for name in RARE_FAMILIES}

            for cap in self.levels:
                keep: list[int] = []
                seen: Counter = Counter()
                for index, family in enumerate(f_train):
                    if family in RARE_FAMILIES:
                        if cap is not None and seen[family] >= cap:
                            continue
                        seen[family] += 1
                    keep.append(index)
                chosen = np.asarray(keep)
                sub_x, sub_f = x_train[chosen], list(f_train[chosen])
                rare_mask = np.isin(np.asarray(sub_f), RARE_FAMILIES)
                generator = TabularDiffusion(
                    n_steps=200,
                    epochs=self.epochs,
                    hidden=(160, 160),
                    seed=1,
                    min_rows_per_family=8,
                ).fit(sub_x[rare_mask], list(np.asarray(sub_f)[rare_mask]))
                augmented = augment_training_set(
                    sub_x,
                    sub_f,
                    generator=generator,
                    multiplier=6.0 if cap else 3.0,
                    rng=np.random.default_rng(11),
                )
                before = recall(FamilyClassifier(seed=5).fit(sub_x, sub_f))
                after = recall(FamilyClassifier(seed=5).fit(augmented.x, list(augmented.families)))
                row = {
                    "level": "all data" if cap is None else f"{cap} per family",
                    "real_rare_rows": int(rare_mask.sum()),
                    "synthetic_rows": int(augmented.n_synthetic),
                    "before": before,
                    "after": after,
                    "before_macro": float(np.mean(list(before.values()))),
                    "after_macro": float(np.mean(list(after.values()))),
                    "final_loss": float(generator.history_.train_loss[-1]),
                }
                with self._lock:
                    self.results.append(row)
                    if not self.loss_curve:
                        self.loss_curve = _downsample(generator.history_.train_loss)
            with self._lock:
                self.status = "done"
        except Exception as exc:  # reported on the page rather than killing the server
            with self._lock:
                self.status = "failed"
                self.error = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            self.seconds = time.perf_counter() - started

    def view(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "levels_total": len(self.levels),
                "levels_done": len(self.results),
                "results": [dict(r) for r in self.results],
                "loss_curve": list(self.loss_curve),
                "seconds": self.seconds,
                "rare_families": list(RARE_FAMILIES),
            }


# --------------------------------------------------------------------------- #
# Everything, for the Models page
# --------------------------------------------------------------------------- #


def _autoencoder(models) -> dict[str, Any]:
    ensemble = models.triage_model.ensemble
    detectors = []
    history = None
    for detector, weight in zip(ensemble.detectors, ensemble.weights, strict=True):
        detectors.append({"name": detector.name, "weight": float(weight)})
        if getattr(detector, "history_", None) is not None:
            history = detector.history_
    if history is None:
        return {"detectors": detectors, "available": False}
    return {
        "available": True,
        "detectors": detectors,
        "n_parameters": history.n_parameters,
        "epochs_run": history.epochs_run,
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "train_loss": _downsample(history.train_loss),
        "validation_loss": _downsample(history.validation_loss),
    }


def _gnn(models) -> dict[str, Any]:
    report = models.gnn.report_
    return {
        **report.as_dict(),
        "train_loss": _downsample(report.train_loss),
        "validation_loss": _downsample(report.validation_loss),
        "evaluation": models.graph_evaluation,
    }


def model_report(models, *, policy: ServingPolicy | None = None) -> dict[str, Any]:
    """Every model's training record, from this process's own start-up run."""
    real = models.mode == "real"
    supply = models.real_supply
    return {
        "mode": models.mode,
        "real_supply": None
        if not (real and supply)
        else {
            "truth": supply["truth"],
            "sources": supply["sources"],
            "exploited_in_the_wild": supply.get("exploited_in_the_wild"),
            "epss_date": supply.get("epss_date"),
            "generated_at": supply["generated_at"],
            "projects": supply["projects"],
            "cycle_edges_cut": supply["cycle_edges_cut"],
        },
        "seed": models.seed,
        "autoencoder": _autoencoder(models),
        "gnn": _gnn(models),
        "policy": {
            **models.policy_training.as_dict(),
            "live_decisions": 0 if policy is None else policy.decisions,
            "serving": "greedy (posterior mean); no online learning from dashboard clicks",
            "trained_on": "the project's response simulator, in every workspace: there is no "
            "public dataset of real analyst decisions and their outcomes",
        },
        # The diffusion study generates synthetic attack rows; it says nothing about real data.
        "diffusion": {
            "status": "not_applicable",
            "error": None,
            "levels_total": 0,
            "levels_done": 0,
            "results": [],
            "loss_curve": [],
            "seconds": None,
            "rare_families": [],
        }
        if real
        else models.diffusion.view(),
    }
