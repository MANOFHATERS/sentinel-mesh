"""Offline evaluation (PRD F-12, Section 9.3).

The single evaluation pipeline. PRD Section 9.3: *"there is exactly one evaluation
pipeline, not a separate 'for the report' version, so the numbers presented are the
numbers the system actually produces."* Anything that quotes a metric — the report
view in the dashboard, the pitch deck, the tests — reads it from here.

Usage::

    python scripts/evaluate.py --n 20000
    python scripts/evaluate.py --n 20000 --cross-dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import contextlib

import numpy as np

from sentinel.ml.anomaly import build_default_ensemble
from sentinel.ml.datasets.synthetic import generate_alerts
from sentinel.ml.deep import build_deep_ensemble
from sentinel.ml.featurestore import AlertVectorizer
from sentinel.ml.metrics import (
    SOC_ATTACK_BASE_RATE,
    detection_report,
    roc_auc,
    three_way_split,
)

BENIGN = "benign"


def labels_of(alerts: list[Any]) -> np.ndarray:
    return np.array([0 if a.ground_truth_label == BENIGN else 1 for a in alerts], dtype=int)


def run(
    *,
    n: int,
    seed: int,
    separability: float,
    target_fpr: float,
    cross_dataset: bool,
    deep: bool = True,
    min_weight: float = 0.10,
) -> dict[str, Any]:
    alerts = generate_alerts(n, seed=seed, dataset="cic", separability=separability)
    y = labels_of(alerts)
    families = [a.ground_truth_label for a in alerts]

    split = three_way_split(y, seed=seed)
    train = [alerts[i] for i in split.train_benign]
    validation = [alerts[i] for i in split.validation]
    test = [alerts[i] for i in split.test]

    # Fitted on the training split only. Fitting the scaler on all the data is the
    # most common silent leak in a security-ML pipeline, so it happens here, once,
    # visibly.
    vectorizer = AlertVectorizer().fit(train)
    x_train = vectorizer.transform(train)
    x_validation = vectorizer.transform(validation)
    x_test = vectorizer.transform(test)

    # The PRD Section 5.5.2 ensemble (Isolation Forest + nonlinear denoising
    # autoencoder) is the default. `--shallow` selects the Part 1 linear baseline,
    # so the two are comparable from one command rather than from a git checkout.
    ensemble = build_deep_ensemble(random_state=seed) if deep else build_default_ensemble()
    ensemble.fit(x_train)
    # min_weight keeps the second detector's signal alive: unconstrained tuning
    # assigns the Isolation Forest weight 0.0 against the autoencoder, which is
    # AUC-optimal and defeats the point of an ensemble. See
    # WeightedEnsemble.tune_weights for the measured cost of the floor.
    tuning = ensemble.tune_weights(
        x_validation, y[split.validation], min_weight=min_weight if deep else 0.0
    )
    threshold = ensemble.threshold_for_fpr(x_train, target_fpr)

    detail = ensemble.score_detail(x_test)
    report = detection_report(
        split="test",
        y_true=y[split.test],
        scores=detail.combined,
        threshold=threshold,
        families=[families[i] for i in split.test],
        per_detector_scores=detail.per_detector,
    )

    result: dict[str, Any] = {
        "dataset": "cic-ids2017-synthetic",
        "ensemble": "deep (IF + denoising autoencoder)" if deep else "shallow (IF + PCA)",
        "detectors": [d.name for d in ensemble.detectors],
        "min_weight": min_weight if deep else 0.0,
        "n_alerts": len(alerts),
        "separability": separability,
        "split_sizes": split.sizes,
        "spec": vectorizer.spec.name,
        "spec_fingerprint": vectorizer.spec.fingerprint,
        "n_features": vectorizer.spec.width,
        "degenerate_columns": list(vectorizer.degenerate_columns),
        "weights": tuning,
        "threshold": threshold,
        "target_fpr": target_fpr,
        "test": {
            "roc_auc": report.roc_auc,
            "pr_auc": report.average_precision,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "alert_reduction": report.alert_reduction,
            "false_positive_rate": report.false_positive_rate,
            # The split is ~33% attacks, where better recall mechanically lowers the
            # reduction figure. Projected onto a realistic feed as well, so PRD 9.1
            # is assessed against the deployment it describes rather than the split.
            "alert_reduction_at_soc_base_rate": report.reduction_at_base_rate(
                SOC_ATTACK_BASE_RATE
            ),
            "soc_base_rate": SOC_ATTACK_BASE_RATE,
            "per_detector_auc": report.per_detector_auc,
            "per_family_recall": report.per_family_recall,
        },
        "f03_auc_pass": report.meets_f03_auc,
    }

    # The autoencoder's architecture and training curve belong in the artifact:
    # F-12 requires the report to come from the same pipeline as the demo, and an
    # architecture nobody recorded is not reproducible.
    for detector in ensemble.detectors:
        if hasattr(detector, "training_report"):
            result["autoencoder"] = detector.training_report()

    print(f"[{result['ensemble']}]")
    print(report.summary())

    if cross_dataset:
        # PRD Section 10: cross-dataset validation, so the detector is not merely
        # memorising one lab's traffic generator. Train on CIC, score UNSW —
        # different schema, different units, different label vocabulary, and only
        # the unified feature space in common.
        other = generate_alerts(n, seed=seed + 1, dataset="unsw", separability=separability)
        y_other = labels_of(other)
        x_other = vectorizer.transform(other)
        cross_auc = roc_auc(y_other, ensemble.score(x_other))
        result["cross_dataset"] = {
            "dataset": "unsw-nb15-synthetic",
            "n_alerts": len(other),
            "roc_auc": cross_auc,
            "degradation_vs_in_domain": report.roc_auc - cross_auc,
        }
        print(
            f"[cross-dataset unsw] ROC-AUC {cross_auc:.4f} "
            f"(in-domain {report.roc_auc:.4f}, drop {report.roc_auc - cross_auc:+.4f})"
        )

    return result


def run_graph(*, seed: int, top_k: int) -> dict[str, Any]:
    """Supply-chain graph evaluation (PRD F-06, Section 5.5.3).

    Lives in the same script as the detection evaluation because PRD Section 9.3 is
    explicit that there is *"exactly one evaluation pipeline, not a separate 'for the
    report' version"*. The graph is a different model on different data, but the
    number quoted for it has to come from the same place as every other number.
    """
    from sentinel.graph.explain import top_risk_explanations
    from sentinel.graph.gnn import GraphSplit, SupplyChainGNN, top_k_precision
    from sentinel.graph.schema import NodeKind
    from sentinel.graph.synthetic import SyntheticGraphGenerator
    from sentinel.ml.metrics import spearman_correlation

    graph, truth = SyntheticGraphGenerator(seed=seed).generate()
    node_ids = graph.node_ids()
    labels = truth.labels(node_ids)
    risk = truth.risk_vector(node_ids)
    split = GraphSplit.stratified(labels, seed=seed)

    model = SupplyChainGNN(random_state=seed).fit(graph, labels, split, exposure=risk)
    scores = model.risk_scores(graph)
    test = split.test

    # Features-only baseline. Reported alongside, not tucked away, because it is the
    # number that says whether the graph is load-bearing: a GNN that only matches a
    # logistic regression on node features has not earned its place in the stack.
    from sklearn.linear_model import LogisticRegression

    raw = graph.feature_matrix()
    mean = raw[split.train].mean(axis=0)
    deviation = raw[split.train].std(axis=0)
    scaled = (raw - mean) / np.where(deviation > 1e-12, deviation, 1.0)
    baseline_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    baseline_model.fit(scaled[split.train], labels[split.train])
    baseline = baseline_model.predict_proba(scaled)[:, 1]

    inherited = np.array([1.0 if n in truth.inherited else 0.0 for n in node_ids])
    inherited_subset = np.array(
        [i for i in test if inherited[i] > 0 or labels[i] == 0], dtype=int
    )

    per_kind: dict[str, dict[str, float]] = {}
    test_set = set(test.tolist())
    for kind in NodeKind:
        indices = np.array(
            [i for i in graph.indices_of_kind(kind) if i in test_set], dtype=int
        )
        if indices.size < 5 or labels[indices].sum() == 0:
            continue
        per_kind[kind.value] = {
            "n_test": int(indices.size),
            "prevalence": float(labels[indices].mean()),
            "top5_precision": top_k_precision(scores[indices], labels[indices], 5),
        }

    explanations = top_risk_explanations(graph, scores, model=model, k=top_k)
    result = {
        "graph": graph.describe(),
        "ground_truth": {
            "n_high_risk": len(truth.high_risk),
            "n_intrinsic": len(truth.intrinsic),
            "n_inherited": len(truth.inherited),
            "prevalence": len(truth.high_risk) / graph.n_nodes,
        },
        "split_sizes": split.sizes,
        "model": model.training_report(),
        "test": {
            f"top{top_k}_precision": top_k_precision(scores[test], labels[test], top_k),
            "spearman_vs_true_risk": spearman_correlation(scores[test], risk[test]),
            "per_node_kind": per_kind,
        },
        "features_only_baseline": {
            f"top{top_k}_precision": top_k_precision(baseline[test], labels[test], top_k),
            "top10_precision_inherited_only": top_k_precision(
                baseline[inherited_subset], inherited[inherited_subset], 10
            ),
        },
        "gnn_on_inherited_only": {
            "top10_precision": top_k_precision(
                scores[inherited_subset], inherited[inherited_subset], 10
            )
        },
        "top_flagged": [
            {
                "node_id": e.node_id,
                "kind": e.kind.value,
                "risk": e.risk_score,
                "driver": e.dominant_driver,
                "paths": [p.describe() for p in e.paths[:2]],
            }
            for e in explanations
        ],
    }
    result["f06_top_k_pass"] = (
        result["test"][f"top{top_k}_precision"] >= 0.80  # type: ignore[index]
    )

    print(f"\n[supply-chain graph] {graph.n_nodes} nodes, {len(graph.edges)} edges")
    print(f"  {truth.summary()}")
    print(
        f"  top-{top_k} precision   "
        f"{result['test'][f'top{top_k}_precision']:.4f}   "  # type: ignore[index]
        f"{'PASS' if result['f06_top_k_pass'] else 'FAIL'} (F-06 needs >= 0.80)"
    )
    print(
        f"  rank corr vs truth {result['test']['spearman_vs_true_risk']:.4f}"  # type: ignore[index]
    )
    print(
        f"  features-only      "
        f"{result['features_only_baseline'][f'top{top_k}_precision']:.4f}   "  # type: ignore[index]
        "<- the graph's contribution is the gap"
    )
    print(
        f"  inherited-risk only: GNN "
        f"{result['gnn_on_inherited_only']['top10_precision']:.4f} vs "  # type: ignore[index]
        f"features-only "
        f"{result['features_only_baseline']['top10_precision_inherited_only']:.4f}"  # type: ignore[index]
    )
    print("  top flagged nodes:")
    for entry in result["top_flagged"][:3]:  # type: ignore[index]
        print(f"    {entry['node_id']:<12} risk={entry['risk']:.3f} driver={entry['driver']}")
        for path in entry["paths"]:
            print(f"      {path}")
    return result


def run_kb(*, k: int) -> dict[str, Any]:
    """Measure the retrieval knowledge base (PRD F-05, Section 5.5.6).

    Every component retriever is reported alongside the fusion, on the held-out query
    split. The components are printed because the fusion's margin over the best of
    them is smaller than one query out of 61, and a report showing only the
    configuration that shipped would imply a confidence the measurement does not
    support. Both splits are printed for the same reason: the gap between them is the
    most useful number here.
    """
    from sentinel.kb import KnowledgeBase, evaluate_retrieval, load_eval_queries
    from sentinel.kb.eval import GATE_MRR, GATE_RECALL_AT_5, GATE_RECALL_WITH_LINKS
    from sentinel.kb.index import LexicalIndex, LsaIndex

    print()
    print("=" * 72)
    print("F-05 / Section 5.5.6 - retrieval knowledge base")
    print("=" * 72)

    kb = KnowledgeBase.build()
    stats = kb.stats()
    print(
        f"  corpus: {stats['documents']} documents -> {stats['chunks']} chunks "
        f"{stats['chunks_by_kind']}"
    )
    density = float(stats["density"])  # type: ignore[arg-type]
    print(
        f"  index:  {stats['features']} features, {stats['nonzeros']} non-zeros, "
        f"density {density:.4f}"
    )
    if stats["suspicious_chunks"]:
        print(
            f"  note:   {stats['suspicious_chunks']} chunk(s) matched injection "
            "heuristics and are reported for audit"
        )

    tune = load_eval_queries(split="tune")
    test = load_eval_queries(split="test")
    bm25 = LexicalIndex(encoder=kb.lexical.encoder, scorer="bm25")
    components = {
        "bm25": bm25,
        "tfidf": LexicalIndex(encoder=kb.lexical.encoder, scorer="tfidf"),
        "lsa": LsaIndex.build(bm25),
    }

    print()
    print(f"  held-out split, {len(test)} queries, k={k}:")
    per_component: dict[str, Any] = {}
    for name, retriever in components.items():
        report = evaluate_retrieval(kb.with_retriever(retriever), test, k=k)
        per_component[name] = report.as_dict()
        print(f"    {report.summary()}")
    fused = evaluate_retrieval(kb, test, k=k)
    print(f"    {fused.summary()}")
    by_kind = {name: round(value, 3) for name, value in fused.per_kind_recall.items()}
    print(f"  per-kind recall: {by_kind}")

    tuned = evaluate_retrieval(kb, tune, k=k)
    print()
    print(f"  tuning split, {len(tune)} queries, shown for the generalisation gap:")
    print(f"    {tuned.summary()}")

    checks = {
        f"recall@{k}": (fused.recall_at_k, GATE_RECALL_AT_5),
        "mrr": (fused.mrr, GATE_MRR),
        f"recall@{k}+links": (fused.recall_with_links, GATE_RECALL_WITH_LINKS),
    }
    print()
    passed = True
    for label, (value, gate) in checks.items():
        ok = value >= gate
        passed = passed and ok
        verdict = "PASS" if ok else "FAIL"
        print(f"  {verdict}  {label:<18} {value:.3f}   gate {gate:.2f}")
    if fused.misses:
        print()
        print("  queries with no correct document in the top k:")
        for miss in fused.misses:
            print(f"    - {miss}")

    return {
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "suspicious_chunks": stats["suspicious_chunks"],
        "test": fused.as_dict(),
        "tune": tuned.as_dict(),
        "components": per_component,
        "f05_retrieval_pass": passed,
    }


def run_policy(*, n_episodes: int) -> dict[str, Any]:
    """Measure the contextual-bandit response policy (PRD F-09, Section 5.5.4).

    Reported across :data:`~sentinel.rl.simulate.REPORTING_SEEDS` only, against a
    non-learning baseline on the same seeds. Both are needed: "regret fell" is only
    interesting next to "a policy that learns nothing does not produce that curve".

    Per-tier results are printed because the action mask changes the problem, not just
    the permissions -- at ``observe`` the policy has two arms instead of four, so its
    regret is measured against a different oracle and the numbers are not comparable
    across rows.
    """
    from sentinel.core.schemas import RiskTier
    from sentinel.rl import (
        ALL_ACTIONS,
        REPORTING_SEEDS,
        aggregate,
        assert_f09_gates,
        replay_many_seeds,
    )
    from sentinel.rl.simulate import (
        GATE_MEAN_SUBLINEARITY,
        GATE_OPTIMAL_ACTION_RATE,
        GATE_REGRET_RATIO,
        GATE_WORST_SUBLINEARITY,
    )

    print()
    print("=" * 72)
    print("F-09 / Section 5.5.4 - contextual bandit response policy")
    print("=" * 72)

    learned = replay_many_seeds(seeds=REPORTING_SEEDS, n_episodes=n_episodes)
    baseline = replay_many_seeds(
        seeds=REPORTING_SEEDS, n_episodes=n_episodes, random_policy=True
    )
    stats = aggregate(learned)
    reference = aggregate(baseline)
    ratio = stats["mean_total_regret"] / reference["mean_total_regret"]

    print(f"  {len(REPORTING_SEEDS)} held-out seeds, {n_episodes} episodes each:")
    for result in learned:
        print(f"    {result.summary()}")

    print()
    print(f"  {'metric':<26} {'policy':>10} {'no-learning':>12}")
    for label, key in (
        ("total regret", "mean_total_regret"),
        ("first-quarter regret", "mean_first_quarter_regret"),
        ("last-quarter regret", "mean_last_quarter_regret"),
        ("sublinearity", "mean_sublinearity"),
        ("optimal-action rate", "mean_optimal_action_rate"),
        ("mean reward", "mean_reward"),
    ):
        print(f"  {label:<26} {stats[key]:>10.3f} {reference[key]:>12.3f}")

    counts: dict[str, int] = {}
    for result in learned:
        for action in ALL_ACTIONS:
            counts[action.value] = counts.get(action.value, 0) + result.action_counts[
                action
            ]
    print(f"  action mix: {counts}")

    print()
    checks = {
        "mean sublinearity": (stats["mean_sublinearity"], GATE_MEAN_SUBLINEARITY, True),
        "worst sublinearity": (
            stats["worst_sublinearity"],
            GATE_WORST_SUBLINEARITY,
            True,
        ),
        "regret vs baseline": (ratio, GATE_REGRET_RATIO, False),
        "optimal-action rate": (
            stats["mean_optimal_action_rate"],
            GATE_OPTIMAL_ACTION_RATE,
            True,
        ),
        "tier violations": (stats["total_violations"], 0.0, False),
    }
    passed = True
    for label, (value, gate, higher_is_better) in checks.items():
        ok = value >= gate if higher_is_better else value <= gate
        passed = passed and ok
        arrow = ">=" if higher_is_better else "<="
        print(
            f"  {'PASS' if ok else 'FAIL'}  {label:<22} {value:.3f}   "
            f"gate {arrow} {gate:.2f}"
        )

    per_tier: dict[str, Any] = {}
    print()
    print("  per trust tier (regret is against that tier's own masked oracle):")
    for tier in RiskTier:
        tier_learned = replay_many_seeds(
            seeds=REPORTING_SEEDS, n_episodes=n_episodes, tier=tier
        )
        tier_stats = aggregate(tier_learned)
        per_tier[tier.value] = tier_stats
        print(
            f"    {tier.value:<18} regret={tier_stats['mean_total_regret']:7.2f} "
            f"sublinear={tier_stats['mean_sublinearity']:+.1%} "
            f"optimal={tier_stats['mean_optimal_action_rate']:.1%} "
            f"violations={int(tier_stats['total_violations'])}"
        )
        passed = passed and tier_stats["total_violations"] == 0.0

    try:
        assert_f09_gates(learned, baseline)
    except Exception as exc:  # reported below rather than swallowed
        passed = False
        print(f"\n  gate assertion failed: {exc}")

    # Section 9.1 asks for the cumulative-regret *curve* against the oracle, not only
    # its total; the mean over seeds is what the dashboard's Evaluation page draws.
    curves = {
        "episode": list(range(1, n_episodes + 1)),
        "policy": [float(v) for v in np.mean(
            [r.cumulative_regret for r in learned], axis=0)],
        "no_learning": [float(v) for v in np.mean(
            [r.cumulative_regret for r in baseline], axis=0)],
    }

    return {
        "episodes": n_episodes,
        "seeds": list(REPORTING_SEEDS),
        "curves": curves,
        "policy": stats,
        "baseline": reference,
        "regret_ratio": ratio,
        "action_mix": counts,
        "per_tier": per_tier,
        "f09_regret_pass": passed,
    }


def run_augmentation(*, n: int, seed: int) -> dict[str, Any]:
    """Measure diffusion augmentation and the calibration probe (PRD Section 5.5.5).

    Reports the augmentation delta at several scarcity levels rather than one, because
    the single-number version of this result is misleading: at full data the effect is
    approximately zero, and the sign flips depending on how much real data the
    generator had to learn from.

    The calibration probe is reported alongside it because that is where the generator
    earns its place. The raw softmax confidence fails its gates; the same classifier
    with a novelty-gated confidence passes them.
    """
    from collections import Counter

    from sentinel.ml.anomaly import build_default_ensemble
    from sentinel.ml.classify import FamilyClassifier
    from sentinel.ml.datasets.synthetic import generate_alerts
    from sentinel.ml.diffusion import TabularDiffusion
    from sentinel.ml.featurestore import AlertVectorizer
    from sentinel.ml.robustness import (
        GATE_MAX_OVERCONFIDENCE,
        NoveltyGate,
        RobustnessError,
        assert_calibration_holds,
        augment_training_set,
        boundary_adjacent_samples,
        calibration_report,
        gated_calibration_report,
        gated_robustness_curve,
        robustness_curve,
    )

    rare = ("web_attack", "botnet", "infiltration", "brute_force")

    print()
    print("=" * 72)
    print("Section 5.5.5 - diffusion augmentation and the calibration probe")
    print("=" * 72)

    alerts = generate_alerts(n, seed=seed)
    vectorizer = AlertVectorizer().fit(alerts)
    matrix = vectorizer.transform(alerts)
    families = np.asarray([a.ground_truth_label or "benign" for a in alerts])
    order = np.random.default_rng(7).permutation(len(families))
    cut = int(0.6 * len(order))
    x_train, f_train = matrix[order[:cut]], families[order[:cut]]
    x_test, f_test = matrix[order[cut:]], list(families[order[cut:]])
    print(f"  {cut} training rows, {len(f_test)} held out; no synthetic row reaches the")
    print("  held-out split, and the majority class is passed through unchanged.")

    def rare_macro(model: Any, xs: np.ndarray, fs: list[str]) -> float:
        per_family = model.recall_by_family(xs, fs)
        return float(np.mean([per_family.get(name, 0.0) for name in rare]))

    print()
    print("  augmentation delta by scarcity level (held-out rare-family macro recall):")
    print(f"    {'cap':>6} {'real':>6} {'synth':>6} {'base':>8} {'augmented':>10} {'delta':>8}")
    levels: dict[str, Any] = {}
    for cap in (None, 200, 60, 25, 12):
        if cap is None:
            subset_x, subset_f = x_train, list(f_train)
        else:
            keep: list[int] = []
            seen: Counter = Counter()
            for index, family in enumerate(f_train):
                if family in rare:
                    if seen[family] >= cap:
                        continue
                    seen[family] += 1
                keep.append(index)
            chosen = np.asarray(keep)
            subset_x, subset_f = x_train[chosen], list(f_train[chosen])

        mask = np.isin(np.asarray(subset_f), rare)
        generator = TabularDiffusion(
            n_steps=200, epochs=250, hidden=(160, 160), seed=1, min_rows_per_family=8
        ).fit(subset_x[mask], list(np.asarray(subset_f)[mask]))
        augmented = augment_training_set(
            subset_x, subset_f, generator=generator, multiplier=6.0 if cap else 3.0,
            rng=np.random.default_rng(11),
        )
        baseline = FamilyClassifier(seed=5).fit(subset_x, subset_f)
        boosted = FamilyClassifier(seed=5).fit(augmented.x, list(augmented.families))
        before = rare_macro(baseline, x_test, f_test)
        after = rare_macro(boosted, x_test, f_test)
        label = "full" if cap is None else str(cap)
        levels[label] = {
            "real_rare_rows": int(mask.sum()),
            "synthetic_rows": augmented.n_synthetic,
            "baseline_rare_macro": before,
            "augmented_rare_macro": after,
            "delta": after - before,
        }
        print(
            f"    {label:>6} {int(mask.sum()):>6} {augmented.n_synthetic:>6} "
            f"{before:>8.3f} {after:>10.3f} {after - before:>+8.3f}"
        )
    print("  Augmentation is not a remedy for genuine rarity: the regime that wants it")
    print("  most is the regime where the generator has too little to model.")

    classifier = FamilyClassifier(seed=5).fit(x_train, list(f_train))
    weighted = FamilyClassifier(seed=5, class_weight="balanced").fit(
        x_train, list(f_train)
    )
    print()
    print("  the one-line alternative, for comparison:")
    print(
        f"    unweighted macro={classifier.macro_recall(x_test, f_test):.3f}   "
        f"class-weighted macro={weighted.macro_recall(x_test, f_test):.3f}"
    )

    support = build_default_ensemble()
    support.fit(x_train)
    gate = NoveltyGate(quantile=0.99).fit(support.score(x_train))
    raw = robustness_curve(classifier, x_test, f_test, rng=np.random.default_rng(13))
    gated = gated_robustness_curve(
        classifier, x_test, f_test, novelty_fn=support.score, gate=gate,
        rng=np.random.default_rng(13),
    )
    rows, left, _ = boundary_adjacent_samples(
        classifier, x_test, f_test, n_samples=300, rng=np.random.default_rng(17)
    )

    print()
    print("  calibration under perturbation - raw softmax confidence:")
    for report in raw:
        print(f"    {report.summary()}")
    print(f"    {calibration_report(classifier, rows, left, label='boundary').summary()}")
    raw_peak = max(report.overconfidence for report in raw)
    print(f"    peak overconfidence {raw_peak:+.3f}")

    print()
    print("  calibration under perturbation - novelty-gated confidence:")
    for report in gated:
        print(f"    {report.summary()}")
    boundary = gated_calibration_report(
        classifier, rows, left, novelty=support.score(rows), gate=gate, label="boundary"
    )
    print(f"    {boundary.summary()}")
    gated_peak = max(report.overconfidence for report in gated)
    print(f"    peak overconfidence {gated_peak:+.3f}")

    raw_fails = True
    try:
        assert_calibration_holds(raw)
        raw_fails = False
    except RobustnessError:
        pass

    passed = True
    stats: dict[str, float] = {}
    print()
    try:
        stats = assert_calibration_holds(gated, boundary=boundary)
        print(f"  PASS  novelty-gated calibration     peak {gated_peak:+.3f}   "
              f"gate <= {GATE_MAX_OVERCONFIDENCE:.2f}")
    except RobustnessError as exc:
        passed = False
        print(f"  FAIL  novelty-gated calibration     {exc}")
    if raw_fails:
        print("  PASS  the probe has teeth           raw softmax correctly rejected")
    else:
        passed = False
        print("  FAIL  the probe has teeth           raw softmax passed; probe is vacuous")

    return {
        "levels": levels,
        "unweighted_macro": classifier.macro_recall(x_test, f_test),
        "class_weighted_macro": weighted.macro_recall(x_test, f_test),
        "raw_peak_overconfidence": raw_peak,
        "gated_peak_overconfidence": gated_peak,
        "raw_curve": [
            {"label": r.label, "accuracy": r.accuracy, "confidence": r.mean_confidence,
             "ece": r.ece, "overconfidence": r.overconfidence}
            for r in raw
        ],
        "gated_curve": [
            {"label": r.label, "accuracy": r.accuracy, "confidence": r.mean_confidence,
             "ece": r.ece, "overconfidence": r.overconfidence}
            for r in gated
        ],
        "gated_stats": stats,
        "raw_correctly_rejected": raw_fails,
        "s555_calibration_pass": passed,
    }


def _spread(items: list[Any], count: int) -> list[Any]:
    """``count`` items spread evenly across ``items``, in order.

    The agent-layer checks drive a bounded number of incidents. Taking the *first* N of a
    test split is a sample of one stretch of the timeline, and under some seeds that
    stretch is almost all benign: nothing reaches the approval gate, and the gate,
    resume and timing checks fail for a reason that has nothing to do with the system.
    An evenly spaced sample covers the whole split, deterministically.
    """
    if count >= len(items):
        return list(items)
    return [items[(i * len(items)) // count] for i in range(count)]


def run_agents(*, n: int, seed: int, incidents: int) -> dict[str, Any]:
    """Evaluate the agent layer (PRD F-02, F-04, F-05, F-08, Section 9.1).

    Runs real alerts through the real graph with the real audit log, so every
    number here is produced by the same code path the demo uses — Section 9.3's
    one-pipeline rule applied to Layer 4.

    F-08 is reported from :func:`verify_no_ungated_execution`, which reads the
    audit chain rather than the :class:`ActionRequest` objects. Those objects
    enforce the rule, so asking them whether it held would be circular.
    """
    import tempfile
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    from sentinel.agents.contain import (
        ContainmentAgent,
        SimulatedConnector,
        verify_no_ungated_execution,
    )
    from sentinel.agents.investigate import InvestigationAgent
    from sentinel.agents.orchestrator import (
        build_incident_graph,
        incident_timings,
        new_incident,
    )
    from sentinel.agents.state import HumanDecision, IncidentStatus
    from sentinel.agents.triage import TECHNIQUE_BY_FAMILY, TriageAgent, TriageModel
    from sentinel.audit.log import HashChainedAuditLog
    from sentinel.core.clock import SimulationClock
    from sentinel.core.schemas import TriageDecision
    from sentinel.kb.retrieve import KnowledgeBase
    from sentinel.ml.metrics import four_way_split

    print()
    print("=" * 72)
    print("Layer 4 - agents and orchestration (F-02, F-04, F-05, F-08)")
    print("=" * 72)

    alerts = generate_alerts(n, seed=seed)
    y = labels_of(alerts)
    split = four_way_split(y, seed=seed)
    train_benign = [alerts[i] for i in split.train_benign]
    train_labelled = [alerts[i] for i in split.train_labelled]
    validation = [alerts[i] for i in split.validation]
    test = [alerts[i] for i in split.test]

    model = TriageModel.fit(
        train_benign, train_labelled=train_labelled, validation=validation, seed=seed
    )
    calibration = model.calibration
    assert calibration is not None
    print(f"  split {split.sizes}")
    print(f"  calibration (chosen on validation): {calibration.summary()}")

    # --- F-02, on the split neither the fit nor the calibration touched -----
    results = TriageAgent(model=model).triage_batch(test)
    truth = y[split.test]
    kept = np.array(
        [0 if r.decision is TriageDecision.AUTO_DISMISS else 1 for r in results]
    )
    agreement = float((kept == truth).mean())
    true_positive = int(((kept == 1) & (truth == 1)).sum())
    false_positive = int(((kept == 1) & (truth == 0)).sum())
    false_negative = int(((kept == 0) & (truth == 1)).sum())
    recall = true_positive / max(1, true_positive + false_negative)
    precision = true_positive / max(1, true_positive + false_positive)
    dismiss_rate = float((kept == 0).mean())

    technique_hits = technique_total = 0
    for alert, result in zip(test, results, strict=True):
        if alert.ground_truth_label == BENIGN:
            continue
        technique_total += 1
        technique_hits += result.technique_id == TECHNIQUE_BY_FAMILY.get(
            alert.ground_truth_label
        )
    technique_agreement = technique_hits / max(1, technique_total)

    f02_pass = agreement >= 0.85
    print()
    print(f"[triage (F-02)] n={len(test):,} ({truth.mean():.1%} attack)")
    print(
        f"  label agreement  {agreement:.4f}   "
        f"{'PASS' if f02_pass else 'FAIL'} (F-02 needs >= 0.85)"
    )
    print(f"  recall           {recall:.4f}   (Section 9.1 needs >= 0.80)")
    print(f"  precision        {precision:.4f}   (Section 9.1 needs >= 0.85)")
    print(f"  alert reduction  {dismiss_rate:.1%}   (Section 9.1 needs >= 60%)")
    print(f"  technique match  {technique_agreement:.4f} on {technique_total:,} attacks")
    print(f"  max latency      {max(r.latency_ms for r in results):.2f} ms (budget 5,000)")

    # --- F-04 / F-05 / F-08, through the live graph ------------------------
    # Real node durations, with the analyst's click injected on top. A frozen
    # clock would report MTTD as 0.00s and pass its budget without measuring.
    clock = SimulationClock(datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC))
    connector = SimulatedConnector()
    kb = KnowledgeBase.build()
    graph = build_incident_graph(
        triage=TriageAgent(model=model, clock=clock),
        investigation=InvestigationAgent(kb=kb, clock=clock),
        containment=ContainmentAgent(clock=clock),
        connector=connector,
    )
    temp_dir = _Path(tempfile.mkdtemp(prefix="sentinel-eval-"))
    log = HashChainedAuditLog(temp_dir / "audit.sqlite", clock=clock)

    counts = {"dismissed": 0, "completed": 0, "gated": 0, "failed": 0}
    detect: list[float] = []
    contain: list[float] = []
    ungrounded_claims = 0
    unresolvable_refs = 0
    reports = 0

    for alert in _spread(test, incidents):
        run_result = graph.invoke(
            new_incident(alert, at=clock.now()), clock=clock, audit=log
        )
        state = run_result.state
        timings = incident_timings(state)
        if timings.pipeline_detect_seconds is not None:
            detect.append(timings.pipeline_detect_seconds)

        if state.status is IncidentStatus.DISMISSED:
            counts["dismissed"] += 1
        elif state.status is IncidentStatus.FAILED:
            counts["failed"] += 1

        if run_result.interrupted:
            counts["gated"] += 1
            clock.advance(12.0)  # the scripted analyst click from Section 9.1
            run_result = graph.resume(
                state.incident_id,
                HumanDecision(
                    approver="analyst@acme", approved=True, decided_at=clock.now()
                ),
                clock=clock,
                audit=log,
            )
            state = run_result.state
            timings = incident_timings(state)
            if timings.contain_seconds is not None:
                contain.append(timings.contain_seconds)
        if state.status is IncidentStatus.COMPLETED:
            counts["completed"] += 1

        if state.report is not None:
            reports += 1
            known = {item.ref for item in state.report.evidence}
            for _statement, refs in state.report.claims:
                if not refs:
                    ungrounded_claims += 1
                elif not set(refs) <= known:
                    unresolvable_refs += 1
            kb_refs = [ref for ref in known if ref.startswith("kb://")]
            if not kb.resolves_all(kb_refs):
                unresolvable_refs += 1

    ungated = verify_no_ungated_execution(log)
    chain = log.verify()
    f08_pass = not ungated and not chain.findings
    f05_pass = reports > 0 and ungrounded_claims == 0 and unresolvable_refs == 0
    f04_pass = counts["gated"] > 0 and counts["failed"] == 0

    print()
    print(f"[orchestration (F-04)] {min(incidents, len(test)):,} incidents")
    print(f"  dismissed at triage    {counts['dismissed']:,}")
    print(f"  stopped at the gate    {counts['gated']:,}")
    print(f"  completed              {counts['completed']:,}")
    print(f"  failed                 {counts['failed']:,}")
    print(
        f"  interrupt/resume       {'PASS' if f04_pass else 'FAIL'} "
        "(F-04 needs a reachable gate and no failed runs)"
    )
    print()
    print(f"[investigation (F-05)] {reports:,} reports")
    print(f"  uncited claims         {ungrounded_claims}")
    print(f"  unresolvable refs      {unresolvable_refs}")
    print(
        f"  grounding              {'PASS' if f05_pass else 'FAIL'} "
        "(F-05 needs every claim to trace to a chunk or log line)"
    )
    print()
    print("[approval gate (F-08)]")
    print(f"  ungated executions     {len(ungated)}")
    print(f"  audit chain findings   {len(chain.findings)}")
    print(f"  connector executions   {len(connector.executed):,}")
    print(
        f"  gate                   {'PASS' if f08_pass else 'FAIL'} "
        "(F-08 needs zero actions executed without a logged approval)"
    )
    print()
    print("[Section 9.1 timing]")
    if detect:
        print(
            f"  MTTD (pipeline)        {float(np.mean(detect)):.2f}s mean, "
            f"{max(detect):.2f}s worst   (target < 30s)"
        )
    if contain:
        print(
            f"  MTTC                   {float(np.mean(contain)):.2f}s mean, "
            f"{max(contain):.2f}s worst   (target < 180s)"
        )
    log.close()

    return {
        "split_sizes": split.sizes,
        "calibration": {
            "escalate_fpr": calibration.escalate_fpr,
            "monitor_fpr": calibration.monitor_fpr,
            "validation_agreement": calibration.agreement,
            "grid_points": calibration.n_considered,
        },
        "triage": {
            "n_test": len(test),
            "agreement": agreement,
            "recall": recall,
            "precision": precision,
            "alert_reduction": dismiss_rate,
            "technique_agreement": technique_agreement,
            "max_latency_ms": max(r.latency_ms for r in results),
        },
        "orchestration": counts,
        "investigation": {
            "reports": reports,
            "uncited_claims": ungrounded_claims,
            "unresolvable_refs": unresolvable_refs,
        },
        "approval_gate": {
            "ungated_executions": list(ungated),
            "audit_findings": len(chain.findings),
            "connector_executions": len(connector.executed),
        },
        "timing": {
            "mttd_pipeline_mean_s": float(np.mean(detect)) if detect else None,
            "mttd_pipeline_max_s": max(detect) if detect else None,
            "mttc_mean_s": float(np.mean(contain)) if contain else None,
            "mttc_max_s": max(contain) if contain else None,
        },
        "f02_agreement_pass": f02_pass,
        "f04_orchestration_pass": f04_pass,
        "f05_grounding_pass": f05_pass,
        "f08_approval_gate_pass": f08_pass,
        "s91_mttd_pass": bool(detect) and max(detect) < 30.0,
        "s91_mttc_pass": bool(contain) and max(contain) < 180.0,
    }


def run_codescan(*, seed: int) -> dict[str, Any]:
    """Evaluate the Code-Scan / Patch Agent (PRD F-07).

    F-07: *"Run Semgrep, map findings to CVEs, draft a patch PR. At least 3 seeded
    vulnerabilities detected and a syntactically valid patch PR opened for each."*

    Measured against ``data/vulnerable_app``, whose ground truth is marker comments in
    the source itself rather than a manifest that drifts away from it. Recall is
    reported beside the **false-positive count on the SAFE controls**, and the gate
    fails on either: a scanner that finds every seeded defect and also flags the
    correct version next to it has a recall of 1.0 and no value, because the first
    thing a team does with a tool that cries wolf is switch it off.

    The whole thing runs through the real code-scan graph with the real audit log, so
    F-08 is re-verified on a second graph rather than assumed to carry over.
    """
    import tempfile
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    from sentinel.agents.codescan import (
        CodeScanAgent,
        DraftPullRequestConnector,
        build_code_scan_graph,
        code_scan_timings,
        new_code_scan_incident,
        synthesize_scan_alert,
    )
    from sentinel.agents.contain import verify_no_ungated_execution
    from sentinel.agents.state import HumanDecision, IncidentStatus
    from sentinel.audit.log import HashChainedAuditLog
    from sentinel.core.clock import SimulationClock
    from sentinel.core.schemas import EvidenceKind
    from sentinel.kb.retrieve import KnowledgeBase
    from sentinel.scan.analyzer import AstAnalyzer
    from sentinel.scan.patch import apply_unified_diff
    from sentinel.scan.repo import RepoSnapshot
    from sentinel.scan.rules import RULES, rule_ids
    from sentinel.scan.seeded import FIXTURE_DIR, load_seed_manifest, score_scan

    del seed  # the scan is deterministic; nothing here is sampled

    print()
    print("=" * 72)
    print("Code-Scan / Patch Agent (F-07)")
    print("=" * 72)

    snapshot = RepoSnapshot.from_dir(FIXTURE_DIR)
    manifest = load_seed_manifest(snapshot)
    manifest.validate(frozenset(rule_ids()))
    analyzer = AstAnalyzer()
    scan = analyzer.scan(snapshot)
    score = score_scan(manifest, scan.findings)

    # Every patch is re-verified here rather than trusted from the scan: the diff is
    # the artifact that would reach a pull request, so the check that matters is that
    # it applies and that the result parses.
    import ast as _ast

    diff_failures = 0
    parse_failures = 0
    for patch in scan.valid_patches:
        try:
            if apply_unified_diff(patch.before, patch.diff) != patch.after:
                diff_failures += 1
        except Exception:
            diff_failures += 1
        try:
            _ast.parse(patch.after)
        except SyntaxError:
            parse_failures += 1

    f07_detection_pass = len(score.detected) >= 3 and not score.missed
    f07_patch_pass = (
        len(scan.valid_patches) >= 3
        and not scan.rejected_patches
        and diff_failures == 0
        and parse_failures == 0
    )
    noise_pass = not score.false_positives and not score.over_escalated

    print()
    print(
        f"[static analysis] {len(RULES)} rules, {scan.files_scanned} file(s), "
        f"{scan.lines_scanned:,} lines"
    )
    print(f"  findings               {len(scan.findings)}")
    print(
        f"  seeded detected        {len(score.detected)}/{score.n_seeded}   "
        f"recall {score.recall:.3f}   "
        f"{'PASS' if f07_detection_pass else 'FAIL'} (F-07 needs >= 3)"
    )
    print(
        f"  false positives        {len(score.false_positives)} of "
        f"{len(manifest.controls)} SAFE controls   "
        f"{'PASS' if noise_pass else 'FAIL'} (needs 0)"
    )
    print(f"  INFO escalated         {len(score.over_escalated)} (needs 0)")
    print(f"  unmarked findings      {len(score.unmarked)}")
    print(
        f"  validated patches      {len(scan.valid_patches)}   "
        f"{'PASS' if f07_patch_pass else 'FAIL'} (F-07 needs >= 3, each valid)"
    )
    print(f"  rejected patches       {len(scan.rejected_patches)} (needs 0)")
    print(f"  diff replay failures   {diff_failures} (needs 0)")
    print(f"  patched-parse failures {parse_failures} (needs 0)")
    if score.missed or score.false_positives or score.over_escalated:
        print(score.describe())
    by_rule: dict[str, int] = {}
    for finding in scan.findings:
        by_rule[finding.rule_id] = by_rule.get(finding.rule_id, 0) + 1
    print("  findings by rule:")
    for rule_id, count in sorted(by_rule.items()):
        patched = sum(1 for p in scan.valid_patches if p.rule_id == rule_id)
        print(f"    {rule_id:<44} {count} found, {patched} patched")

    # --- through the live graph, with the gate and the audit chain -------------
    clock = SimulationClock(datetime(2026, 9, 29, 9, 0, 0, tzinfo=UTC))
    kb = KnowledgeBase.build()
    connector = DraftPullRequestConnector()
    graph = build_code_scan_graph(
        agent=CodeScanAgent(kb=kb, clock=clock),
        snapshot=snapshot,
        connector=connector,
    )
    temp_dir = _Path(tempfile.mkdtemp(prefix="sentinel-codescan-"))
    log = HashChainedAuditLog(temp_dir / "audit.sqlite", clock=clock)

    alert = synthesize_scan_alert(
        snapshot,
        tenant_id="acme",
        repository="acme/vulnerable-app",
        at=clock.now(),
        commit="HEAD",
    )
    run = graph.invoke(new_code_scan_incident(alert, at=clock.now()), clock=clock, audit=log)
    gated = run.interrupted
    opened_before_approval = len(connector.opened)
    if gated:
        clock.advance(25.0)  # the scripted developer review
        run = graph.resume(
            run.state.incident_id,
            HumanDecision(approver="dev@acme", approved=True, decided_at=clock.now()),
            clock=clock,
            audit=log,
        )
    state = run.state
    report = state.report
    ungated = verify_no_ungated_execution(log)
    chain = log.verify()

    uncited = 0
    unresolvable = 0
    cve_citations = 0
    if report is not None:
        known = {item.ref for item in report.evidence}
        for _statement, refs in report.claims:
            if not refs:
                uncited += 1
            elif not set(refs) <= known:
                unresolvable += 1
        cve_citations = sum(
            1 for item in report.evidence if item.kind is EvidenceKind.CVE_RECORD
        )
        kb_refs = [ref for ref in known if ref.startswith("kb://")]
        if kb_refs and not kb.resolves_all(kb_refs):
            unresolvable += 1

    grounded_pass = report is not None and uncited == 0 and unresolvable == 0
    gate_pass = (
        gated
        and opened_before_approval == 0
        and len(connector.opened) == 1
        and not ungated
        and not chain.findings
    )
    scan_seconds, to_pr = code_scan_timings(state)

    print()
    print("[code-scan graph (F-07, F-08)]")
    print(f"  run status             {state.status.value}")
    print(f"  stopped at the gate    {gated}")
    print(f"  PRs before approval    {opened_before_approval} (needs 0)")
    print(f"  draft PRs opened       {len(connector.opened)}")
    print(f"  ungated executions     {len(ungated)}")
    print(f"  audit chain findings   {len(chain.findings)}")
    print(
        f"  gate                   {'PASS' if gate_pass else 'FAIL'} "
        "(F-08 on a second graph)"
    )
    print(f"  cited claims           {0 if report is None else len(report.claims)}")
    print(f"  uncited claims         {uncited}")
    print(f"  unresolvable refs      {unresolvable}")
    print(f"  CVE citations          {cve_citations}")
    print(
        f"  grounding              {'PASS' if grounded_pass else 'FAIL'} "
        "(every claim traces to a finding, a CVE or a log line)"
    )
    if state.status is not IncidentStatus.COMPLETED:
        print(f"  error                  {state.error}")
    if connector.opened:
        _action, draft = connector.opened[0]
        print(f"  branch                 {draft.branch}")
        print(f"  files touched          {', '.join(draft.files_touched)}")
        print(f"  left for a human       {len(draft.unpatched_refs)} finding(s)")
    if scan_seconds is not None:
        print(f"  scan latency           {scan_seconds:.2f}s")
    if to_pr is not None:
        print(f"  time to draft PR       {to_pr:.2f}s (includes a 25s human review)")
    log.close()

    return {
        "rules": len(RULES),
        "files_scanned": scan.files_scanned,
        "lines_scanned": scan.lines_scanned,
        "findings": len(scan.findings),
        "findings_by_rule": by_rule,
        "seeded": score.n_seeded,
        "detected": len(score.detected),
        "recall": score.recall,
        "missed": [item.describe() for item in score.missed],
        "safe_controls": len(manifest.controls),
        "false_positives": [item.describe() for item in score.false_positives],
        "over_escalated": [item.describe() for item in score.over_escalated],
        "unmarked_findings": len(score.unmarked),
        "validated_patches": len(scan.valid_patches),
        "rejected_patches": [
            {"rule_id": p.rule_id, "reason": p.rejection} for p in scan.rejected_patches
        ],
        "diff_replay_failures": diff_failures,
        "patched_parse_failures": parse_failures,
        "graph": {
            "status": state.status.value,
            "gated": gated,
            "prs_before_approval": opened_before_approval,
            "prs_opened": len(connector.opened),
            "ungated_executions": list(ungated),
            "audit_findings": len(chain.findings),
            "claims": 0 if report is None else len(report.claims),
            "uncited_claims": uncited,
            "unresolvable_refs": unresolvable,
            "cve_citations": cve_citations,
            "scan_seconds": scan_seconds,
            "seconds_to_draft_pr": to_pr,
        },
        "f07_detection_pass": f07_detection_pass,
        "f07_patch_pass": f07_patch_pass,
        "f07_noise_pass": noise_pass,
        "f07_grounding_pass": grounded_pass,
        "f08_second_graph_pass": gate_pass,
    }


def run_supplychain_agent(*, seed: int, top_k: int) -> dict[str, Any]:
    """Evaluate the Supply-Chain Agent — F-06's *guardrail* (PRD Sections 3.4, 5.4).

    F-06's metric (top-10 precision) is measured by ``--graph``, which is the model's
    number. This measures the guardrail: *"flags are explainable via the specific graph
    path that drove the score."* So the figure that gates here is the share of flagged
    nodes carrying a concrete exposure path or an intrinsic reason, and the run goes
    through the real review graph so the approval gate and the audit chain are exercised
    a third time.

    Both routes are driven. A flagged package proposes ``OPEN_PATCH_PR``, which is
    destructive and therefore gated; a flagged vendor proposes ``NOTIFY_ANALYST``, which
    is not. Reporting only the gated path would leave half the routing unmeasured.
    """
    import tempfile
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    import numpy as _np

    from sentinel.agents.contain import SimulatedConnector, verify_no_ungated_execution
    from sentinel.agents.state import HumanDecision
    from sentinel.agents.supplychain import (
        DEPENDENCY_TECHNIQUE,
        SupplyChainAgent,
        SupplyChainMonitor,
        build_supply_chain_review_graph,
        supply_chain_timings,
    )
    from sentinel.audit.log import HashChainedAuditLog
    from sentinel.core.clock import SimulationClock
    from sentinel.core.schemas import EvidenceKind
    from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
    from sentinel.graph.schema import NodeKind
    from sentinel.graph.synthetic import SyntheticGraphGenerator
    from sentinel.kb.retrieve import KnowledgeBase

    print()
    print("=" * 72)
    print("Supply-Chain Agent (F-06 guardrail)")
    print("=" * 72)

    graph, truth = SyntheticGraphGenerator(seed=seed).generate()
    node_ids = graph.node_ids()
    labels = truth.labels(node_ids)
    split = GraphSplit.stratified(labels, seed=seed)
    model = SupplyChainGNN(random_state=seed).fit(
        graph, labels, split, exposure=truth.risk_vector(node_ids)
    )
    scores = model.risk_scores(graph)

    clock = SimulationClock(datetime(2026, 9, 29, 3, 0, 0, tzinfo=UTC))
    kb = KnowledgeBase.build()
    agent = SupplyChainAgent(kb=kb, clock=clock, top_k=top_k)
    connector = SimulatedConnector()
    compiled = build_supply_chain_review_graph(
        agent=agent, graph=graph, model=model, scores=scores, connector=connector
    )
    temp_dir = _Path(tempfile.mkdtemp(prefix="sentinel-supplychain-"))
    log = HashChainedAuditLog(temp_dir / "audit.sqlite", clock=clock)
    monitor = SupplyChainMonitor(
        agent=agent, graph=graph, model=model, scores=scores, clock=clock
    )

    run = monitor.tick(
        tenant_id="acme", compiled=compiled, audit=log, assessment_id="2026-09-29"
    )
    gated = run.interrupted
    executed_before_approval = len(connector.executed)
    if gated:
        clock.advance(45.0)  # the scripted vCISO review
        run = compiled.resume(
            run.state.incident_id,
            HumanDecision(approver="vciso@acme", approved=True, decided_at=clock.now()),
            clock=clock,
            audit=log,
        )
    state = run.state
    report = state.report
    assert report is not None

    assessment = agent.assess(graph, alert=state.alert, model=model, scores=scores)
    explainable = len(assessment.explainable)
    flagged = len(assessment.findings)
    explainability = explainable / flagged if flagged else 0.0

    uncited = 0
    unresolvable = 0
    known = {item.ref for item in report.evidence}
    for _statement, refs in report.claims:
        if not refs:
            uncited += 1
        elif not set(refs) <= known:
            unresolvable += 1
    kb_refs = [ref for ref in known if ref.startswith("kb://")]
    if kb_refs and not kb.resolves_all(kb_refs):
        unresolvable += 1
    path_citations = sum(
        1 for item in report.evidence if item.kind is EvidenceKind.GRAPH_PATH
    )

    # The ungated route: force vendors and organisations to the top so the
    # NOTIFY_ANALYST branch is measured rather than assumed.
    forced = _np.array(
        [0.99 if node.kind is not NodeKind.PACKAGE else 0.01 for node in graph.nodes],
        dtype=float,
    )
    notify_connector = SimulatedConnector()
    notify_clock = SimulationClock(datetime(2026, 9, 29, 4, 0, 0, tzinfo=UTC))
    notify_graph = build_supply_chain_review_graph(
        agent=SupplyChainAgent(kb=kb, clock=notify_clock, top_k=top_k),
        graph=graph,
        scores=forced,
        connector=notify_connector,
    )
    from sentinel.agents.supplychain import new_vendor_risk_incident, synthesize_vendor_alert

    notify_alert = synthesize_vendor_alert(
        tenant_id="acme", graph=graph, at=notify_clock.now(), assessment_id="vendor-route"
    )
    notify_run = notify_graph.invoke(
        new_vendor_risk_incident(notify_alert, at=notify_clock.now()),
        clock=notify_clock,
        audit=log,
    )

    ungated = verify_no_ungated_execution(log)
    chain = log.verify()
    assessed_s, acted_s = supply_chain_timings(state)

    explainability_pass = explainability >= 0.80
    grounding_pass = uncited == 0 and unresolvable == 0 and path_citations > 0
    gate_pass = (
        gated
        and executed_before_approval == 0
        and len(connector.executed) == 1
        and not ungated
        and not chain.findings
    )
    route_pass = (
        not notify_run.interrupted
        and len(notify_connector.executed) == 1
        and notify_connector.executed[0].action_type.value == "notify_analyst"
    )

    by_kind = {kind.value: count for kind, count in assessment.by_kind.items()}
    print()
    print(f"[supply-chain agent] {graph.n_nodes} nodes, {len(graph.edges)} edges")
    print(f"  flagged                {flagged} (top-{top_k})")
    print(f"  by kind                {by_kind}")
    print(
        f"  explainable            {explainable}/{flagged} = {explainability:.3f}   "
        f"{'PASS' if explainability_pass else 'FAIL'} (F-06 guardrail needs >= 0.80)"
    )
    print(f"  graph-path citations   {path_citations}")
    print(f"  cited claims           {len(report.claims)}")
    print(f"  uncited claims         {uncited}")
    print(f"  unresolvable refs      {unresolvable}")
    print(
        f"  grounding              {'PASS' if grounding_pass else 'FAIL'} "
        "(every claim cites a path, a CVE or a technique)"
    )
    print(f"  technique asserted     {report.techniques}")
    print(f"  attribution conflicts  {len(assessment.disagreements)} reported, not hidden")
    print()
    print("[supply-chain graph (F-08)]")
    print(f"  package route          gated={gated}, executions={len(connector.executed)}")
    print(f"  executed pre-approval  {executed_before_approval} (needs 0)")
    print(
        f"  vendor route           gated={notify_run.interrupted}, "
        f"executions={len(notify_connector.executed)} "
        f"({'notify_analyst' if route_pass else 'unexpected'})"
    )
    print(
        f"  routing                {'PASS' if route_pass else 'FAIL'} "
        "(a notification is not destructive, so it is not gated)"
    )
    print(f"  ungated executions     {len(ungated)}")
    print(f"  audit chain findings   {len(chain.findings)}")
    print(
        f"  gate                   {'PASS' if gate_pass else 'FAIL'} "
        "(F-08 on a third graph)"
    )
    if assessed_s is not None:
        print(f"  assessment latency     {assessed_s:.2f}s")
    if acted_s is not None:
        print(f"  time to remediation    {acted_s:.2f}s (includes a 45s human review)")
    print("  top flagged nodes:")
    for finding in assessment.findings[:5]:
        print(f"    {finding.describe().splitlines()[0]}")
        for path in finding.explanation.paths[:1]:
            print(f"      {path.describe()}")
    log.close()

    return {
        "n_nodes": graph.n_nodes,
        "n_edges": len(graph.edges),
        "flagged": flagged,
        "by_kind": by_kind,
        "explainable": explainable,
        "explainability": explainability,
        "graph_path_citations": path_citations,
        "claims": len(report.claims),
        "uncited_claims": uncited,
        "unresolvable_refs": unresolvable,
        "techniques": list(report.techniques),
        "dependency_technique": DEPENDENCY_TECHNIQUE,
        "attribution_disagreements": list(assessment.disagreements),
        "package_route": {
            "gated": gated,
            "executed_before_approval": executed_before_approval,
            "executions": len(connector.executed),
        },
        "vendor_route": {
            "gated": notify_run.interrupted,
            "executions": len(notify_connector.executed),
            "action": (
                notify_connector.executed[0].action_type.value
                if notify_connector.executed
                else None
            ),
        },
        "ungated_executions": list(ungated),
        "audit_findings": len(chain.findings),
        "assessment_seconds": assessed_s,
        "seconds_to_remediation": acted_s,
        "f06_explainability_pass": explainability_pass,
        "f06_grounding_pass": grounding_pass,
        "f08_third_graph_pass": gate_pass,
        "routing_pass": route_pass,
    }


def run_connectors(*, n: int, seed: int, incidents: int) -> dict[str, Any]:
    """Evaluate the connector layer (Part 4: PRD Figure 2 Layer 5, Section 5.7, F-08).

    All three graphs run through :class:`~sentinel.connectors.router.ConnectorRouter`
    in front of the Wazuh, SCIM, GitHub and Slack connectors, speaking HTTP to the
    local emulators in :mod:`sentinel.connectors.sandbox`. Every effect is checked
    against the *remote system's* state rather than the connector's own report, and
    F-08 is re-read from the audit chain with one extra, stronger question: did any
    connector touch the wire for a gated action before its approval row?

    Five gates:

    *   ``s57_least_privilege_pass`` — every out-of-scope request (merge, PR edit,
        another repository, a plaintext origin, a redirect) and every over-scoped
        credential is refused, with zero requests reaching the emulators.
    *   ``f08_wire_pass`` — ``verify_no_ungated_execution`` is empty, no
        ``connector_called`` row for a gated action precedes its grant, and no
        rejected action produced a single request.
    *   ``state_agreement_pass`` — the hosts isolated and addresses blocked in Wazuh
        are exactly the approved targets; no action failed; the router refused none.
    *   ``draft_only_pass`` — one draft PR, zero merges, zero PR edits, only GET and
        POST on the wire, and the pushed files re-scan with the patched findings gone.
    *   ``exactly_once_pass`` — a process killed between the connector call and the
        checkpoint recovers with the host isolated once; the control without the
        durable journal isolates it twice.
    """
    import tempfile
    from datetime import UTC, datetime, timedelta
    from pathlib import Path as _Path

    from sentinel.agents.checkpoint import SqliteCheckpointer
    from sentinel.agents.codescan import (
        CodeScanAgent,
        build_code_scan_graph,
        new_code_scan_incident,
        synthesize_scan_alert,
    )
    from sentinel.agents.contain import ContainmentAgent, verify_no_ungated_execution
    from sentinel.agents.investigate import InvestigationAgent
    from sentinel.agents.orchestrator import build_incident_graph, new_incident
    from sentinel.agents.state import HumanDecision, IncidentStatus
    from sentinel.agents.supplychain import (
        SupplyChainAgent,
        SupplyChainMonitor,
        build_supply_chain_review_graph,
    )
    from sentinel.agents.triage import TriageAgent, TriageModel
    from sentinel.audit.log import HashChainedAuditLog
    from sentinel.connectors.base import Capability, Credential, Secret
    from sentinel.connectors.github import GitHubConnector
    from sentinel.connectors.http import EgressPolicy, HttpError, Route, ScopedHttpClient
    from sentinel.connectors.journal import BlastRadiusLimiter, MemoryJournal, SqliteJournal
    from sentinel.connectors.sandbox import EmulatedService, LiveServer, Sandbox
    from sentinel.core.clock import SimulationClock
    from sentinel.core.errors import GuardrailViolation
    from sentinel.core.schemas import ActionType, ApprovalStatus, AuditEventType
    from sentinel.graph.gnn import GraphSplit, SupplyChainGNN
    from sentinel.graph.synthetic import SyntheticGraphGenerator
    from sentinel.kb.retrieve import KnowledgeBase
    from sentinel.ml.metrics import four_way_split
    from sentinel.scan.analyzer import AstAnalyzer
    from sentinel.scan.repo import RepoSnapshot
    from sentinel.scan.seeded import FIXTURE_DIR

    print()
    print("=" * 72)
    print("Layer 5 - connectors (Part 4: Section 5.7 least privilege, F-08 on the wire)")
    print("=" * 72)

    alerts = generate_alerts(n, seed=seed)
    y = labels_of(alerts)
    split = four_way_split(y, seed=seed)
    model = TriageModel.fit(
        [alerts[i] for i in split.train_benign],
        train_labelled=[alerts[i] for i in split.train_labelled],
        validation=[alerts[i] for i in split.validation],
        seed=seed,
    )
    feed = _spread([alerts[i] for i in split.test], incidents)
    tenant = feed[0].tenant_id
    kb = KnowledgeBase.build()
    snapshot = RepoSnapshot.from_dir(FIXTURE_DIR)
    graph, truth = SyntheticGraphGenerator(seed=seed).generate()
    node_ids = graph.node_ids()
    node_labels = truth.labels(node_ids)
    gnn = SupplyChainGNN(random_state=seed).fit(
        graph, node_labels, GraphSplit.stratified(node_labels, seed=seed),
        exposure=truth.risk_vector(node_ids),
    )
    scores = gnn.risk_scores(graph)

    temp_dir = _Path(tempfile.mkdtemp(prefix="sentinel-connectors-"))
    clock = SimulationClock(datetime(2026, 9, 29, 9, 0, 0, tzinfo=UTC))
    log = HashChainedAuditLog(temp_dir / "audit.sqlite", clock=clock)
    sandbox = Sandbox(
        repo_files={f.path: f.text for f in snapshot.files},
        hosts=[a.asset_id for a in feed],
        clock=clock,
    ).start()
    journal = SqliteJournal(temp_dir / "journal.sqlite")
    router = sandbox.router(
        tenant_id=tenant,
        audit=log,
        journal=journal,
        limiter=BlastRadiusLimiter(max_actions=max(25, incidents), window=timedelta(hours=1),
                                   clock=clock),
    )

    # --- graph 1: the alert feed -----------------------------------------------
    incident_graph = build_incident_graph(
        triage=TriageAgent(model=model, clock=clock),
        investigation=InvestigationAgent(kb=kb, clock=clock),
        containment=ContainmentAgent(clock=clock),
        connector=router,
    )
    gated = approved = rejected = 0
    approved_targets: dict[ActionType, set[str]] = {}
    rejected_ids: set[str] = set()
    failed_actions: list[str] = []
    for index, alert in enumerate(feed):
        run = incident_graph.invoke(new_incident(alert, at=clock.now()), clock=clock, audit=log)
        if run.interrupted:
            gated += 1
            pending = run.state.actions[-1]
            approve = gated % 4 != 0  # every fourth gated action is rejected
            clock.advance(12.0)
            run = incident_graph.resume(
                run.state.incident_id,
                HumanDecision(approver="soc@acme", approved=approve, decided_at=clock.now()),
                clock=clock, audit=log,
            )
            if approve:
                approved += 1
                approved_targets.setdefault(pending.action_type, set()).add(pending.target)
            else:
                rejected += 1
                rejected_ids.add(pending.action_id)
        failed_actions += [
            f"{a.action_type.value}: {a.failure_reason}"
            for a in run.state.actions
            if a.approval_status is ApprovalStatus.FAILED
        ]
        del index

    # --- graph 2: a code scan ---------------------------------------------------
    code_graph = build_code_scan_graph(
        agent=CodeScanAgent(kb=kb, clock=clock), snapshot=snapshot, connector=router
    )
    scan_alert = synthesize_scan_alert(snapshot, tenant_id=tenant, repository="acme/billing",
                                       at=clock.now(), commit="HEAD")
    draft = CodeScanAgent(kb=kb, clock=clock).assess(
        snapshot, alert=scan_alert, now=clock.now()
    ).draft
    scan_run = code_graph.invoke(new_code_scan_incident(scan_alert, at=clock.now()),
                                 clock=clock, audit=log)
    prs_before_approval = len(sandbox.github.pulls)
    if scan_run.interrupted:
        clock.advance(25.0)
        scan_run = code_graph.resume(
            scan_run.state.incident_id,
            HumanDecision(approver="dev@acme", approved=True, decided_at=clock.now()),
            clock=clock, audit=log,
        )

    # --- graph 3: a scheduled supply-chain review --------------------------------
    supply_agent = SupplyChainAgent(kb=kb, clock=clock)
    review_graph = build_supply_chain_review_graph(
        agent=supply_agent, graph=graph, model=gnn, scores=scores, connector=router
    )
    monitor = SupplyChainMonitor(agent=supply_agent, graph=graph, model=gnn, scores=scores,
                                 clock=clock)
    vendor_run = monitor.tick(tenant_id=tenant, compiled=review_graph, audit=log,
                              assessment_id="2026-09-29")
    issues_before_approval = len(sandbox.github.issues)
    if vendor_run.interrupted:
        clock.advance(45.0)
        vendor_run = review_graph.resume(
            vendor_run.state.incident_id,
            HumanDecision(approver="vciso@acme", approved=True, decided_at=clock.now()),
            clock=clock, audit=log,
        )

    records = list(log.iter_records())
    calls = [r for r in records if r.event_type is AuditEventType.CONNECTOR_CALLED]

    # --- F-08 on the wire ---------------------------------------------------------
    granted: dict[str, int] = {}
    wire_before_approval = 0
    for record in records:
        if record.event_type is AuditEventType.APPROVAL_GRANTED:
            granted.setdefault(record.subject_id, record.seq)
        elif (record.event_type is AuditEventType.CONNECTOR_CALLED
              and record.payload.get("requires_human_approval")
              and granted.get(record.subject_id, 10**12) > record.seq):
            wire_before_approval += 1
    touched = {r.subject_id for r in calls}
    rejected_on_wire = len(rejected_ids & touched)
    ungated = verify_no_ungated_execution(log)
    chain = log.verify()

    # --- state agreement ------------------------------------------------------------
    isolated_ok = sandbox.wazuh.isolated_hosts() == approved_targets.get(
        ActionType.ISOLATE_HOST, set()
    )
    blocked_ok = sandbox.wazuh.blocked_addresses() == approved_targets.get(
        ActionType.BLOCK_IP, set()
    )

    # --- draft PR ---------------------------------------------------------------------
    github = sandbox.github
    analyzer = AstAnalyzer()
    rescan_ok = draft is not None and len(github.pulls) == 1
    if rescan_ok:
        for file in snapshot.files:
            pushed = github.file_at(draft.branch, file.path)
            if pushed is None:
                rescan_ok = False
                break
            before = analyzer.rule_counts(file)
            after = analyzer.rule_counts(file.with_text(pushed))
            for patch_rule in {p.rule_id for p in draft.patches if p.path == file.path}:
                fixed = sum(1 for p in draft.patches
                            if p.path == file.path and p.rule_id == patch_rule)
                if after.get(patch_rule, 0) != before[patch_rule] - fixed:
                    rescan_ok = False
            if any(count > before.get(rule, 0) for rule, count in after.items()):
                rescan_ok = False

    # --- secrets --------------------------------------------------------------------
    rendered = " ".join(repr(r.payload) for r in records)
    secret_leaks = sum(1 for value in sandbox._secrets.values() if value in rendered)
    secret_leaks += rendered.count("Bearer ") + rendered.count("Basic ")

    # --- least-privilege probes ---------------------------------------------------------
    probes_denied = 0
    probes_total = 0
    before_probe = len(github.requests)
    probe_connector = sandbox.github_connector()
    for method, path in (
        ("PUT", "/repos/acme/billing/pulls/1/merge"),
        ("PATCH", "/repos/acme/billing/pulls/1"),
        ("DELETE", "/repos/acme/billing/git/refs/heads/main"),
        ("GET", "/repos/acme/other-repo"),
        ("POST", "/repos/acme/billing/hooks"),
        ("GET", "/repos/acme/billing/git/ref/heads/../../../admin"),
    ):
        probes_total += 1
        try:
            probe_connector.http.request(method, path)
        except GuardrailViolation:
            probes_denied += 1
    for scopes in (
        {"contents:write", "pull_requests:write", "administration:write"},
        {"contents:write", "pull_requests:write", "workflow"},
        {"contents:write"},
    ):
        probes_total += 1
        try:
            GitHubConnector(owner="acme", repo="billing",
                            credential=Credential(Secret("x"), frozenset(scopes)),
                            capabilities=(Capability.PR_OPEN_DRAFT,))
        except GuardrailViolation:
            probes_denied += 1
    probes_total += 1
    try:
        EgressPolicy("http://wazuh.example:55000", (Route("r", "GET", "/"),),
                     allow_insecure_loopback=True)
    except GuardrailViolation:
        probes_denied += 1

    class _Redirect(EmulatedService):
        def route(self, request):
            return 302, {"Location": "http://127.0.0.1:9/steal"}, b""

    probes_total += 1
    with LiveServer(_Redirect()) as redirector:
        client = ScopedHttpClient(
            connector="probe",
            policy=EgressPolicy(redirector.url, (Route("r", "GET", "/x"),),
                                allow_insecure_loopback=True),
            auth=lambda: {"Authorization": "Bearer probe"},
        )
        try:
            client.request("GET", "/x")
        except HttpError:
            probes_denied += 1
    probes_reached_remote = len(github.requests) - before_probe

    sandbox.stop()
    journal.close()
    log.close()

    # --- exactly once across a crash ----------------------------------------------------
    class _Crash(BaseException):
        pass

    class _CrashAfterGatedCall:
        def __init__(self, inner):
            self.inner = inner
            self.armed = True

        def execute(self, action):
            outcome = self.inner.execute(action)
            if self.armed and action.requires_human_approval:
                self.armed = False
                raise _Crash
            return outcome

    def crash_and_recover(durable: bool) -> int:
        work = _Path(tempfile.mkdtemp(prefix="sentinel-crash-"))
        crash_clock = SimulationClock(datetime(2026, 9, 29, 9, 0, 0, tzinfo=UTC))
        crash_log = HashChainedAuditLog(work / "audit.sqlite", clock=crash_clock)
        # Search past the driven feed for a gated incident: whether one of the first N
        # flows happens to escalate depends on the seed, and a crash test that can only
        # run for some seeds cannot back a "run it under any seed" claim.
        pool = [alerts[i] for i in split.test]
        box = Sandbox(hosts=[a.asset_id for a in pool], clock=crash_clock).start()

        def build(connector, store):
            return build_incident_graph(
                triage=TriageAgent(model=model, clock=crash_clock),
                investigation=InvestigationAgent(kb=kb, clock=crash_clock),
                containment=ContainmentAgent(clock=crash_clock),
                connector=connector,
                checkpointer=store,
            )

        try:
            j1 = SqliteJournal(work / "j.sqlite") if durable else MemoryJournal()
            with SqliteCheckpointer(work / "cp.sqlite") as store:
                g = build(_CrashAfterGatedCall(box.router(tenant_id=tenant, audit=crash_log,
                                                          journal=j1)), store)
                paused = None
                for alert in pool:
                    run = g.invoke(new_incident(alert, at=crash_clock.now()),
                                   clock=crash_clock, audit=crash_log)
                    if run.interrupted:
                        paused = run
                        break
                assert paused is not None, "no gated incident in the whole test split"
                crash_clock.advance(12.0)
                with contextlib.suppress(_Crash):
                    g.resume(paused.state.incident_id,
                             HumanDecision(approver="soc@acme", approved=True,
                                           decided_at=crash_clock.now()),
                             clock=crash_clock, audit=crash_log)
            if durable:
                j1.close()
            j2 = SqliteJournal(work / "j.sqlite") if durable else MemoryJournal()
            with SqliteCheckpointer(work / "cp.sqlite") as store:
                g = build(box.router(tenant_id=tenant, audit=crash_log, journal=j2), store)
                recovered = g.recover(paused.state.incident_id, clock=crash_clock,
                                      audit=crash_log)
                assert recovered.state.status is IncidentStatus.COMPLETED
            if durable:
                j2.close()
            return len(box.wazuh.executed)
        finally:
            box.stop()
            crash_log.close()

    runs_with_journal = crash_and_recover(True)
    runs_without = crash_and_recover(False)

    # --- report ---------------------------------------------------------------------
    by_connector: dict[str, int] = {}
    retries = 0
    durations: dict[str, list[float]] = {}
    for record in calls:
        name = str(record.payload["connector"])
        by_connector[name] = by_connector.get(name, 0) + 1
        retries += int(record.payload["attempt"]) > 1
        durations.setdefault(name, []).append(float(record.payload["duration_ms"]))
    latency = {
        name: {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
        }
        for name, values in durations.items()
    }

    least_privilege_pass = probes_denied == probes_total and probes_reached_remote == 0
    f08_wire_pass = (
        not ungated and wire_before_approval == 0 and rejected_on_wire == 0
        and not chain.findings and gated > 0 and rejected > 0
    )
    state_pass = isolated_ok and blocked_ok and not failed_actions and not router.refusals
    draft_pass = (
        rescan_ok and prs_before_approval == 0 and github.merges == 0
        and github.pull_edits == 0
        and {r.method for r in github.requests} <= {"GET", "POST"}
        and bool(github.pulls) and github.pulls[0]["draft"] is True
        and scan_run.state.status is IncidentStatus.COMPLETED
        and issues_before_approval == 0 and len(github.issues) == 1
        and vendor_run.state.status is IncidentStatus.COMPLETED
    )
    exactly_once_pass = runs_with_journal == 1 and runs_without == 2

    print()
    print(f"[incident graph over HTTP] {len(feed)} incidents, tenant {tenant!r}")
    print(f"  gated {gated} · approved {approved} · rejected {rejected}")
    print(f"  isolated in Wazuh      {sorted(sandbox.wazuh.isolated_hosts())}")
    print(f"  blocked in Wazuh       {len(sandbox.wazuh.blocked_addresses())} address(es)")
    print(f"  Slack messages         {len(sandbox.slack.messages)}")
    print(f"  failed actions         {len(failed_actions)} · router refusals "
          f"{len(router.refusals)}")
    print(f"  state agreement        {'PASS' if state_pass else 'FAIL'} "
          "(remote state == approved targets)")
    print()
    print("[code scan -> GitHub]")
    print(f"  PRs before approval    {prs_before_approval} · draft PRs {len(github.pulls)} · "
          f"merges {github.merges} · PR edits {github.pull_edits}")
    print(f"  methods on the wire    {sorted({r.method for r in github.requests})}")
    print(f"  pushed files re-scan   {'clean' if rescan_ok else 'NOT clean'} for "
          f"{0 if draft is None else len(draft.patches)} patch(es)")
    print(f"  supply-chain issues    {len(github.issues)} (before approval "
          f"{issues_before_approval})")
    print(f"  draft only             {'PASS' if draft_pass else 'FAIL'}")
    print()
    print("[F-08 on the wire]")
    print(f"  ungated executions     {len(ungated)} (Part 3 reader, unmodified)")
    print(f"  wire before approval   {wire_before_approval} (needs 0)")
    print(f"  rejected on the wire   {rejected_on_wire} (needs 0)")
    print(f"  audit chain findings   {len(chain.findings)}")
    print(f"  credential leaks       {secret_leaks} (needs 0)")
    print(f"  F-08 on the wire       {'PASS' if f08_wire_pass and not secret_leaks else 'FAIL'}")
    print()
    print("[Section 5.7 least privilege]")
    print(f"  out-of-scope probes    {probes_denied}/{probes_total} refused, "
          f"{probes_reached_remote} reached the remote")
    print(f"  least privilege        {'PASS' if least_privilege_pass else 'FAIL'}")
    print()
    print("[exactly once across a crash]")
    print(f"  with the SQLite journal   approved action reached Wazuh {runs_with_journal}x")
    print(f"  control, no journal       approved action reached Wazuh {runs_without}x")
    print(f"  exactly once           {'PASS' if exactly_once_pass else 'FAIL'}")
    print()
    print(f"[wire] {len(calls)} request(s) · retries {retries} · by connector {by_connector}")
    for name, stats in sorted(latency.items()):
        print(f"  {name:<8} p50 {stats['p50_ms']:.2f} ms · p95 {stats['p95_ms']:.2f} ms")

    return {
        "incidents": len(feed),
        "tenant": tenant,
        "gated": gated,
        "approved": approved,
        "rejected": rejected,
        "failed_actions": failed_actions,
        "router_refusals": len(router.refusals),
        "isolated_hosts": sorted(sandbox.wazuh.isolated_hosts()),
        "blocked_addresses": len(sandbox.wazuh.blocked_addresses()),
        "slack_messages": len(sandbox.slack.messages),
        "draft_prs": len(github.pulls),
        "merges": github.merges,
        "pull_edits": github.pull_edits,
        "issues": len(github.issues),
        "rescan_clean": rescan_ok,
        "ungated_executions": list(ungated),
        "wire_before_approval": wire_before_approval,
        "rejected_on_wire": rejected_on_wire,
        "audit_findings": len(chain.findings),
        "credential_leaks": secret_leaks,
        "probes_total": probes_total,
        "probes_denied": probes_denied,
        "probes_reached_remote": probes_reached_remote,
        "crash_runs_with_journal": runs_with_journal,
        "crash_runs_without_journal": runs_without,
        "wire_requests": len(calls),
        "wire_retries": retries,
        "wire_by_connector": by_connector,
        "wire_latency": latency,
        "s57_least_privilege_pass": least_privilege_pass,
        "f08_wire_pass": f08_wire_pass and secret_leaks == 0,
        "state_agreement_pass": state_pass,
        "draft_only_pass": draft_pass,
        "exactly_once_pass": exactly_once_pass,
    }


def run_dashboard(*, seed: int, n: int, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the Analyst Copilot (Part 5: PRD F-10, Figure 2 Layer 6).

    F-10's criterion is *"all 3 demo scenarios completable end-to-end from this UI
    alone"*. The UI is a static page whose every action is an HTTP call, so this
    starts the real app under uvicorn on a loopback port and drives it with nothing
    but ``urllib`` and bearer tokens: launch each scenario, read the approval queue,
    post the decisions a human would, and read the outcome back from ``/api/wire``,
    ``/api/overview`` and ``/api/audit``. No workspace method is called directly.

    Five gates:

    *   ``f10_scenarios_pass`` — all three scenarios complete (every run terminal, no
        run failed), the queue empties, all seven gated decisions were made over
        HTTP, and the remote systems hold exactly what was approved.
    *   ``f10_guardrails_visible_pass`` — Part 4's two dashboard requirements: the
        FAILED action inside a completed run and the router's refusal are both
        returned, with the protected-network reason.
    *   ``f10_boundary_pass`` — every route refuses a missing or wrong token, a viewer
        cannot decide, a body naming its own approver is rejected, and another
        tenant's incident is indistinguishable from a nonexistent one.
    *   ``f08_dashboard_pass`` — zero ungated executions read back from the chain, the
        chain verifies, and every approval row names the token's principal.
    *   ``f12_same_pipeline_pass`` — the evaluation report the dashboard serves has
        exactly the gates this run computed (Section 9.3: one pipeline).
    """
    import tempfile
    import threading
    import time as _time
    import urllib.error
    import urllib.request
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    import uvicorn

    from sentinel.core.clock import SimulationClock
    from sentinel.dashboard.app import create_app
    from sentinel.dashboard.auth import Identity, Role, TokenRegistry
    from sentinel.dashboard.views import evaluation_view
    from sentinel.dashboard.workspace import MeshModels, Workspace

    print()
    print("=" * 72)
    print("Layer 6 - Analyst Copilot dashboard (Part 5: F-10, over HTTP)")
    print("=" * 72)

    started = _time.perf_counter()
    models = MeshModels.build(seed=seed, n_alerts=n)
    build_seconds = _time.perf_counter() - started
    work = _Path(tempfile.mkdtemp(prefix="sentinel-dashboard-eval-"))
    snapshot = work / "evaluation.json"
    snapshot.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    clock = SimulationClock(datetime(2026, 9, 29, 9, 0, 0, tzinfo=UTC))
    workspaces = {
        tenant: Workspace(models, tenant_id=tenant, workdir=work / tenant, clock=clock)
        for tenant in ("acme", "globex")
    }
    maya = Identity("maya@acme.example", "acme", Role.ANALYST)
    tokens, issued = TokenRegistry.generate(
        [maya, Identity("auditor@acme.example", "acme", Role.VIEWER),
         Identity("ops@globex.example", "globex", Role.ANALYST)]
    )
    analyst = issued["maya@acme.example@acme"]
    viewer = issued["auditor@acme.example@acme"]
    other_tenant = issued["ops@globex.example@globex"]

    config = uvicorn.Config(create_app(workspaces, tokens, evaluation_path=snapshot),
                            host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = _time.monotonic() + 30
    while not server.started and _time.monotonic() < deadline:
        _time.sleep(0.02)
    if not server.started:
        raise RuntimeError("the dashboard server did not start within 30 s")
    port = server.servers[0].sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def call(method: str, path: str, token: str | None = None,
             body: dict[str, Any] | None = None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method=method)
        if token is not None:
            request.add_header("Authorization", f"Bearer {token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    try:
        # --- the boundary, before anything exists ------------------------------------
        probes: list[bool] = []
        for path in ("/api/session", "/api/overview", "/api/queue", "/api/incidents",
                     "/api/wire", "/api/audit", "/api/evaluation", "/api/code-scan",
                     "/api/supply-chain/graph"):
            probes.append(call("GET", path)[0] == 401)
            probes.append(call("GET", path, "not-a-real-token-" + "x" * 20)[0] == 401)

        # --- the three scenarios, from the API alone -----------------------------------
        scenario_started = _time.perf_counter()
        opened: dict[str, list[str]] = {}
        for name in ("phishing-lateral", "vendor-cve", "malicious-package"):
            status, body = call("POST", f"/api/scenarios/{name}/launch", analyst)
            opened[name] = body["incidents"] if status == 200 else []
        first_queue = call("GET", "/api/queue", analyst)[1]

        # Boundary probes that need a live gate.
        target = first_queue[0]
        gate = f"/api/incidents/{target['incident_id']}/decision"
        pending_id = target["pending_action"]["action_id"]
        probes.append(call("POST", gate, viewer,
                           {"action_id": pending_id, "approved": True})[0] == 403)
        probes.append(call("POST", gate, analyst,
                           {"action_id": pending_id, "approved": True,
                            "approver": "ciso@acme.example"})[0] == 422)
        foreign = call("GET", f"/api/incidents/{target['incident_id']}", other_tenant)
        missing = call("GET", "/api/incidents/does-not-exist", other_tenant)
        probes.append(foreign[0] == missing[0] == 404
                      and foreign[1]["error"] == missing[1]["error"])
        probes.append(call("POST", gate, other_tenant,
                           {"action_id": pending_id, "approved": True})[0] == 404)

        decisions = 0
        rejected = 0
        while True:
            queue = call("GET", "/api/queue", analyst)[1]
            if not queue:
                break
            item = queue[0]
            action = item["pending_action"]
            # The walkthrough: the discovery-sweep block of the already isolated
            # workstation is redundant, so the analyst rejects it.
            approve = not (action["action_type"] == "block_ip"
                           and action["target"] == "10.20.4.17")
            clock.advance(20.0)  # the analyst reads the evidence
            status, _ = call("POST", f"/api/incidents/{item['incident_id']}/decision",
                             analyst, {"action_id": action["action_id"], "approved": approve,
                                       "note": "evaluate.py --dashboard"})
            if status != 200:
                break
            decisions += 1
            rejected += not approve
        scenario_seconds = _time.perf_counter() - scenario_started

        scenarios = call("GET", "/api/scenarios", analyst)[1]
        wire = call("GET", "/api/wire", analyst)[1]
        overview = call("GET", "/api/overview", analyst)[1]
        verify = call("GET", "/api/audit/verify", analyst)[1]
        audit = call("GET", "/api/audit?limit=500", analyst)[1]
        served_report = call("GET", "/api/evaluation", analyst)[1]
        globex_audit = call("GET", "/api/audit", other_tenant)[1]
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        for workspace in workspaces.values():
            workspace.close()

    complete = [s["name"] for s in scenarios if s["complete"] and not s["failed_runs"]]
    approvals = [r for r in audit["items"]
                 if r["event_type"] in ("approval_granted", "approval_denied")]
    approvers = sorted({r["payload"]["approver"] for r in approvals})
    refusal_reasons = [r["reason"] for r in wire["refusals"]]
    failed = wire["failed_actions"]
    remote = wire["remote"]

    scenarios_pass = (
        len(complete) == 3 and decisions == 7 and rejected == 1
        and overview["queue"] == 0 and overview["stalled"] == 0
        and all(opened.values())
        and remote["wazuh"]["isolated_hosts"] == ["10.20.4.17", "10.20.8.30"]
        and remote["wazuh"]["blocked_addresses"] == []
        and len(remote["github"]["issues"]) == 2
        and [p["draft"] for p in remote["github"]["pulls"]] == [True]
        and remote["github"]["merges"] == 0
    )
    guardrails_pass = (
        len(failed) == 1 and failed[0]["target"] == "10.20.0.5"
        and bool(failed[0]["failure_reason"])
        and len(refusal_reasons) == 1 and "10.20.0.0/28" in refusal_reasons[0]
        and overview["failed_actions"] == 1 and overview["refusals"] == 1
    )
    boundary_pass = all(probes) and globex_audit["total"] == 0
    f08_pass = (
        verify["ok"] and verify["ungated_executions"] == [] and len(approvals) == 7
        and approvers == [maya.principal]
    )
    expected_gates = evaluation_view(snapshot)["gates"]
    same_pipeline_pass = (
        served_report["available"] and served_report["gates"] == expected_gates
        and len(expected_gates) > 0
    )
    mttd = overview["mttd_seconds"]
    mttc = overview["mttc_seconds"]
    failed_pairs = [(f["action_type"], f["target"]) for f in failed]

    print()
    print(f"[models] built in {build_seconds:.1f}s from {n} flows; server on {base}")
    print(f"[scenarios over HTTP] {len(complete)}/3 complete in {scenario_seconds:.2f}s "
          f"wall · {decisions} decisions ({rejected} rejected)")
    for scenario in scenarios:
        state = "complete" if scenario["complete"] else "NOT complete"
        print(f"  {scenario['name']:<18} {scenario['finished']}/{scenario['total']} runs "
              f"finished · {scenario['decisions']} decision(s) · "
              f"{scenario['failed_actions']} failed action(s) · {state}")
    print(f"  Wazuh isolated {remote['wazuh']['isolated_hosts']} · blocked "
          f"{remote['wazuh']['blocked_addresses']} · GitHub issues "
          f"{len(remote['github']['issues'])} · draft PRs {len(remote['github']['pulls'])} "
          f"· merges {remote['github']['merges']}")
    print(f"  scenarios              {'PASS' if scenarios_pass else 'FAIL'}")
    print()
    print("[Part 4 requirements, as the dashboard shows them]")
    print(f"  failed action in a completed run  {failed_pairs}")
    first_reason = refusal_reasons[0][:100] if refusal_reasons else None
    print(f"  router refusal reason             {first_reason}")
    print(f"  guardrails visible     {'PASS' if guardrails_pass else 'FAIL'}")
    print()
    print(f"[boundary] {sum(probes)}/{len(probes)} probes refused as required "
          "(401 without a token, 403 viewer, 422 forged approver, 404 cross-tenant)")
    print(f"  boundary               {'PASS' if boundary_pass else 'FAIL'}")
    print()
    chain_state = "verified" if verify["ok"] else "BROKEN"
    print(f"[F-08 through the dashboard] ungated {len(verify['ungated_executions'])} · chain "
          f"{chain_state} ({verify['rows']} rows) · {len(approvals)} decisions, by {approvers}")
    print(f"  F-08 dashboard         {'PASS' if f08_pass else 'FAIL'}")
    print(f"[F-12] dashboard serves {len(served_report['gates'])} gates, identical to this "
          f"run's: {'PASS' if same_pipeline_pass else 'FAIL'}")
    print(f"[Section 9.1 as the dashboard reports it] MTTD {mttd:.3f}s · MTTC {mttc:.1f}s "
          "(MTTC includes the simulated 20 s analyst read)")

    return {
        "models_build_seconds": build_seconds,
        "scenario_wall_seconds": scenario_seconds,
        "scenarios_complete": complete,
        "decisions": decisions,
        "rejected": rejected,
        "isolated_hosts": remote["wazuh"]["isolated_hosts"],
        "blocked_addresses": remote["wazuh"]["blocked_addresses"],
        "github_issues": len(remote["github"]["issues"]),
        "draft_prs": len(remote["github"]["pulls"]),
        "merges": remote["github"]["merges"],
        "failed_actions": [[f["action_type"], f["target"], f["failure_reason"]]
                           for f in failed],
        "refusal_reasons": refusal_reasons,
        "boundary_probes": len(probes),
        "boundary_refused": sum(probes),
        "audit_rows": verify["rows"],
        "ungated_executions": verify["ungated_executions"],
        "approvals": len(approvals),
        "approvers": approvers,
        "mttd_seconds": mttd,
        "mttc_seconds": mttc,
        "served_gates": len(served_report["gates"]),
        "f10_scenarios_pass": scenarios_pass,
        "f10_guardrails_visible_pass": guardrails_pass,
        "f10_boundary_pass": boundary_pass,
        "f08_dashboard_pass": f08_pass,
        "f12_same_pipeline_pass": same_pipeline_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel Mesh offline evaluation")
    parser.add_argument("--n", type=int, default=20000, help="alerts to generate")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--separability", type=float, default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument(
        "--shallow",
        action="store_true",
        help="use the Part 1 linear baseline (Isolation Forest + PCA) instead of the "
        "PRD Section 5.5.2 denoising autoencoder ensemble",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.10,
        help="floor on every detector's ensemble weight, so AUC-optimal tuning cannot "
        "discard a detector entirely (ignored with --shallow)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also evaluate the supply-chain risk graph (PRD F-06)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="k for F-06 top-k precision")
    parser.add_argument(
        "--kb",
        action="store_true",
        help="also evaluate the retrieval knowledge base (PRD F-05)",
    )
    parser.add_argument(
        "--kb-k", type=int, default=5, help="cut-off for the F-05 retrieval metrics"
    )
    parser.add_argument(
        "--policy",
        action="store_true",
        help="also evaluate the contextual-bandit response policy (PRD F-09)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="also evaluate diffusion augmentation and the calibration probe "
        "(PRD Section 5.5.5)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="episodes per seed for the F-09 replay; the PRD names 200",
    )
    parser.add_argument(
        "--agents",
        action="store_true",
        help="also evaluate the agent layer end to end (PRD F-02, F-04, F-05, F-08)",
    )
    parser.add_argument(
        "--incidents",
        type=int,
        default=200,
        help="incidents to drive through the orchestration graph for --agents",
    )
    parser.add_argument(
        "--codescan",
        action="store_true",
        help="run the Code-Scan / Patch Agent against data/vulnerable_app (F-07)",
    )
    parser.add_argument(
        "--supplychain",
        action="store_true",
        help="run the Supply-Chain Agent over the risk graph (F-06's guardrail)",
    )
    parser.add_argument(
        "--connectors",
        action="store_true",
        help="drive all three graphs through the real connector layer against the local "
        "API emulators (Part 4: Section 5.7, F-08 on the wire)",
    )
    parser.add_argument(
        "--connector-alerts",
        type=int,
        default=4000,
        help="alerts to generate for the --connectors triage model",
    )
    parser.add_argument(
        "--connector-incidents",
        type=int,
        default=300,
        help="incidents to drive through the real connectors for --connectors",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="serve the Analyst Copilot on a loopback port and complete all three demo "
        "scenarios over HTTP alone (Part 5: F-10)",
    )
    parser.add_argument(
        "--dashboard-alerts",
        type=int,
        default=12000,
        help="flows to generate for the dashboard's models (the dashboard's default)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/artifacts/evaluation.json"))
    args = parser.parse_args()

    result = run(
        n=args.n,
        seed=args.seed,
        separability=args.separability,
        target_fpr=args.target_fpr,
        cross_dataset=args.cross_dataset,
        deep=not args.shallow,
        min_weight=args.min_weight,
    )
    if args.graph:
        result["supply_chain"] = run_graph(seed=args.seed, top_k=args.top_k)
    if args.kb:
        result["knowledge_base"] = run_kb(k=args.kb_k)
    if args.policy:
        result["response_policy"] = run_policy(n_episodes=args.episodes)
    if args.augment:
        result["augmentation"] = run_augmentation(n=args.n, seed=args.seed)
    if args.agents:
        result["agents"] = run_agents(
            n=args.n, seed=args.seed, incidents=args.incidents
        )
    if args.codescan:
        result["code_scan"] = run_codescan(seed=args.seed)
    if args.supplychain:
        result["supply_chain_agent"] = run_supplychain_agent(
            seed=args.seed, top_k=args.top_k
        )

    if args.connectors:
        result["connectors"] = run_connectors(
            n=args.connector_alerts, seed=args.seed, incidents=args.connector_incidents
        )

    if args.dashboard:
        # Last, so the report the dashboard serves is this run's own (Section 9.3).
        result["dashboard"] = run_dashboard(
            seed=args.seed, n=args.dashboard_alerts, evaluation=result
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # Non-zero exit if any acceptance criterion this run measured has failed, so the
    # script is usable as a CI gate rather than something whose output must be read.
    passed = bool(result["f03_auc_pass"])
    if args.graph:
        passed = passed and bool(result["supply_chain"]["f06_top_k_pass"])
    if args.kb:
        passed = passed and bool(result["knowledge_base"]["f05_retrieval_pass"])
    if args.policy:
        passed = passed and bool(result["response_policy"]["f09_regret_pass"])
    if args.augment:
        passed = passed and bool(result["augmentation"]["s555_calibration_pass"])
    if args.agents:
        agents = result["agents"]
        passed = passed and all(
            bool(agents[gate])
            for gate in (
                "f02_agreement_pass",
                "f04_orchestration_pass",
                "f05_grounding_pass",
                "f08_approval_gate_pass",
                "s91_mttd_pass",
                "s91_mttc_pass",
            )
        )
    if args.codescan:
        code_scan = result["code_scan"]
        passed = passed and all(
            bool(code_scan[gate])
            for gate in (
                "f07_detection_pass",
                "f07_patch_pass",
                "f07_noise_pass",
                "f07_grounding_pass",
                "f08_second_graph_pass",
            )
        )
    if args.supplychain:
        supply_chain = result["supply_chain_agent"]
        passed = passed and all(
            bool(supply_chain[gate])
            for gate in (
                "f06_explainability_pass",
                "f06_grounding_pass",
                "f08_third_graph_pass",
                "routing_pass",
            )
        )
    if args.connectors:
        connectors = result["connectors"]
        passed = passed and all(
            bool(connectors[gate])
            for gate in (
                "s57_least_privilege_pass",
                "f08_wire_pass",
                "state_agreement_pass",
                "draft_only_pass",
                "exactly_once_pass",
            )
        )
    if args.dashboard:
        dashboard = result["dashboard"]
        passed = passed and all(
            bool(dashboard[gate])
            for gate in (
                "f10_scenarios_pass",
                "f10_guardrails_visible_pass",
                "f10_boundary_pass",
                "f08_dashboard_pass",
                "f12_same_pipeline_pass",
            )
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
