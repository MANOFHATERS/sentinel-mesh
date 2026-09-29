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

    return {
        "episodes": n_episodes,
        "seeds": list(REPORTING_SEEDS),
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

    for alert in test[:incidents]:
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
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
