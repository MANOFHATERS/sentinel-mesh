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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # Non-zero exit if any acceptance criterion this run measured has failed, so the
    # script is usable as a CI gate rather than something whose output must be read.
    passed = bool(result["f03_auc_pass"])
    if args.graph:
        passed = passed and bool(result["supply_chain"]["f06_top_k_pass"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
