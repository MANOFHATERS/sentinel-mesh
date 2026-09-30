"""The detector and triage agent, run on real public network-flow data.

Everything else in the dashboard runs on flows the synthetic generator made. This is the
same triage model — trained the same way, scored by the same Triage Agent — on a real
labelled capture: UNSW-NB15 (Australian Centre for Cyber Security) or CIC-IDS2017
(Canadian Institute for Cybersecurity), through the same normalizers the synthetic
generator's rows go through.

What it is honest about:

*   **Real data is not the synthetic corpus.** The scores are usually lower and always
    less tidy, and a real-data run reports whatever it measured, not a target.
*   **The UNSW train/test CSVs carry no capture timestamps**, so the session-context
    features the synthetic corpus gets (a host's burst behaviour over a window) are
    absent. This is per-flow detection only, which is the harder problem.
*   **It samples.** ``limit`` rows are drawn uniformly (seeded, so the run is
    reproducible) from the whole file, then split into train / validation / test the same
    way the synthetic evaluation splits. The test split is never seen by the fit or the
    calibration.
"""

from __future__ import annotations

import csv
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import Alert, TriageDecision
from sentinel.ingest.normalizer import CICIDS2017Normalizer, Normalizer, UNSWNB15Normalizer

__all__ = ["RealDataError", "analyse_flows", "detect_format", "find_dataset"]

BENIGN: Final[str] = "benign"
MIN_ROWS: Final[int] = 400
_SEARCH_DIRS: Final[tuple[str, ...]] = ("data/raw/unsw-nb15", "data/raw/cic-ids2017", "data/raw")


class RealDataError(SentinelError):
    """The real dataset is missing, unreadable, or too small to evaluate on."""


def find_dataset(root: Path | str = ".") -> Path | None:
    """The first real flow CSV found under the conventional ``data/raw`` folders."""
    base = Path(root)
    for folder in _SEARCH_DIRS:
        directory = base / folder
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.csv")):
            if detect_format(candidate) is not None:
                return candidate
    return None


def _open(path: Path):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as probe:
                probe.readline()
            return path.open("r", encoding=encoding, newline="")
        except UnicodeDecodeError:
            continue
    raise RealDataError(f"{path} is not readable as text")


def detect_format(path: Path) -> str | None:
    """``"unsw"`` or ``"cic"`` from the header row, else ``None``."""
    try:
        with _open(path) as handle:
            header = next(csv.reader(handle), [])
    except (OSError, RealDataError, StopIteration):
        return None
    names = {h.strip().lower() for h in header}
    if {"sbytes", "dbytes", "dur"} <= names:
        return "unsw"
    if "flow duration" in names:
        return "cic"
    return None


def _normalizer(fmt: str) -> Normalizer:
    if fmt == "unsw":
        return UNSWNB15Normalizer(tenant_id="real", strict=False)
    return CICIDS2017Normalizer(tenant_id="real", strict=False)


def _sample_rows(path: Path, limit: int, seed: int) -> tuple[list[tuple[int, dict[str, str]]], int]:
    """Uniform reservoir sample of ``limit`` rows, in file order. Returns (rows, rows seen)."""
    rng = random.Random(seed)
    kept: list[tuple[int, dict[str, str]]] = []
    seen = 0
    with _open(path) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            seen += 1
            if len(kept) < limit:
                kept.append((index, row))
            else:
                slot = rng.randint(0, index)
                if slot < limit:
                    kept[slot] = (index, row)
    kept.sort(key=lambda item: item[0])
    return kept, seen


def _is_attack(alert: Alert) -> bool:
    return alert.ground_truth_label != BENIGN


def analyse_flows(
    path: Path | str, *, limit: int = 20_000, seed: int = 20260928, samples: int = 24
) -> dict[str, Any]:
    """Train and evaluate the triage model on a real flow CSV. Returns a JSON-able report."""
    return _fit_and_evaluate(path, limit=limit, seed=seed, samples=samples)[0]


def train_real(
    path: Path | str, *, limit: int = 20_000, seed: int = 20260928
) -> tuple[Any, list[Alert], dict[str, Any]]:
    """The fitted triage model, the held-out real flows (file order), and the report.

    What the "real" workspace is built from: the model was trained on the real training
    split only, and the flows returned are the test split it never saw, so replaying them
    through the incident pipeline is a fair run of the real model on real traffic.
    """
    report, model, test = _fit_and_evaluate(path, limit=limit, seed=seed, samples=24)
    return model, test, report


def _fit_and_evaluate(
    path: Path | str, *, limit: int, seed: int, samples: int
) -> tuple[dict[str, Any], Any, list[Alert]]:
    from sentinel.agents.triage import TriageAgent, TriageModel
    from sentinel.ml.metrics import four_way_split

    started = time.perf_counter()
    file = Path(path)
    fmt = detect_format(file)
    if fmt is None:
        raise RealDataError(f"{file.name} is not a UNSW-NB15 or CIC-IDS2017 flow CSV")
    if not 1_000 <= limit <= 200_000:
        raise RealDataError("limit must be between 1,000 and 200,000 rows")

    rows, total_rows = _sample_rows(file, limit, seed)
    normalizer = _normalizer(fmt)
    alerts: list[Alert] = []
    for index, row in rows:
        alert = normalizer.normalize(row, row_index=index)
        if alert is not None:
            alerts.append(alert)
    if len(alerts) < MIN_ROWS:
        raise RealDataError(f"only {len(alerts)} usable rows in {file.name}; need {MIN_ROWS}")

    labels = np.array([1 if _is_attack(a) else 0 for a in alerts])
    if labels.sum() < 50 or (1 - labels).sum() < 50:
        raise RealDataError("the sample needs at least 50 benign and 50 attack flows")
    split = four_way_split(labels, seed=seed)
    model = TriageModel.fit(
        [alerts[i] for i in split.train_benign],
        train_labelled=[alerts[i] for i in split.train_labelled],
        validation=[alerts[i] for i in split.validation],
        seed=seed,
    )
    test = [alerts[i] for i in split.test]
    truth = labels[split.test]
    results = TriageAgent(model=model).triage_batch(test)
    kept = np.array([0 if r.decision is TriageDecision.AUTO_DISMISS else 1 for r in results])
    tp = int(((kept == 1) & (truth == 1)).sum())
    fp = int(((kept == 1) & (truth == 0)).sum())
    fn = int(((kept == 0) & (truth == 1)).sum())
    tn = int(((kept == 0) & (truth == 0)).sum())

    scores = np.array([r.anomaly_score if r.anomaly_score is not None else 0.0 for r in results])
    roc_auc = _roc_auc(truth, scores)

    families: dict[str, list[int]] = {}
    for alert, flag in zip(test, kept, strict=True):
        if _is_attack(alert):
            families.setdefault(alert.ground_truth_label or "unlabelled", []).append(int(flag))
    per_family = {
        name: {"flows": len(flags), "recall": sum(flags) / len(flags)}
        for name, flags in sorted(families.items(), key=lambda kv: -len(kv[1]))
    }

    report = {
        "dataset": normalizer.dataset,
        "file": file.name,
        "format": fmt,
        "rows_in_file": total_rows,
        "rows_sampled": len(alerts),
        "rows_rejected": len(rows) - len(alerts),
        "attack_share": float(labels.mean()),
        "split": split.sizes,
        "test_flows": len(test),
        "metrics": {
            "roc_auc": roc_auc,
            "agreement": float((kept == truth).mean()),
            "recall": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp),
            "alert_reduction": float((kept == 0).mean()),
        },
        "confusion": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "per_family": per_family,
        "label_counts": dict(Counter(a.ground_truth_label or "unlabelled" for a in alerts)),
        "samples": _samples(test, results, samples),
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 1),
        "notes": [
            "Real, public, labelled network flows; not generated by this project.",
            "Per-flow detection only: the train/test CSVs have no capture timestamps, so no "
            "session-context features exist for this data."
            if fmt == "unsw"
            else "Session-context features are not computed in this run.",
            "The test split was never seen by the fit or the calibration.",
        ],
    }
    return report, model, test


def _roc_auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if truth.min() == truth.max():
        return None
    return float(roc_auc_score(truth, scores))


def _samples(test: list[Alert], results: list[Any], count: int) -> list[dict[str, Any]]:
    """A readable slice of real decisions: a mix of attacks and benign, deterministic."""
    attacks = [i for i, a in enumerate(test) if _is_attack(a)]
    benign = [i for i, a in enumerate(test) if not _is_attack(a)]
    chosen = attacks[: count // 2 + count % 2] + benign[: count // 2]
    out: list[dict[str, Any]] = []
    for i in chosen:
        alert, result = test[i], results[i]
        out.append(
            {
                "protocol": alert.protocol,
                "dst_port": alert.dst_port,
                "bytes_out": _feature(alert, "src_bytes"),
                "duration_s": _feature(alert, "duration_seconds"),
                "true_label": alert.ground_truth_label or "unlabelled",
                "decision": result.decision.value,
                "severity": result.severity.value,
                "confidence": float(result.confidence),
                "correct": (result.decision is not TriageDecision.AUTO_DISMISS)
                == _is_attack(alert),
            }
        )
    return out


def _feature(alert: Alert, *names: str) -> float | None:
    for name in names:
        value = alert.features.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None
