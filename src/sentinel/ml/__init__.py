"""Layer 3 — the intelligence core."""

from sentinel.ml.anomaly import (
    AnomalyDetector,
    DetectorScores,
    IsolationForestDetector,
    PCAReconstructionDetector,
    WeightedEnsemble,
    build_default_ensemble,
)
from sentinel.ml.featurestore import DEFAULT_SPEC, AlertVectorizer, FeatureSpec, default_spec
from sentinel.ml.metrics import DetectionReport, detection_report, roc_auc, three_way_split

__all__ = [
    "DEFAULT_SPEC",
    "AlertVectorizer",
    "AnomalyDetector",
    "DetectionReport",
    "DetectorScores",
    "FeatureSpec",
    "IsolationForestDetector",
    "PCAReconstructionDetector",
    "WeightedEnsemble",
    "build_default_ensemble",
    "default_spec",
    "detection_report",
    "roc_auc",
    "three_way_split",
]
