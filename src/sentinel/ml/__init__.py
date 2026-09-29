"""Layer 3 — the intelligence core."""

from sentinel.ml.anomaly import (
    AnomalyDetector,
    DetectorScores,
    IsolationForestDetector,
    PCAReconstructionDetector,
    WeightedEnsemble,
    build_default_ensemble,
)
from sentinel.ml.classify import ClassifierError, FamilyClassifier
from sentinel.ml.diffusion import DiffusionError, TabularDiffusion
from sentinel.ml.featurestore import DEFAULT_SPEC, AlertVectorizer, FeatureSpec, default_spec
from sentinel.ml.metrics import DetectionReport, detection_report, roc_auc, three_way_split
from sentinel.ml.robustness import (
    AugmentationResult,
    CalibrationReport,
    NoveltyGate,
    RobustnessError,
    assert_calibration_holds,
    augment_training_set,
    boundary_adjacent_samples,
    calibration_report,
    gated_robustness_curve,
    robustness_curve,
)

__all__ = [
    "DEFAULT_SPEC",
    "AlertVectorizer",
    "AnomalyDetector",
    "AugmentationResult",
    "CalibrationReport",
    "ClassifierError",
    "DetectionReport",
    "DetectorScores",
    "DiffusionError",
    "FamilyClassifier",
    "FeatureSpec",
    "IsolationForestDetector",
    "NoveltyGate",
    "PCAReconstructionDetector",
    "RobustnessError",
    "TabularDiffusion",
    "WeightedEnsemble",
    "assert_calibration_holds",
    "augment_training_set",
    "boundary_adjacent_samples",
    "build_default_ensemble",
    "calibration_report",
    "default_spec",
    "detection_report",
    "gated_robustness_curve",
    "robustness_curve",
    "roc_auc",
    "three_way_split",
]
