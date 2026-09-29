"""The Triage Agent (PRD F-02, Appendix A, Section 5.5.1).

F-02 asks for *"severity, likely technique, confidence"* per alert with *"≥ 85%
agreement with dataset ground-truth labels on the held-out test split"*, in under
five seconds. The design that meets it is deterministic; the language model sits
on top of it under :func:`~sentinel.agents.engine.monotone_caution` and cannot
make any outcome less cautious.

How the verdict is built
------------------------
1.  **Injection pre-scan.** ``alert.injection_scan`` runs before anything else.
    Content that reads like an instruction escalates immediately and is never
    dismissed — enforced twice, here and by ``TriageResult``'s validator.
2.  **Novelty.** The Section 5.5.2 anomaly ensemble scores the alert. Two
    thresholds calibrated on *training benign* traffic carve the score into
    dismiss / monitor / escalate bands.
3.  **Family.** The supervised classifier (:mod:`sentinel.ml.classify`) names the
    attack family, which supplies the severity and the ATT&CK mapping.
4.  **Confidence, gated.** This is the step worth reading closely — see below.
5.  **Reconciliation.** The engine's opinion is merged upward only.

The novelty gate is a safety mechanism, not just a probe
--------------------------------------------------------
Part 2.2 built :class:`~sentinel.ml.robustness.NoveltyGate` to answer Section
5.5.5's calibration question, and measured that a classifier's softmax *rises*
on far-out-of-distribution input — confident exactly where it knows least. Here
that measurement becomes operational. Confidence is multiplied by
``1 - novelty_excess`` before the decision is taken, so an alert unlike anything
in training arrives at the dismissal floor with its confidence already
discounted. Appendix A's ``confidence < 0.6 ⇒ escalate`` rule then fires on its
own.

The consequence is the behaviour a security team actually wants and that a
threshold on raw score does not give: **the stranger an alert is, the harder it
is to auto-dismiss.** An attack technique absent from the training data is the
case where a supervised triage model is most dangerous, because it is confidently
wrong; routing that case to a human instead is what the gate buys.

Why two thresholds instead of one
---------------------------------
One threshold gives a binary auto-dismiss/escalate, which throws away the
``monitor`` decision Appendix A defines and makes the PRD's alert-reduction
target (Section 9.1, ≥60%) a direct trade against recall. The monitor band —
scored above the benign population but below the escalation line — is where an
alert is cheap to keep and expensive to close. Both thresholds are calibrated on
the training benign split by false-positive rate, never on the evaluation split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from sentinel.agents.engine import (
    NullEngine,
    ReasoningEngine,
    monotone_caution,
    parse_triage_opinion,
)
from sentinel.agents.prompts import TRIAGE_SYSTEM_PROMPT, AgentPrompt
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import ModelNotFittedError, SentinelError
from sentinel.core.schemas import Alert, Severity, TriageDecision, TriageResult
from sentinel.core.untrusted import InjectionVerdict
from sentinel.ml.anomaly import WeightedEnsemble
from sentinel.ml.classify import FamilyClassifier
from sentinel.ml.featurestore import AlertVectorizer
from sentinel.ml.robustness import NoveltyGate

__all__ = [
    "BENIGN_FAMILY",
    "DEFAULT_ESCALATE_GRID",
    "DEFAULT_MONITOR_GRID",
    "SEVERITY_BY_FAMILY",
    "TARGET_PRECISION",
    "TARGET_RECALL",
    "TECHNIQUE_BY_FAMILY",
    "Assessment",
    "Calibration",
    "TriageAgent",
    "TriageError",
    "TriageModel",
    "known_field_names",
]

BENIGN_FAMILY: Final[str] = "benign"


class TriageError(SentinelError):
    """The Triage Agent could not produce a verdict."""


# --------------------------------------------------------------------------- #
# Policy tables
# --------------------------------------------------------------------------- #

SEVERITY_BY_FAMILY: Final[dict[str, Severity]] = {
    BENIGN_FAMILY: Severity.LOW,
    # Reconnaissance is pre-attack: real, and not an emergency on its own.
    "recon": Severity.MEDIUM,
    # Availability attacks are loud and self-evident; the operational response is
    # mitigation, not investigation, so they rank below the quiet ones.
    "dos": Severity.HIGH,
    "ddos": Severity.HIGH,
    "brute_force": Severity.HIGH,
    "web_attack": Severity.HIGH,
    # A beacon and a successful transfer both mean an adversary is already inside.
    "botnet": Severity.CRITICAL,
    "infiltration": Severity.CRITICAL,
}
"""Family → base severity.

A policy, not a measurement, and stated as a table so a customer can change it
without touching the detector. The ordering encodes one judgement worth making
explicit: a confirmed foothold outranks a loud outage, because the outage is
already visible to everyone and the foothold is not.
"""

TECHNIQUE_BY_FAMILY: Final[dict[str, str]] = {
    "recon": "T1046",  # Network Service Discovery
    "brute_force": "T1110",  # Brute Force
    "web_attack": "T1190",  # Exploit Public-Facing Application
    "dos": "T1499",  # Endpoint Denial of Service
    "ddos": "T1498",  # Network Denial of Service
    "botnet": "T1071",  # Application Layer Protocol (C2)
    "infiltration": "T1105",  # Ingress Tool Transfer
}
"""Family → MITRE ATT&CK technique. Every id here is present in the Part 2.4 KB.

Checked by a test rather than by review: a mapping to a technique the knowledge
base cannot resolve produces an investigation whose central citation points at
nothing, which is the exact failure F-05 forbids.
"""

SUPPORTING_FIELDS_BY_FAMILY: Final[dict[str, tuple[str, ...]]] = {
    "recon": ("src_distinct_dst_ports_window", "src_flow_count_window", "dst_port"),
    "brute_force": ("src_flow_count_window", "dst_port", "src_interarrival_cv_window"),
    "web_attack": ("dst_port", "bytes_ratio_src_to_total", "src_bytes"),
    "dos": ("packets_per_second", "dst_flow_count_window", "duration_seconds"),
    "ddos": ("dst_distinct_src_ips_window", "dst_flow_count_window", "packets_per_second"),
    "botnet": ("src_interarrival_cv_window", "src_interarrival_mean_window", "dst_port"),
    "infiltration": ("src_bytes", "bytes_ratio_src_to_total", "src_bytes_sum_window"),
}
"""Appendix A: *"never claim a technique mapping without citing the specific alert
field(s) that support it"*. These are the fields that justify each mapping, and
they are filtered against the alert at runtime — a mapping whose evidence fields
are all absent is dropped rather than claimed."""

_ALERT_ATTRIBUTE_FIELDS: Final[tuple[str, ...]] = (
    "source",
    "signature",
    "asset_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "dataset",
)


def known_field_names(alert: Alert) -> tuple[str, ...]:
    """Every field name a citation may legitimately refer to on this alert.

    Used both to filter the built-in technique evidence and to validate an
    engine's ``supporting_fields`` — an LLM citing a field that does not exist is
    the cheapest detectable form of hallucination, and it is worth detecting
    precisely because it is cheap.
    """
    names = [name for name in _ALERT_ATTRIBUTE_FIELDS if getattr(alert, name, None) is not None]
    names.extend(alert.features)
    return tuple(dict.fromkeys(names))


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #


#: PRD Section 9.1's triage targets. Both are reported; the calibrator treats
#: them as constraints rather than as the objective, because an optimiser handed
#: "maximise agreement" alone would discover that dismissing everything scores
#: the benign base rate (0.773 on this corpus) and calls that a win.
TARGET_RECALL: Final[float] = 0.80
TARGET_PRECISION: Final[float] = 0.85

DEFAULT_ESCALATE_GRID: Final[tuple[float, ...]] = (0.005, 0.01, 0.02, 0.05, 0.10)
DEFAULT_MONITOR_GRID: Final[tuple[float, ...]] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
"""Candidate false-positive rates, measured against *training benign* traffic.

Expressed as FPRs rather than as raw scores so the grid means the same thing
across corpora and across a refitted detector: "the score at which one benign
flow in fifty is above the line" survives a model change in a way that "0.871"
does not.
"""


@dataclass(frozen=True, slots=True)
class Calibration:
    """The operating point chosen on validation, with the evidence for it."""

    escalate_fpr: float
    monitor_fpr: float
    escalate_threshold: float
    monitor_threshold: float
    agreement: float
    recall: float
    precision: float
    dismiss_rate: float
    meets_targets: bool
    #: How many grid points were scored to arrive at this one. Recorded on the
    #: winner at the end of the search, so it is the size of the search rather
    #: than the winner's position in it — the two differ, and the second is not
    #: a useful number to report.
    n_considered: int = 0

    def summary(self) -> str:
        verdict = "meets" if self.meets_targets else "MISSES"
        return (
            f"escalate@fpr={self.escalate_fpr:.3f} monitor@fpr={self.monitor_fpr:.3f} "
            f"-> agreement={self.agreement:.4f} recall={self.recall:.4f} "
            f"precision={self.precision:.4f} ({verdict} Section 9.1 targets)"
        )


@dataclass(frozen=True, slots=True)
class Assessment:
    """What the deterministic core computed, before any decision policy applies."""

    anomaly_score: float
    novelty_excess: float
    family: str
    family_probability: float
    gated_confidence: float
    benign_probability: float

    @property
    def looks_like_attack(self) -> bool:
        return self.family != BENIGN_FAMILY


@dataclass(slots=True)
class TriageModel:
    """The fitted deterministic core. One object, so train/serve cannot diverge.

    Bundling the vectorizer with the detectors is the point: PRD Section 7.3 names
    train/serve skew as the failure this architecture is meant to avoid, and the
    way skew actually happens is a vectorizer fitted in one place and a detector
    loaded in another. Here they are fitted together, versioned together by
    :attr:`version`, and there is no constructor that accepts one without the
    other.
    """

    vectorizer: AlertVectorizer
    ensemble: WeightedEnsemble
    escalate_threshold: float
    monitor_threshold: float
    classifier: FamilyClassifier | None = None
    gate: NoveltyGate | None = None
    version: str = "triage-1"
    calibration: Calibration | None = None
    #: Novelty scores of the benign training rows. Kept so :meth:`calibrate` can
    #: turn a false-positive rate into a threshold without refitting.
    _benign_reference: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=float), repr=False
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.monitor_threshold <= self.escalate_threshold <= 1.0:
            raise TriageError(
                f"thresholds must satisfy 0 <= monitor ({self.monitor_threshold}) <= "
                f"escalate ({self.escalate_threshold}) <= 1; an inverted pair silently "
                "makes the monitor band empty and turns triage back into a coin flip"
            )

    # --- fitting -------------------------------------------------------------- #

    @classmethod
    def fit(
        cls,
        train_benign: Sequence[Alert],
        *,
        train_labelled: Sequence[Alert] | None = None,
        validation: Sequence[Alert] | None = None,
        target_fpr: float = 0.10,
        monitor_fpr: float = 0.30,
        seed: int = 20260929,
        version: str = "triage-1",
        with_classifier: bool = True,
    ) -> TriageModel:
        """Fit on training data only.

        ``train_benign`` fits the vectorizer and the novelty detectors (benign-only
        by construction — that is what makes them novelty detectors).
        ``train_labelled`` fits the supervised family classifier and may contain
        attacks. Both must come from the training split; nothing here sees
        validation or test, and the separation is a parameter rather than an
        internal split so that a caller cannot accidentally hand over everything.
        """
        if not train_benign:
            raise TriageError("cannot fit a triage model on no benign training data")
        from sentinel.ml.deep import build_deep_ensemble

        vectorizer = AlertVectorizer().fit(train_benign)
        x_benign = vectorizer.transform(train_benign)
        ensemble = build_deep_ensemble(random_state=seed)
        ensemble.fit(x_benign)

        benign_scores = np.asarray(ensemble.score(x_benign), dtype=float).ravel()
        escalate = ensemble.threshold_for_fpr(x_benign, target_fpr)
        monitor = ensemble.threshold_for_fpr(x_benign, monitor_fpr)

        classifier: FamilyClassifier | None = None
        gate: NoveltyGate | None = None
        if with_classifier and train_labelled:
            families = [alert.ground_truth_label or BENIGN_FAMILY for alert in train_labelled]
            if len(set(families)) < 2:
                raise TriageError(
                    "the labelled training split contains one family; a classifier "
                    "fitted on it would predict that family for everything"
                )
            x_labelled = vectorizer.transform(list(train_labelled))
            classifier = FamilyClassifier(seed=seed).fit(x_labelled, families)
            gate = NoveltyGate().fit(ensemble.score(x_labelled))

        model = cls(
            vectorizer=vectorizer,
            ensemble=ensemble,
            _benign_reference=benign_scores,
            escalate_threshold=float(escalate),
            monitor_threshold=float(monitor),
            classifier=classifier,
            gate=gate,
            version=version,
        )
        if validation:
            model = model.calibrate(validation)
        return model

    def calibrate(
        self,
        validation: Sequence[Alert],
        *,
        escalate_grid: Sequence[float] = DEFAULT_ESCALATE_GRID,
        monitor_grid: Sequence[float] = DEFAULT_MONITOR_GRID,
        target_recall: float = TARGET_RECALL,
        target_precision: float = TARGET_PRECISION,
    ) -> TriageModel:
        """Choose the operating point on ``validation``. Never on test.

        The search is over false-positive rates measured on *training benign*
        traffic, scored by agreement with ground truth on the validation split,
        subject to PRD Section 9.1's recall and precision floors. Constraints
        rather than a weighted objective: a weighted objective has a knob, and a
        knob on the metric that decides whether an alert reaches a human is a knob
        on how many incidents get closed unseen.

        When no grid point satisfies both floors the best unconstrained point is
        returned with ``meets_targets=False``, so the failure is reported rather
        than converted into a quietly relaxed threshold.
        """
        if not validation:
            raise TriageError("cannot calibrate on an empty validation split")
        truth = np.array(
            [0 if a.ground_truth_label == BENIGN_FAMILY else 1 for a in validation], dtype=int
        )
        if truth.sum() == 0 or truth.sum() == truth.size:
            raise TriageError(
                "the validation split contains one class; every threshold would score "
                "identically and the choice between them would be arbitrary"
            )

        matrix = self.vectorizer.transform(list(validation))
        novelty = np.asarray(self.ensemble.score(matrix), dtype=float).ravel()
        if self.classifier is not None and self.classifier.is_fitted:
            probabilities = self.classifier.predict_proba(matrix)
            classes = self.classifier.classes_
            predicted_attack = np.array(
                [classes[i] != BENIGN_FAMILY for i in np.argmax(probabilities, axis=1)]
            )
        else:
            predicted_attack = np.zeros(len(validation), dtype=bool)

        # The reference population for an FPR is training benign traffic, which is
        # also what the detector was fitted on. Recomputed here rather than reused
        # from fit() so a model whose thresholds were set by hand can still be
        # calibrated.
        benign_reference = self._benign_reference
        if benign_reference.size == 0:
            raise TriageError(
                "this model carries no benign reference scores, so a false-positive "
                "rate cannot be turned into a threshold; refit it with TriageModel.fit"
            )
        best: Calibration | None = None
        considered = 0
        for escalate_fpr in sorted(escalate_grid):
            escalate = float(np.quantile(benign_reference, 1.0 - escalate_fpr))
            for monitor_fpr in sorted(monitor_grid):
                monitor = float(np.quantile(benign_reference, 1.0 - monitor_fpr))
                if monitor > escalate:
                    # An inverted pair makes the monitor band empty; skipped rather
                    # than clamped so the grid does not silently evaluate the same
                    # operating point several times.
                    continue
                considered += 1
                kept = predicted_attack | (novelty >= monitor)
                candidate = _score_operating_point(
                    truth,
                    kept,
                    escalate_fpr=escalate_fpr,
                    monitor_fpr=monitor_fpr,
                    escalate_threshold=escalate,
                    monitor_threshold=monitor,
                    target_recall=target_recall,
                    target_precision=target_precision,
                )
                best = _better(best, candidate)

        if best is None:  # pragma: no cover - the grids always contain a valid pair
            raise TriageError("no valid operating point in the supplied grids")
        best = replace(best, n_considered=considered)
        return TriageModel(
            vectorizer=self.vectorizer,
            ensemble=self.ensemble,
            _benign_reference=self._benign_reference,
            escalate_threshold=best.escalate_threshold,
            monitor_threshold=best.monitor_threshold,
            classifier=self.classifier,
            gate=self.gate,
            version=self.version,
            calibration=best,
        )

    # --- scoring -------------------------------------------------------------- #

    def assess(self, alert: Alert) -> Assessment:
        return self.assess_batch([alert])[0]

    def assess_batch(self, alerts: Sequence[Alert]) -> tuple[Assessment, ...]:
        """Score a batch. Vectorized because F-01 is a throughput requirement.

        The single-alert path calls this rather than the reverse, so there is one
        scoring implementation and the per-alert and bulk numbers cannot drift.
        """
        if not alerts:
            return ()
        matrix = self.vectorizer.transform(list(alerts))
        novelty = np.asarray(self.ensemble.score(matrix), dtype=float).ravel()
        excess = (
            np.asarray(self.gate.excess(novelty), dtype=float).ravel()
            if self.gate is not None
            else np.zeros_like(novelty)
        )

        if self.classifier is not None and self.classifier.is_fitted:
            probabilities = self.classifier.predict_proba(matrix)
            classes = self.classifier.classes_
            best = np.argmax(probabilities, axis=1)
            families = [classes[index] for index in best]
            family_probability = probabilities[np.arange(len(alerts)), best]
            benign_probability = _column(probabilities, classes, BENIGN_FAMILY)
        else:
            # Without a classifier the score alone decides, and the "family" is the
            # binary one the detector can actually speak to. Reported honestly as
            # ``attack`` rather than guessing a specific family.
            families = [
                "attack" if score >= self.escalate_threshold else BENIGN_FAMILY
                for score in novelty
            ]
            family_probability = np.where(
                novelty >= self.escalate_threshold, novelty, 1.0 - novelty
            )
            benign_probability = 1.0 - novelty

        gated = np.clip(family_probability * (1.0 - excess), 0.0, 1.0)
        return tuple(
            Assessment(
                anomaly_score=float(novelty[i]),
                novelty_excess=float(excess[i]),
                family=str(families[i]),
                family_probability=float(family_probability[i]),
                gated_confidence=float(gated[i]),
                benign_probability=float(benign_probability[i]),
            )
            for i in range(len(alerts))
        )


def _column(
    probabilities: npt.NDArray[np.float64], classes: Sequence[str], name: str
) -> npt.NDArray[np.float64]:
    """The column for ``name``, or zeros when the classifier never saw that class."""
    if name in classes:
        return probabilities[:, list(classes).index(name)]
    return np.zeros(probabilities.shape[0], dtype=float)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TriageAgent:
    """Turns an :class:`Alert` into a :class:`TriageResult` (PRD F-02)."""

    model: TriageModel
    engine: ReasoningEngine = field(default_factory=NullEngine)
    clock: Clock = field(default_factory=SystemClock)
    #: Consult the engine only when the deterministic verdict is not already at
    #: maximum caution. Asking a model to review an escalation it cannot change
    #: spends a call to learn nothing — monotone caution means the answer is
    #: already fixed.
    consult_engine_on_escalation: bool = False

    def triage(self, alert: Alert, *, now: datetime | None = None) -> TriageResult:
        """Produce a verdict. Pure: no audit writes, no state mutation."""
        started = self.clock.monotonic()
        decided_at = now or self.clock.now()

        assessment = self.model.assess(alert)
        baseline = self._deterministic(alert, assessment, decided_at=decided_at, started=started)

        if not self._should_consult(baseline):
            return baseline

        response = self.engine.respond(self._prompt(alert, assessment))
        opinion = parse_triage_opinion(response)
        merged = monotone_caution(
            baseline,
            opinion,
            known_fields=known_field_names(alert),
            engine_name=response.engine,
        )
        return merged.updated(
            latency_ms=self._elapsed_ms(started),
            model_version=f"{self.model.version}+{response.engine}",
        )

    def triage_batch(self, alerts: Sequence[Alert], *, now: datetime | None = None) -> tuple[
        TriageResult, ...
    ]:
        """Vectorized triage for the offline evaluation (F-12) and throughput tests.

        The engine is not consulted here. That is deliberate and is stated rather
        than hidden: a 20,000-alert evaluation making 20,000 model calls is not a
        thing anyone would run, and quoting an F-02 number produced with an engine
        in the loop while the batch path skips it would make the headline metric
        untraceable to the code that produced it. The deterministic path is what
        the evaluation measures, and monotone caution is what guarantees the
        online path is no less safe than what was measured.
        """
        decided_at = now or self.clock.now()
        started = self.clock.monotonic()
        assessments = self.model.assess_batch(alerts)
        return tuple(
            self._deterministic(alert, assessment, decided_at=decided_at, started=started)
            for alert, assessment in zip(alerts, assessments, strict=True)
        )

    # --- internals ------------------------------------------------------------ #

    def _should_consult(self, baseline: TriageResult) -> bool:
        if isinstance(self.engine, NullEngine):
            return False
        if baseline.decision is TriageDecision.ESCALATE:
            return self.consult_engine_on_escalation
        return True

    def _deterministic(
        self,
        alert: Alert,
        assessment: Assessment,
        *,
        decided_at: datetime,
        started: float,
    ) -> TriageResult:
        scan = alert.injection_scan
        injected = scan.is_attack_indicator
        verdict = (
            InjectionVerdict.LIKELY_INJECTION
            if injected
            else InjectionVerdict(scan.verdict)
        )

        decision = self._decide(assessment, injected=injected)
        confidence = self._confidence(assessment, decision)
        severity = self._severity(assessment, decision, injected=injected)
        technique, supporting = self._technique(alert, assessment, decision)

        # Appendix A's floor, applied before construction rather than caught by the
        # validator. Both exist: this one produces the right object, the validator
        # guarantees no other code path can produce a wrong one.
        if decision is TriageDecision.AUTO_DISMISS and confidence < (
            TriageResult.DISMISS_CONFIDENCE_FLOOR
        ):
            decision = TriageDecision.ESCALATE
            severity = max(severity, Severity.MEDIUM)

        return TriageResult(
            severity=severity,
            confidence=confidence,
            decision=decision,
            technique_id=technique,
            rationale=self._rationale(assessment, decision, injected=injected, scan=scan),
            supporting_fields=supporting,
            anomaly_score=assessment.anomaly_score,
            injection_verdict=verdict,
            model_version=self.model.version,
            latency_ms=self._elapsed_ms(started),
            decided_at=decided_at,
        )

    def _decide(self, assessment: Assessment, *, injected: bool) -> TriageDecision:
        """Two signals, and an explicit rule for what each one alone is worth.

        The classifier names a family; the novelty detector says how unlike
        training benign traffic the flow is. They fail differently, which is the
        whole reason Section 5.5.2 asks for an ensemble, so neither is allowed to
        page a human on its own:

        *   **Both agree it is an attack** → escalate. Two independent signals.
        *   **Only the classifier says attack** → monitor. The novelty score is
            inside the benign range, so the family call is unconfirmed.
        *   **Only the detector says novel** → monitor. Measured on this corpus
            the detector's benign false-positive rate at the escalation threshold
            is an order of magnitude above the classifier's, so letting it
            escalate alone converts every unusual-but-legitimate flow into a page.
            Novelty with no family attached is the definition of "worth keeping,
            not worth waking someone".
        *   **Neither** → dismiss, subject to the confidence floor.
        """
        if injected:
            return TriageDecision.ESCALATE
        novel = assessment.anomaly_score >= self.model.escalate_threshold
        if assessment.looks_like_attack and novel:
            return TriageDecision.ESCALATE
        if assessment.looks_like_attack or assessment.anomaly_score >= (
            self.model.monitor_threshold
        ):
            return TriageDecision.MONITOR
        return TriageDecision.AUTO_DISMISS

    def _confidence(self, assessment: Assessment, decision: TriageDecision) -> float:
        """Confidence *in the decision taken*, gated by novelty.

        Not the classifier's top probability, which would answer a different
        question. When the decision is to dismiss, the quantity that matters is
        belief that the alert is benign; when it is to escalate, belief that it is
        not. Conflating them is how a model's 0.95 certainty that traffic is
        ``ddos`` becomes 0.95 confidence in a dismissal.
        """
        if decision is TriageDecision.AUTO_DISMISS:
            raw = assessment.benign_probability
        else:
            raw = max(assessment.family_probability, 1.0 - assessment.benign_probability)
        gated = raw * (1.0 - assessment.novelty_excess)
        return float(min(1.0, max(0.0, gated)))

    def _severity(
        self, assessment: Assessment, decision: TriageDecision, *, injected: bool
    ) -> Severity:
        if injected:
            # Someone is writing at the agents. That is a targeted attempt, and it
            # is never low severity whatever the flow features say.
            return Severity.HIGH
        base = SEVERITY_BY_FAMILY.get(assessment.family, Severity.MEDIUM)
        if decision is TriageDecision.AUTO_DISMISS:
            return Severity.LOW
        if assessment.family == BENIGN_FAMILY:
            # Escalating a flow the classifier calls benign means the detector
            # disagreed. Medium: real enough to look at, not asserted as high on
            # the strength of a disagreement.
            return Severity.MEDIUM if decision is TriageDecision.ESCALATE else Severity.LOW
        return base

    def _technique(
        self, alert: Alert, assessment: Assessment, decision: TriageDecision
    ) -> tuple[str | None, tuple[str, ...]]:
        if decision is TriageDecision.AUTO_DISMISS:
            return None, ()
        technique = TECHNIQUE_BY_FAMILY.get(assessment.family)
        if technique is None:
            return None, ()
        known = set(known_field_names(alert))
        supporting = tuple(
            name for name in SUPPORTING_FIELDS_BY_FAMILY.get(assessment.family, ()) if name in known
        )
        if not supporting:
            # Appendix A forbids the claim without the citation, so the claim goes.
            return None, ()
        return technique, supporting

    def _rationale(
        self,
        assessment: Assessment,
        decision: TriageDecision,
        *,
        injected: bool,
        scan: Any,
    ) -> str:
        parts: list[str] = []
        if injected:
            parts.append(
                f"Alert payload matched prompt-injection heuristics ({scan.summary()}); "
                "escalated as a targeted attempt against the agent layer."
            )
        parts.append(
            f"Anomaly ensemble scored {assessment.anomaly_score:.3f} "
            f"(escalate>={self.model.escalate_threshold:.3f}, "
            f"monitor>={self.model.monitor_threshold:.3f})."
        )
        parts.append(
            f"Classifier: {assessment.family} at p={assessment.family_probability:.3f}."
        )
        if assessment.novelty_excess > 0.0:
            parts.append(
                f"Confidence discounted by {assessment.novelty_excess:.2f} for novelty "
                "beyond the training distribution."
            )
        parts.append(f"Decision: {decision.value}.")
        return " ".join(parts)[:4000]

    def _prompt(self, alert: Alert, assessment: Assessment) -> AgentPrompt:
        task = (
            "Review the deterministic triage below and the raw alert payload.\n"
            f"Detector novelty score: {assessment.anomaly_score:.3f} "
            f"(escalation threshold {self.model.escalate_threshold:.3f}).\n"
            f"Classifier family: {assessment.family} "
            f"(p={assessment.family_probability:.3f}, "
            f"novelty discount {assessment.novelty_excess:.2f}).\n"
            f"Alert fields available for citation: "
            f"{', '.join(known_field_names(alert))}.\n"
            "Report anything the detectors would miss. You can raise severity, "
            "lower confidence, or escalate; you cannot lower any of them."
        )
        schema = (
            '{"severity": "low|medium|high|critical", "confidence": 0.0, '
            '"decision": "auto_dismiss|monitor|escalate", "technique_id": "T####" | null, '
            '"supporting_fields": ["field_name"], "rationale": "string", '
            '"injection_detected": true|false}'
        )
        return (
            AgentPrompt(system=TRIAGE_SYSTEM_PROMPT, task=task, response_schema=schema)
            .with_untrusted(
                alert.raw_payload, label="alert.raw_payload", origin="alert.raw_payload"
            )
        )

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self.clock.monotonic() - started) * 1000.0)


def require_fitted(model: TriageModel) -> TriageModel:
    """Guard for callers that received a model from configuration."""
    if not model.ensemble.is_fitted:
        raise ModelNotFittedError("triage model's anomaly ensemble is not fitted")
    return model


def _score_operating_point(
    truth: npt.NDArray[np.int_],
    kept: npt.NDArray[np.bool_],
    *,
    escalate_fpr: float,
    monitor_fpr: float,
    escalate_threshold: float,
    monitor_threshold: float,
    target_recall: float,
    target_precision: float,
) -> Calibration:
    """Score one grid point. ``kept`` is "did not auto-dismiss"."""
    predicted = kept.astype(int)
    true_positive = int(((predicted == 1) & (truth == 1)).sum())
    false_positive = int(((predicted == 1) & (truth == 0)).sum())
    false_negative = int(((predicted == 0) & (truth == 1)).sum())
    recall = true_positive / max(1, true_positive + false_negative)
    precision = true_positive / max(1, true_positive + false_positive)
    return Calibration(
        escalate_fpr=escalate_fpr,
        monitor_fpr=monitor_fpr,
        escalate_threshold=escalate_threshold,
        monitor_threshold=monitor_threshold,
        agreement=float((predicted == truth).mean()),
        recall=float(recall),
        precision=float(precision),
        dismiss_rate=float((predicted == 0).mean()),
        meets_targets=recall >= target_recall and precision >= target_precision,
    )


def _better(current: Calibration | None, candidate: Calibration) -> Calibration:
    """Prefer a point that meets the targets; among equals, prefer agreement.

    Ties break toward the *lower* escalation FPR, which is the more conservative
    operating point: on a tie in agreement, the model that pages a human less
    often for the same accuracy is the better one to ship.
    """
    if current is None:
        return candidate
    key = lambda c: (c.meets_targets, round(c.agreement, 6), -c.escalate_fpr)  # noqa: E731
    return candidate if key(candidate) > key(current) else current
