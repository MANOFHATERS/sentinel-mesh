"""Canonical data contracts — PRD Appendix B, hardened.

There is exactly one definition of what an Alert is, and both the real-time agent
pipeline and the offline training/evaluation jobs import it from here. PRD
Section 7.3 calls that out as the discipline that prevents train/serve skew, and
it is enforced structurally: the feature store (:mod:`sentinel.ml.featurestore`)
can only vectorize an :class:`Alert`, so there is no second path by which
training data could be shaped differently from serving data.

Design rules applied throughout:

*Frozen models.* Every contract is immutable. Enrichment returns a new object via
helpers such as :meth:`Alert.with_triage`, so an object already written to the
audit log can never be mutated behind the log's back.

*extra="forbid".* An unexpected field is a bug or an injection attempt, not a
convenience. Silent field-dropping is how a ``requires_human_approval=false``
sneaks through a schema migration.

*Hash-safe by construction.* Feature values are validated finite, timestamps are
validated timezone-aware, and every model round-trips through
:func:`~sentinel.core.canonical.canonical_bytes`. A contract that cannot be
hashed cannot be audited.

*Untrusted payloads are typed.* ``Alert.raw_payload`` is an
:class:`~sentinel.core.untrusted.UntrustedText`, so it cannot be interpolated
into a prompt by accident.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Final, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from sentinel.core.canonical import canonical_bytes, sha256_hex
from sentinel.core.clock import to_utc
from sentinel.core.ids import deterministic_id, new_id
from sentinel.core.untrusted import InjectionVerdict, UntrustedText

__all__ = [
    "ActionRequest",
    "ActionType",
    "Alert",
    "AlertSource",
    "ApprovalStatus",
    "AuditEventType",
    "Confidence",
    "Evidence",
    "EvidenceKind",
    "FeatureValue",
    "InvestigationReport",
    "RiskTier",
    "Severity",
    "TriageDecision",
    "TriageResult",
]

# --------------------------------------------------------------------------- #
# Primitive annotated types
# --------------------------------------------------------------------------- #

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""A probability-like score. Bounds are enforced, not documented-and-hoped-for."""

TECHNIQUE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^T\d{4}(?:\.\d{3})?$")
"""MITRE ATT&CK technique or sub-technique, e.g. ``T1566`` or ``T1566.001``."""

CVE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^CVE-\d{4}-\d{4,7}$")

SHA256_HEX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

GENESIS_HASH: Final[str] = "0" * 64
"""``audit_hash_prev`` for the first row in a chain."""


def _aware_utc(value: Any) -> Any:
    """Coerce datetimes to aware UTC; reject naive ones loudly."""
    if isinstance(value, datetime):
        return to_utc(value)
    return value


UtcDatetime = Annotated[datetime, BeforeValidator(_aware_utc)]


def _as_untrusted(origin: str):
    """Build an annotated ``UntrustedText`` field tagged with its provenance."""

    def _coerce(value: Any) -> Any:
        if isinstance(value, UntrustedText):
            return value
        if isinstance(value, str):
            return UntrustedText(value, origin=origin)
        return value

    return Annotated[
        UntrustedText,
        BeforeValidator(_coerce),
        PlainSerializer(lambda v: v.raw, return_type=str, when_used="always"),
        WithJsonSchema(
            {
                "type": "string",
                "description": f"UNTRUSTED data from {origin}; never treated as instructions",
            }
        ),
    ]


FeatureValue = float | int | bool | str | None
"""What may appear in ``Alert.features``. Deliberately narrow so it is hashable."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class AlertSource(StrEnum):
    """Where an alert entered the mesh (PRD Appendix B ``source``)."""

    SIEM = "siem"
    EDR = "edr"
    CLOUD = "cloud"
    CODE_SCAN = "code_scan"
    VENDOR_FEED = "vendor_feed"
    NETWORK_IDS = "network_ids"
    REPLAY = "replay"


class Severity(StrEnum):
    """Ordered severity. Comparisons work, so ``sev >= Severity.HIGH`` is legal."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.rank < other.rank
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.rank <= other.rank
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.rank > other.rank
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.rank >= other.rank
        return NotImplemented


_SEVERITY_RANK: Final[dict[Severity, int]] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class TriageDecision(StrEnum):
    """The Triage Agent's routing decision (PRD Appendix A)."""

    AUTO_DISMISS = "auto_dismiss"
    MONITOR = "monitor"
    ESCALATE = "escalate"


class ActionType(StrEnum):
    """Every action with a real-world side effect (PRD Appendix B)."""

    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    DISABLE_ACCOUNT = "disable_account"
    KILL_PROCESS = "kill_process"
    QUARANTINE_FILE = "quarantine_file"
    OPEN_PATCH_PR = "open_patch_pr"
    NOTIFY_ANALYST = "notify_analyst"
    ENRICH_ONLY = "enrich_only"

    @property
    def is_destructive(self) -> bool:
        """True when the action changes a monitored system's state.

        PRD Section 5.1: *every* destructive or externally-visible action pauses
        at the Human Approval Gate. This property is the single source of truth
        for that decision, so a new action type added without classifying it
        here will fail :func:`test_every_action_type_is_classified`.
        """
        return self in _DESTRUCTIVE_ACTIONS


_DESTRUCTIVE_ACTIONS: Final[frozenset[ActionType]] = frozenset(
    {
        ActionType.ISOLATE_HOST,
        ActionType.BLOCK_IP,
        ActionType.DISABLE_ACCOUNT,
        ActionType.KILL_PROCESS,
        ActionType.QUARANTINE_FILE,
        ActionType.OPEN_PATCH_PR,
    }
)


class RiskTier(StrEnum):
    """Tiered autonomy (PRD Section 5.7). Ordered: trust is earned upward."""

    OBSERVE = "observe"
    RECOMMEND = "recommend"
    AUTO_WITH_NOTIFY = "auto_with_notify"
    AUTONOMOUS = "autonomous"

    @property
    def rank(self) -> int:
        return _TIER_RANK[self]

    @property
    def permits_unattended_execution(self) -> bool:
        """Only the top two tiers may act without a human in the loop."""
        return self in (RiskTier.AUTO_WITH_NOTIFY, RiskTier.AUTONOMOUS)


_TIER_RANK: Final[dict[RiskTier, int]] = {
    RiskTier.OBSERVE: 0,
    RiskTier.RECOMMEND: 1,
    RiskTier.AUTO_WITH_NOTIFY: 2,
    RiskTier.AUTONOMOUS: 3,
}


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    FAILED = "failed"


class AgentName(StrEnum):
    """The five sprint agents plus the orchestrator (PRD Section 5.4)."""

    ORCHESTRATOR = "orchestrator"
    TRIAGE = "triage_agent"
    INVESTIGATION = "investigation_agent"
    CONTAINMENT = "containment_agent"
    CODE_SCAN = "code_scan_agent"
    SUPPLY_CHAIN = "supply_chain_agent"


class EvidenceKind(StrEnum):
    """What kind of thing an evidence citation points at."""

    RAW_LOG = "raw_log"
    ALERT_FIELD = "alert_field"
    KB_CHUNK = "kb_chunk"
    CVE_RECORD = "cve_record"
    ATTACK_TECHNIQUE = "attack_technique"
    MODEL_OUTPUT = "model_output"
    GRAPH_PATH = "graph_path"
    CODE_FINDING = "code_finding"


class AuditEventType(StrEnum):
    """Everything the tamper-evident log records (PRD Section 5.7)."""

    ALERT_INGESTED = "alert_ingested"
    ALERT_NORMALIZED = "alert_normalized"
    TRIAGE_DECIDED = "triage_decided"
    INVESTIGATION_COMPLETED = "investigation_completed"
    ACTION_PROPOSED = "action_proposed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"
    MODEL_SCORED = "model_scored"
    POLICY_UPDATED = "policy_updated"
    INJECTION_DETECTED = "injection_detected"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    TRUST_TIER_CHANGED = "trust_tier_changed"


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #


class Contract(BaseModel):
    """Base for every canonical contract: immutable, strict, hashable."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
        ser_json_timedelta="float",
    )

    def canonical_hash(self) -> str:
        """SHA-256 over this object's canonical bytes.

        Stable across processes and runs, which is what lets the audit log bind a
        row to an exact object rather than to a Python ``repr``.
        """
        return sha256_hex(canonical_bytes(self.model_dump(mode="python")))

    def updated(self, **changes: Any) -> Self:
        """Return a copy with ``changes`` applied, **re-running every validator**.

        ``model_copy(update=...)`` is the obvious way to write this and it is
        wrong here: pydantic deliberately skips validation on ``model_copy``, so
        it will happily produce an object that violates the model's own
        invariants. For ordinary models that is a performance choice; for these
        contracts it would silently defeat the guardrails — a state transition
        could land an ActionRequest in ``EXECUTED`` with no approver, which is
        precisely the state F-08 forbids. Round-tripping through
        ``model_validate`` costs a few microseconds per transition and makes the
        invariants unconditional.
        """
        payload = self.model_dump(mode="python")
        payload.update(changes)
        return type(self).model_validate(payload)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class Evidence(Contract):
    """One citation. PRD Section 5.4: every claim must cite retrieved evidence.

    ``excerpt`` may contain attacker-controlled text (a raw log line), so it is
    typed untrusted even here — an evidence list rendered into a follow-up prompt
    is exactly the kind of second-order injection path that gets missed.
    """

    kind: EvidenceKind
    ref: str = Field(min_length=1, max_length=512)
    excerpt: _as_untrusted("evidence.excerpt")  # type: ignore[valid-type]
    relevance: Confidence = 1.0
    retrieved_at: UtcDatetime | None = None

    @field_validator("ref")
    @classmethod
    def _ref_shape(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("evidence ref must not have leading/trailing whitespace")
        return value


# --------------------------------------------------------------------------- #
# Triage
# --------------------------------------------------------------------------- #


class TriageResult(Contract):
    """The Triage Agent's verdict (PRD Appendix A/B).

    The confidence floor from Appendix A ("if confidence < 0.6 you MUST escalate
    rather than dismiss") is enforced here as a schema invariant, not left to the
    model's good behaviour. A prompt rule an attacker can talk the model out of is
    not a guardrail; a validator that rejects the resulting object is.
    """

    #: Appendix A: below this confidence, auto-dismissal is structurally illegal.
    #: ``ClassVar`` rather than ``Final``: pydantic would otherwise promote it to a
    #: real, settable field in V3, which would let a caller pass a lower floor
    #: alongside the value it is supposed to constrain.
    DISMISS_CONFIDENCE_FLOOR: ClassVar[float] = 0.6

    severity: Severity
    confidence: Confidence
    decision: TriageDecision
    technique_id: str | None = None
    rationale: str = Field(min_length=1, max_length=4000)
    supporting_fields: tuple[str, ...] = Field(default=(), max_length=32)
    anomaly_score: Confidence | None = None
    cluster_id: str | None = None
    injection_verdict: InjectionVerdict = InjectionVerdict.CLEAN
    model_version: str = Field(min_length=1, max_length=64)
    latency_ms: float = Field(ge=0.0, le=600_000.0)
    decided_at: UtcDatetime

    @field_validator("technique_id")
    @classmethod
    def _technique_shape(cls, value: str | None) -> str | None:
        if value is not None and not TECHNIQUE_ID_PATTERN.match(value):
            raise ValueError(
                f"technique_id {value!r} must look like T1566 or T1566.001 (MITRE ATT&CK)"
            )
        return value

    @field_validator("latency_ms")
    @classmethod
    def _latency_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency_ms must be finite")
        return value

    @model_validator(mode="after")
    def _enforce_guardrails(self) -> Self:
        if (
            self.decision is TriageDecision.AUTO_DISMISS
            and self.confidence < self.DISMISS_CONFIDENCE_FLOOR
        ):
            raise ValueError(
                f"auto_dismiss requires confidence >= {self.DISMISS_CONFIDENCE_FLOOR}; "
                f"got {self.confidence}. Appendix A mandates escalation instead."
            )
        if (
            self.injection_verdict is InjectionVerdict.LIKELY_INJECTION
            and self.decision is not TriageDecision.ESCALATE
        ):
            raise ValueError(
                "content flagged as likely prompt injection must escalate, never be "
                "dismissed or merely monitored (PRD Section 5.7)"
            )
        if self.technique_id is not None and not self.supporting_fields:
            raise ValueError(
                "Appendix A forbids claiming a technique mapping without citing the "
                "alert field(s) that support it; supporting_fields is empty"
            )
        return self

    @property
    def requires_human(self) -> bool:
        return self.decision is TriageDecision.ESCALATE


# --------------------------------------------------------------------------- #
# Alert
# --------------------------------------------------------------------------- #


class Alert(Contract):
    """The canonical alert. Every source normalizes to exactly this.

    ``features`` is the numeric/categorical payload the ML layer consumes. It is
    validated finite because CIC-IDS2017 genuinely ships ``Infinity`` and ``NaN``
    in ``Flow Bytes/s`` and ``Flow Packets/s``; letting those through would make
    the alert unhashable (breaking the audit chain) and would poison model
    training with silent NaN propagation.

    ``ground_truth_label`` exists because the sprint replays *labelled* public
    datasets. It is carried on the alert so evaluation can join predictions to
    truth without a side-channel, and it is never read by any agent — a test
    asserts the Triage path does not consume it.
    """

    alert_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=64)
    source: AlertSource
    timestamp: UtcDatetime
    ingested_at: UtcDatetime
    asset_id: str = Field(min_length=1, max_length=128)
    signature: str = Field(default="", max_length=256)
    raw_payload: _as_untrusted("alert.raw_payload")  # type: ignore[valid-type]
    features: dict[str, FeatureValue] = Field(default_factory=dict)
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = Field(default=None, ge=0, le=65535)
    dst_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: str | None = Field(default=None, max_length=16)
    dataset: str | None = Field(default=None, max_length=64)
    ground_truth_label: str | None = Field(default=None, max_length=64)
    triage: TriageResult | None = None

    @field_validator("features")
    @classmethod
    def _features_hashable(cls, value: dict[str, FeatureValue]) -> dict[str, FeatureValue]:
        for name, item in value.items():
            if not name:
                raise ValueError("feature names must be non-empty")
            if isinstance(item, bool) or item is None or isinstance(item, (int, str)):
                continue
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(
                    f"feature {name!r} is {item!r}; non-finite values break canonical "
                    "hashing and poison training. Sanitize in the normalizer "
                    "(CIC-IDS2017 emits Infinity in Flow Bytes/s)."
                )
        return value

    @model_validator(mode="after")
    def _causality(self) -> Self:
        if self.ingested_at < self.timestamp:
            raise ValueError(
                f"ingested_at ({self.ingested_at.isoformat()}) precedes event timestamp "
                f"({self.timestamp.isoformat()}); an alert cannot be ingested before it "
                "occurred. Check source clock skew."
            )
        return self

    # --- helpers ------------------------------------------------------------

    @property
    def injection_scan(self):
        """Injection scan of the raw payload, memoized on the payload object."""
        return self.raw_payload.scan()

    def with_triage(self, triage: TriageResult) -> Alert:
        """Return a copy carrying ``triage``. Never mutates in place."""
        if self.triage is not None:
            raise ValueError(
                f"alert {self.alert_id} already carries a triage result; produce a new "
                "alert or supersede the decision explicitly rather than overwriting an "
                "already-audited verdict"
            )
        return self.updated(triage=triage)

    def dwell_seconds(self) -> float:
        """Seconds between the event happening and the mesh seeing it."""
        return (self.ingested_at - self.timestamp).total_seconds()

    @staticmethod
    def derive_id(*, dataset: str, source: str, row_index: int, tenant_id: str) -> str:
        """Deterministic alert id for replayed dataset rows.

        Replaying the same dataset twice must yield identical ids, or the demo is
        not reproducible and the evaluation cannot be joined across runs.
        """
        return deterministic_id("alert", tenant_id, dataset, source, row_index)


# --------------------------------------------------------------------------- #
# Investigation
# --------------------------------------------------------------------------- #


class InvestigationReport(Contract):
    """A cited, MITRE-mapped root-cause narrative (PRD F-05).

    The acceptance criterion is *"every factual claim traces to a retrieved KB
    chunk or raw log line"*. Enforced structurally: claims are a list of
    ``(statement, evidence_refs)`` pairs and a claim with no refs is rejected, so
    an ungrounded report cannot be constructed at all.
    """

    report_id: str = Field(min_length=1, max_length=64)
    alert_ids: tuple[str, ...] = Field(min_length=1, max_length=512)
    tenant_id: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=8000)
    claims: tuple[tuple[str, tuple[str, ...]], ...] = Field(default=())
    evidence: tuple[Evidence, ...] = Field(default=())
    techniques: tuple[str, ...] = Field(default=(), max_length=64)
    severity: Severity
    confidence: Confidence
    recommended_actions: tuple[ActionType, ...] = Field(default=())
    agent: AgentName = AgentName.INVESTIGATION
    model_version: str = Field(min_length=1, max_length=64)
    created_at: UtcDatetime

    @field_validator("techniques")
    @classmethod
    def _techniques_shape(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for technique in value:
            if not TECHNIQUE_ID_PATTERN.match(technique):
                raise ValueError(f"technique {technique!r} is not a valid ATT&CK id")
        if len(set(value)) != len(value):
            raise ValueError("duplicate technique ids in report")
        return value

    @model_validator(mode="after")
    def _every_claim_is_grounded(self) -> Self:
        known = {item.ref for item in self.evidence}
        for index, (statement, refs) in enumerate(self.claims):
            if not statement.strip():
                raise ValueError(f"claim {index} is empty")
            if not refs:
                raise ValueError(
                    f"claim {index} ({statement[:60]!r}) cites no evidence; F-05 requires "
                    "every factual claim to trace to a KB chunk or raw log line"
                )
            unknown = sorted(set(refs) - known)
            if unknown:
                raise ValueError(
                    f"claim {index} cites unknown evidence refs {unknown}; a citation "
                    "that points at nothing is worse than no citation"
                )
        return self

    @property
    def is_grounded(self) -> bool:
        """True when the report makes at least one claim and all claims cite evidence."""
        return bool(self.claims)


# --------------------------------------------------------------------------- #
# Action requests
# --------------------------------------------------------------------------- #


class ActionRequest(Contract):
    """Anything with a real-world side effect (PRD Appendix B).

    Two invariants are enforced here rather than trusted to the orchestrator:

    1.  A destructive action at or below the ``recommend`` trust tier must carry
        ``requires_human_approval=True``. This is the schema-level half of
        F-08's *"zero actions executed without a logged approval event"*.
    2.  ``approval_status`` may only be ``executed`` once an approval decision
        exists, and only ``approved`` requests may reach ``executed``.
    """

    action_id: str = Field(min_length=1, max_length=64)
    alert_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=64)
    proposed_by: AgentName
    action_type: ActionType
    target: str = Field(min_length=1, max_length=256)
    rationale: str = Field(min_length=1, max_length=4000)
    risk_tier: RiskTier
    requires_human_approval: bool
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: str | None = Field(default=None, max_length=128)
    decided_at: UtcDatetime | None = None
    executed_at: UtcDatetime | None = None
    failure_reason: str | None = Field(default=None, max_length=1000)
    evidence: tuple[Evidence, ...] = Field(default=())
    audit_hash_prev: str = GENESIS_HASH
    created_at: UtcDatetime

    @field_validator("audit_hash_prev")
    @classmethod
    def _hash_shape(cls, value: str) -> str:
        if not SHA256_HEX_PATTERN.match(value):
            raise ValueError("audit_hash_prev must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _approval_invariants(self) -> Self:
        if (
            self.action_type.is_destructive
            and not self.risk_tier.permits_unattended_execution
            and not self.requires_human_approval
        ):
            raise ValueError(
                f"{self.action_type.value} is destructive and the action is only at "
                f"trust tier {self.risk_tier.value}; requires_human_approval cannot "
                "be False (PRD Section 5.1, F-08)"
            )

        decided = self.approval_status in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
        )
        if decided and self.decided_at is None:
            raise ValueError(
                f"approval_status={self.approval_status.value} requires decided_at; an "
                "approval with no timestamp is not an audit record"
            )
        if (
            self.approval_status is ApprovalStatus.APPROVED
            and self.requires_human_approval
            and not self.approved_by
        ):
            raise ValueError(
                "a human-gated action recorded as approved must name the approver"
            )
        if self.approval_status is ApprovalStatus.EXECUTED:
            if self.executed_at is None:
                raise ValueError("executed actions must record executed_at")
            if self.requires_human_approval and not self.approved_by:
                raise ValueError(
                    "human-gated action reached EXECUTED with no approver recorded; this "
                    "is exactly the state F-08 forbids"
                )
        if self.approval_status is ApprovalStatus.FAILED and not self.failure_reason:
            raise ValueError("failed actions must record a failure_reason")
        if self.executed_at is not None and self.executed_at < self.created_at:
            raise ValueError("executed_at precedes created_at")
        if self.decided_at is not None and self.decided_at < self.created_at:
            raise ValueError("decided_at precedes created_at")
        return self

    @classmethod
    def propose(
        cls,
        *,
        alert_id: str,
        tenant_id: str,
        proposed_by: AgentName,
        action_type: ActionType,
        target: str,
        rationale: str,
        risk_tier: RiskTier,
        created_at: datetime,
        evidence: tuple[Evidence, ...] = (),
        audit_hash_prev: str = GENESIS_HASH,
        action_id: str | None = None,
    ) -> ActionRequest:
        """Construct a proposal with ``requires_human_approval`` derived, not asserted.

        Callers cannot accidentally propose an ungated destructive action, because
        they do not get to set the flag.
        """
        needs_human = action_type.is_destructive and not risk_tier.permits_unattended_execution
        return cls(
            action_id=action_id or new_id(),
            alert_id=alert_id,
            tenant_id=tenant_id,
            proposed_by=proposed_by,
            action_type=action_type,
            target=target,
            rationale=rationale,
            risk_tier=risk_tier,
            requires_human_approval=needs_human,
            evidence=evidence,
            audit_hash_prev=audit_hash_prev,
            created_at=created_at,
        )

    def approve(self, *, approver: str, at: datetime) -> ActionRequest:
        """Record a human approval. Rejects anything but a pending request."""
        if self.approval_status is not ApprovalStatus.PENDING:
            raise ValueError(
                f"cannot approve an action already in state {self.approval_status.value}"
            )
        if not approver.strip():
            raise ValueError("approver must be identified")
        return self.updated(
            approval_status=ApprovalStatus.APPROVED,
            approved_by=approver,
            decided_at=to_utc(at),
        )

    def reject(self, *, approver: str, at: datetime) -> ActionRequest:
        if self.approval_status is not ApprovalStatus.PENDING:
            raise ValueError(
                f"cannot reject an action already in state {self.approval_status.value}"
            )
        return self.updated(
            approval_status=ApprovalStatus.REJECTED,
            approved_by=approver,
            decided_at=to_utc(at),
        )

    def mark_executed(self, *, at: datetime) -> ActionRequest:
        """Record execution. Only an approved (or unattended-tier) action may execute."""
        if self.requires_human_approval and self.approval_status is not ApprovalStatus.APPROVED:
            raise ValueError(
                f"refusing to execute {self.action_type.value}: human approval required but "
                f"status is {self.approval_status.value} (F-08)"
            )
        if self.approval_status in (ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED):
            raise ValueError(f"cannot execute a {self.approval_status.value} action")
        return self.updated(
            approval_status=ApprovalStatus.EXECUTED, executed_at=to_utc(at)
        )

    def mark_failed(self, *, reason: str, at: datetime) -> ActionRequest:
        if not reason.strip():
            raise ValueError("failure reason must be non-empty")
        return self.updated(
            approval_status=ApprovalStatus.FAILED,
            failure_reason=reason,
            executed_at=to_utc(at),
        )
