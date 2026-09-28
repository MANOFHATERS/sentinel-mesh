"""Layer 2 — schema normalization (PRD F-01).

Maps heterogeneous source records onto the one canonical :class:`~sentinel.core.schemas.Alert`.
Two real datasets are supported in the sprint, and both of them fight back:

**CIC-IDS2017.** Column names carry leading spaces (``" Destination Port"``),
inconsistent capitalisation, and slashes (``"Flow Bytes/s"``). ``Flow Bytes/s``
and ``Flow Packets/s`` genuinely contain ``Infinity`` and ``NaN`` for
zero-duration flows. The ``Label`` column spells web attacks with a Unicode
en-dash (``"Web Attack – Brute Force"``) which, in the widely-mirrored
copies of the CSVs, is frequently mojibaked to ``"Web Attack \x96 Brute Force"``
by a cp1252/latin-1 round trip. ``Flow Duration`` is in **microseconds**.

**UNSW-NB15.** A different ~47-column schema, ``attack_cat`` values padded with
stray spaces and inconsistently pluralised (``"Backdoor"`` vs ``"Backdoors"``),
blank ``attack_cat`` meaning benign, and ``dur`` in **seconds**.

The microseconds-versus-seconds mismatch is the single most consequential detail
in this module. A model trained on CIC durations and validated on UNSW durations
is off by a factor of a million, which does not crash anything — it just makes the
cross-dataset validation the PRD's risk register depends on completely
meaningless. Both normalizers therefore emit duration in **seconds**, and
``test_duration_units_agree_across_datasets`` pins it.

Unified feature space
---------------------
CIC-IDS2017 ships ~78 flow features, UNSW-NB15 ~47, and they are not the same
features. Cross-dataset validation (PRD Section 10, "overfitting to the same
datasets") is only meaningful in a feature space both datasets genuinely express,
so :data:`UNIFIED_FEATURES` defines that intersection: eleven bidirectional flow
statistics plus port and protocol. Rate features are **recomputed here** from the
unified counters rather than copied from the source's own derived columns, which
is both where the ``Infinity`` values come from and where the two datasets'
definitions quietly disagree.

Native columns are not discarded — the verbatim source row is preserved in
``Alert.raw_payload`` (untrusted-wrapped), so a richer per-dataset feature set
remains available to a later phase without a re-ingest.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from sentinel.core.errors import NormalizationError
from sentinel.core.schemas import Alert, AlertSource, Severity

__all__ = [
    "UNIFIED_FEATURES",
    "AttackFamily",
    "CICIDS2017Normalizer",
    "NormalizationReport",
    "Normalizer",
    "UNSWNB15Normalizer",
    "canonicalize_column",
    "canonicalize_row",
    "get_normalizer",
    "severity_for_family",
]


# --------------------------------------------------------------------------- #
# Column canonicalization
# --------------------------------------------------------------------------- #

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def canonicalize_column(name: str) -> str:
    """Normalize a source column name to ``snake_case``.

    ``" Destination Port"`` -> ``destination_port``;
    ``"Flow Bytes/s"`` -> ``flow_bytes_s``;
    ``"Total Length of Fwd Packets"`` -> ``total_length_of_fwd_packets``.

    Idempotent, so applying it to already-canonical names is safe.
    """
    lowered = name.strip().lower()
    return _NON_ALNUM.sub("_", lowered).strip("_")


def canonicalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize every key in ``row``, refusing silent collisions."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        canonical = canonicalize_column(str(key))
        if canonical in out and out[canonical] != value:
            raise NormalizationError(
                f"columns {key!r} and another collapse to {canonical!r} with different values; "
                "the source header is ambiguous",
                source="canonicalize_row",
            )
        out[canonical] = value
    return out


# --------------------------------------------------------------------------- #
# Unified label taxonomy
# --------------------------------------------------------------------------- #


class AttackFamily(str):
    """Canonical attack families shared by both datasets.

    A plain ``str`` subclass rather than an enum: new datasets bring new families,
    and a hard enum would force either a code change or an ``OTHER`` bucket that
    hides the mismatch. The known set is enumerated below for validation.
    """

    __slots__ = ()


BENIGN: Final[str] = "benign"
DOS: Final[str] = "dos"
DDOS: Final[str] = "ddos"
RECON: Final[str] = "recon"
BRUTE_FORCE: Final[str] = "brute_force"
WEB_ATTACK: Final[str] = "web_attack"
BOTNET: Final[str] = "botnet"
EXPLOIT: Final[str] = "exploit"
BACKDOOR: Final[str] = "backdoor"
WORM: Final[str] = "worm"
INFILTRATION: Final[str] = "infiltration"
SHELLCODE: Final[str] = "shellcode"
FUZZERS: Final[str] = "fuzzers"
ANALYSIS: Final[str] = "analysis"
GENERIC: Final[str] = "generic"
UNKNOWN: Final[str] = "unknown"

KNOWN_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        BENIGN,
        DOS,
        DDOS,
        RECON,
        BRUTE_FORCE,
        WEB_ATTACK,
        BOTNET,
        EXPLOIT,
        BACKDOOR,
        WORM,
        INFILTRATION,
        SHELLCODE,
        FUZZERS,
        ANALYSIS,
        GENERIC,
        UNKNOWN,
    }
)

#: Severity prior per family. Not a model output — a defensible starting point the
#: Triage Agent may override with evidence, and the label the evaluation compares
#: against. Ransomware-adjacent and post-compromise families rank highest because
#: dwell time matters more than volume.
_FAMILY_SEVERITY: Final[dict[str, Severity]] = {
    BENIGN: Severity.INFO,
    ANALYSIS: Severity.LOW,
    FUZZERS: Severity.LOW,
    RECON: Severity.MEDIUM,
    GENERIC: Severity.MEDIUM,
    DOS: Severity.MEDIUM,
    WEB_ATTACK: Severity.HIGH,
    BRUTE_FORCE: Severity.HIGH,
    DDOS: Severity.HIGH,
    EXPLOIT: Severity.HIGH,
    SHELLCODE: Severity.HIGH,
    WORM: Severity.HIGH,
    BOTNET: Severity.CRITICAL,
    BACKDOOR: Severity.CRITICAL,
    INFILTRATION: Severity.CRITICAL,
    UNKNOWN: Severity.MEDIUM,
}


def severity_for_family(family: str) -> Severity:
    """Severity prior for an attack family; unknown families are not silently low."""
    return _FAMILY_SEVERITY.get(family, Severity.MEDIUM)


# Dash-like characters, including the cp1252 mojibake byte 0x96 that appears in
# mirrored copies of the CIC-IDS2017 CSVs.
_DASHES = re.compile("[‐-―−\x96\x97-]+")


def _normalize_label_text(raw: object) -> str:
    """Fold a raw dataset label to a comparable key."""
    text = "" if raw is None else str(raw)
    text = _DASHES.sub("-", text)
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


_CIC_LABEL_FAMILY: Final[dict[str, str]] = {
    "benign": BENIGN,
    "normal": BENIGN,
    "ddos": DDOS,
    "dos hulk": DOS,
    "dos goldeneye": DOS,
    "dos slowloris": DOS,
    "dos slowhttptest": DOS,
    "portscan": RECON,
    "port scan": RECON,
    "ftp-patator": BRUTE_FORCE,
    "ssh-patator": BRUTE_FORCE,
    "web attack-brute force": WEB_ATTACK,
    "web attack - brute force": WEB_ATTACK,
    "web attack-xss": WEB_ATTACK,
    "web attack - xss": WEB_ATTACK,
    "web attack-sql injection": WEB_ATTACK,
    "web attack - sql injection": WEB_ATTACK,
    "bot": BOTNET,
    "botnet": BOTNET,
    "infiltration": INFILTRATION,
    "heartbleed": EXPLOIT,
}

_UNSW_LABEL_FAMILY: Final[dict[str, str]] = {
    "": BENIGN,
    "normal": BENIGN,
    "dos": DOS,
    "reconnaissance": RECON,
    "exploits": EXPLOIT,
    "exploit": EXPLOIT,
    "fuzzers": FUZZERS,
    "generic": GENERIC,
    "backdoor": BACKDOOR,
    "backdoors": BACKDOOR,
    "analysis": ANALYSIS,
    "shellcode": SHELLCODE,
    "worms": WORM,
    "worm": WORM,
}


# --------------------------------------------------------------------------- #
# Unified feature space
# --------------------------------------------------------------------------- #

UNIFIED_FEATURES: Final[tuple[str, ...]] = (
    "duration_seconds",
    "src_bytes",
    "dst_bytes",
    "src_packets",
    "dst_packets",
    "total_bytes",
    "total_packets",
    "bytes_per_second",
    "packets_per_second",
    "src_mean_packet_bytes",
    "dst_mean_packet_bytes",
    "bytes_ratio_src_to_total",
    "dst_port",
    "protocol_number",
)
"""The feature space both datasets genuinely express. Order is part of the contract."""

#: Seconds added to a zero-length flow's duration before dividing. A zero-duration
#: flow with bytes is physically real (one packet, one timestamp resolution), and
#: 1 microsecond is the resolution CIC-IDS2017 reports in — so this floor is the
#: measurement limit, not an arbitrary epsilon. It is what turns the dataset's
#: literal ``Infinity`` values into a large-but-finite rate.
RATE_DENOMINATOR_FLOOR: Final[float] = 1e-6

_PROTOCOL_NUMBERS: Final[dict[str, int]] = {
    "hopopt": 0,
    "icmp": 1,
    "igmp": 2,
    "ggp": 3,
    "ipv4": 4,
    "tcp": 6,
    "egp": 8,
    "igp": 9,
    "udp": 17,
    "rdp": 27,
    "ipv6": 41,
    "rsvp": 46,
    "gre": 47,
    "esp": 50,
    "ah": 51,
    "ospf": 89,
    "sctp": 132,
    "arp": 0,
    "unas": 255,
    "any": 255,
}


def _protocol_number(value: object) -> int | None:
    """Map a protocol column to its IANA number; both datasets spell it differently."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        number = int(value)
        return number if 0 <= number <= 255 else None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        number = int(text)
        return number if 0 <= number <= 255 else None
    return _PROTOCOL_NUMBERS.get(text)


def _protocol_name(value: object) -> str | None:
    number = _protocol_number(value)
    if number is None:
        text = str(value).strip().lower() if value is not None else ""
        return text or None
    for name, candidate in _PROTOCOL_NUMBERS.items():
        if candidate == number and name not in {"arp", "any"}:
            return name
    return str(number)


def _to_float(value: object, *, field_name: str, source: str, row_index: int | None) -> float:
    """Parse a numeric cell, mapping the datasets' Infinity/NaN spellings to 0.0.

    Returning 0.0 rather than propagating a non-finite value is deliberate and
    narrow: it only ever applies to *rate* columns the datasets compute by
    dividing by a zero duration, and this module recomputes every rate it uses
    from the raw counters anyway. Nothing that feeds a model keeps a substituted
    value silently — :class:`NormalizationReport` counts every substitution.
    """
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        lowered = text.lower()
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        if lowered in {"-inf", "-infinity"}:
            return -math.inf
        if lowered in {"nan", "na", "n/a", "null", "-", "?"}:
            return math.nan
        try:
            number = float(text)
        except ValueError as exc:
            raise NormalizationError(
                f"field {field_name!r} has non-numeric value {value!r}",
                source=source,
                row_index=row_index,
            ) from exc
    return number


def _finite(value: float) -> tuple[float, bool]:
    """Return ``(sanitized, was_substituted)``."""
    if math.isfinite(value):
        return value, False
    return 0.0, True


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class NormalizationReport:
    """Observability for a normalization run. F-01 demands zero schema failures."""

    source: str
    rows_seen: int = 0
    alerts_emitted: int = 0
    rows_rejected: int = 0
    non_finite_substitutions: int = 0
    unknown_labels: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        return 0.0 if self.rows_seen == 0 else self.rows_rejected / self.rows_seen

    def note_unknown_label(self, label: str) -> None:
        self.unknown_labels[label] = self.unknown_labels.get(label, 0) + 1

    def note_rejection(self, reason: str) -> None:
        self.rows_rejected += 1
        key = reason[:120]
        self.rejection_reasons[key] = self.rejection_reasons.get(key, 0) + 1

    def summary(self) -> str:
        parts = [
            f"{self.source}: {self.alerts_emitted}/{self.rows_seen} rows normalized",
            f"{self.rows_rejected} rejected ({self.failure_rate:.4%})",
            f"{self.non_finite_substitutions} non-finite values sanitized",
        ]
        if self.unknown_labels:
            top = sorted(self.unknown_labels.items(), key=lambda kv: -kv[1])[:3]
            parts.append("unmapped labels: " + ", ".join(f"{k!r}x{v}" for k, v in top))
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #


class Normalizer(ABC):
    """Base class: one source schema in, canonical :class:`Alert` out."""

    #: Dataset name recorded on every emitted alert.
    dataset: str
    #: Which layer-1 source this stands in for.
    alert_source: AlertSource = AlertSource.NETWORK_IDS

    def __init__(self, *, tenant_id: str = "demo", strict: bool = False) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required; the platform is multi-tenant by design")
        self.tenant_id = tenant_id
        self.strict = strict
        self.report = NormalizationReport(source=self.dataset)

    # --- required per-dataset hooks ----------------------------------------

    @abstractmethod
    def _extract(self, row: Mapping[str, Any], row_index: int) -> tuple[dict[str, Any], str]:
        """Return ``(unified_features_and_meta, raw_label_text)`` for one row."""

    @abstractmethod
    def _label_family(self, raw_label: str) -> str | None:
        """Map a raw dataset label to a unified family, or ``None`` if unmapped."""

    # --- shared pipeline ----------------------------------------------------

    def normalize(
        self,
        row: Mapping[str, Any],
        *,
        row_index: int,
        observed_at: datetime | None = None,
        ingested_at: datetime | None = None,
    ) -> Alert | None:
        """Normalize one source row.

        Returns ``None`` when the row is rejected in lenient mode; raises
        :class:`NormalizationError` in strict mode. F-01's acceptance criterion is
        zero schema-validation failures, so a production run sets ``strict=True``
        and any rejection fails the ingest loudly instead of thinning the dataset
        invisibly.
        """
        self.report.rows_seen += 1
        try:
            canonical = canonicalize_row(row)
            extracted, raw_label = self._extract(canonical, row_index)

            family = self._label_family(raw_label)
            if family is None:
                self.report.note_unknown_label(raw_label)
                family = UNKNOWN

            features, substitutions = self._finalize_features(extracted)
            self.report.non_finite_substitutions += substitutions

            event_time = observed_at or extracted.get("_timestamp") or _EPOCH_FALLBACK
            if not isinstance(event_time, datetime):
                event_time = _EPOCH_FALLBACK
            seen_time = ingested_at or event_time
            if seen_time < event_time:
                seen_time = event_time

            alert = Alert(
                alert_id=Alert.derive_id(
                    dataset=self.dataset,
                    source=self.alert_source.value,
                    row_index=row_index,
                    tenant_id=self.tenant_id,
                ),
                tenant_id=self.tenant_id,
                source=self.alert_source,
                timestamp=event_time,
                ingested_at=seen_time,
                asset_id=str(extracted.get("_asset_id") or "unknown-asset"),
                signature=f"{self.dataset}:{family}",
                raw_payload=json.dumps(
                    {str(k): _jsonable(v) for k, v in row.items()},
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ),
                features=features,
                src_ip=_optional_str(extracted.get("_src_ip")),
                dst_ip=_optional_str(extracted.get("_dst_ip")),
                src_port=_optional_port(extracted.get("_src_port")),
                dst_port=_optional_port(extracted.get("dst_port")),
                protocol=_optional_str(extracted.get("_protocol_name")),
                dataset=self.dataset,
                ground_truth_label=family,
            )
        except NormalizationError as exc:
            self.report.note_rejection(str(exc))
            if self.strict:
                raise
            return None
        except (ValueError, TypeError) as exc:
            self.report.note_rejection(f"{type(exc).__name__}: {exc}")
            if self.strict:
                raise NormalizationError(
                    str(exc), source=self.dataset, row_index=row_index
                ) from exc
            return None

        self.report.alerts_emitted += 1
        return alert

    def _finalize_features(
        self, extracted: Mapping[str, Any]
    ) -> tuple[dict[str, float | int | None], int]:
        """Derive rates, sanitize, and emit exactly :data:`UNIFIED_FEATURES`."""
        substitutions = 0

        duration, sub = _finite(float(extracted.get("duration_seconds") or 0.0))
        substitutions += sub
        duration = max(duration, 0.0)

        src_bytes, sub = _finite(float(extracted.get("src_bytes") or 0.0))
        substitutions += sub
        dst_bytes, sub = _finite(float(extracted.get("dst_bytes") or 0.0))
        substitutions += sub
        src_packets, sub = _finite(float(extracted.get("src_packets") or 0.0))
        substitutions += sub
        dst_packets, sub = _finite(float(extracted.get("dst_packets") or 0.0))
        substitutions += sub

        total_bytes = src_bytes + dst_bytes
        total_packets = src_packets + dst_packets
        denominator = max(duration, RATE_DENOMINATOR_FLOOR)

        features: dict[str, float | int | None] = {
            "duration_seconds": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "src_packets": src_packets,
            "dst_packets": dst_packets,
            "total_bytes": total_bytes,
            "total_packets": total_packets,
            "bytes_per_second": total_bytes / denominator,
            "packets_per_second": total_packets / denominator,
            "src_mean_packet_bytes": src_bytes / src_packets if src_packets > 0 else 0.0,
            "dst_mean_packet_bytes": dst_bytes / dst_packets if dst_packets > 0 else 0.0,
            "bytes_ratio_src_to_total": src_bytes / total_bytes if total_bytes > 0 else 0.0,
            "dst_port": _optional_port(extracted.get("dst_port")),
            "protocol_number": extracted.get("protocol_number"),
        }

        # Guarantee the contract: exactly the declared keys, every value finite.
        missing = set(UNIFIED_FEATURES) - set(features)
        if missing:
            raise NormalizationError(
                f"unified feature contract violated; missing {sorted(missing)}",
                source=self.dataset,
            )
        for name, value in features.items():
            if isinstance(value, float) and not math.isfinite(value):
                features[name] = 0.0
                substitutions += 1
        return features, substitutions


_EPOCH_FALLBACK: Final[datetime] = datetime(2017, 1, 1, tzinfo=UTC)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_port(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        port = int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None
    return port if 0 <= port <= 65535 else None


class CICIDS2017Normalizer(Normalizer):
    """CIC-IDS2017 flow records (Canadian Institute for Cybersecurity)."""

    dataset = "cic-ids2017"
    alert_source = AlertSource.NETWORK_IDS

    #: Duration arrives in microseconds. This is the conversion the cross-dataset
    #: validation in PRD Section 10 lives or dies on.
    DURATION_SCALE_TO_SECONDS: Final[float] = 1e-6

    _LABEL_COLUMNS: Final[tuple[str, ...]] = ("label", "labels", "attack", "class")

    def _extract(self, row: Mapping[str, Any], row_index: int) -> tuple[dict[str, Any], str]:
        def number(*names: str) -> float:
            for name in names:
                if name in row:
                    return _to_float(
                        row[name], field_name=name, source=self.dataset, row_index=row_index
                    )
            return 0.0

        present = set(row)
        if not ({"flow_duration", "total_fwd_packets", "destination_port"} & present):
            raise NormalizationError(
                "row does not look like CIC-IDS2017 (no flow_duration / total_fwd_packets / "
                f"destination_port among {sorted(present)[:8]}…)",
                source=self.dataset,
                row_index=row_index,
            )

        raw_label = ""
        for candidate in self._LABEL_COLUMNS:
            if candidate in row:
                raw_label = _normalize_label_text(row[candidate])
                break

        duration_us = number("flow_duration")
        duration_seconds = (
            duration_us * self.DURATION_SCALE_TO_SECONDS if math.isfinite(duration_us) else 0.0
        )

        protocol_raw = row.get("protocol")
        return (
            {
                "duration_seconds": duration_seconds,
                "src_bytes": number("total_length_of_fwd_packets", "fwd_packets_length_total"),
                "dst_bytes": number("total_length_of_bwd_packets", "bwd_packets_length_total"),
                "src_packets": number("total_fwd_packets", "fwd_packets"),
                "dst_packets": number("total_backward_packets", "bwd_packets"),
                "dst_port": row.get("destination_port"),
                "protocol_number": _protocol_number(protocol_raw),
                "_protocol_name": _protocol_name(protocol_raw),
                "_src_ip": row.get("source_ip"),
                "_dst_ip": row.get("destination_ip"),
                "_src_port": row.get("source_port"),
                "_asset_id": row.get("destination_ip") or row.get("flow_id") or "cic-flow",
                "_timestamp": _parse_cic_timestamp(row.get("timestamp")),
            },
            raw_label,
        )

    def _label_family(self, raw_label: str) -> str | None:
        if raw_label == "":
            return None
        direct = _CIC_LABEL_FAMILY.get(raw_label)
        if direct is not None:
            return direct
        # "web attack-brute force" variants after dash folding may still carry
        # spacing differences; compare with all spaces removed as a last resort.
        squashed = raw_label.replace(" ", "")
        for key, family in _CIC_LABEL_FAMILY.items():
            if key.replace(" ", "") == squashed:
                return family
        return None


def _parse_cic_timestamp(value: Any) -> datetime | None:
    """Parse CIC-IDS2017's ``Timestamp`` column.

    The format is ``D/M/YYYY H:MM(:SS)`` with no timezone and no zero padding, and
    it is genuinely ambiguous for days 1-12: ``5/7/2017`` is 5 July in the capture's
    own notation. The capture ran 3-7 July 2017, so day-first is the correct
    reading, and anything that fails to parse returns ``None`` so the replay
    service assigns a synthetic time rather than inventing a wrong one.
    """
    if value is None or value == "":
        return None
    text = str(value).strip()
    for pattern in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


class UNSWNB15Normalizer(Normalizer):
    """UNSW-NB15 flow records (ACCS). Handles both the full and the train/test CSVs."""

    dataset = "unsw-nb15"
    alert_source = AlertSource.NETWORK_IDS

    #: ``dur`` is already in seconds. Stated explicitly so the asymmetry with
    #: CIC-IDS2017 is visible at the point it matters.
    DURATION_SCALE_TO_SECONDS: Final[float] = 1.0

    def _extract(self, row: Mapping[str, Any], row_index: int) -> tuple[dict[str, Any], str]:
        def number(*names: str) -> float:
            for name in names:
                if name in row:
                    return _to_float(
                        row[name], field_name=name, source=self.dataset, row_index=row_index
                    )
            return 0.0

        present = set(row)
        if not ({"sbytes", "dbytes", "dur"} & present):
            raise NormalizationError(
                "row does not look like UNSW-NB15 (no sbytes / dbytes / dur among "
                f"{sorted(present)[:8]}…)",
                source=self.dataset,
                row_index=row_index,
            )

        raw_label = _normalize_label_text(row.get("attack_cat", ""))
        if raw_label in {"", "-"}:
            # The train/test CSVs use a binary `label` column with attack_cat blank
            # for normal traffic; a blank attack_cat with label=1 is a real record
            # of an attack whose family was not annotated, not a benign flow.
            binary = row.get("label")
            if binary is not None and _to_float(
                binary, field_name="label", source=self.dataset, row_index=row_index
            ) >= 0.5:
                raw_label = "unknown-attack"
            else:
                raw_label = ""

        protocol_raw = row.get("proto")
        return (
            {
                "duration_seconds": number("dur") * self.DURATION_SCALE_TO_SECONDS,
                "src_bytes": number("sbytes"),
                "dst_bytes": number("dbytes"),
                "src_packets": number("spkts", "s_pkts"),
                "dst_packets": number("dpkts", "d_pkts"),
                "dst_port": row.get("dsport", row.get("dst_port")),
                "protocol_number": _protocol_number(protocol_raw),
                "_protocol_name": _protocol_name(protocol_raw),
                "_src_ip": row.get("srcip"),
                "_dst_ip": row.get("dstip"),
                "_src_port": row.get("sport"),
                "_asset_id": row.get("dstip") or f"unsw-flow-{row.get('id', row_index)}",
                "_timestamp": _parse_unsw_timestamp(row.get("stime")),
            },
            raw_label,
        )

    def _label_family(self, raw_label: str) -> str | None:
        if raw_label == "unknown-attack":
            return UNKNOWN
        return _UNSW_LABEL_FAMILY.get(raw_label)


def _parse_unsw_timestamp(value: Any) -> datetime | None:
    """``Stime`` is a Unix epoch second count in the full UNSW-NB15 CSVs."""
    if value is None or value == "":
        return None
    try:
        epoch = float(str(value).strip())
    except ValueError:
        return None
    # Guard against the train/test CSVs, which have no Stime and where a stray
    # numeric column could be mistaken for one.
    if not (946_684_800 <= epoch <= 4_102_444_800):  # 2000-01-01 .. 2100-01-01
        return None
    return datetime.fromtimestamp(epoch, tz=UTC)


_REGISTRY: Final[dict[str, type[Normalizer]]] = {
    CICIDS2017Normalizer.dataset: CICIDS2017Normalizer,
    UNSWNB15Normalizer.dataset: UNSWNB15Normalizer,
    "cicids2017": CICIDS2017Normalizer,
    "unswnb15": UNSWNB15Normalizer,
}


def get_normalizer(dataset: str, **kwargs: Any) -> Normalizer:
    """Look up a normalizer by dataset name."""
    key = dataset.strip().lower()
    cls = _REGISTRY.get(key) or _REGISTRY.get(key.replace("_", "-"))
    if cls is None:
        raise NormalizationError(
            f"no normalizer registered for {dataset!r}; known: {sorted(set(_REGISTRY))}",
            source="registry",
        )
    return cls(**kwargs)
