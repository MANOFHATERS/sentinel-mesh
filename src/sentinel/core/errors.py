"""Typed error hierarchy.

Every failure mode a caller may reasonably want to branch on gets its own class.
Nothing in Sentinel raises a bare ``Exception`` or ``ValueError`` across a module
boundary: a security platform that cannot distinguish "the data was malformed"
from "the tamper-evidence chain is broken" is not auditable.
"""

from __future__ import annotations


class SentinelError(Exception):
    """Base class for every error Sentinel raises deliberately."""


# --- data / contract errors -------------------------------------------------


class SchemaError(SentinelError):
    """A payload did not satisfy a canonical contract (Appendix B)."""


class NormalizationError(SchemaError):
    """A source record could not be mapped onto the canonical Alert schema."""

    def __init__(self, message: str, *, source: str, row_index: int | None = None) -> None:
        self.source = source
        self.row_index = row_index
        location = f" (row {row_index})" if row_index is not None else ""
        super().__init__(f"[{source}]{location} {message}")


class CanonicalizationError(SchemaError):
    """A value cannot be canonically serialized, so it cannot be hashed or signed.

    Raised for NaN/Infinity, non-finite floats, unsupported types, and any other
    value whose byte representation is not stable across runs and interpreters.
    """


# --- audit / integrity errors -----------------------------------------------


class AuditError(SentinelError):
    """Base class for audit-log failures."""


class ChainIntegrityError(AuditError):
    """The append-only hash chain does not verify — the log has been tampered with."""

    def __init__(self, message: str, *, seq: int | None = None) -> None:
        self.seq = seq
        where = f" at seq={seq}" if seq is not None else ""
        super().__init__(f"audit chain integrity failure{where}: {message}")


class AppendOnlyViolation(AuditError):
    """An attempt was made to mutate or delete an already-committed audit row."""


# --- pipeline / model errors ------------------------------------------------


class BusError(SentinelError):
    """The event bus could not accept or deliver an event."""


class FeatureError(SentinelError):
    """A feature vector could not be produced — the train/serve contract was violated."""


class ModelNotFittedError(SentinelError):
    """A detector or transformer was used for inference before being fitted."""


class GuardrailViolation(SentinelError):
    """An agent or connector attempted an action its guardrails forbid."""
