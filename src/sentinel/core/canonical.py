"""Canonical serialization — byte-exact, stable, and hashable.

Everything in Sentinel that gets hashed, chained or signed goes through
:func:`canonical_bytes` first. The contract is strict:

*Determinism*
    The same logical value produces the same bytes on every run, every platform
    and every CPython 3.12+ build. Object keys are sorted by codepoint, there is
    no insignificant whitespace, and floats use CPython's shortest
    round-tripping ``repr``, which is platform-stable.

*Total rejection of ambiguity*
    ``NaN``, ``Infinity`` and ``-Infinity`` are rejected: JSON has no
    representation for them, so every library invents one and the bytes stop
    being portable. Sets are rejected because iteration order is not part of
    their identity. Naive datetimes are rejected because their instant is
    unknowable. A value that cannot be canonicalized raises
    :class:`~sentinel.core.errors.CanonicalizationError` rather than being
    silently coerced — a silent coercion inside a hash chain is a forged record
    waiting to happen.

*No Unicode folding*
    Strings are encoded UTF-8 with their exact codepoints preserved. It is
    tempting to NFC-normalize, but that would map distinct byte sequences onto
    one hash, handing an attacker a way to mutate stored text without breaking
    the chain. Lone surrogates are rejected instead, since they cannot be
    encoded at all.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum
from typing import Any

from sentinel.core.errors import CanonicalizationError

__all__ = ["canonical_bytes", "canonical_json", "canonicalize", "sha256_hex"]

_MAX_DEPTH = 64


def canonicalize(value: Any, *, _depth: int = 0, _path: str = "$") -> Any:
    """Reduce ``value`` to the JSON-native subset that serializes deterministically.

    Accepts: ``None``, ``bool``, ``int``, finite ``float``, ``str``, ``bytes``
    (base64), ``Decimal`` (decimal string), ``Enum`` (its value), ``uuid.UUID``
    (canonical hyphenated string), aware ``datetime``/``date``/``time``
    (ISO-8601, UTC for datetimes), mappings with string-coercible keys, and
    sequences. Everything else raises.
    """
    if _depth > _MAX_DEPTH:
        raise CanonicalizationError(
            f"at {_path}: nesting deeper than {_MAX_DEPTH} levels; refusing to hash "
            "a structure this deep (likely a cycle or a hostile payload)"
        )

    # bool must precede int: bool is a subclass of int.
    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CanonicalizationError(
                f"at {_path}: non-finite float {value!r} has no portable JSON form. "
                "Sanitize upstream (e.g. CIC-IDS2017 'Flow Bytes/s' can be Infinity)."
            )
        return value

    if isinstance(value, str):
        _reject_lone_surrogates(value, _path)
        return value

    if isinstance(value, Enum):
        return canonicalize(value.value, _depth=_depth + 1, _path=f"{_path}(enum)")

    if isinstance(value, (bytes, bytearray, memoryview)):
        # Base64 with padding: one representation per byte string, both directions.
        return {"__bytes_b64__": base64.b64encode(bytes(value)).decode("ascii")}

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CanonicalizationError(f"at {_path}: non-finite Decimal {value!r}")
        # Normalized decimal string: exact, and never loses precision to binary float.
        return {"__decimal__": format(value.normalize(), "f")}

    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            raise CanonicalizationError(
                f"at {_path}: naive datetime {value!r}; its instant is ambiguous"
            )
        as_utc = value.astimezone(_dt.UTC)
        # Fixed-width microseconds + literal 'Z': one spelling per instant.
        return as_utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{as_utc.microsecond:06d}Z"

    if isinstance(value, _dt.date):  # after datetime: datetime subclasses date
        return value.isoformat()

    if isinstance(value, _dt.time):
        if value.tzinfo is None:
            raise CanonicalizationError(f"at {_path}: naive time {value!r}")
        return value.isoformat()

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, raw_val in value.items():
            key = _canonical_key(raw_key, _path)
            if key in out:
                raise CanonicalizationError(
                    f"at {_path}: duplicate key {key!r} after coercion; "
                    "distinct source keys must not collide"
                )
            out[key] = canonicalize(raw_val, _depth=_depth + 1, _path=f"{_path}.{key}")
        return out

    if isinstance(value, (set, frozenset)):
        raise CanonicalizationError(
            f"at {_path}: sets have no defined order; pass a sorted list instead"
        )

    if isinstance(value, Sequence):
        return [
            canonicalize(item, _depth=_depth + 1, _path=f"{_path}[{i}]")
            for i, item in enumerate(value)
        ]

    # Pydantic models and other objects: ask them for a plain dict, once.
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        return canonicalize(dumper(mode="python"), _depth=_depth + 1, _path=_path)

    raise CanonicalizationError(
        f"at {_path}: {type(value).__name__} has no canonical form. "
        "Convert it explicitly at the call site so the mapping is reviewable."
    )


def _canonical_key(raw_key: object, path: str) -> str:
    if isinstance(raw_key, str):
        _reject_lone_surrogates(raw_key, path)
        return raw_key
    if isinstance(raw_key, bool):
        # "true"/"false" would collide with the literal strings; refuse.
        raise CanonicalizationError(f"at {path}: bool mapping keys are ambiguous")
    if isinstance(raw_key, int):
        return str(raw_key)
    if isinstance(raw_key, Enum) and isinstance(raw_key.value, str):
        return raw_key.value
    raise CanonicalizationError(
        f"at {path}: mapping key of type {type(raw_key).__name__} is not canonical; "
        "use str or int keys"
    )


def _reject_lone_surrogates(text: str, path: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:  # lone surrogate from e.g. surrogateescape decoding
        raise CanonicalizationError(
            f"at {path}: string contains un-encodable codepoints ({exc.reason}); "
            "decode source bytes strictly before hashing"
        ) from exc


def canonical_json(value: Any) -> str:
    """Canonical JSON text for ``value``."""
    reduced = canonicalize(value)
    return json.dumps(
        reduced,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        check_circular=True,
    )


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 bytes for ``value`` — the input to every hash in Sentinel."""
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Lowercase hex SHA-256 of ``data``."""
    import hashlib

    return hashlib.sha256(data).hexdigest()
