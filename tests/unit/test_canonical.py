"""Canonical serialization: determinism, rejection of ambiguity, hash stability.

These tests matter more than their size suggests. Every tamper-evidence guarantee
in the platform reduces to "the same logical value always produces the same bytes",
so if this module drifts, the audit chain silently stops proving anything.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from enum import Enum, StrEnum
from uuid import UUID

import pytest

from sentinel.core.canonical import canonical_bytes, canonical_json, canonicalize, sha256_hex
from sentinel.core.errors import CanonicalizationError


class Colour(StrEnum):
    RED = "red"


class Number(Enum):
    ONE = 1


class TestDeterminism:
    def test_key_order_does_not_affect_bytes(self):
        assert canonical_bytes({"a": 1, "b": 2}) == canonical_bytes({"b": 2, "a": 1})

    def test_nested_key_order_does_not_affect_bytes(self):
        left = {"outer": {"z": [1, {"b": 2, "a": 1}], "a": 0}}
        right = {"outer": {"a": 0, "z": [1, {"a": 1, "b": 2}]}}
        assert canonical_bytes(left) == canonical_bytes(right)

    def test_list_order_does_affect_bytes(self):
        # Sequences are ordered data, not sets. Folding them would be a bug.
        assert canonical_bytes([1, 2]) != canonical_bytes([2, 1])

    def test_no_insignificant_whitespace(self):
        assert canonical_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'

    def test_repeated_calls_are_byte_identical(self):
        value = {"m": [1.5, "x", True, None], "n": {"deep": {"deeper": 3}}}
        assert len({canonical_bytes(value) for _ in range(50)}) == 1

    def test_int_and_float_of_equal_value_differ(self):
        # 1 and 1.0 are different types carrying different information; conflating
        # them would let a schema change slip past the hash unnoticed.
        assert canonical_bytes({"x": 1}) != canonical_bytes({"x": 1.0})


class TestRejection:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_floats_rejected(self, bad):
        with pytest.raises(CanonicalizationError, match="non-finite"):
            canonical_bytes({"rate": bad})

    def test_error_message_names_the_path(self):
        with pytest.raises(CanonicalizationError, match=r"\$\.outer\.inner"):
            canonical_bytes({"outer": {"inner": math.inf}})

    def test_error_message_names_the_list_index(self):
        with pytest.raises(CanonicalizationError, match=r"\$\.items\[2\]"):
            canonical_bytes({"items": [1, 2, math.nan]})

    def test_sets_rejected(self):
        with pytest.raises(CanonicalizationError, match="no defined order"):
            canonical_bytes({"tags": {"a", "b"}})

    def test_naive_datetime_rejected(self):
        with pytest.raises(CanonicalizationError, match="naive datetime"):
            canonical_bytes({"at": dt.datetime(2026, 1, 1)})

    def test_unknown_type_rejected(self):
        class Opaque:
            pass

        with pytest.raises(CanonicalizationError, match="no canonical form"):
            canonical_bytes({"thing": Opaque()})

    def test_bool_mapping_key_rejected(self):
        with pytest.raises(CanonicalizationError, match="bool mapping keys"):
            canonical_bytes({True: 1})

    def test_colliding_coerced_keys_rejected(self):
        # {1: "a", "1": "b"} both coerce to "1"; silently keeping one would drop data
        # from the hashed material.
        with pytest.raises(CanonicalizationError, match="duplicate key"):
            canonical_bytes({1: "a", "1": "b"})

    def test_deep_nesting_rejected(self):
        value: object = 1
        for _ in range(80):
            value = {"n": value}
        with pytest.raises(CanonicalizationError, match="nesting deeper"):
            canonical_bytes(value)

    def test_cycle_is_caught_by_depth_limit(self):
        node: dict = {}
        node["self"] = node
        with pytest.raises(CanonicalizationError):
            canonical_bytes(node)

    def test_lone_surrogate_rejected(self):
        with pytest.raises(CanonicalizationError, match="un-encodable"):
            canonical_bytes({"text": "ok\ud800bad"})


class TestTypeHandling:
    def test_datetime_normalized_to_utc_z(self):
        tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
        local = dt.datetime(2026, 1, 1, 5, 30, tzinfo=tz)
        assert canonicalize(local) == "2026-01-01T00:00:00.000000Z"

    def test_equal_instants_in_different_zones_hash_identically(self):
        a = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC)
        b = dt.datetime(2026, 1, 1, 5, 30, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
        assert canonical_bytes(a) == canonical_bytes(b)

    def test_microseconds_are_fixed_width(self):
        value = dt.datetime(2026, 1, 1, 0, 0, 0, 5, tzinfo=dt.UTC)
        assert canonicalize(value) == "2026-01-01T00:00:00.000005Z"

    def test_bytes_are_base64_tagged(self):
        assert canonicalize(b"hi") == {"__bytes_b64__": "aGk="}

    def test_decimal_keeps_exact_value(self):
        assert canonicalize(Decimal("0.10")) == {"__decimal__": "0.1"}

    def test_uuid_is_hyphenated_string(self):
        value = UUID("12345678-1234-5678-1234-567812345678")
        assert canonicalize(value) == "12345678-1234-5678-1234-567812345678"

    def test_str_enum_uses_value(self):
        assert canonicalize(Colour.RED) == "red"

    def test_int_enum_uses_value(self):
        assert canonicalize(Number.ONE) == 1

    def test_bool_is_not_treated_as_int(self):
        assert canonical_json({"x": True}) == '{"x":true}'

    def test_pydantic_model_is_dumped(self):
        from pydantic import BaseModel

        class Thing(BaseModel):
            b: int
            a: str

        assert canonical_json(Thing(b=2, a="x")) == '{"a":"x","b":2}'

    def test_unicode_is_not_escaped_but_is_utf8(self):
        assert canonical_bytes({"t": "café"}) == '{"t":"café"}'.encode()

    def test_unicode_codepoints_are_not_folded(self):
        # NFC-normalizing here would map two distinct byte sequences to one hash,
        # letting an attacker alter stored text without breaking the chain.
        composed = "é"  # é
        decomposed = "é"  # e + combining acute
        assert canonical_bytes(composed) != canonical_bytes(decomposed)


class TestHashing:
    def test_sha256_hex_is_stable_and_lowercase(self):
        digest = sha256_hex(b"")
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert digest == digest.lower()

    def test_hash_changes_with_any_field_change(self):
        base = {"severity": "high", "confidence": 0.9}
        changed = {"severity": "high", "confidence": 0.90001}
        assert sha256_hex(canonical_bytes(base)) != sha256_hex(canonical_bytes(changed))
