"""Passwords, one-time codes and throttling: the parts of a login that must not be wrong."""

from __future__ import annotations

import base64

import pytest

from sentinel.dashboard.credentials import (
    Account,
    AccountStore,
    LoginThrottle,
    hash_password,
    new_totp_secret,
    otpauth_uri,
    totp_code,
    verify_password,
    verify_totp,
)

# RFC 6238, Appendix B: the SHA-1 secret is the ASCII string "12345678901234567890".
RFC_SECRET = base64.b32encode(b"12345678901234567890").decode()


@pytest.mark.parametrize(
    "at, eight_digit",
    [
        (59, "94287082"),
        (1111111109, "07081804"),
        (1111111111, "14050471"),
        (1234567890, "89005924"),
        (2000000000, "69279037"),
        (20000000000, "65353130"),
    ],
)
def test_totp_matches_the_rfc_6238_test_vectors(at, eight_digit):
    assert totp_code(RFC_SECRET, at=at, digits=8) == eight_digit
    assert totp_code(RFC_SECRET, at=at) == eight_digit[-6:]  # six digits are the last six


def test_a_code_is_valid_for_its_step_and_one_either_side_and_no_further():
    at = 1_700_000_000.0
    code = totp_code(RFC_SECRET, at=at)
    assert verify_totp(RFC_SECRET, code, at=at) is not None
    assert verify_totp(RFC_SECRET, code, at=at + 30) is not None  # a step late (clock drift)
    assert verify_totp(RFC_SECRET, code, at=at - 30) is not None  # a step early
    assert verify_totp(RFC_SECRET, code, at=at + 95) is None
    assert verify_totp(RFC_SECRET, code, at=at - 95) is None


def test_a_used_step_cannot_be_used_again():
    at = 1_700_000_000.0
    code = totp_code(RFC_SECRET, at=at)
    step = verify_totp(RFC_SECRET, code, at=at)
    assert verify_totp(RFC_SECRET, code, at=at, last_step=step) is None
    # the next step's code still works
    later = totp_code(RFC_SECRET, at=at + 30)
    assert verify_totp(RFC_SECRET, later, at=at + 30, last_step=step) == step + 1


@pytest.mark.parametrize("bad", ["", "12345", "1234567", "abcdef", "12 34 5", None])
def test_malformed_codes_are_refused(bad):
    assert verify_totp(RFC_SECRET, bad or "", at=1_700_000_000.0) is None


def test_spaces_in_a_typed_code_are_forgiven():
    at = 1_700_000_000.0
    code = totp_code(RFC_SECRET, at=at)
    assert verify_totp(RFC_SECRET, f"{code[:3]} {code[3:]}", at=at) is not None


def test_secrets_are_random_base32_of_160_bits():
    a, b = new_totp_secret(), new_totp_secret()
    assert a != b and len(base64.b32decode(a)) == 20 and a.isalnum()


def test_the_otpauth_uri_is_what_an_authenticator_app_reads():
    uri = otpauth_uri("ABCDEF", "maya@acme.example")
    assert uri.startswith(
        "otpauth://totp/Sentinel%20Mesh%20%28demo%20IdP%29%3Amaya%40acme.example?"
    )
    assert "secret=ABCDEF" in uri and "digits=6" in uri and "period=30" in uri


def test_passwords_are_salted_scrypt_and_never_stored_as_text():
    one, two = hash_password("hunter2 is bad"), hash_password("hunter2 is bad")
    assert one != two, "each hash has its own random salt"
    assert one.startswith("scrypt$") and "hunter2" not in one
    assert verify_password("hunter2 is bad", one) and verify_password("hunter2 is bad", two)
    assert not verify_password("hunter2 is bad!", one)
    assert not verify_password("", one)


def test_an_unknown_account_and_a_garbled_hash_both_fail_safely():
    assert verify_password("anything", None) is False
    assert verify_password("anything", "not a hash") is False
    assert verify_password("anything", "md5$1$1$1$AA==$AA==") is False


def test_the_account_store_round_trips_and_looks_names_up_case_insensitively(tmp_path):
    path = tmp_path / "accounts.json"
    store = AccountStore([], path)
    store.add(
        Account("Maya@Acme.example", "Maya", ("SOC-Analyst",), "acme", hash_password("x"), "SECRET")
    )
    loaded = AccountStore.load(path)
    account = loaded.get("  maya@ACME.example ")
    assert (
        account is not None
        and account.groups == ("SOC-Analyst",)
        and account.totp_secret == "SECRET"
    )
    assert loaded.get("nobody@acme.example") is None
    loaded.use_totp_step(account, 42)
    assert AccountStore.load(path).get("maya@acme.example").last_totp_step == 42


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_throttle_locks_after_five_failures_and_a_success_clears_the_count():
    clock = Clock()
    throttle = LoginThrottle(now=clock)
    for _ in range(4):
        throttle.failure("Maya")
    assert throttle.locked_for("maya") == 0
    throttle.success("maya")  # a good sign-in resets the count
    for _ in range(4):
        throttle.failure("maya")
    assert throttle.locked_for("maya") == 0
    throttle.failure("maya")
    assert throttle.locked_for("MAYA") == 300  # case does not dodge the lock
    clock.t += 299
    assert throttle.locked_for("maya") == 1
    clock.t += 2
    assert throttle.locked_for("maya") == 0


def test_old_failures_age_out_and_accounts_are_throttled_separately():
    clock = Clock()
    throttle = LoginThrottle(now=clock)
    for _ in range(4):
        throttle.failure("omar")
    clock.t += 301  # outside the window
    throttle.failure("omar")
    assert throttle.locked_for("omar") == 0
    for _ in range(5):
        throttle.failure("nina")
    assert throttle.locked_for("nina") > 0 and throttle.locked_for("omar") == 0
