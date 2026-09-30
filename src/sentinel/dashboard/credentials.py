"""Passwords, authenticator-app codes and sign-in throttling for the demo identity provider.

The demo identity provider used to list its users as links, which is a picker and not a login.
This is what a login is made of, using only the standard library:

*   **Passwords** are stored as scrypt hashes with a per-account random salt, never as text, and
    compared in constant time. An unknown username is verified against a dummy hash so that
    "no such user" and "wrong password" take the same time and give the same answer.
*   **Second factor** is a time-based one-time password (RFC 6238: HMAC-SHA1, 30-second steps, six
    digits) that any authenticator app produces from the account's secret. A code is accepted for
    the current step and one step either side (clock drift), and **never twice**: a step that has
    been used cannot be used again, so a code read over someone's shoulder is worthless.
*   **Throttling**: five failures in five minutes lock the account for five minutes, whoever is
    typing, so a password cannot be guessed at speed.

Accounts live in a JSON file (hashes and TOTP secrets, git-ignored) so passwords and authenticator
enrolments survive a restart. This is a stand-in for Okta / Azure AD; a real deployment does none of
this itself, which is the point of having an identity provider.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import quote

__all__ = [
    "Account",
    "AccountStore",
    "LoginThrottle",
    "hash_password",
    "new_totp_secret",
    "otpauth_uri",
    "totp_code",
    "verify_password",
    "verify_totp",
]

_SCRYPT_N: Final[int] = 2**14
_SCRYPT_R: Final[int] = 8
_SCRYPT_P: Final[int] = 1
TOTP_STEP: Final[int] = 30
TOTP_DIGITS: Final[int] = 6


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """``scrypt$n$r$p$salt$hash`` (base64). A new random salt unless one is given."""
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32
    )
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")  # noqa: E731
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt)}${b64(digest)}"


_dummy_lock = threading.Lock()
_dummy: list[str] = []


def _dummy_hash() -> str:
    with _dummy_lock:
        if not _dummy:
            _dummy.append(hash_password(secrets.token_urlsafe(16)))
        return _dummy[0]


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time check. ``stored=None`` (no such account) still does the full work."""
    real = stored is not None
    try:
        scheme, n, r, p, salt, expected = (stored if real else _dummy_hash()).split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=32,
        )
        return hmac.compare_digest(digest, base64.b64decode(expected)) and real
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Time-based one-time passwords (RFC 6238)
# --------------------------------------------------------------------------- #


def new_totp_secret() -> str:
    """A 160-bit secret, base32 (what an authenticator app is given)."""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii")


def _hotp(secret: str, counter: int, digits: int = TOTP_DIGITS) -> str:
    key = base64.b32decode(secret.replace(" ", "").upper() + "=" * (-len(secret) % 8))
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    number = (struct.unpack(">I", mac[offset : offset + 4])[0] & 0x7FFFFFFF) % 10**digits
    return str(number).zfill(digits)


def totp_code(secret: str, *, at: float | None = None, digits: int = TOTP_DIGITS) -> str:
    return _hotp(secret, int((time.time() if at is None else at) // TOTP_STEP), digits)


def verify_totp(
    secret: str, code: str, *, at: float | None = None, window: int = 1, last_step: int = -1
) -> int | None:
    """The matched time step, or ``None``. A step at or before ``last_step`` is refused (replay)."""
    code = (code or "").strip().replace(" ", "")
    if len(code) != TOTP_DIGITS or not code.isdigit():
        return None
    now_step = int((time.time() if at is None else at) // TOTP_STEP)
    for step in range(now_step - window, now_step + window + 1):
        if step > last_step and hmac.compare_digest(_hotp(secret, step), code):
            return step
    return None


def otpauth_uri(secret: str, account: str, issuer: str = "Sentinel Mesh (demo IdP)") -> str:
    """The ``otpauth://`` URI an authenticator app's QR scanner reads."""
    label = quote(f"{issuer}:{account}")
    return f"otpauth://totp/{label}?secret={secret}&issuer={quote(issuer)}&digits=6&period=30"


# --------------------------------------------------------------------------- #
# Accounts and throttling
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Account:
    username: str
    display: str
    groups: tuple[str, ...]
    tenant: str
    password_hash: str
    #: ``None`` means the account has no second factor enrolled: it can sign in with a password
    #: alone, and the application then refuses it because it requires MFA.
    totp_secret: str | None = None
    last_totp_step: int = field(default=-1, repr=False)


class AccountStore:
    """Accounts by username, persisted as JSON (hashes and TOTP secrets, never passwords)."""

    def __init__(self, accounts: list[Account] | None = None, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._path = path
        self._accounts: dict[str, Account] = {a.username.lower(): a for a in accounts or []}

    def get(self, username: str) -> Account | None:
        return self._accounts.get((username or "").strip().lower())

    def all(self) -> list[Account]:
        return list(self._accounts.values())

    def add(self, account: Account) -> None:
        with self._lock:
            self._accounts[account.username.lower()] = account
            self.save()

    def use_totp_step(self, account: Account, step: int) -> None:
        with self._lock:
            account.last_totp_step = step
            self.save()

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{**asdict(a), "groups": list(a.groups)} for a in self._accounts.values()]
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"accounts": payload}, indent=1), encoding="utf-8")
        tmp.replace(self._path)

    @classmethod
    def load(cls, path: Path) -> AccountStore:
        data = json.loads(path.read_text(encoding="utf-8"))
        accounts = [
            Account(**{**item, "groups": tuple(item["groups"])})
            for item in data.get("accounts", [])
        ]
        return cls(accounts, path)


class LoginThrottle:
    """Five failures inside five minutes lock the username for five minutes."""

    def __init__(
        self,
        *,
        max_failures: int = 5,
        window: float = 300.0,
        lockout: float = 300.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._max = max_failures
        self._window = window
        self._lockout = lockout
        self._now = now
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def locked_for(self, username: str) -> float:
        """Seconds left on a lock, else 0."""
        key = username.strip().lower()
        with self._lock:
            return max(0.0, self._locked_until.get(key, 0.0) - self._now())

    def failure(self, username: str) -> None:
        key = username.strip().lower()
        with self._lock:
            now = self._now()
            recent = [t for t in self._failures.get(key, []) if now - t < self._window]
            recent.append(now)
            self._failures[key] = recent
            if len(recent) >= self._max:
                self._locked_until[key] = now + self._lockout
                self._failures[key] = []

    def success(self, username: str) -> None:
        key = username.strip().lower()
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)
