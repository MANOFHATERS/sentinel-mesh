"""A stand-in OpenID Connect provider for local demos and tests.

This is **not** part of the product. It plays the role Okta / Azure AD / Google
Workspace plays in a deployment, so the dashboard's SSO can be exercised end to end on a
laptop with no external account. It speaks the real protocol — discovery, JWKS,
authorization-code with PKCE, RS256-signed ID tokens carrying ``groups`` and ``amr`` —
so the relying party in :mod:`sentinel.dashboard.sso` is the same code that would face
a real provider.

The sign-in is a real one: a username, a password (stored as an scrypt hash) and, for accounts
with a second factor, a six-digit code from an authenticator app, with per-account throttling
and lockout (see :mod:`sentinel.dashboard.credentials`). There is no sign-up and no list of
users: accounts exist only in the provider's store. Two accounts exist to show refusals — one
with no second factor enrolled and one in no mapped group.

Started only by ``python -m sentinel.dashboard --dev-idp``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, urlencode

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from sentinel.dashboard.credentials import (
    Account,
    AccountStore,
    LoginThrottle,
    hash_password,
    new_totp_secret,
    verify_password,
    verify_totp,
)

__all__ = [
    "DEMO_USERS",
    "REAL_USERS",
    "DemoUser",
    "make_dev_idp",
    "provision_accounts",
]


@dataclass(frozen=True, slots=True)
class DemoUser:
    """A template for a demo account. Passwords are generated when the account is created."""

    email: str
    label: str
    groups: tuple[str, ...]
    mfa: bool = True
    tenant: str = "acme"


#: People whose workspace runs on real public data (tenant ``real``): a real network capture,
#: MITRE's real ATT&CK catalogue and real repositories. Offered only when those files exist.
REAL_USERS: Final[tuple[DemoUser, ...]] = (
    DemoUser(
        "real.analyst@acme.example",
        "Real-data SOC analyst (group SOC-Analyst, MFA)",
        ("SOC-Analyst",),
        tenant="real",
    ),
    DemoUser(
        "real.auditor@acme.example",
        "Real-data auditor, read-only (group Auditor, MFA)",
        ("Auditor",),
        tenant="real",
    ),
)

DEMO_USERS: Final[tuple[DemoUser, ...]] = (
    DemoUser(
        "maya.analyst@acme.example", "Maya, SOC analyst (group SOC-Analyst, MFA)", ("SOC-Analyst",)
    ),
    DemoUser("omar.auditor@acme.example", "Omar, auditor (group Auditor, MFA)", ("Auditor",)),
    DemoUser(
        "nina.nomfa@acme.example",
        "Nina, SOC analyst with no second factor enrolled (refused by MFA policy)",
        ("SOC-Analyst",),
        mfa=False,
    ),
    DemoUser(
        "carl.contractor@acme.example",
        "Carl, contractor in no mapped group (refused: no role)",
        ("Contractors",),
    ),
)


def provision_accounts(
    path: Path, templates: tuple[DemoUser, ...]
) -> tuple[AccountStore, dict[str, tuple[str, str | None]]]:
    """Load the account store, creating any template account that is missing.

    Returns the store and ``{username: (password, totp_secret)}`` for the accounts created *now*.
    A password exists in plain text only in that return value (it is printed once by the server);
    the file holds hashes and TOTP secrets, so a restart keeps every password and every
    authenticator enrolment.
    """
    store = AccountStore.load(path) if path.is_file() else AccountStore([], path)
    created: dict[str, tuple[str, str | None]] = {}
    for template in templates:
        if store.get(template.email) is not None:
            continue
        password = secrets.token_urlsafe(12)
        secret = new_totp_secret() if template.mfa else None
        store.add(
            Account(
                username=template.email,
                display=template.label,
                groups=template.groups,
                tenant=template.tenant,
                password_hash=hash_password(password),
                totp_secret=secret,
            )
        )
        created[template.email] = (password, secret)
    return store, created


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


_CSP: Final[str] = (
    "default-src 'self'; style-src 'self'; img-src 'self' data:; base-uri 'none'; "
    "object-src 'none'; frame-ancestors 'none'; form-action 'self'"
)

#: Largest values the sign-in form accepts. Generated demo passwords are 16 characters, so 16 is
#: the smallest limit that leaves them working.
MAX_USERNAME: Final[int] = 30
MAX_PASSWORD: Final[int] = 16
MAX_CODE: Final[int] = 7
MAX_FORM_BYTES: Final[int] = 4096


def make_dev_idp(
    *,
    issuer: str,
    client_id: str,
    redirect_uri: str,
    accounts: AccountStore,
    client_secret: str | None = None,
    throttle: LoginThrottle | None = None,
    now: Any = time.time,
) -> FastAPI:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = key.public_key().public_numbers()
    kid = secrets.token_hex(4)
    jwk = {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": _b64url_uint(numbers.n),
        "e": _b64url_uint(numbers.e),
    }
    throttle = throttle or LoginThrottle(now=now)
    requests: dict[str, dict[str, str]] = {}
    codes: dict[str, dict[str, Any]] = {}
    base = issuer.rstrip("/")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def page(rid: str, *, error: str = "", username: str = "", status: int = 200) -> Response:
        message = f'<p class="alert tone-bad" role="alert">{escape(error)}</p>' if error else ""
        html = (
            '<!doctype html><meta charset="utf-8"><title>Sign in · demo identity provider</title>'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<link rel="stylesheet" href="/static/app.css">'
            '<div class="login-wrap"><form class="login" method="post" '
            f'action="{escape(base)}/login" autocomplete="off">'
            "<h1>Sign in</h1>"
            '<p class="sub">Demo identity provider, standing in for Okta or Azure AD. '
            "There is no sign-up here: accounts are created by an administrator.</p>"
            f'{message}<input type="hidden" name="rid" value="{escape(rid)}">'
            '<label for="username">Username</label>'
            '<input class="wide-input" id="username" name="username" type="email" required '
            f'maxlength="{MAX_USERNAME}" '
            f'autocomplete="username" value="{escape(username)}" autofocus>'
            '<label for="password">Password</label>'
            '<input class="wide-input" id="password" name="password" type="password" required '
            f'maxlength="{MAX_PASSWORD}" '
            'autocomplete="current-password">'
            '<label for="code">Authenticator code</label>'
            '<input class="wide-input" id="code" name="code" inputmode="numeric" '
            'pattern="[0-9 ]*" maxlength="7" placeholder="6 digits (if you have enrolled one)" '
            'autocomplete="one-time-code">'
            '<button class="btn primary" type="submit">Sign in</button></form></div>'
        )
        return HTMLResponse(
            html,
            status_code=status,
            headers={"Content-Security-Policy": _CSP, "Cache-Control": "no-store"},
        )

    @app.get("/.well-known/openid-configuration")
    def discovery() -> dict[str, Any]:
        return {
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "jwks_uri": f"{base}/jwks",
            "response_types_supported": ["code"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "code_challenge_methods_supported": ["S256"],
            "subject_types_supported": ["public"],
        }

    @app.get("/jwks")
    def jwks() -> dict[str, Any]:
        return {"keys": [jwk]}

    @app.get("/authorize")
    def authorize(request: Request) -> Response:
        q = request.query_params
        if (
            q.get("client_id") != client_id
            or q.get("redirect_uri") != redirect_uri
            or q.get("response_type") != "code"
            or q.get("code_challenge_method") != "S256"
            or not q.get("code_challenge")
            or not q.get("state")
        ):
            return HTMLResponse("invalid authorization request", status_code=400)
        rid = secrets.token_urlsafe(16)
        requests[rid] = {k: q.get(k, "") for k in ("state", "nonce", "code_challenge")}
        return page(rid)

    @app.post("/login")
    async def login(request: Request) -> Response:
        raw = await request.body()
        if len(raw) > MAX_FORM_BYTES:
            return HTMLResponse("request too large", status_code=413)
        parsed = parse_qs(raw.decode("utf-8", "ignore"))
        form = {k: v[0] for k, v in parsed.items()}
        rid = form.get("rid", "")
        submitted = form.get("username", "").strip()
        # Over-long fields can never be right: they are refused before any hashing, and the
        # username is cut so an attacker cannot fill the throttle table with huge keys.
        too_long = (
            len(submitted) > MAX_USERNAME
            or len(form.get("password", "")) > MAX_PASSWORD
            or len(form.get("code", "")) > MAX_CODE
        )
        username = submitted[:MAX_USERNAME]
        if rid not in requests:
            return HTMLResponse(
                "This sign-in expired. Start again from the application.", status_code=400
            )
        if too_long:
            throttle.failure(username)
            return page(
                rid,
                error="Incorrect username, password or authenticator code.",
                username=username,
                status=401,
            )
        wait = throttle.locked_for(username)
        if wait > 0:
            return page(
                rid,
                error=f"Too many attempts. Try again in {int(wait // 60) + 1} minute(s).",
                username=username,
                status=429,
            )
        account = accounts.get(username)
        # The password is checked whether or not the account exists, and every failure reads the
        # same, so neither the answer nor the timing says which part was wrong.
        password_ok = verify_password(
            form.get("password", ""), account.password_hash if account else None
        )
        step = None
        if account is not None and password_ok and account.totp_secret:
            step = verify_totp(
                account.totp_secret,
                form.get("code", ""),
                at=now(),
                last_step=account.last_totp_step,
            )
        good = account is not None and password_ok and (not account.totp_secret or step is not None)
        if not good:
            throttle.failure(username)
            return page(
                rid,
                error="Incorrect username, password or authenticator code.",
                username=username,
                status=401,
            )
        assert account is not None
        if step is not None:
            accounts.use_totp_step(account, step)  # this code can never be used again
        throttle.success(username)
        entry = requests.pop(rid)
        code = secrets.token_urlsafe(24)
        codes[code] = {
            **entry,
            "account": account,
            "amr": ["pwd", "otp"] if account.totp_secret else ["pwd"],
            "exp": time.time() + 60,
        }
        return RedirectResponse(
            f"{redirect_uri}?{urlencode({'code': code, 'state': entry['state']})}", status_code=303
        )

    @app.post("/token")
    async def token(request: Request) -> JSONResponse:
        parsed = parse_qs((await request.body()).decode("utf-8", "ignore"))
        form = {k: v[0] for k, v in parsed.items()}
        entry = codes.pop(str(form.get("code", "")), None)  # single use
        if entry is None or time.time() > entry["exp"]:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        if form.get("client_id") != client_id or form.get("redirect_uri") != redirect_uri:
            return JSONResponse({"error": "invalid_client"}, status_code=400)
        if client_secret and not hmac.compare_digest(
            str(form.get("client_secret", "")), client_secret
        ):
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        verifier = str(form.get("code_verifier", ""))
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii", "ignore")).digest())
            .rstrip(b"=")
            .decode()
        )
        if not hmac.compare_digest(challenge, entry["code_challenge"]):
            return JSONResponse({"error": "invalid_grant", "detail": "PKCE"}, status_code=400)
        account: Account = entry["account"]
        issued = int(time.time())
        claims = {
            "iss": base,
            "aud": client_id,
            "sub": hashlib.sha256(account.username.encode()).hexdigest()[:16],
            "iat": issued,
            "exp": issued + 300,
            "nonce": entry["nonce"],
            "email": account.username,
            "email_verified": True,
            "groups": list(account.groups),
            "tenant": account.tenant,
            "amr": entry["amr"],
        }
        id_token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})
        return JSONResponse({"id_token": id_token, "token_type": "Bearer", "expires_in": 300})

    return app
