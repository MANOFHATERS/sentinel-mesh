"""A stand-in OpenID Connect provider for local demos and tests.

This is **not** part of the product. It plays the role Okta / Azure AD / Google
Workspace plays in a deployment, so the dashboard's SSO can be exercised end to end on a
laptop with no external account. It speaks the real protocol — discovery, JWKS,
authorization-code with PKCE, RS256-signed ID tokens carrying ``groups`` and ``amr`` —
so the relying party in :mod:`sentinel.dashboard.sso` is the same code that would face
a real provider. Only the "log in" step is fake: instead of a password and MFA prompt it
lists a few demo people, one of whom has no second factor and one of whom has no group,
so the refusals can be shown as well as the successes.

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
from typing import Any, Final
from urllib.parse import parse_qs, urlencode

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

__all__ = ["DEMO_USERS", "DemoUser", "make_dev_idp"]


@dataclass(frozen=True, slots=True)
class DemoUser:
    email: str
    label: str
    groups: tuple[str, ...]
    mfa: bool = True
    tenant: str = "acme"


DEMO_USERS: Final[tuple[DemoUser, ...]] = (
    DemoUser(
        "maya.analyst@acme.example", "Maya — SOC analyst (group SOC-Analyst, MFA)", ("SOC-Analyst",)
    ),
    DemoUser("omar.auditor@acme.example", "Omar — Auditor (group Auditor, MFA)", ("Auditor",)),
    DemoUser(
        "nina.nomfa@acme.example",
        "Nina — SOC analyst but signed in WITHOUT a second factor",
        ("SOC-Analyst",),
        mfa=False,
    ),
    DemoUser(
        "carl.contractor@acme.example", "Carl — contractor, in no mapped group", ("Contractors",)
    ),
)


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def make_dev_idp(
    *,
    issuer: str,
    client_id: str,
    redirect_uri: str,
    users: tuple[DemoUser, ...] = DEMO_USERS,
    client_secret: str | None = None,
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
    by_email = {u.email: u for u in users}
    requests: dict[str, dict[str, str]] = {}
    codes: dict[str, dict[str, Any]] = {}
    base = issuer.rstrip("/")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

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

    @app.get("/authorize", response_class=HTMLResponse)
    def authorize(request: Request) -> HTMLResponse:
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
        rows = "".join(
            f'<li><a href="{base}/login?{urlencode({"rid": rid, "u": u.email})}">'
            f"{escape(u.label)}</a></li>"
            for u in users
        )
        page = (
            '<!doctype html><meta charset="utf-8"><title>Demo identity provider</title>'
            '<link rel="stylesheet" href="/static/app.css">'
            '<div class="login-wrap"><div class="login"><h1>Demo identity provider</h1>'
            '<p class="sub">Stands in for Okta / Azure AD for local demos. '
            f"Pick who is signing in to Sentinel Mesh.</p><ul>{rows}</ul></div></div>"
        )
        return HTMLResponse(page)

    @app.get("/login")
    def login(rid: str, u: str) -> RedirectResponse:
        request = requests.pop(rid, None)
        user = by_email.get(u)
        if request is None or user is None:
            return RedirectResponse(f"{redirect_uri}?error=access_denied", status_code=303)
        code = secrets.token_urlsafe(24)
        codes[code] = {**request, "user": user, "exp": time.time() + 60}
        return RedirectResponse(
            f"{redirect_uri}?{urlencode({'code': code, 'state': request['state']})}",
            status_code=303,
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
        user: DemoUser = entry["user"]
        now = int(time.time())
        claims = {
            "iss": base,
            "aud": client_id,
            "sub": hashlib.sha256(user.email.encode()).hexdigest()[:16],
            "iat": now,
            "exp": now + 300,
            "nonce": entry["nonce"],
            "email": user.email,
            "email_verified": True,
            "groups": list(user.groups),
            "tenant": user.tenant,
            "amr": ["pwd", "otp"] if user.mfa else ["pwd"],
        }
        id_token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})
        return JSONResponse({"id_token": id_token, "token_type": "Bearer", "expires_in": 300})

    return app
