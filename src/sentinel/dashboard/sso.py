"""Single sign-on for the dashboard: OIDC sign-in, group→role mapping, SCIM, sessions.

What a product sold to a SOC is expected to do, and where each piece lives:

*   **SSO (OIDC).** The browser is sent to the company's identity provider (Okta,
    Azure AD, Google Workspace, ...) using the authorization-code flow with PKCE,
    ``state`` and ``nonce``. The product never sees a password. The ID token is a
    signed JWT and is validated for signature (JWKS), issuer, audience, expiry and
    nonce before anything in it is believed. See :class:`OidcClient`.
*   **Roles from groups.** ``SOC-Analyst`` → analyst, ``Auditor`` → viewer, from the
    token's ``groups`` claim (:class:`SsoConfig.group_roles`). A user in no mapped
    group is refused rather than defaulted to viewer: access is opt-in.
*   **Provisioning and removal (SCIM 2.0).** The identity provider pushes users to
    ``/scim/v2/Users``; deactivating one revokes their live sessions immediately
    (:class:`Directory`, :func:`install_sso`).
*   **Short-lived sessions.** A session is a 15-minute access token plus a rotating
    refresh token, with an absolute lifetime. A refresh token presented twice is a
    stolen one, and the whole session is revoked (:class:`SessionStore`).
*   **MFA.** The ID token's ``amr``/``acr`` must show a second factor, or the sign-in
    is refused (``require_mfa``, on by default).
*   **Audit trail of who signed in.** Every success, denial, refresh, revocation and
    SCIM change is appended to a hash-chained log (:class:`AuthAudit`).

Static bearer tokens (``auth.py``) still work and are now what they should be: API keys
for machines and break-glass, not the way people sign in.

The browser still holds a bearer token in memory/sessionStorage and sends it in a
header, not a cookie, so the CSRF argument in ``auth.py`` still holds. The token
reaches the page through a one-time handoff code in the URL *fragment* (never sent to
a server or written to a log) that the page exchanges over ``POST``.

Deliberate limits: sessions, the directory and the audit chain live in memory (a
restart signs everyone out; a real deployment puts them in a database), and roles are
read from the ID token at sign-in, so a group change takes effect at the next sign-in,
not mid-session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from sentinel.core.errors import SentinelError
from sentinel.dashboard.auth import MIN_TOKEN_LENGTH, AuthError, Identity, Role

__all__ = [
    "AuthAudit",
    "Directory",
    "OidcClient",
    "SessionStore",
    "SsoConfig",
    "SsoError",
    "SsoService",
    "install_sso",
]

#: ``amr`` values (RFC 8176) that show a second factor was used.
MFA_AMR: Final[frozenset[str]] = frozenset({"mfa", "otp", "hwk", "swk", "fpt", "face", "sms"})
_PRINCIPAL: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._%+\-@]{0,127}$")
_PENDING_TTL: Final[float] = 300.0
_HANDOFF_TTL: Final[float] = 60.0


class SsoError(SentinelError):
    """A sign-in was refused. ``reason`` is what the audit log records; the browser is
    told only the short ``code`` so an attacker learns nothing about which check failed."""

    def __init__(self, reason: str, *, code: str = "denied") -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SsoConfig:
    issuer: str
    client_id: str
    redirect_uri: str
    #: IdP group name → role. A user with several mapped groups gets the most
    #: privileged one (the union of what each group allows).
    group_roles: Mapping[str, Role]
    client_secret: str | None = None
    scopes: str = "openid email profile"
    group_claim: str = "groups"
    tenant_claim: str = "tenant"
    default_tenant: str | None = None
    require_mfa: bool = True
    #: With ``False``, only users SCIM has provisioned may sign in; with ``True`` an
    #: unknown but validly-authenticated user is created on first sign-in.
    jit_provisioning: bool = True
    access_ttl: float = 900.0
    refresh_ttl: float = 8 * 3600.0
    session_max: float = 12 * 3600.0
    #: Overrides for IdPs that skip discovery. Left ``None``, they come from
    #: ``{issuer}/.well-known/openid-configuration``.
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    jwks_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.group_roles:
            raise AuthError("SSO needs at least one group→role mapping")
        if not self.redirect_uri.startswith(("http://127.0.0.1", "http://localhost", "https://")):
            raise AuthError("SSO redirect_uri must be https (or loopback for local demos)")

    @classmethod
    def parse_group_roles(cls, spec: str) -> dict[str, Role]:
        """``"SOC-Analyst=analyst,Auditor=viewer"``."""
        mapping: dict[str, Role] = {}
        for raw in spec.split(","):
            entry = raw.strip()
            if not entry:
                continue
            group, sep, role = entry.partition("=")
            if not sep or not group.strip():
                raise AuthError(f"group mapping {entry!r} is not group=role")
            try:
                mapping[group.strip()] = Role(role.strip())
            except ValueError as exc:
                raise AuthError(f"unknown role {role!r} in group mapping") from exc
        return mapping


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #


class AuthAudit:
    """Append-only, hash-chained log of authentication events."""

    def __init__(self, path: Path | None = None, *, now: Callable[[], float] = time.time) -> None:
        self._events: list[dict[str, Any]] = []
        self._path = path
        self._now = now

    def record(
        self,
        event: str,
        *,
        principal: str | None,
        tenant: str | None,
        outcome: str,
        detail: str = "",
    ) -> dict[str, Any]:
        prev = self._events[-1]["hash"] if self._events else "0" * 64
        body = {
            "seq": len(self._events) + 1,
            "ts": round(self._now(), 3),
            "event": event,
            "principal": principal,
            "tenant": tenant,
            "outcome": outcome,
            "detail": detail[:300],
            "prev": prev,
        }
        body["hash"] = self._hash(body)
        self._events.append(body)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(body, sort_keys=True) + "\n")
        return body

    @staticmethod
    def _hash(body: Mapping[str, Any]) -> str:
        material = {k: v for k, v in body.items() if k != "hash"}
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def verify(self) -> bool:
        prev = "0" * 64
        for index, event in enumerate(self._events, start=1):
            if event["seq"] != index or event["prev"] != prev or event["hash"] != self._hash(event):
                return False
            prev = event["hash"]
        return True

    def events(self, *, tenant: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Newest first. With ``tenant``, only that tenant's events (never the
        tenant-less ones, which are pre-authentication and could be anyone's)."""
        rows = [e for e in self._events if tenant is None or e["tenant"] == tenant]
        return list(reversed(rows))[:limit]

    def __len__(self) -> int:
        return len(self._events)


# --------------------------------------------------------------------------- #
# SCIM directory
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DirectoryUser:
    id: str
    user_name: str
    active: bool = True
    external_id: str | None = None
    display_name: str | None = None


class Directory:
    """Users the identity provider has provisioned, by SCIM."""

    def __init__(self) -> None:
        self._by_id: dict[str, DirectoryUser] = {}

    def create(
        self,
        user_name: str,
        *,
        active: bool = True,
        external_id: str | None = None,
        display_name: str | None = None,
    ) -> DirectoryUser:
        if not _PRINCIPAL.match(user_name):
            raise SsoError(f"userName {user_name!r} is not a valid login", code="invalid")
        if self.by_name(user_name) is not None:
            raise SsoError(f"userName {user_name!r} already exists", code="conflict")
        user = DirectoryUser(secrets.token_hex(8), user_name, active, external_id, display_name)
        self._by_id[user.id] = user
        return user

    def get(self, user_id: str) -> DirectoryUser | None:
        return self._by_id.get(user_id)

    def by_name(self, user_name: str) -> DirectoryUser | None:
        wanted = user_name.lower()
        return next((u for u in self._by_id.values() if u.user_name.lower() == wanted), None)

    def all(self) -> list[DirectoryUser]:
        return list(self._by_id.values())

    def permits(self, user_name: str, *, jit: bool) -> bool:
        """May this principal sign in? Deactivated: never. Unknown: only with JIT."""
        user = self.by_name(user_name)
        if user is None:
            return jit
        return user.active

    def ensure(self, user_name: str) -> None:
        if self.by_name(user_name) is None:
            self.create(user_name)


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Session:
    sid: str
    identity: Identity
    access: str
    refresh: str
    access_exp: float
    refresh_exp: float
    hard_exp: float
    used_refresh: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class IssuedTokens:
    access_token: str
    refresh_token: str
    expires_in: int
    principal: str


class SessionStore:
    """Short-lived access tokens with rotating refresh tokens. Holds only digests."""

    def __init__(
        self,
        *,
        access_ttl: float,
        refresh_ttl: float,
        session_max: float,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._access_ttl = access_ttl
        self._refresh_ttl = refresh_ttl
        self._session_max = session_max
        self._now = now
        self._sessions: dict[str, _Session] = {}
        self._by_access: dict[str, str] = {}
        self._by_refresh: dict[str, str] = {}
        self._reused: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def _mint(self, session: _Session) -> IssuedTokens:
        now = self._now()
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        self._by_access.pop(session.access, None)
        self._by_refresh.pop(session.refresh, None)
        session.access, session.refresh = _digest(access), _digest(refresh)
        session.access_exp = min(now + self._access_ttl, session.hard_exp)
        session.refresh_exp = min(now + self._refresh_ttl, session.hard_exp)
        self._by_access[session.access] = session.sid
        self._by_refresh[session.refresh] = session.sid
        return IssuedTokens(
            access, refresh, int(session.access_exp - now), session.identity.principal
        )

    def create(self, identity: Identity) -> IssuedTokens:
        session = _Session(
            secrets.token_hex(8), identity, "", "", 0.0, 0.0, self._now() + self._session_max
        )
        self._sessions[session.sid] = session
        return self._mint(session)

    def authenticate(self, header: str | None) -> Identity:
        if not header:
            raise AuthError("missing bearer token")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token or token != token.strip():
            raise AuthError("expected 'Authorization: Bearer <token>'")
        sid = self._by_access.get(_digest(token))
        session = self._sessions.get(sid) if sid else None
        if session is None:
            raise AuthError("unknown token")
        if self._now() >= session.access_exp:
            raise AuthError("session expired")
        return session.identity

    def refresh(self, refresh_token: str) -> tuple[IssuedTokens, Identity]:
        """Rotate. A refresh token that was already used revokes its whole session."""
        digest = _digest(refresh_token)
        if digest in self._reused:
            self.revoke(self._reused[digest])
            raise SsoError("refresh token reuse detected; session revoked", code="reuse")
        sid = self._by_refresh.get(digest)
        session = self._sessions.get(sid) if sid else None
        if session is None:
            raise SsoError("unknown refresh token", code="denied")
        if self._now() >= session.refresh_exp:
            self.revoke(session.sid)
            raise SsoError("refresh token expired", code="expired")
        self._reused[digest] = session.sid
        return self._mint(session), session.identity

    def revoke(self, sid: str) -> None:
        session = self._sessions.pop(sid, None)
        if session is None:
            return
        self._by_access.pop(session.access, None)
        self._by_refresh.pop(session.refresh, None)
        for used in [d for d, s in self._reused.items() if s == sid]:
            del self._reused[used]

    def revoke_token(self, access_token: str) -> Identity | None:
        sid = self._by_access.get(_digest(access_token))
        if sid is None:
            return None
        identity = self._sessions[sid].identity
        self.revoke(sid)
        return identity

    def revoke_principal(self, principal: str) -> int:
        doomed = [
            s.sid
            for s in self._sessions.values()
            if s.identity.principal.lower() == principal.lower()
        ]
        for sid in doomed:
            self.revoke(sid)
        return len(doomed)


# --------------------------------------------------------------------------- #
# OIDC relying party
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Claims:
    principal: str
    tenant: str
    role: Role


class OidcClient:
    """Discovery, JWKS, code exchange and ID-token validation."""

    def __init__(
        self,
        config: SsoConfig,
        *,
        http: httpx.AsyncClient | None = None,
        tenants: frozenset[str] = frozenset(),
    ) -> None:
        self._config = config
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._tenants = tenants
        self._meta: dict[str, Any] | None = None
        self._jwks: dict[str, Any] | None = None

    async def _metadata(self) -> dict[str, Any]:
        if self._meta is None:
            cfg = self._config
            if cfg.authorization_endpoint and cfg.token_endpoint and cfg.jwks_uri:
                self._meta = {
                    "authorization_endpoint": cfg.authorization_endpoint,
                    "token_endpoint": cfg.token_endpoint,
                    "jwks_uri": cfg.jwks_uri,
                }
            else:
                url = cfg.issuer.rstrip("/") + "/.well-known/openid-configuration"
                response = await self._http.get(url)
                response.raise_for_status()
                meta = response.json()
                if meta.get("issuer", "").rstrip("/") != cfg.issuer.rstrip("/"):
                    raise SsoError("discovery document issuer does not match", code="config")
                self._meta = meta
        return self._meta

    async def authorization_url(self, *, state: str, nonce: str, challenge: str) -> str:
        meta = await self._metadata()
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self._config.client_id,
                "redirect_uri": self._config.redirect_uri,
                "scope": self._config.scopes,
                "state": state,
                "nonce": nonce,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{meta['authorization_endpoint']}?{query}"

    async def exchange(self, code: str, verifier: str) -> str:
        meta = await self._metadata()
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._config.redirect_uri,
            "client_id": self._config.client_id,
            "code_verifier": verifier,
        }
        if self._config.client_secret:
            form["client_secret"] = self._config.client_secret
        response = await self._http.post(meta["token_endpoint"], data=form)
        if response.status_code != 200:
            raise SsoError(f"token endpoint refused the code ({response.status_code})")
        token = response.json().get("id_token")
        if not isinstance(token, str):
            raise SsoError("token response carried no id_token")
        return token

    async def _key(self, kid: str | None, *, refetch: bool = False) -> Any:
        meta = await self._metadata()
        if self._jwks is None or refetch:
            response = await self._http.get(meta["jwks_uri"])
            response.raise_for_status()
            self._jwks = response.json()
        for jwk in self._jwks.get("keys", []):
            if kid is None or jwk.get("kid") == kid:
                return jwt.PyJWK(jwk).key
        if not refetch:
            return await self._key(kid, refetch=True)  # key rotation
        raise SsoError("no signing key matches the token's kid")

    async def validate(self, id_token: str, nonce: str) -> Claims:
        cfg = self._config
        try:
            header = jwt.get_unverified_header(id_token)
        except jwt.PyJWTError as exc:
            raise SsoError("id_token is not a JWT") from exc
        # Only asymmetric algorithms: accepting "none" or an HMAC keyed by the public
        # key is the classic JWT forgery.
        if header.get("alg") not in ("RS256", "ES256"):
            raise SsoError(f"id_token alg {header.get('alg')!r} is not accepted")
        key = await self._key(header.get("kid"))
        try:
            claims = jwt.decode(
                id_token,
                key,
                algorithms=["RS256", "ES256"],
                audience=cfg.client_id,
                issuer=cfg.issuer,
                leeway=30,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise SsoError(f"id_token rejected: {exc}") from exc
        if not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
            raise SsoError("id_token nonce does not match this sign-in")
        if cfg.require_mfa and not self._mfa(claims):
            raise SsoError("no second factor in the id_token", code="mfa_required")
        principal = str(claims.get("email") or claims.get("preferred_username") or "")
        if not _PRINCIPAL.match(principal):
            raise SsoError("id_token carries no usable email")
        if claims.get("email") and claims.get("email_verified") is False:
            raise SsoError("email is not verified by the provider")
        roles = self._roles(claims.get(cfg.group_claim))
        if not roles:
            raise SsoError(f"{principal} is in no group mapped to a role", code="no_role")
        role = Role.ANALYST if Role.ANALYST in roles else Role.VIEWER
        tenant = str(claims.get(cfg.tenant_claim) or cfg.default_tenant or "")
        if tenant not in self._tenants:
            raise SsoError(f"tenant {tenant!r} has no workspace here", code="no_tenant")
        return Claims(principal, tenant, role)

    @staticmethod
    def _mfa(claims: Mapping[str, Any]) -> bool:
        amr = claims.get("amr")
        if isinstance(amr, list) and MFA_AMR.intersection(str(a) for a in amr):
            return True
        return str(claims.get("acr", "")).lower() in {"mfa", "urn:okta:loa:2fa:any"}

    def _roles(self, groups: Any) -> set[Role]:
        if isinstance(groups, str):
            groups = [groups]
        if not isinstance(groups, list):
            return set()
        return {self._config.group_roles[g] for g in groups if g in self._config.group_roles}


# --------------------------------------------------------------------------- #
# The service and its routes
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Pending:
    nonce: str
    verifier: str
    created: float


class SsoService:
    def __init__(
        self,
        config: SsoConfig,
        *,
        tenants: frozenset[str],
        scim_token: str | None = None,
        http: httpx.AsyncClient | None = None,
        audit: AuthAudit | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        if scim_token is not None and (
            len(scim_token) < MIN_TOKEN_LENGTH or scim_token != scim_token.strip()
        ):
            raise AuthError(f"SCIM token must be at least {MIN_TOKEN_LENGTH} characters")
        self.config = config
        self.now = now
        self.oidc = OidcClient(config, http=http, tenants=tenants)
        self.sessions = SessionStore(
            access_ttl=config.access_ttl,
            refresh_ttl=config.refresh_ttl,
            session_max=config.session_max,
            now=now,
        )
        self.directory = Directory()
        self.audit = audit or AuthAudit(now=now)
        self.scim_digest = _digest(scim_token) if scim_token else None
        self._pending: dict[str, _Pending] = {}
        self._handoff: dict[str, tuple[float, IssuedTokens]] = {}

    # -- sign-in ---------------------------------------------------------------- #

    async def begin(self) -> str:
        self._sweep()
        state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        self._pending[state] = _Pending(nonce, verifier, self.now())
        return await self.oidc.authorization_url(state=state, nonce=nonce, challenge=challenge)

    async def complete(self, code: str, state: str) -> str:
        """Finish the redirect. Returns a one-time handoff code for the page."""
        self._sweep()
        pending = self._pending.pop(state, None)  # single use: a replayed state finds nothing
        if pending is None:
            raise SsoError("unknown or already-used state")
        id_token = await self.oidc.exchange(code, pending.verifier)
        claims = await self.oidc.validate(id_token, pending.nonce)
        if not self.directory.permits(claims.principal, jit=self.config.jit_provisioning):
            raise SsoError(
                f"{claims.principal} is deactivated or not provisioned", code="deprovisioned"
            )
        self.directory.ensure(claims.principal)
        issued = self.sessions.create(Identity(claims.principal, claims.tenant, claims.role))
        self.audit.record(
            "sso_login",
            principal=claims.principal,
            tenant=claims.tenant,
            outcome="success",
            detail=f"role={claims.role.value}",
        )
        handoff = secrets.token_urlsafe(24)
        self._handoff[_digest(handoff)] = (self.now() + _HANDOFF_TTL, issued)
        return handoff

    def redeem(self, handoff: str) -> IssuedTokens:
        self._sweep()
        entry = self._handoff.pop(_digest(handoff), None)
        if entry is None or self.now() >= entry[0]:
            raise SsoError("unknown or expired handoff code")
        return entry[1]

    def _sweep(self) -> None:
        now = self.now()
        for state in [s for s, p in self._pending.items() if now - p.created > _PENDING_TTL]:
            del self._pending[state]
        for key in [k for k, (exp, _) in self._handoff.items() if now >= exp]:
            del self._handoff[key]

    # -- SCIM ------------------------------------------------------------------- #

    def scim_authorised(self, header: str | None) -> bool:
        if self.scim_digest is None or not header:
            return False
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(_digest(token), self.scim_digest)

    def deactivate(self, user: DirectoryUser, *, why: str) -> None:
        user.active = False
        revoked = self.sessions.revoke_principal(user.user_name)
        self.audit.record(
            "scim_deactivate",
            principal=user.user_name,
            tenant=None,
            outcome="success",
            detail=f"{why}; {revoked} session(s) revoked",
        )


_SCIM_USER = "urn:ietf:params:scim:schemas:core:2.0:User"
_SCIM_LIST = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
_SCIM_ERROR = "urn:ietf:params:scim:api:messages:2.0:Error"


def _scim_user(user: DirectoryUser) -> dict[str, Any]:
    return {
        "schemas": [_SCIM_USER],
        "id": user.id,
        "userName": user.user_name,
        "active": user.active,
        "externalId": user.external_id,
        "displayName": user.display_name,
        "meta": {"resourceType": "User"},
    }


def _scim(payload: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, media_type="application/scim+json")


def _scim_error(status: int, detail: str) -> JSONResponse:
    return _scim({"schemas": [_SCIM_ERROR], "status": str(status), "detail": detail}, status)


def install_sso(app: FastAPI, service: SsoService, *, static_tokens: bool) -> None:
    """Add ``/auth/*`` and ``/scim/v2/*`` to ``app``."""

    def _fail(reason: str, code: str) -> RedirectResponse:
        service.audit.record(
            "sso_login", principal=None, tenant=None, outcome="denied", detail=reason
        )
        return RedirectResponse(f"/#sso_error={code}", status_code=303)

    @app.get("/auth/config")
    def auth_config() -> dict[str, Any]:
        return {
            "sso": True,
            "token_login": static_tokens,
            "mfa_required": service.config.require_mfa,
        }

    @app.get("/auth/login", include_in_schema=False)
    async def login() -> RedirectResponse:
        try:
            return RedirectResponse(await service.begin(), status_code=303)
        except (httpx.HTTPError, SsoError) as exc:
            return _fail(f"cannot reach the identity provider: {exc}", "idp_unreachable")

    @app.get("/auth/callback", include_in_schema=False)
    async def callback(request: Request) -> RedirectResponse:
        params = request.query_params
        if "error" in params:
            return _fail(f"identity provider returned {params['error'][:60]}", "denied")
        code, state = params.get("code"), params.get("state")
        if not code or not state:
            return _fail("callback without code and state", "denied")
        try:
            handoff = await service.complete(code, state)
        except SsoError as exc:
            return _fail(exc.reason, exc.code)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return _fail(
                f"identity provider exchange failed: {type(exc).__name__}", "idp_unreachable"
            )
        return RedirectResponse(f"/#sso={handoff}", status_code=303)

    @app.post("/auth/exchange")
    async def exchange(request: Request) -> JSONResponse:
        body = await _json(request)
        try:
            issued = service.redeem(str(body.get("code", "")))
        except SsoError as exc:
            return JSONResponse({"error": "unauthorized", "detail": exc.reason}, status_code=401)
        return JSONResponse(_tokens(issued))

    @app.post("/auth/refresh")
    async def refresh(request: Request) -> JSONResponse:
        body = await _json(request)
        try:
            issued, identity = service.sessions.refresh(str(body.get("refresh_token", "")))
        except SsoError as exc:
            service.audit.record(
                "session_refresh", principal=None, tenant=None, outcome="denied", detail=exc.reason
            )
            return JSONResponse({"error": "unauthorized", "detail": exc.reason}, status_code=401)
        if not service.directory.permits(identity.principal, jit=True):
            service.sessions.revoke_principal(identity.principal)
            return JSONResponse(
                {"error": "unauthorized", "detail": "account deactivated"}, status_code=401
            )
        service.audit.record(
            "session_refresh",
            principal=identity.principal,
            tenant=identity.tenant_id,
            outcome="success",
        )
        return JSONResponse(_tokens(issued))

    @app.post("/auth/logout")
    async def logout(request: Request) -> JSONResponse:
        header = request.headers.get("authorization", "")
        identity = service.sessions.revoke_token(header.partition(" ")[2])
        if identity is not None:
            service.audit.record(
                "logout", principal=identity.principal, tenant=identity.tenant_id, outcome="success"
            )
        return JSONResponse({"ok": True})

    # --- SCIM 2.0 ------------------------------------------------------------- #

    def _guard(request: Request) -> JSONResponse | None:
        if not service.scim_authorised(request.headers.get("authorization")):
            return _scim_error(401, "invalid or missing SCIM bearer token")
        return None

    @app.get("/scim/v2/ServiceProviderConfig", include_in_schema=False)
    def scim_config(request: Request) -> JSONResponse:
        return _guard(request) or _scim(
            {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"],
                "patch": {"supported": True},
                "bulk": {"supported": False},
                "filter": {"supported": True, "maxResults": 200},
                "authenticationSchemes": [{"type": "oauthbearertoken", "name": "Bearer token"}],
            }
        )

    @app.get("/scim/v2/Users", include_in_schema=False)
    def scim_list(request: Request, filter: str | None = None) -> JSONResponse:
        if (denied := _guard(request)) is not None:
            return denied
        users = service.directory.all()
        if filter:
            match = re.fullmatch(r'\s*userName\s+eq\s+"([^"]{1,128})"\s*', filter)
            if match is None:
                return _scim_error(400, 'only: userName eq "value"')
            users = [u for u in users if u.user_name.lower() == match.group(1).lower()]
        return _scim(
            {
                "schemas": [_SCIM_LIST],
                "totalResults": len(users),
                "Resources": [_scim_user(u) for u in users],
            }
        )

    @app.post("/scim/v2/Users", include_in_schema=False)
    async def scim_create(request: Request) -> JSONResponse:
        if (denied := _guard(request)) is not None:
            return denied
        body = await _json(request)
        try:
            user = service.directory.create(
                str(body.get("userName", "")),
                active=bool(body.get("active", True)),
                external_id=body.get("externalId"),
                display_name=body.get("displayName"),
            )
        except SsoError as exc:
            return _scim_error(409 if exc.code == "conflict" else 400, exc.reason)
        service.audit.record(
            "scim_create", principal=user.user_name, tenant=None, outcome="success"
        )
        return _scim(_scim_user(user), 201)

    @app.get("/scim/v2/Users/{user_id}", include_in_schema=False)
    def scim_get(user_id: str, request: Request) -> JSONResponse:
        if (denied := _guard(request)) is not None:
            return denied
        user = service.directory.get(user_id)
        return _scim(_scim_user(user)) if user else _scim_error(404, "no such user")

    @app.patch("/scim/v2/Users/{user_id}", include_in_schema=False)
    async def scim_patch(user_id: str, request: Request) -> JSONResponse:
        if (denied := _guard(request)) is not None:
            return denied
        user = service.directory.get(user_id)
        if user is None:
            return _scim_error(404, "no such user")
        body = await _json(request)
        for op in body.get("Operations", []):
            path, value = op.get("path"), op.get("value")
            if isinstance(value, dict) and path is None:
                value = value.get("active")
                path = "active"
            if str(op.get("op", "")).lower() == "replace" and path == "active":
                active = value if isinstance(value, bool) else str(value).lower() == "true"
                if active:
                    user.active = True
                    service.audit.record(
                        "scim_activate", principal=user.user_name, tenant=None, outcome="success"
                    )
                else:
                    service.deactivate(user, why="SCIM PATCH active=false")
        return _scim(_scim_user(user))

    @app.delete("/scim/v2/Users/{user_id}", include_in_schema=False)
    def scim_delete(user_id: str, request: Request) -> JSONResponse:
        if (denied := _guard(request)) is not None:
            return denied
        user = service.directory.get(user_id)
        if user is None:
            return _scim_error(404, "no such user")
        # Deactivate rather than erase, so the audit trail still resolves the name.
        service.deactivate(user, why="SCIM DELETE")
        return JSONResponse(None, status_code=204)


async def _json(request: Request) -> dict[str, Any]:
    try:
        body = json.loads((await request.body()) or b"{}")
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _tokens(issued: IssuedTokens) -> dict[str, Any]:
    return {
        "access_token": issued.access_token,
        "refresh_token": issued.refresh_token,
        "expires_in": issued.expires_in,
        "principal": issued.principal,
    }
