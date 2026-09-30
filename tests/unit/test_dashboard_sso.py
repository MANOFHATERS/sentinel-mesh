"""SSO: OIDC sign-in, group→role, MFA, SCIM, short-lived sessions, the sign-in audit.

The OIDC tests run the real relying party against the real demo identity provider over
an in-process ASGI transport, so the protocol is exercised end to end (discovery, PKCE,
JWKS, RS256) with no network. The ID-token validation tests mint their own tokens so
each forgery a real provider would never send can be tried.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from sentinel.dashboard.app import create_app
from sentinel.dashboard.auth import AuthError, Identity, Role, TokenRegistry
from sentinel.dashboard.devidp import DemoUser, make_dev_idp
from sentinel.dashboard.sso import (
    AuthAudit,
    Directory,
    OidcClient,
    SessionStore,
    SsoConfig,
    SsoError,
    SsoService,
)

ISSUER = "http://idp.test"
CLIENT = "sentinel-mesh-dashboard"
REDIRECT = "http://127.0.0.1:8765/auth/callback"
SCIM = "scim-token-" + "s" * 30
MACHINE = "machine-key-" + "m" * 30
GROUPS = {"SOC-Analyst": Role.ANALYST, "Auditor": Role.VIEWER}


def config(**overrides) -> SsoConfig:
    base = dict(
        issuer=ISSUER,
        client_id=CLIENT,
        redirect_uri=REDIRECT,
        group_roles=GROUPS,
        default_tenant="acme",
    )
    base.update(overrides)
    return SsoConfig(**base)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def stack():
    idp = make_dev_idp(issuer=ISSUER, client_id=CLIENT, redirect_uri=REDIRECT)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=idp), base_url=ISSUER)
    service = SsoService(config(), tenants=frozenset({"acme"}), scim_token=SCIM, http=http)
    tokens = TokenRegistry({MACHINE: Identity("ci-bot@acme.example", "acme", Role.VIEWER)})
    app = create_app({"acme": SimpleNamespace(tenant_id="acme")}, tokens, sso=service)
    with (
        TestClient(app, follow_redirects=False) as client,
        TestClient(idp, follow_redirects=False) as idp_client,
    ):
        yield SimpleNamespace(app=client, idp=idp_client, service=service)


def sign_in(stack, email: str) -> str:
    """Drive the whole redirect dance for ``email``; returns the final Location."""
    start = stack.app.get("/auth/login")
    assert start.status_code == 303
    authorize = urlparse(start.headers["location"])
    page = stack.idp.get(f"{authorize.path}?{authorize.query}")
    assert page.status_code == 200
    pick = re.search(
        rf'href="[^"]*?/login\?([^"]*u={re.escape(quote(email))}[^"]*)"',
        page.text.replace("&amp;", "&"),
    )
    assert pick, f"{email} not offered by the demo IdP"
    back = stack.idp.get(f"/login?{pick.group(1)}")
    callback = urlparse(back.headers["location"])
    done = stack.app.get(f"{callback.path}?{callback.query}")
    return done.headers["location"]


def tokens_for(stack, email: str) -> dict:
    location = sign_in(stack, email)
    assert "#sso=" in location, location
    handoff = location.split("#sso=")[1]
    response = stack.app.post("/auth/exchange", json={"code": handoff})
    assert response.status_code == 200
    return response.json()


def bearer(tokens: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


# --------------------------------------------------------------------------- #
# Sign-in and roles
# --------------------------------------------------------------------------- #


def test_analyst_group_signs_in_as_analyst(stack):
    tokens = tokens_for(stack, "maya.analyst@acme.example")
    me = stack.app.get("/api/session", headers=bearer(tokens)).json()
    assert me == {
        "principal": "maya.analyst@acme.example",
        "tenant_id": "acme",
        "role": "analyst",
        "can_act": True,
        "data_mode": "synthetic",
        "dataset": None,
        "workspace": "ready",
        "workspace_error": None,
    }


def test_auditor_group_signs_in_as_viewer(stack):
    me = stack.app.get(
        "/api/session", headers=bearer(tokens_for(stack, "omar.auditor@acme.example"))
    ).json()
    assert me["role"] == "viewer" and me["can_act"] is False


def test_no_second_factor_is_refused(stack):
    assert sign_in(stack, "nina.nomfa@acme.example").endswith("#sso_error=mfa_required")


def test_a_user_in_no_mapped_group_is_refused(stack):
    assert sign_in(stack, "carl.contractor@acme.example").endswith("#sso_error=no_role")


def test_handoff_code_is_single_use(stack):
    handoff = sign_in(stack, "maya.analyst@acme.example").split("#sso=")[1]
    assert stack.app.post("/auth/exchange", json={"code": handoff}).status_code == 200
    assert stack.app.post("/auth/exchange", json={"code": handoff}).status_code == 401


def test_a_replayed_callback_is_refused(stack):
    start = stack.app.get("/auth/login")
    authorize = urlparse(start.headers["location"])
    page = stack.idp.get(f"{authorize.path}?{authorize.query}").text.replace("&amp;", "&")
    pick = re.search(r'href="[^"]*?/login\?([^"]*maya[^"]*)"', page).group(1)
    callback = urlparse(stack.idp.get(f"/login?{pick}").headers["location"])
    first = stack.app.get(f"{callback.path}?{callback.query}")
    second = stack.app.get(f"{callback.path}?{callback.query}")
    assert "#sso=" in first.headers["location"]
    assert second.headers["location"].endswith("#sso_error=denied")


def test_a_callback_with_an_unknown_state_is_refused(stack):
    done = stack.app.get("/auth/callback?code=abc&state=never-issued")
    assert done.headers["location"].endswith("#sso_error=denied")


def test_the_identity_provider_saying_no_is_refused(stack):
    assert (
        stack.app.get("/auth/callback?error=access_denied")
        .headers["location"]
        .endswith("#sso_error=denied")
    )


def test_the_authorization_request_uses_pkce_state_and_nonce(stack):
    query = parse_qs(urlparse(stack.app.get("/auth/login").headers["location"]).query)
    assert query["code_challenge_method"] == ["S256"]
    assert len(query["state"][0]) >= 24 and len(query["nonce"][0]) >= 24
    assert len(query["code_challenge"][0]) >= 43


def test_login_state_is_different_every_time(stack):
    a = parse_qs(urlparse(stack.app.get("/auth/login").headers["location"]).query)
    b = parse_qs(urlparse(stack.app.get("/auth/login").headers["location"]).query)
    assert a["state"] != b["state"] and a["nonce"] != b["nonce"]


def test_static_tokens_still_work_as_machine_keys(stack):
    assert (
        stack.app.get("/api/session", headers={"Authorization": f"Bearer {MACHINE}"}).status_code
        == 200
    )


def test_config_reports_sso_and_token_login(stack):
    assert stack.app.get("/auth/config").json() == {
        "sso": True,
        "token_login": True,
        "mfa_required": True,
    }


def test_without_sso_the_config_says_so():
    app = create_app(
        {"acme": SimpleNamespace(tenant_id="acme")},
        TokenRegistry({MACHINE: Identity("a@acme.example", "acme", Role.ANALYST)}),
    )
    body = TestClient(app).get("/auth/config").json()
    assert body["sso"] is False and body["token_login"] is True
    assert TestClient(app).get("/auth/login").status_code == 404


def test_an_sso_session_is_not_accepted_without_the_header(stack):
    assert stack.app.get("/api/session").status_code == 401


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


def make_store(clock: Clock, **kw) -> SessionStore:
    return SessionStore(
        access_ttl=kw.get("access", 900),
        refresh_ttl=kw.get("refresh", 3600),
        session_max=kw.get("hard", 7200),
        now=clock,
    )


ME = Identity("maya@acme.example", "acme", Role.ANALYST)


def test_access_token_expires():
    clock = Clock()
    store = make_store(clock)
    issued = store.create(ME)
    assert store.authenticate(f"Bearer {issued.access_token}") == ME
    clock.t += 901
    with pytest.raises(AuthError, match="expired"):
        store.authenticate(f"Bearer {issued.access_token}")


def test_refresh_rotates_and_the_old_access_token_dies():
    clock = Clock()
    store = make_store(clock)
    first = store.create(ME)
    second, who = store.refresh(first.refresh_token)
    assert who == ME and second.access_token != first.access_token
    with pytest.raises(AuthError):
        store.authenticate(f"Bearer {first.access_token}")
    assert store.authenticate(f"Bearer {second.access_token}") == ME


def test_a_reused_refresh_token_revokes_the_whole_session():
    clock = Clock()
    store = make_store(clock)
    first = store.create(ME)
    second, _ = store.refresh(first.refresh_token)
    with pytest.raises(SsoError) as err:
        store.refresh(first.refresh_token)  # the thief, or the victim, replays it
    assert err.value.code == "reuse"
    with pytest.raises(AuthError):
        store.authenticate(f"Bearer {second.access_token}")
    with pytest.raises(SsoError):
        store.refresh(second.refresh_token)


def test_refresh_token_expires():
    clock = Clock()
    store = make_store(clock)
    issued = store.create(ME)
    clock.t += 3601
    with pytest.raises(SsoError, match="expired"):
        store.refresh(issued.refresh_token)


def test_session_has_an_absolute_lifetime_however_often_it_refreshes():
    clock = Clock()
    store = make_store(clock, access=900, refresh=3600, hard=2000)
    issued = store.create(ME)
    for _ in range(2):
        clock.t += 800
        issued, _ = store.refresh(issued.refresh_token)
    clock.t += 500  # 2100s since sign-in, past the 2000s cap
    with pytest.raises(AuthError):
        store.authenticate(f"Bearer {issued.access_token}")


def test_revoking_a_principal_kills_every_session_of_theirs():
    store = make_store(Clock())
    a, b = store.create(ME), store.create(ME)
    other = store.create(Identity("omar@acme.example", "acme", Role.VIEWER))
    assert store.revoke_principal("MAYA@acme.example") == 2
    for issued in (a, b):
        with pytest.raises(AuthError):
            store.authenticate(f"Bearer {issued.access_token}")
    assert store.authenticate(f"Bearer {other.access_token}").role is Role.VIEWER


def test_the_store_holds_only_digests():
    store = make_store(Clock())
    issued = store.create(ME)
    dump = repr(vars(store))
    assert issued.access_token not in dump and issued.refresh_token not in dump


def test_refresh_endpoint_rotates_and_logout_revokes(stack):
    tokens = tokens_for(stack, "maya.analyst@acme.example")
    rotated = stack.app.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert rotated.status_code == 200
    fresh = rotated.json()
    assert stack.app.get("/api/session", headers=bearer(fresh)).status_code == 200
    assert stack.app.get("/api/session", headers=bearer(tokens)).status_code == 401
    assert stack.app.post("/auth/logout", headers=bearer(fresh)).status_code == 200
    assert stack.app.get("/api/session", headers=bearer(fresh)).status_code == 401
    replay = stack.app.post("/auth/refresh", json={"refresh_token": fresh["refresh_token"]})
    assert replay.status_code == 401


# --------------------------------------------------------------------------- #
# ID-token validation: forgeries the identity provider would never send
# --------------------------------------------------------------------------- #


class Minter:
    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        n = self.key.public_key().public_numbers()

        def b64(v: int) -> str:
            return (
                base64.urlsafe_b64encode(v.to_bytes((v.bit_length() + 7) // 8, "big"))
                .rstrip(b"=")
                .decode()
            )

        self.jwks = {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "k1",
                    "alg": "RS256",
                    "use": "sig",
                    "n": b64(n.n),
                    "e": b64(n.e),
                }
            ]
        }

    def mint(self, key=None, headers=None, algorithm="RS256", **claims) -> str:
        now = int(time.time())
        body = {
            "iss": ISSUER,
            "aud": CLIENT,
            "sub": "u1",
            "iat": now,
            "exp": now + 300,
            "nonce": "N",
            "email": "maya@acme.example",
            "email_verified": True,
            "groups": ["SOC-Analyst"],
            "tenant": "acme",
            "amr": ["pwd", "otp"],
        }
        body.update(claims)
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(
            body, key or self.key, algorithm=algorithm, headers=headers or {"kid": "k1"}
        )


def client_for(minter: Minter, **overrides) -> OidcClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=minter.jwks)

    cfg = config(
        authorization_endpoint=f"{ISSUER}/a",
        token_endpoint=f"{ISSUER}/t",
        jwks_uri=f"{ISSUER}/jwks",
        **overrides,
    )
    return OidcClient(
        cfg,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        tenants=frozenset({"acme"}),
    )


@pytest.fixture(scope="module")
def minter() -> Minter:
    return Minter()


def test_a_good_token_validates(minter):
    claims = asyncio.run(client_for(minter).validate(minter.mint(), "N"))
    assert (claims.principal, claims.tenant, claims.role) == (
        "maya@acme.example",
        "acme",
        Role.ANALYST,
    )


@pytest.mark.parametrize(
    "claims, why",
    [
        ({"exp": int(time.time()) - 3600, "iat": int(time.time()) - 4000}, "expired"),
        ({"aud": "someone-else"}, "wrong audience"),
        ({"iss": "http://evil.test"}, "wrong issuer"),
        ({"nonce": "OTHER"}, "nonce"),
        ({"email": None, "preferred_username": None}, "no email"),
        ({"email_verified": False}, "unverified email"),
        ({"groups": ["Interns"]}, "no mapped group"),
        ({"groups": None}, "no groups claim"),
        ({"tenant": "globex"}, "unknown tenant"),
        ({"amr": ["pwd"]}, "no MFA"),
        ({"amr": None}, "no amr"),
    ],
)
def test_forged_or_unqualified_tokens_are_refused(minter, claims, why):
    with pytest.raises(SsoError):
        asyncio.run(client_for(minter).validate(minter.mint(**claims), "N"))


def test_a_token_signed_by_another_key_is_refused(minter):
    stranger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(SsoError):
        asyncio.run(client_for(minter).validate(minter.mint(key=stranger), "N"))


def test_alg_none_and_hmac_confusion_are_refused(minter):
    unsigned = jwt.encode(
        {
            "iss": ISSUER,
            "aud": CLIENT,
            "sub": "x",
            "exp": int(time.time()) + 99,
            "iat": int(time.time()),
            "email": "a@acme.example",
        },
        None,
        algorithm="none",
        headers={"kid": "k1"},
    )
    with pytest.raises(SsoError, match="alg"):
        asyncio.run(client_for(minter).validate(unsigned, "N"))
    hmac_token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": CLIENT,
            "sub": "x",
            "exp": int(time.time()) + 99,
            "iat": int(time.time()),
        },
        "secret" * 8,
        algorithm="HS256",
        headers={"kid": "k1"},
    )
    with pytest.raises(SsoError, match="alg"):
        asyncio.run(client_for(minter).validate(hmac_token, "N"))


def test_garbage_is_not_a_token(minter):
    with pytest.raises(SsoError):
        asyncio.run(client_for(minter).validate("not.a.jwt", "N"))


def test_mfa_can_be_switched_off_explicitly(minter):
    claims = asyncio.run(client_for(minter, require_mfa=False).validate(minter.mint(amr=None), "N"))
    assert claims.role is Role.ANALYST


def test_both_groups_grant_the_more_privileged_role(minter):
    claims = asyncio.run(
        client_for(minter).validate(minter.mint(groups=["Auditor", "SOC-Analyst"]), "N")
    )
    assert claims.role is Role.ANALYST


def test_group_mapping_parses_and_rejects_nonsense():
    assert SsoConfig.parse_group_roles("A=analyst, B=viewer") == {
        "A": Role.ANALYST,
        "B": Role.VIEWER,
    }
    with pytest.raises(AuthError):
        SsoConfig.parse_group_roles("A=root")
    with pytest.raises(AuthError):
        SsoConfig.parse_group_roles("justagroup")
    with pytest.raises(AuthError):
        SsoConfig(issuer=ISSUER, client_id=CLIENT, redirect_uri=REDIRECT, group_roles={})
    with pytest.raises(AuthError, match="https"):
        config(redirect_uri="http://example.com/auth/callback")


# --------------------------------------------------------------------------- #
# SCIM
# --------------------------------------------------------------------------- #


def scim(token: str | None = SCIM) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def test_scim_requires_its_own_token(stack):
    assert stack.app.get("/scim/v2/Users").status_code == 401
    assert stack.app.get("/scim/v2/Users", headers=scim("wrong" * 8)).status_code == 401
    # neither an API key nor a user's session is a SCIM credential
    assert stack.app.get("/scim/v2/Users", headers=scim(MACHINE)).status_code == 401
    session = bearer(tokens_for(stack, "maya.analyst@acme.example"))
    assert stack.app.get("/scim/v2/Users", headers=session).status_code == 401


def test_scim_is_off_when_no_token_is_configured():
    idp = make_dev_idp(issuer=ISSUER, client_id=CLIENT, redirect_uri=REDIRECT)
    service = SsoService(
        config(),
        tenants=frozenset({"acme"}),
        scim_token=None,
        http=httpx.AsyncClient(transport=httpx.ASGITransport(app=idp)),
    )
    app = create_app({"acme": SimpleNamespace(tenant_id="acme")}, TokenRegistry({}), sso=service)
    assert TestClient(app).get("/scim/v2/Users", headers=scim()).status_code == 401


def test_scim_create_list_filter_and_conflict(stack):
    made = stack.app.post(
        "/scim/v2/Users",
        headers=scim(),
        json={"userName": "new.hire@acme.example", "externalId": "e1"},
    )
    assert made.status_code == 201 and made.json()["active"] is True
    assert made.headers["content-type"].startswith("application/scim+json")
    again = stack.app.post(
        "/scim/v2/Users", headers=scim(), json={"userName": "NEW.hire@acme.example"}
    )
    assert again.status_code == 409
    found = stack.app.get(
        '/scim/v2/Users?filter=userName eq "new.hire@acme.example"', headers=scim()
    ).json()
    assert found["totalResults"] == 1
    assert stack.app.get("/scim/v2/Users?filter=active eq true", headers=scim()).status_code == 400
    assert (
        stack.app.post("/scim/v2/Users", headers=scim(), json={"userName": "bad name"}).status_code
        == 400
    )
    assert stack.app.get("/scim/v2/Users/nope", headers=scim()).status_code == 404


def test_deactivating_a_user_ends_their_session_at_once(stack):
    tokens = tokens_for(stack, "maya.analyst@acme.example")  # JIT-provisioned on first sign-in
    assert stack.app.get("/api/session", headers=bearer(tokens)).status_code == 200
    user = stack.app.get(
        '/scim/v2/Users?filter=userName eq "maya.analyst@acme.example"', headers=scim()
    ).json()["Resources"][0]
    patched = stack.app.patch(
        f"/scim/v2/Users/{user['id']}",
        headers=scim(),
        json={
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "Replace", "path": "active", "value": False}],
        },
    )
    assert patched.json()["active"] is False
    assert stack.app.get("/api/session", headers=bearer(tokens)).status_code == 401
    assert (
        stack.app.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code
        == 401
    )
    assert sign_in(stack, "maya.analyst@acme.example").endswith("#sso_error=deprovisioned")


def test_reactivating_lets_them_back_in(stack):
    tokens_for(stack, "maya.analyst@acme.example")
    user = stack.app.get("/scim/v2/Users", headers=scim()).json()["Resources"][0]
    stack.app.patch(
        f"/scim/v2/Users/{user['id']}",
        headers=scim(),
        json={"Operations": [{"op": "replace", "value": {"active": False}}]},
    )
    assert sign_in(stack, "maya.analyst@acme.example").endswith("deprovisioned")
    stack.app.patch(
        f"/scim/v2/Users/{user['id']}",
        headers=scim(),
        json={"Operations": [{"op": "replace", "path": "active", "value": "True"}]},
    )
    assert "#sso=" in sign_in(stack, "maya.analyst@acme.example")


def test_scim_delete_deactivates_and_keeps_the_record(stack):
    made = stack.app.post(
        "/scim/v2/Users", headers=scim(), json={"userName": "leaver@acme.example"}
    ).json()
    assert stack.app.delete(f"/scim/v2/Users/{made['id']}", headers=scim()).status_code == 204
    assert stack.app.get(f"/scim/v2/Users/{made['id']}", headers=scim()).json()["active"] is False


def test_with_jit_off_only_provisioned_users_may_sign_in():
    idp = make_dev_idp(issuer=ISSUER, client_id=CLIENT, redirect_uri=REDIRECT)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=idp), base_url=ISSUER)
    service = SsoService(
        config(jit_provisioning=False), tenants=frozenset({"acme"}), scim_token=SCIM, http=http
    )
    app = create_app({"acme": SimpleNamespace(tenant_id="acme")}, TokenRegistry({}), sso=service)
    with (
        TestClient(app, follow_redirects=False) as client,
        TestClient(idp, follow_redirects=False) as idp_client,
    ):
        s = SimpleNamespace(app=client, idp=idp_client, service=service)
        assert sign_in(s, "maya.analyst@acme.example").endswith("deprovisioned")
        client.post(
            "/scim/v2/Users", headers=scim(), json={"userName": "maya.analyst@acme.example"}
        )
        assert "#sso=" in sign_in(s, "maya.analyst@acme.example")


def test_directory_matches_names_case_insensitively():
    d = Directory()
    d.create("Maya@Acme.example")
    assert d.permits("maya@acme.example", jit=False)
    assert not d.permits("someone@acme.example", jit=False)
    assert d.permits("someone@acme.example", jit=True)


# --------------------------------------------------------------------------- #
# The sign-in audit trail
# --------------------------------------------------------------------------- #


def test_every_outcome_is_audited_and_the_chain_verifies(stack):
    tokens = tokens_for(stack, "maya.analyst@acme.example")
    sign_in(stack, "nina.nomfa@acme.example")
    stack.app.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    stack.app.post("/auth/logout", headers=bearer(tokens))
    events = [(e["event"], e["outcome"]) for e in stack.service.audit.events()]
    assert ("sso_login", "success") in events
    assert ("sso_login", "denied") in events
    assert ("session_refresh", "success") in events
    assert stack.service.audit.verify()


def test_tampering_with_the_audit_chain_is_detected(tmp_path):
    audit = AuthAudit(tmp_path / "a.jsonl")
    for who in ("a@x.example", "b@x.example", "c@x.example"):
        audit.record("sso_login", principal=who, tenant="acme", outcome="success")
    assert audit.verify()
    audit._events[1]["principal"] = "mallory@x.example"
    assert not audit.verify()
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["prev"] == "0" * 64


def test_the_audit_api_needs_a_token_and_is_tenant_scoped(stack):
    assert stack.app.get("/api/auth/audit").status_code == 401
    tokens = tokens_for(stack, "omar.auditor@acme.example")  # a viewer may read it
    body = stack.app.get("/api/auth/audit", headers=bearer(tokens)).json()
    assert body["enabled"] and body["intact"]
    assert all(e["tenant"] == "acme" for e in body["items"])
    # pre-authentication refusals have no tenant and are not shown to any tenant
    sign_in(stack, "carl.contractor@acme.example")
    after = stack.app.get("/api/auth/audit", headers=bearer(tokens)).json()["items"]
    assert all(e["tenant"] is not None for e in after)


def test_the_audit_api_is_disabled_without_sso():
    app = create_app(
        {"acme": SimpleNamespace(tenant_id="acme")},
        TokenRegistry({MACHINE: Identity("a@acme.example", "acme", Role.VIEWER)}),
    )
    body = (
        TestClient(app)
        .get("/api/auth/audit", headers={"Authorization": f"Bearer {MACHINE}"})
        .json()
    )
    assert body == {"enabled": False, "intact": True, "items": []}


# --------------------------------------------------------------------------- #
# The demo identity provider
# --------------------------------------------------------------------------- #


def test_dev_idp_rejects_a_wrong_pkce_verifier_and_a_reused_code():
    idp = make_dev_idp(
        issuer=ISSUER,
        client_id=CLIENT,
        redirect_uri=REDIRECT,
        users=(DemoUser("a@acme.example", "A", ("SOC-Analyst",)),),
    )
    with TestClient(idp, follow_redirects=False) as c:
        challenge = (
            base64.urlsafe_b64encode(__import__("hashlib").sha256(b"v" * 50).digest())
            .rstrip(b"=")
            .decode()
        )
        page = c.get(
            "/authorize",
            params={
                "client_id": CLIENT,
                "redirect_uri": REDIRECT,
                "response_type": "code",
                "state": "s",
                "nonce": "n",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        ).text
        pick = re.search(r'/login\?([^"]+)"', page.replace("&amp;", "&")).group(1)
        code = parse_qs(urlparse(c.get(f"/login?{pick}").headers["location"]).query)["code"][0]
        bad = c.post(
            "/token",
            data={
                "code": code,
                "client_id": CLIENT,
                "redirect_uri": REDIRECT,
                "code_verifier": "wrong",
            },
        )
        assert bad.status_code == 400
        # the failed attempt consumed the code: it cannot be retried with the right verifier
        again = c.post(
            "/token",
            data={
                "code": code,
                "client_id": CLIENT,
                "redirect_uri": REDIRECT,
                "code_verifier": "v" * 50,
            },
        )
        assert again.status_code == 400
        assert c.get("/authorize", params={"client_id": "other"}).status_code == 400
