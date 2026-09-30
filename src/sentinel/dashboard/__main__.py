"""``python -m sentinel.dashboard`` — serve the Analyst Copilot locally.

Binds to ``127.0.0.1`` unless told otherwise, because the API can isolate hosts and
open pull requests (against the local sandbox emulators here, but the shape is the
same one a real deployment would have). Tokens come from
``SENTINEL_DASHBOARD_TOKENS`` (``token:principal:tenant:role,...``); when that is
unset (and SSO is off), fresh tokens are generated for one analyst and one viewer per
tenant and printed once. They are not written anywhere.

Single sign-on is configured with ``SENTINEL_OIDC_ISSUER``, ``SENTINEL_OIDC_CLIENT_ID``,
``SENTINEL_OIDC_CLIENT_SECRET`` (optional), ``SENTINEL_OIDC_GROUPS``
(``SOC-Analyst=analyst,Auditor=viewer``), ``SENTINEL_OIDC_TENANT`` (default tenant) and
``SENTINEL_SCIM_TOKEN``; ``--public-url`` is the address users reach the dashboard at.
``--dev-idp`` mounts a stand-in identity provider for demos. With SSO on, static tokens
exist only if ``SENTINEL_DASHBOARD_TOKENS`` is set explicitly.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from sentinel.dashboard.auth import Identity, Role, TokenRegistry
from sentinel.dashboard.workspace import DEFAULT_ALERTS, DEFAULT_SEED, MeshModels, Workspace


def _build_sso(args, public: str, tenants: frozenset[str], root: Path):
    """``(SsoService | None, dev IdP app | None)`` from flags and environment."""
    from sentinel.dashboard.sso import AuthAudit, SsoConfig, SsoService

    env = os.environ
    redirect = f"{public}/auth/callback"
    scim = env.get("SENTINEL_SCIM_TOKEN")
    idp = None
    if args.dev_idp:
        from sentinel.dashboard.devidp import make_dev_idp

        issuer = f"{public}/devidp"
        client_id = "sentinel-mesh-dashboard"
        idp = make_dev_idp(issuer=issuer, client_id=client_id, redirect_uri=redirect)
        config = SsoConfig(
            issuer=issuer, client_id=client_id, redirect_uri=redirect,
            group_roles=SsoConfig.parse_group_roles("SOC-Analyst=analyst,Auditor=viewer"),
            default_tenant=sorted(tenants)[0],
            # The demo IdP lives in this process; discovery is fetched over loopback.
        )
    elif env.get("SENTINEL_OIDC_ISSUER"):
        config = SsoConfig(
            issuer=env["SENTINEL_OIDC_ISSUER"],
            client_id=env["SENTINEL_OIDC_CLIENT_ID"],
            client_secret=env.get("SENTINEL_OIDC_CLIENT_SECRET"),
            redirect_uri=redirect,
            group_roles=SsoConfig.parse_group_roles(
                env.get("SENTINEL_OIDC_GROUPS", "SOC-Analyst=analyst,Auditor=viewer")),
            default_tenant=env.get("SENTINEL_OIDC_TENANT"),
        )
    else:
        return None, None
    audit = AuthAudit(root / "auth-audit.jsonl")
    return SsoService(config, tenants=tenants, scim_token=scim, audit=audit), idp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workdir", type=Path, default=None,
                        help="where the audit chain, checkpoints and journal live "
                             "(default: a fresh temporary directory)")
    parser.add_argument("--tenants", default="acme",
                        help="comma-separated tenant ids, one workspace each")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alerts", type=int, default=DEFAULT_ALERTS)
    parser.add_argument("--public-url", default=None,
                        help="the URL browsers use (default: http://HOST:PORT); the SSO "
                             "redirect URI is derived from it")
    parser.add_argument("--dev-idp", action="store_true",
                        help="mount a demo OpenID Connect provider at /devidp and use it "
                             "for SSO (local demos only)")
    parser.add_argument("--evaluation", type=Path,
                        default=Path("data/artifacts/evaluation.json"))
    args = parser.parse_args(argv)

    import uvicorn

    from sentinel.dashboard.app import create_app

    tenants = [t.strip() for t in args.tenants.split(",") if t.strip()]
    started = time.perf_counter()
    print(f"training models (seed {args.seed}, {args.alerts} flows)...", flush=True)
    models = MeshModels.build(seed=args.seed, n_alerts=args.alerts)
    # The diffusion study trains in the background; the Models page fills in when
    # it finishes (about half a minute on a laptop).
    models.start_background()
    root = args.workdir or Path(tempfile.mkdtemp(prefix="sentinel-dashboard-"))
    workspaces = {
        tenant: Workspace(models, tenant_id=tenant, workdir=root / tenant) for tenant in tenants
    }
    public = (args.public_url or f"http://{args.host}:{args.port}").rstrip("/")
    sso, idp = _build_sso(args, public, frozenset(tenants), root)
    spec = os.environ.get("SENTINEL_DASHBOARD_TOKENS")
    if spec:
        tokens = TokenRegistry.parse(spec)
        issued: dict[str, str] = {}
    elif sso is not None:
        tokens, issued = TokenRegistry({}), {}
    else:
        tokens, issued = TokenRegistry.generate(
            [Identity(f"analyst@{t}.example", t, Role.ANALYST) for t in tenants]
            + [Identity(f"viewer@{t}.example", t, Role.VIEWER) for t in tenants]
        )
    app = create_app(workspaces, tokens, evaluation_path=args.evaluation, sso=sso)
    if idp is not None:
        app.mount("/devidp", idp)
    print(f"ready in {time.perf_counter() - started:.1f}s; state in {root}")
    for who, token in issued.items():
        print(f"  token for {who}: {token}")
    if sso is not None:
        print("single sign-on is on; people sign in through the identity provider")
        if args.dev_idp:
            print("  demo identity provider: maya.analyst@ (analyst), omar.auditor@ (viewer),")
            print("  nina.nomfa@ (refused: no MFA), carl.contractor@ (refused: no group)")
        if os.environ.get("SENTINEL_SCIM_TOKEN"):
            print("  SCIM provisioning at /scim/v2/Users")
    print(f"open http://{args.host}:{args.port}/", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        for workspace in workspaces.values():
            workspace.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
