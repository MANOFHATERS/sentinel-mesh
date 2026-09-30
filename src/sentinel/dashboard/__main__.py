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


def _real_files(root: Path = Path(".")):
    """``(dataset csv, attack json, label)`` when the real public data is on disk, else ``None``."""
    from sentinel.real.network import detect_format, find_dataset

    dataset = find_dataset(root)
    attack = root / "data" / "real" / "enterprise-attack.json"
    if dataset is None or not attack.is_file():
        return None
    label = {"unsw": "UNSW-NB15", "cic": "CIC-IDS2017"}[detect_format(dataset)]
    return dataset, attack, label


def _build_sso(args, public: str, tenants: frozenset[str], root: Path, *, real: bool = False):
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
        from sentinel.dashboard.devidp import DEMO_USERS, REAL_USERS

        idp = make_dev_idp(
            issuer=issuer,
            client_id=client_id,
            redirect_uri=redirect,
            users=DEMO_USERS + REAL_USERS if real else DEMO_USERS,
        )
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
    from sentinel.dashboard.registry import WorkspaceRegistry

    workspaces = WorkspaceRegistry(
        {tenant: Workspace(models, tenant_id=tenant, workdir=root / tenant) for tenant in tenants}
    )
    real_files = _real_files()
    if real_files is not None:
        # The real-data workspace trains on a real capture and indexes the real ATT&CK catalogue.
        # That takes about a minute, so it builds in the background and is served as "preparing"
        # until it is ready; the synthetic demo is not held back.
        dataset, attack, label = real_files
        workspaces.declare("real", mode="real", dataset=f"{label} (real)")

        def build_real() -> None:
            try:
                real_models = MeshModels.build_real(
                    models, dataset_path=dataset, attack_path=attack, seed=args.seed
                )
                workspaces.set_ready(
                    "real", Workspace(real_models, tenant_id="real", workdir=root / "real")
                )
                print("real-data workspace ready", flush=True)
            except Exception as exc:  # reported to the user by the registry, never fatal
                workspaces.set_failed("real", f"{type(exc).__name__}: {exc}")
                print(f"real-data workspace failed: {exc}", flush=True)

        import threading

        threading.Thread(target=build_real, name="real-workspace", daemon=True).start()
    public = (args.public_url or f"http://{args.host}:{args.port}").rstrip("/")
    sso_tenants = frozenset(tenants) | ({"real"} if real_files else frozenset())
    sso, idp = _build_sso(args, public, sso_tenants, root, real=real_files is not None)
    spec = os.environ.get("SENTINEL_DASHBOARD_TOKENS")
    if spec:
        tokens = TokenRegistry.parse(spec)
        issued: dict[str, str] = {}
    elif sso is not None:
        tokens, issued = TokenRegistry({}), {}
    else:
        every = [*tenants, *(["real"] if real_files else [])]
        tokens, issued = TokenRegistry.generate(
            [Identity(f"analyst@{t}.example", t, Role.ANALYST) for t in every]
            + [Identity(f"viewer@{t}.example", t, Role.VIEWER) for t in every]
        )
    from sentinel.dashboard.lab import model_report
    from sentinel.dashboard.runs import RunManager
    from sentinel.dashboard.views import evaluation_view
    from sentinel.real.service import RealData

    real = RealData(Path("."), curated_kb=models.kb)
    real.warm()
    runs = RunManager(
        workdir=root / "runs",
        retrain=lambda seed: model_report(MeshModels.build(seed=seed, n_alerts=args.alerts)),
        read_evaluation=evaluation_view,
        handlers={"real_network": real.run_network, "real_scan": real.run_scan},
        availability=lambda: {"real_network": real.network_path() is not None},
    )
    app = create_app(
        workspaces, tokens, evaluation_path=args.evaluation, sso=sso, runs=runs, real=real
    )
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
            if real_files:
                print("  REAL DATA users: real.analyst@ (analyst), real.auditor@ (viewer) — the")
                print("  whole dashboard shows the real capture and real ATT&CK for these two")
        if os.environ.get("SENTINEL_SCIM_TOKEN"):
            print("  SCIM provisioning at /scim/v2/Users")
    print(f"open http://{args.host}:{args.port}/", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        for _tenant, workspace in workspaces.ready_items():
            workspace.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
