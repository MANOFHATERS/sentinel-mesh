"""``python -m sentinel.dashboard`` — serve the Analyst Copilot locally.

Binds to ``127.0.0.1`` unless told otherwise, because the API can isolate hosts and
open pull requests (against the local sandbox emulators here, but the shape is the
same one a real deployment would have). Tokens come from
``SENTINEL_DASHBOARD_TOKENS`` (``token:principal:tenant:role,...``); when that is
unset, fresh tokens are generated for one analyst and one viewer per tenant and
printed once. They are not written anywhere.
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
    parser.add_argument("--evaluation", type=Path,
                        default=Path("data/artifacts/evaluation.json"))
    args = parser.parse_args(argv)

    import uvicorn

    from sentinel.dashboard.app import create_app

    tenants = [t.strip() for t in args.tenants.split(",") if t.strip()]
    started = time.perf_counter()
    print(f"training models (seed {args.seed}, {args.alerts} flows)...", flush=True)
    models = MeshModels.build(seed=args.seed, n_alerts=args.alerts)
    root = args.workdir or Path(tempfile.mkdtemp(prefix="sentinel-dashboard-"))
    workspaces = {
        tenant: Workspace(models, tenant_id=tenant, workdir=root / tenant) for tenant in tenants
    }
    spec = os.environ.get("SENTINEL_DASHBOARD_TOKENS")
    if spec:
        tokens = TokenRegistry.parse(spec)
        issued: dict[str, str] = {}
    else:
        tokens, issued = TokenRegistry.generate(
            [Identity(f"analyst@{t}.example", t, Role.ANALYST) for t in tenants]
            + [Identity(f"viewer@{t}.example", t, Role.VIEWER) for t in tenants]
        )
    app = create_app(workspaces, tokens, evaluation_path=args.evaluation)
    print(f"ready in {time.perf_counter() - started:.1f}s; state in {root}")
    for who, token in issued.items():
        print(f"  token for {who}: {token}")
    print(f"open http://{args.host}:{args.port}/", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        for workspace in workspaces.values():
            workspace.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
