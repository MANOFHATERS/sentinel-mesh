"""The Analyst Copilot's HTTP API and static front end (PRD F-10, Figure 2 Layer 6).

Everything the browser can do goes through the routes below, and every route is a
thin call into :class:`~sentinel.dashboard.workspace.Workspace`. There is no
business logic here that a test of the workspace would not also cover; what lives
here is the boundary — authentication, tenant resolution, input validation, error
mapping and response headers.

Error mapping
-------------
=====================================  ======  =====================================
Condition                              Status  Why this status
=====================================  ======  =====================================
no/unknown bearer token                401     with ``WWW-Authenticate: Bearer``
viewer tries to act                    403     authenticated, not authorised
unknown id, or another tenant's id     404     the same answer for both, so an id
                                               from another tenant is not confirmed
decision raced / scenario relaunched   409     the request was valid a moment ago
malformed body, extra fields           422     including an ``approver`` field
=====================================  ======  =====================================

Response headers
----------------
The page loads scripts and styles only from its own origin and runs no inline code,
so the Content-Security-Policy can be strict (no ``'unsafe-inline'``). That makes
an injected ``<script>`` inert even if some future view forgot to use
``textContent`` — defence in depth behind the rule in ``static/js/dom.js``. API
responses are ``no-store``: they contain incident data and approval state, and a
cached approval queue is a stale approval queue.
"""

# No ``from __future__ import annotations`` here, deliberately: FastAPI resolves
# string annotations against module globals, and the dependency aliases below
# (``Who``, ``Actor``, ``Ws``) are local to create_app, so postponed evaluation would
# silently turn them into required *query parameters*.

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from sentinel.dashboard import views
from sentinel.dashboard.auth import AuthError, Identity, TokenRegistry
from sentinel.dashboard.lab import model_report
from sentinel.dashboard.registry import WorkspaceUnavailable
from sentinel.dashboard.runs import Busy, RunManager, RunsUnavailable
from sentinel.dashboard.scenarios import ASSET_INVENTORY, ScenarioName
from sentinel.dashboard.sso import SsoService, install_sso
from sentinel.dashboard.workspace import (
    Conflict,
    NotFound,
    Workspace,
    WorkspaceError,
    workspace_features,
)
from sentinel.real.repos import RepoError, parse_github_url
from sentinel.real.service import RealData

__all__ = ["STATIC_DIR", "create_app"]

STATIC_DIR: Final[Path] = Path(__file__).parent / "static"

_SECURITY_HEADERS: Final[dict[str, str]] = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class Forbidden(WorkspaceError):
    """Authenticated, but the role does not permit the action."""


class DecisionBody(BaseModel):
    """An analyst's answer to the gate. Deliberately has no approver field."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=64)
    approved: StrictBool
    note: str = Field(default="", max_length=2000)


class RunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(pattern="^(retrain|eval_quick|eval_full|real_network|real_scan)$")
    seed: int = Field(default=20260928, ge=0, le=2**31 - 1)
    url: str | None = Field(default=None, max_length=300)
    limit: int | None = Field(default=None, ge=1_000, le=200_000)


class ReplayBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1, le=500)


def create_app(
    workspaces: Mapping[str, Workspace],
    tokens: TokenRegistry,
    *,
    evaluation_path: Path | None = None,
    sso: SsoService | None = None,
    runs: RunManager | None = None,
    real: RealData | None = None,
) -> FastAPI:
    """Build the app. Every tenant a token names must have a workspace."""
    missing = sorted(tokens.tenants - set(workspaces))
    if missing:
        raise AuthError(f"tokens name tenants with no workspace: {missing}")
    ready = workspaces.ready_items() if hasattr(workspaces, "ready_items") else workspaces.items()
    for tenant, workspace in ready:
        if workspace.tenant_id != tenant:
            raise AuthError(f"workspace for {tenant!r} serves tenant {workspace.tenant_id!r}")
    evaluation = evaluation_path or Path("data/artifacts/evaluation.json")

    app = FastAPI(
        title="Sentinel Mesh — Analyst Copilot",
        version="0.5.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def _headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        else:
            # Revalidate (ETag) on every load: a browser running last deploy's
            # approval UI against this deploy's API is a bug nobody can reproduce.
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(AuthError)
    async def _auth(_request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            {"error": "unauthorized", "detail": str(exc)},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(Forbidden)
    async def _forbidden(_request: Request, exc: Forbidden) -> JSONResponse:
        return JSONResponse({"error": "forbidden", "detail": str(exc)}, status_code=403)

    @app.exception_handler(NotFound)
    async def _not_found(_request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"error": "not_found", "detail": str(exc)}, status_code=404)

    @app.exception_handler(Conflict)
    async def _conflict(_request: Request, exc: Conflict) -> JSONResponse:
        return JSONResponse({"error": "conflict", "detail": str(exc)}, status_code=409)

    @app.exception_handler(WorkspaceUnavailable)
    async def _unavailable(_request: Request, exc: WorkspaceUnavailable) -> JSONResponse:
        return JSONResponse(
            {"error": "failed" if exc.failed else "preparing", "detail": str(exc)},
            status_code=503,
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(WorkspaceError)
    async def _workspace(_request: Request, exc: WorkspaceError) -> JSONResponse:
        return JSONResponse({"error": "bad_request", "detail": str(exc)}, status_code=400)

    # --- dependencies ------------------------------------------------------- #

    def identity(authorization: Annotated[str | None, Header()] = None) -> Identity:
        # Static tokens are API keys for machines; people arrive with an SSO session.
        try:
            return tokens.authenticate(authorization)
        except AuthError:
            if sso is None:
                raise
            return sso.sessions.authenticate(authorization)

    def workspace(who: Annotated[Identity, Depends(identity)]) -> Workspace:
        # Resolved from the identity alone: there is no tenant in any request.
        return workspaces[who.tenant_id]

    def actor(who: Annotated[Identity, Depends(identity)]) -> Identity:
        if not who.can_act:
            raise Forbidden(f"{who.principal} is a {who.role.value}; viewers cannot act")
        return who

    def requires(feature: str):
        def dependency(ws: Annotated[Workspace, Depends(workspace)]) -> Workspace:
            # A real workspace serves a page only if real data backs it; otherwise it refuses
            # rather than serve synthetic content under a real-data label.
            if not workspace_features(ws.models)[feature]:
                raise NotFound("not available in the real-data workspace: no real data behind it")
            return ws

        return dependency

    Who = Annotated[Identity, Depends(identity)]
    Actor = Annotated[Identity, Depends(actor)]
    Ws = Annotated[Workspace, Depends(workspace)]
    SupplyWs = Annotated[Workspace, Depends(requires("supply_chain"))]
    CodeWs = Annotated[Workspace, Depends(requires("code_scan"))]

    # --- routes ----------------------------------------------------------------- #

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    ALL_FEATURES = {
        "scenarios": True,
        "supply_chain": True,
        "code_scan": True,
        "models": True,
        "kb": False,
    }

    def describe_tenant(tenant: str) -> dict[str, Any]:
        if hasattr(workspaces, "describe"):
            return workspaces.describe(tenant)
        models = getattr(workspaces.get(tenant), "models", None)
        return {
            "mode": getattr(models, "mode", "synthetic"),
            "dataset": None,
            "state": "ready",
            "error": None,
            "features": ALL_FEATURES,
        }

    @app.get("/api/session")
    def session(who: Who) -> dict[str, Any]:
        info = describe_tenant(who.tenant_id)
        return {
            "principal": who.principal,
            "tenant_id": who.tenant_id,
            "role": who.role.value,
            "can_act": who.can_act,
            "data_mode": info["mode"],
            "dataset": info["dataset"],
            "workspace": info["state"],
            "workspace_error": info["error"],
            "features": info["features"],
        }

    @app.get("/api/auth/audit")
    def auth_audit(who: Who) -> dict[str, Any]:
        """Who signed in, was refused, refreshed or was deprovisioned — this tenant's rows."""
        if sso is None:
            return {"enabled": False, "intact": True, "items": []}
        return {
            "enabled": True,
            "intact": sso.audit.verify(),
            "items": sso.audit.events(tenant=who.tenant_id),
        }

    @app.get("/api/overview")
    def overview(ws: Ws) -> dict[str, Any]:
        return views.overview(ws)

    @app.get("/api/assets")
    def assets(_who: Who) -> list[dict[str, Any]]:
        return [
            {"address": a.address, "hostname": a.hostname, "role": a.role, "protected": a.protected}
            for a in ASSET_INVENTORY.values()
        ]

    @app.get("/api/scenarios")
    def scenarios(ws: Ws) -> list[dict[str, Any]]:
        return views.scenarios_view(ws)

    @app.post("/api/scenarios/{name}/launch")
    def launch(name: str, ws: Ws, who: Actor) -> dict[str, Any]:
        if ws.models.mode == "real":
            opened = ws.launch_real(name, launched_by=who.principal)
            return {"scenario": name, "incidents": list(opened)}
        try:
            scenario = ScenarioName(name)
        except ValueError as exc:
            raise NotFound(f"no scenario {name!r}") from exc
        opened = ws.launch(scenario, launched_by=who.principal)
        return {"scenario": scenario.value, "incidents": list(opened)}

    @app.get("/api/queue")
    def queue(ws: Ws) -> list[dict[str, Any]]:
        return views.queue_view(ws)

    @app.get("/api/incidents")
    def incidents(
        ws: Ws,
        status: Annotated[str | None, Query(max_length=32)] = None,
        kind: Annotated[str | None, Query(max_length=32)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> dict[str, Any]:
        return views.incidents_view(ws, status=status, kind=kind, limit=limit)

    @app.get("/api/incidents/{incident_id}")
    def incident(incident_id: str, ws: Ws) -> dict[str, Any]:
        return views.incident_detail(ws, incident_id)

    @app.post("/api/incidents/{incident_id}/decision")
    def decide(incident_id: str, body: DecisionBody, ws: Ws, who: Actor) -> dict[str, Any]:
        ws.decide(
            incident_id,
            action_id=body.action_id,
            approved=body.approved,
            approver=who.principal,
            note=body.note,
        )
        return views.incident_detail(ws, incident_id)

    @app.post("/api/incidents/{incident_id}/recover")
    def recover(incident_id: str, ws: Ws, _who: Actor) -> dict[str, Any]:
        ws.recover(incident_id)
        return views.incident_detail(ws, incident_id)

    @app.post("/api/feed/replay")
    def replay(body: ReplayBody, ws: Ws, _who: Actor) -> dict[str, Any]:
        return {**ws.replay(body.count), "remaining": ws.feed_remaining()}

    @app.get("/api/wire")
    def wire(ws: Ws) -> dict[str, Any]:
        return views.wire_view(ws)

    @app.get("/api/supply-chain/graph")
    def supply_graph(
        ws: SupplyWs,
        scope: Annotated[str, Query(max_length=96)] = "top",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> dict[str, Any]:
        if scope not in ("top", "all") and not scope.startswith("advisory:"):
            raise NotFound(f"no scope {scope!r}")
        return views.graph_view(ws, scope=scope, k=k)

    @app.get("/api/supply-chain/nodes/{node_id}")
    def supply_node(
        node_id: str,
        ws: SupplyWs,
        advisory: Annotated[str | None, Query(max_length=64)] = None,
    ) -> dict[str, Any]:
        explanation = ws.explain(node_id, advisory_id=advisory)
        issues = (ws.models.real_supply or {}).get("issues", {}).get(node_id, [])
        return {**views.node_view(explanation, ws.graph), "advisory_id": advisory, "issues": issues}

    @app.get("/api/code-scan")
    def code_scan(ws: CodeWs) -> dict[str, Any]:
        view = views.code_scan_view(ws)
        return {"available": view is not None, "scan": view}

    @app.get("/api/audit")
    def audit(
        ws: Ws,
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 200,
        subject: Annotated[str | None, Query(max_length=64)] = None,
    ) -> dict[str, Any]:
        records = ws.records(None if subject is None else [subject])
        page = [r for r in records if r.seq > after][:limit]
        return {"total": len(records), "items": [views.audit_view(r) for r in page]}

    @app.get("/api/audit/verify")
    def audit_verify(ws: Ws) -> dict[str, Any]:
        return {**views.audit_chain_view(ws), "ungated_executions": list(ws.ungated())}

    @app.get("/api/evaluation")
    def evaluation_report(_who: Who) -> dict[str, Any]:
        return views.evaluation_view(evaluation)

    @app.get("/api/runs")
    def runs_state(_who: Who) -> dict[str, Any]:
        if runs is None:
            return {"enabled": False, "available": {}, "busy": None, "jobs": {}}
        return {"enabled": True, **runs.latest()}

    @app.post("/api/runs", status_code=202)
    def start_run(body: RunBody, who: Actor) -> dict[str, Any]:
        if runs is None:
            raise NotFound("live runs are not enabled on this server")
        params: dict[str, Any] = {}
        if body.kind == "real_scan":
            try:
                parse_github_url(body.url or "")
            except RepoError as exc:
                raise WorkspaceError(str(exc)) from exc
            params["url"] = (body.url or "").strip()
        if body.kind == "real_network" and body.limit:
            params["limit"] = body.limit
        try:
            job = runs.start(body.kind, body.seed, by=who.principal, params=params)
        except Busy as exc:
            raise Conflict(str(exc)) from exc
        except RunsUnavailable as exc:
            raise NotFound(str(exc)) from exc
        return runs.describe(job)

    @app.post("/api/runs/{job_id}/cancel")
    def cancel_run(job_id: str, _who: Actor) -> dict[str, Any]:
        job = runs.cancel(job_id) if runs is not None else None
        if job is None:
            raise NotFound(f"no run {job_id!r}")
        return runs.describe(job)

    @app.get("/api/real/report")
    def real_report(ws: Ws) -> dict[str, Any]:
        """The real-data evaluation this workspace was built from (real workspaces only)."""
        if ws.models.mode != "real":
            raise NotFound("this is the synthetic workspace; it has no real-data report")
        return {"report": ws.models.real_report}

    @app.get("/api/real/status")
    def real_status(_who: Who) -> dict[str, Any]:
        if real is None:
            return {"enabled": False}
        return {"enabled": True, **real.status()}

    @app.get("/api/real/grounding")
    def real_grounding(_who: Who) -> dict[str, Any]:
        if real is None:
            raise NotFound("real data is not enabled on this server")
        return real.grounding()

    @app.get("/api/real/scanner-quality")
    def real_scanner_quality(_who: Who) -> dict[str, Any]:
        if real is None:
            raise NotFound("real data is not enabled on this server")
        return real.scanner_quality()

    @app.get("/api/real/kb/search")
    def real_kb_search(
        _who: Who,
        q: Annotated[str, Query(min_length=2, max_length=200)],
        k: Annotated[int, Query(ge=1, le=10)] = 5,
    ) -> dict[str, Any]:
        if real is None:
            raise NotFound("real data is not enabled on this server")
        return real.kb_search(q, k=k)

    @app.get("/api/models")
    def models_report(ws: Ws) -> dict[str, Any]:
        # Tenant-agnostic training records, but still behind the token: the page
        # says what this deployment trained, which is not public information.
        return model_report(ws.models, policy=ws.models.policy)

    # --- sign-in ------------------------------------------------------------------- #

    if sso is not None:
        install_sso(app, sso, static_tokens=len(tokens) > 0)
    else:

        @app.get("/auth/config")
        def auth_config() -> dict[str, Any]:
            return {"sso": False, "token_login": True, "mfa_required": False}

    # --- the page ------------------------------------------------------------------ #

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
