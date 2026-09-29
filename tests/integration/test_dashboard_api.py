"""F-10 acceptance: all three demo scenarios, end to end, through the HTTP API alone.

The PRD's criterion is *"all 3 demo scenarios completable end-to-end from this UI
alone"*. The UI is a static page that does nothing but call these routes, so the
routes are what is tested: :func:`test_f10_all_three_scenarios_complete_over_http`
signs in with a bearer token, launches each scenario, works the approval queue by
reading what the queue shows and posting decisions, and then checks the outcome on
the remote systems and in the audit chain — never by reaching into the workspace.

The rest of the file is the boundary the ledger asked for: authenticated,
tenant-scoped, every approval naming a real approver, untrusted text inert.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from sentinel.core.clock import SimulationClock
from sentinel.dashboard.app import create_app
from sentinel.dashboard.auth import Identity, Role, TokenRegistry
from sentinel.dashboard.workspace import Workspace

ANALYST = "analyst-token-" + "a" * 30
VIEWER = "viewer-token-" + "v" * 30
GLOBEX = "globex-token-" + "g" * 30


@pytest.fixture
def stack(mesh_models, tmp_path):
    clock = SimulationClock(datetime(2026, 9, 29, 9, 0, tzinfo=UTC))
    acme = Workspace(mesh_models, tenant_id="acme", workdir=tmp_path / "acme", clock=clock)
    globex = Workspace(mesh_models, tenant_id="globex", workdir=tmp_path / "globex",
                       clock=clock)
    tokens = TokenRegistry({
        ANALYST: Identity("maya@acme.example", "acme", Role.ANALYST),
        VIEWER: Identity("auditor@acme.example", "acme", Role.VIEWER),
        GLOBEX: Identity("ops@globex.example", "globex", Role.ANALYST),
    })
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "n_alerts": 20000, "dataset": "cic-ids2017-synthetic", "f03_auc_pass": True,
        "test": {"roc_auc": 0.997, "per_family_recall": {"botnet": 1.0}},
        "agents": {"f08_approval_gate_pass": True},
        "connectors": {"exactly_once_pass": False},
    }))
    app = create_app({"acme": acme, "globex": globex}, tokens, evaluation_path=evaluation)
    with TestClient(app) as client:
        yield client, acme, globex, clock
    acme.close()
    globex.close()


def auth(token: str = ANALYST) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def work_the_queue(client, clock, *, reject=lambda item: False, token=ANALYST):
    """Decide everything the queue shows, the way the UI does. Returns decisions made."""
    made = []
    while True:
        queue = client.get("/api/queue", headers=auth(token)).json()
        if not queue:
            return made
        item = queue[0]
        detail = client.get(f"/api/incidents/{item['incident_id']}", headers=auth(token)).json()
        assert detail["interrupt"]["subject_id"] == item["pending_action"]["action_id"]
        approved = not reject(item)
        clock.advance(20.0)
        response = client.post(
            f"/api/incidents/{item['incident_id']}/decision",
            headers=auth(token),
            json={"action_id": item["pending_action"]["action_id"], "approved": approved,
                  "note": "reviewed in the dashboard"},
        )
        assert response.status_code == 200, response.text
        made.append((item, approved, response.json()))


# --------------------------------------------------------------------------- #
# F-10
# --------------------------------------------------------------------------- #


def test_f10_all_three_scenarios_complete_over_http(stack):
    client, _acme, _globex, clock = stack

    scenarios = client.get("/api/scenarios", headers=auth()).json()
    assert [s["name"] for s in scenarios] == [
        "phishing-lateral", "vendor-cve", "malicious-package",
    ]
    assert not any(s["launched"] for s in scenarios)
    fresh = client.get("/api/overview", headers=auth()).json()
    assert fresh["audit"]["status"] == "empty", "a fresh workspace is empty, not broken"

    for scenario in scenarios:
        response = client.post(f"/api/scenarios/{scenario['name']}/launch", headers=auth())
        assert response.status_code == 200, response.text
        assert response.json()["incidents"]

    queue = client.get("/api/queue", headers=auth()).json()
    assert len(queue) == 7, [q["caption"] for q in queue]
    assert {q["kind"] for q in queue} == {"incident", "code_scan", "supply_chain"}

    # The redundant block of the already-isolated workstation is rejected, as in the
    # walkthrough; everything else is approved.
    def redundant(item):
        action = item["pending_action"]
        return action["action_type"] == "block_ip" and action["target"] == "10.20.4.17"

    decisions = work_the_queue(client, clock, reject=redundant)
    assert len(decisions) == 7
    assert sum(1 for _, approved, _ in decisions if not approved) == 1

    scenarios = client.get("/api/scenarios", headers=auth()).json()
    for scenario in scenarios:
        assert scenario["launched"] and scenario["complete"], scenario
        assert scenario["failed_runs"] == 0
        assert scenario["waiting"] == 0
    assert client.get("/api/queue", headers=auth()).json() == []

    # What actually happened, read from the remote systems' side of the wire.
    wire = client.get("/api/wire", headers=auth()).json()
    assert wire["remote"]["wazuh"]["isolated_hosts"] == ["10.20.4.17", "10.20.8.30"]
    assert wire["remote"]["wazuh"]["blocked_addresses"] == []
    assert len(wire["remote"]["github"]["issues"]) == 2
    assert [p["draft"] for p in wire["remote"]["github"]["pulls"]] == [True]
    assert wire["remote"]["github"]["merges"] == 0

    # Part 4 finding 5: the failed action inside a completed run is visible...
    assert len(wire["failed_actions"]) == 1
    failed = wire["failed_actions"][0]
    assert (failed["action_type"], failed["target"]) == ("block_ip", "10.20.0.5")
    assert failed["approved_by"] == "maya@acme.example"
    # ...and the router's refusal is shown with its reason.
    assert len(wire["refusals"]) == 1
    assert "protected" in wire["refusals"][0]["reason"]
    assert "10.20.0.0/28" in wire["refusals"][0]["reason"]

    overview = client.get("/api/overview", headers=auth()).json()
    assert overview["audit"]["ok"] and overview["audit"]["status"] == "verified"
    assert overview["ungated_executions"] == []
    assert overview["queue"] == 0 and overview["stalled"] == 0
    assert overview["failed_actions"] == 1 and overview["refusals"] == 1
    assert overview["mttd_seconds"] is not None and overview["mttd_seconds"] < 30
    assert overview["mttc_seconds"] is not None and overview["mttc_seconds"] < 180

    verify = client.get("/api/audit/verify", headers=auth()).json()
    assert verify["ok"] and verify["ungated_executions"] == []

    # Every approval in the chain names the analyst who clicked.
    audit = client.get("/api/audit?limit=500", headers=auth()).json()
    grants = [r for r in audit["items"] if r["event_type"] in ("approval_granted",
                                                                 "approval_denied")]
    assert len(grants) == 7
    assert {r["payload"]["approver"] for r in grants} == {"maya@acme.example"}


def test_incident_detail_carries_what_the_analyst_needs(stack):
    client, *_ = stack
    ids = client.post("/api/scenarios/phishing-lateral/launch", headers=auth()).json()
    detail = client.get(f"/api/incidents/{ids['incidents'][0]}", headers=auth()).json()
    assert detail["status"] == "awaiting_approval"
    assert detail["triage"]["decision"] == "escalate"
    assert detail["report"]["grounded"] and detail["report"]["claims"]
    refs = {e["ref"] for e in detail["report"]["evidence"]}
    for claim in detail["report"]["claims"]:
        assert set(claim["refs"]) <= refs, "every claim cites evidence in the report"
    assert detail["alert"]["untrusted"] is True
    assert detail["alert"]["asset"]["hostname"] == "ws-fin-07"
    assert detail["checkpoints"]["verified"] is True
    assert [s["node"] for s in detail["history"]] == ["triage", "investigate", "contain"]
    assert {r["event_type"] for r in detail["timeline"]} >= {
        "triage_decided", "investigation_completed", "action_proposed", "approval_requested",
    }
    assert detail["pending_action"]["target_asset"]["hostname"] == "ws-fin-07"


def test_supply_chain_and_code_scan_views(stack):
    client, *_ = stack
    client.post("/api/scenarios/vendor-cve/launch", headers=auth())
    graph = client.get("/api/supply-chain/graph?scope=advisory:CVE-2026-41822",
                       headers=auth()).json()
    package = graph["advisory"]["package_id"]
    ids = {n["id"] for n in graph["nodes"]}
    assert package in ids
    assert any(path[0] == package and len(path) == 5 for path in graph["highlight"]), (
        "the fourth-order path is highlighted"
    )
    for edge in graph["edges"]:
        assert edge["source"] in ids and edge["target"] in ids
    org = next(n for n in graph["nodes"] if n["kind"] == "organization")
    node = client.get(f"/api/supply-chain/nodes/{org['id']}?advisory=CVE-2026-41822",
                      headers=auth()).json()
    assert node["paths"] and node["paths"][0]["nodes"][0] == package
    assert node["paths"][0]["hops"] == 4

    top = client.get("/api/supply-chain/graph", headers=auth()).json()
    assert len(top["flagged"]) == 10
    whole = client.get("/api/supply-chain/graph?scope=all", headers=auth()).json()
    assert len(whole["nodes"]) == 500 and len(whole["edges"]) == 1080

    scan = client.get("/api/code-scan", headers=auth()).json()
    assert scan["available"]
    body = scan["scan"]
    assert body["findings"] and body["draft"]["patches"]
    assert body["digest_matches"], "the diff shown is the diff the approval binds"
    assert body["status"] == "awaiting_approval"


def test_evaluation_report_is_read_not_recomputed(stack):
    client, *_ = stack
    report = client.get("/api/evaluation", headers=auth()).json()
    assert report["available"] and report["n_alerts"] == 20000
    gates = {g["gate"]: g["passed"] for g in report["gates"]}
    assert gates == {"f03_auc_pass": True, "agents.f08_approval_gate_pass": True,
                     "connectors.exactly_once_pass": False}
    assert report["passed"] == 2


# --------------------------------------------------------------------------- #
# The boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path", ["/api/session", "/api/overview", "/api/queue", "/api/incidents", "/api/wire",
             "/api/audit", "/api/evaluation", "/api/supply-chain/graph", "/api/code-scan"],
)
def test_every_api_route_requires_a_token(stack, path):
    client, *_ = stack
    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": ANALYST}):
        response = client.get(path, headers=headers)
        assert response.status_code == 401, (path, headers)
        assert response.headers["www-authenticate"] == "Bearer"


def test_health_is_the_only_open_route(stack):
    client, *_ = stack
    assert client.get("/api/health").json() == {"ok": True}


def test_session_reports_the_token_s_identity(stack):
    client, *_ = stack
    assert client.get("/api/session", headers=auth()).json() == {
        "principal": "maya@acme.example", "tenant_id": "acme", "role": "analyst",
        "can_act": True,
    }


def test_a_viewer_reads_but_cannot_act(stack):
    client, acme, *_ = stack
    assert client.get("/api/overview", headers=auth(VIEWER)).status_code == 200
    assert client.post("/api/scenarios/phishing-lateral/launch",
                       headers=auth(VIEWER)).status_code == 403
    assert client.post("/api/feed/replay", headers=auth(VIEWER),
                       json={"count": 5}).status_code == 403
    ids = client.post("/api/scenarios/phishing-lateral/launch", headers=auth()).json()
    thread = ids["incidents"][0]
    action = client.get(f"/api/incidents/{thread}", headers=auth(VIEWER)).json()[
        "pending_action"]["action_id"]
    response = client.post(f"/api/incidents/{thread}/decision", headers=auth(VIEWER),
                           json={"action_id": action, "approved": True})
    assert response.status_code == 403
    assert client.post(f"/api/incidents/{thread}/recover",
                       headers=auth(VIEWER)).status_code == 403
    assert acme.sandbox.wazuh.executed == []


def test_the_approver_comes_from_the_token_never_the_body(stack):
    client, acme, *_ = stack
    thread = client.post("/api/scenarios/phishing-lateral/launch",
                         headers=auth()).json()["incidents"][0]
    action = client.get(f"/api/incidents/{thread}", headers=auth()).json()[
        "pending_action"]["action_id"]
    forged = client.post(
        f"/api/incidents/{thread}/decision", headers=auth(),
        json={"action_id": action, "approved": True, "approver": "ciso@acme.example"},
    )
    assert forged.status_code == 422
    assert acme.state(thread).is_waiting
    ok = client.post(f"/api/incidents/{thread}/decision", headers=auth(),
                     json={"action_id": action, "approved": True})
    assert ok.status_code == 200
    assert ok.json()["actions_detail"][-1]["approved_by"] == "maya@acme.example"


@pytest.mark.parametrize(
    "body",
    [{"action_id": "x", "approved": "yes"}, {"action_id": "x", "approved": 1},
     {"action_id": "x"}, {"approved": True}, {"action_id": "", "approved": True},
     {"action_id": "x", "approved": True, "note": "n" * 2001}],
)
def test_malformed_decisions_are_rejected(stack, body):
    client, *_ = stack
    thread = client.post("/api/scenarios/phishing-lateral/launch",
                         headers=auth()).json()["incidents"][0]
    response = client.post(f"/api/incidents/{thread}/decision", headers=auth(), json=body)
    assert response.status_code == 422


def test_conflicts_are_409(stack):
    client, *_ = stack
    thread = client.post("/api/scenarios/phishing-lateral/launch",
                         headers=auth()).json()["incidents"][0]
    assert client.post("/api/scenarios/phishing-lateral/launch",
                       headers=auth()).status_code == 409
    stale = client.post(f"/api/incidents/{thread}/decision", headers=auth(),
                        json={"action_id": "not-the-pending-one", "approved": True})
    assert stale.status_code == 409
    action = client.get(f"/api/incidents/{thread}", headers=auth()).json()[
        "pending_action"]["action_id"]
    first = client.post(f"/api/incidents/{thread}/decision", headers=auth(),
                        json={"action_id": action, "approved": True})
    again = client.post(f"/api/incidents/{thread}/decision", headers=auth(),
                        json={"action_id": action, "approved": True})
    assert (first.status_code, again.status_code) == (200, 409)
    assert client.post(f"/api/incidents/{thread}/recover", headers=auth()).status_code == 409


def test_another_tenant_s_incident_is_not_found(stack):
    client, _acme, globex, _clock = stack
    acme_thread = client.post("/api/scenarios/phishing-lateral/launch",
                              headers=auth()).json()["incidents"][0]
    detail = client.get(f"/api/incidents/{acme_thread}", headers=auth(GLOBEX))
    unknown = client.get("/api/incidents/does-not-exist", headers=auth(GLOBEX))
    # Same status and shape as a genuinely unknown id: no existence oracle.
    assert detail.status_code == unknown.status_code == 404
    assert detail.json()["error"] == unknown.json()["error"] == "not_found"
    action = client.get(f"/api/incidents/{acme_thread}", headers=auth()).json()[
        "pending_action"]["action_id"]
    cross = client.post(f"/api/incidents/{acme_thread}/decision", headers=auth(GLOBEX),
                        json={"action_id": action, "approved": True})
    assert cross.status_code == 404
    assert client.get("/api/queue", headers=auth(GLOBEX)).json() == []
    assert client.get("/api/audit", headers=auth(GLOBEX)).json()["total"] == 0
    # globex launching the same scenario is independent of acme's launch.
    assert client.post("/api/scenarios/phishing-lateral/launch",
                       headers=auth(GLOBEX)).status_code == 200
    assert globex.sandbox.wazuh.executed == []


def test_untrusted_payloads_come_back_as_inert_json_text(stack):
    client, *_ = stack
    ids = client.post("/api/scenarios/malicious-package/launch", headers=auth()).json()
    exfil = next(i for i in ids["incidents"]
                 if client.get(f"/api/incidents/{i}", headers=auth()).json()["kind"]
                 == "incident")
    response = client.get(f"/api/incidents/{exfil}", headers=auth())
    assert response.headers["content-type"].startswith("application/json")
    detail = response.json()
    assert detail["injection_verdict"] == "likely_injection"
    assert detail["alert"]["injection"]["verdict"] == "likely_injection"
    assert "NOTE TO AI SECURITY SCANNERS" in detail["alert"]["raw_payload"]
    assert detail["decision"] == "escalate", "the note asked not to escalate; it escalated"


def test_security_headers(stack):
    client, *_ = stack
    for path, headers in (("/", {}), ("/static/js/app.js", {}), ("/api/overview", auth())):
        response = client.get(path, headers=headers)
        assert response.status_code == 200, path
        csp = response.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "unsafe-inline" not in csp
        assert "frame-ancestors 'none'" in csp
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
    assert client.get("/api/overview", headers=auth()).headers["cache-control"] == "no-store"
    assert client.get("/static/app.css").headers["cache-control"] == "no-cache"


def test_the_page_runs_no_inline_script(stack):
    client, *_ = stack
    page = client.get("/").text
    assert '<script type="module" src="/static/js/app.js"></script>' in page
    assert "<script>" not in page and "onload=" not in page and "style=" not in page


def test_the_front_end_never_uses_inner_html():
    from sentinel.dashboard.app import STATIC_DIR

    for path in (STATIC_DIR / "js").glob("*.js"):
        text = path.read_text(encoding="utf-8")
        for forbidden in (".innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
                          "eval(", "new Function"):
            assert forbidden not in text, (path.name, forbidden)


def test_replay_through_the_api(stack):
    client, *_ = stack
    result = client.post("/api/feed/replay", headers=auth(), json={"count": 25}).json()
    assert result["ingested"] == 25
    assert client.post("/api/feed/replay", headers=auth(),
                       json={"count": 501}).status_code == 422
    assert client.post("/api/feed/replay", headers=auth(),
                       json={"count": 5, "tenant": "globex"}).status_code == 422
    incidents = client.get("/api/incidents?limit=1000", headers=auth()).json()
    assert incidents["total"] == 25


def test_unknown_things_are_404(stack):
    client, *_ = stack
    assert client.post("/api/scenarios/nope/launch", headers=auth()).status_code == 404
    assert client.get("/api/supply-chain/nodes/pkg-9999", headers=auth()).status_code == 404
    assert client.get("/api/supply-chain/graph?scope=advisory:CVE-0",
                      headers=auth()).status_code == 404
    assert client.get("/api/supply-chain/graph?scope=weird", headers=auth()).status_code == 404


def test_tokens_for_a_tenant_without_a_workspace_are_a_configuration_error(mesh_models,
                                                                          tmp_path):
    from sentinel.dashboard.auth import AuthError

    with Workspace(mesh_models, tenant_id="acme", workdir=tmp_path / "w") as acme:
        tokens = TokenRegistry({GLOBEX: Identity("ops@globex.example", "globex",
                                                 Role.ANALYST)})
        with pytest.raises(AuthError, match="no workspace"):
            create_app({"acme": acme}, tokens)
        with pytest.raises(AuthError, match="serves tenant"):
            create_app({"globex": acme}, tokens)


# --------------------------------------------------------------------------- #
# Part 5.1: the response policy in the live flow, and the Models page
# --------------------------------------------------------------------------- #


def test_each_incident_records_the_policy_s_reasoning(stack):
    client, *_ = stack
    ids = client.post("/api/scenarios/phishing-lateral/launch", headers=auth()).json()
    for incident_id in ids["incidents"]:
        detail = client.get(f"/api/incidents/{incident_id}", headers=auth()).json()
        policy = detail["policy"]
        assert policy is not None and policy["fitted"] is True
        assert policy["response"] == "escalate"
        assert policy["choice"] in ("escalate", "monitor", "dismiss")
        assert set(policy["expected"]) == {"escalate", "monitor", "dismiss"}, (
            "at the recommend tier auto-contain is masked out of the policy entirely"
        )
        assert 0.0 <= policy["confidence"] <= 1.0
        assert policy["exploratory"] is False, "served greedily"
        assert policy["seq"] >= 1, "read from the audit chain"


def test_models_page_reports_live_training(stack):
    client, *_ = stack
    assert client.get("/api/models").status_code == 401
    client.post("/api/scenarios/phishing-lateral/launch", headers=auth())
    report = client.get("/api/models", headers=auth()).json()
    assert report["autoencoder"]["available"]
    assert len(report["autoencoder"]["train_loss"]) >= 2
    assert {d["name"] for d in report["autoencoder"]["detectors"]} >= {"denoising_autoencoder"}
    gnn = report["gnn"]
    assert len(gnn["train_loss"]) == len(gnn["validation_loss"]) >= 2
    assert 0.0 <= gnn["evaluation"]["gnn_top_k_precision"] <= 1.0
    policy = report["policy"]
    assert policy["live_decisions"] >= 3, "the scenario's three incidents went through it"
    assert policy["policy_total_regret"] < policy["no_learning_total_regret"]
    assert len(policy["curve"]["policy"]) == len(policy["curve"]["no_learning"])
    # The fixture never starts the background study, and the page says so honestly.
    assert report["diffusion"]["status"] == "pending"
    assert report["diffusion"]["results"] == []
