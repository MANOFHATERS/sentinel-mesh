"""The real-data workspace: a login decides which data the whole dashboard shows.

The fixtures are generated (a CSV in UNSW-NB15's layout, a small STIX bundle); they exist so
these tests need no download and are never presented as real data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from sentinel.agents.triage import TECHNIQUE_BY_FAMILY
from sentinel.core.clock import SimulationClock
from sentinel.dashboard import views
from sentinel.dashboard.app import create_app
from sentinel.dashboard.auth import Identity, Role, TokenRegistry
from sentinel.dashboard.registry import WorkspaceRegistry, WorkspaceUnavailable
from sentinel.dashboard.scenarios import ScenarioName
from sentinel.dashboard.workspace import MeshModels, Workspace, WorkspaceError
from tests.unit.test_real_data import bundle, technique, write_unsw

ACME = {"Authorization": "Bearer " + "acme-analyst-token-" + "a" * 30}
REAL_ANALYST = {"Authorization": "Bearer " + "real-analyst-token-" + "r" * 30}
REAL_VIEWER = {"Authorization": "Bearer " + "real-viewer-token-" + "w" * 30}


def tokens() -> TokenRegistry:
    return TokenRegistry(
        {
            "acme-analyst-token-" + "a" * 30: Identity("maya@acme.example", "acme", Role.ANALYST),
            "real-analyst-token-" + "r" * 30: Identity(
                "real.analyst@acme.example", "real", Role.ANALYST
            ),
            "real-viewer-token-" + "w" * 30: Identity(
                "real.auditor@acme.example", "real", Role.VIEWER
            ),
        }
    )


def stix_for_workspace():
    """Every technique the triage families map to, so an investigation can cite its mapping."""
    ids = sorted(set(TECHNIQUE_BY_FAMILY.values()) | {"T1566", "T1059"})
    return bundle(*[technique(tid, f"Technique {tid}") for tid in ids])


@pytest.fixture(scope="module")
def real_models(mesh_models, tmp_path_factory):
    folder = tmp_path_factory.mktemp("real")
    dataset = write_unsw(folder / "UNSW_NB15_training-set.csv", rows=1600)
    attack = folder / "enterprise-attack.json"
    attack.write_text(json.dumps(stix_for_workspace()))
    return MeshModels.build_real(
        mesh_models, dataset_path=dataset, attack_path=attack, seed=11, limit=1400
    )


@pytest.fixture
def real_workspace(real_models, tmp_path):
    workspace = Workspace(real_models, tenant_id="real", workdir=tmp_path / "real")
    yield workspace
    workspace.close()


# --------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------- #


def test_real_models_replace_the_data_and_leave_the_synthetic_ones_alone(mesh_models, real_models):
    assert real_models.mode == "real" and mesh_models.mode == "synthetic"
    assert real_models.story_flows == {} and mesh_models.story_flows
    assert real_models.triage_model is not mesh_models.triage_model
    assert real_models.kb is not mesh_models.kb and real_models.feed != mesh_models.feed
    assert real_models.real_report["dataset"] == "unsw-nb15"
    assert len(real_models.feed) == real_models.real_report["test_flows"]
    assert all(a.dataset == "unsw-nb15" and a.tenant_id == "real" for a in real_models.feed)


def test_the_real_feed_mixes_benign_and_attack_from_the_first_flows(real_models):
    head = real_models.feed[:40]
    assert {a.ground_truth_label == "benign" for a in head} == {True, False}


def test_real_flows_run_through_the_whole_incident_pipeline(real_workspace):
    outcome = real_workspace.replay(80)
    assert outcome["ingested"] == 80 and outcome["failed"] == 0
    overview = views.overview(real_workspace)
    assert overview["incidents"] == 80 and overview["scenarios"] == []
    assert overview["audit"]["status"] == "verified" and overview["ungated_executions"] == []
    reports = [s.report for _r, s in real_workspace.states() if s.report is not None]
    assert reports, "an escalated real flow should have been investigated"
    kb = real_workspace.models.kb
    # A family with no ATT&CK mapping (a generic exploit, say) cites only the alert's own fields;
    # a mapped one cites the catalogue. Whatever is cited must resolve.
    refs = [str(e.ref) for r in reports for e in r.evidence if str(e.ref).startswith("kb://")]
    assert refs, "some investigation should cite the catalogue"
    assert kb.resolves_all(refs), "every catalogue citation must resolve"


def test_a_real_workspace_has_no_scripted_scenarios(real_workspace):
    with pytest.raises(WorkspaceError, match="real data"):
        real_workspace.launch(next(iter(ScenarioName)), launched_by="maya")


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_a_declared_workspace_is_preparing_until_it_is_ready():
    registry = WorkspaceRegistry({})
    registry.declare("real", mode="real", dataset="UNSW-NB15 (real)")
    assert "real" in registry and list(registry) == ["real"] and len(registry) == 1
    with pytest.raises(WorkspaceUnavailable, match="being prepared") as info:
        registry["real"]
    assert not info.value.failed
    assert registry.describe("real") == {
        "mode": "real",
        "dataset": "UNSW-NB15 (real)",
        "state": "preparing",
        "error": None,
    }
    assert registry.ready_items() == []
    registry.set_failed("real", "disk full")
    with pytest.raises(WorkspaceUnavailable, match="could not be built") as info:
        registry["real"]
    assert info.value.failed and registry.describe("real")["state"] == "failed"
    with pytest.raises(KeyError):
        registry["nobody"]


def test_a_ready_workspace_is_served_and_described(real_workspace):
    registry = WorkspaceRegistry({})
    registry.declare("real", mode="real", dataset="x")
    registry.set_ready("real", real_workspace)
    assert registry["real"] is real_workspace and registry.ready_items()[0][0] == "real"
    info = registry.describe("real")
    assert info["mode"] == "real" and info["dataset"] == "UNSW-NB15" and info["state"] == "ready"
    assert registry.describe("unknown")["mode"] == "synthetic"


# --------------------------------------------------------------------------- #
# The API: one login system, different data
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_worlds(mesh_models, real_models, tmp_path):
    clock = SimulationClock(datetime(2026, 9, 29, 9, 0, tzinfo=UTC))
    acme = Workspace(mesh_models, tenant_id="acme", workdir=tmp_path / "acme", clock=clock)
    real = Workspace(real_models, tenant_id="real", workdir=tmp_path / "real", clock=clock)
    registry = WorkspaceRegistry({"acme": acme, "real": real})
    with TestClient(create_app(registry, tokens())) as client:
        yield client
    acme.close()
    real.close()


def test_the_session_says_which_data_this_login_sees(two_worlds):
    real = two_worlds.get("/api/session", headers=REAL_ANALYST).json()
    assert real["data_mode"] == "real" and real["dataset"] == "UNSW-NB15"
    assert real["workspace"] == "ready"
    synthetic = two_worlds.get("/api/session", headers=ACME).json()
    assert synthetic["data_mode"] == "synthetic" and synthetic["dataset"] is None


@pytest.mark.parametrize(
    "path",
    [
        "/api/scenarios",
        "/api/supply-chain/graph",
        "/api/supply-chain/nodes/x",
        "/api/code-scan",
        "/api/models",
    ],
)
def test_a_real_workspace_refuses_what_has_no_real_data_behind_it(two_worlds, path):
    refused = two_worlds.get(path, headers=REAL_ANALYST)
    assert refused.status_code == 404
    assert "real-data workspace" in refused.json()["detail"]


def test_the_synthetic_login_is_untouched_by_the_real_workspace(two_worlds):
    overview = two_worlds.get("/api/overview", headers=ACME).json()
    assert len(overview["scenarios"]) == 3
    assert two_worlds.get("/api/scenarios", headers=ACME).status_code == 200
    assert two_worlds.get("/api/real/report", headers=ACME).status_code == 404


def test_the_real_login_gets_the_real_report_and_a_viewer_cannot_act(two_worlds):
    report = two_worlds.get("/api/real/report", headers=REAL_VIEWER).json()["report"]
    assert report["dataset"] == "unsw-nb15" and report["metrics"]["roc_auc"] is not None
    denied = two_worlds.post("/api/feed/replay", json={"count": 5}, headers=REAL_VIEWER)
    assert denied.status_code == 403
    replay = two_worlds.post("/api/feed/replay", json={"count": 5}, headers=REAL_ANALYST)
    assert replay.status_code == 200 and replay.json()["ingested"] == 5
    # the tenants are isolated: the real replay is invisible to the synthetic login
    assert two_worlds.get("/api/overview", headers=ACME).json()["incidents"] == 0


def test_a_workspace_still_being_built_answers_503_and_the_login_still_works():
    registry = WorkspaceRegistry({})
    registry.declare("real", mode="real", dataset="UNSW-NB15 (real)")
    only_real = TokenRegistry(
        {
            "real-analyst-token-" + "r" * 30: Identity(
                "real.analyst@acme.example", "real", Role.ANALYST
            )
        }
    )
    with TestClient(create_app(registry, only_real)) as client:
        session = client.get("/api/session", headers=REAL_ANALYST).json()
        assert session["workspace"] == "preparing" and session["data_mode"] == "real"
        busy = client.get("/api/overview", headers=REAL_ANALYST)
        assert busy.status_code == 503 and busy.json()["error"] == "preparing"
        assert busy.headers["retry-after"] == "5"
        registry.set_failed("real", "no space left")
        failed = client.get("/api/overview", headers=REAL_ANALYST)
        assert failed.status_code == 503 and failed.json()["error"] == "failed"
        assert client.get("/api/session", headers=REAL_ANALYST).json()["workspace"] == "failed"


# --------------------------------------------------------------------------- #
# What turns the real workspace on
# --------------------------------------------------------------------------- #


def test_the_real_workspace_is_offered_only_when_the_real_files_are_on_disk(tmp_path):
    from sentinel.dashboard.__main__ import _real_files

    assert _real_files(tmp_path) is None
    folder = tmp_path / "data" / "raw" / "unsw-nb15"
    folder.mkdir(parents=True)
    write_unsw(folder / "UNSW_NB15_training-set.csv", rows=10)
    assert _real_files(tmp_path) is None  # the dataset alone is not enough
    (tmp_path / "data" / "real").mkdir(parents=True)
    (tmp_path / "data" / "real" / "enterprise-attack.json").write_text("{}")
    dataset, attack, label = _real_files(tmp_path)
    assert dataset.name == "UNSW_NB15_training-set.csv" and label == "UNSW-NB15"
    assert attack.name == "enterprise-attack.json"


def test_the_demo_identity_provider_has_separate_real_users():
    from sentinel.dashboard.devidp import DEMO_USERS, REAL_USERS

    assert {u.tenant for u in REAL_USERS} == {"real"}
    assert {u.tenant for u in DEMO_USERS} == {"acme"}
    assert {u.groups for u in REAL_USERS} == {("SOC-Analyst",), ("Auditor",)}
    assert all(u.mfa for u in REAL_USERS)
