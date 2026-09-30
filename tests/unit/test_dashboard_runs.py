"""Live runs: one at a time, isolated, never overwrites the saved report."""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sentinel.dashboard.app import create_app
from sentinel.dashboard.auth import Identity, Role, TokenRegistry
from sentinel.dashboard.runs import Busy, RunKind, RunManager, RunsUnavailable

ANALYST = "analyst-token-" + "a" * 30
VIEWER = "viewer-token-" + "v" * 30


def wait(job, timeout: float = 30.0):
    end = time.time() + timeout
    while job.state == "running" and time.time() < end:
        time.sleep(0.02)
    return job


def fake_script(tmp_path: Path, body: str) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    path = scripts / "evaluate.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


WRITES_REPORT = """
    import argparse, json, sys
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int)
    a, _ = p.parse_known_args()
    print("stage one"); print("stage two")
    a.out.write_text(json.dumps({"seed": a.seed, "argv": sys.argv[1:]}))
    sys.exit(1 if a.seed == 13 else 0)
"""


def manager(tmp_path, script: Path | None, retrain=None) -> RunManager:
    return RunManager(
        workdir=tmp_path / "runs",
        retrain=retrain or (lambda seed: {"seed": seed}),
        read_evaluation=lambda path: json.loads(path.read_text()),
        script=script,
    )


def test_retrain_runs_in_process_and_reports(tmp_path):
    m = manager(tmp_path, None, retrain=lambda seed: {"seed": seed, "gnn": "ok"})
    job = wait(m.start(RunKind.RETRAIN, 7, by="maya"))
    assert job.state == "done" and job.result == {"seed": 7, "gnn": "ok"}
    assert m.latest()["busy"] is None


def test_a_failed_retrain_is_reported_not_raised(tmp_path):
    def boom(seed):
        raise RuntimeError("bad seed")

    m = manager(tmp_path, None, retrain=boom)
    job = wait(m.start(RunKind.RETRAIN, 1, by="maya"))
    assert job.state == "failed" and "RuntimeError: bad seed" in job.error
    assert m.latest()["busy"] is None  # a failure frees the slot


def test_evaluation_runs_the_script_and_reads_its_own_report(tmp_path):
    m = manager(tmp_path, fake_script(tmp_path, WRITES_REPORT))
    job = wait(m.start(RunKind.QUICK, 5, by="maya"))
    assert job.state == "done" and job.exit_code == 0
    assert job.result["seed"] == 5
    argv = job.result["argv"]
    assert "--n" in argv and argv[argv.index("--n") + 1] == "5000"
    assert list(job.log) == ["stage one", "stage two"]


def test_a_failed_gate_is_a_result_not_an_error(tmp_path):
    # the real evaluation exits non-zero when a gate fails, but still writes its report
    m = manager(tmp_path, fake_script(tmp_path, WRITES_REPORT))
    job = wait(m.start(RunKind.FULL, 13, by="maya"))
    assert job.state == "done" and job.exit_code == 1 and job.result["seed"] == 13


def test_no_report_means_failed(tmp_path):
    m = manager(tmp_path, fake_script(tmp_path, "import sys; sys.exit(3)"))
    job = wait(m.start(RunKind.QUICK, 1, by="maya"))
    assert job.state == "failed" and "exited 3" in job.error


def test_only_one_run_at_a_time_and_the_slot_frees_afterwards(tmp_path):
    m = manager(tmp_path, fake_script(tmp_path, "import time; time.sleep(30)"))
    first = m.start(RunKind.QUICK, 1, by="maya")
    with pytest.raises(Busy):
        m.start(RunKind.RETRAIN, 2, by="omar")
    m.cancel(first.id)
    wait(first)
    assert first.state == "cancelled"
    assert wait(m.start(RunKind.RETRAIN, 2, by="maya")).state == "done"


def test_seed_and_kind_are_validated(tmp_path):
    m = manager(tmp_path, None)
    for bad in (-1, 2**31, True, "5", 1.5):
        with pytest.raises(ValueError):
            m.start(RunKind.RETRAIN, bad, by="maya")
    with pytest.raises(ValueError):
        m.start("rm -rf", 1, by="maya")


def test_evaluation_is_unavailable_without_the_script(tmp_path):
    m = manager(tmp_path, tmp_path / "missing.py")
    state = m.latest()["available"]
    assert (state["retrain"], state["eval_quick"], state["eval_full"]) == (True, False, False)
    assert state["real_network"] is False and state["real_scan"] is False  # no handler wired
    with pytest.raises(RunsUnavailable):
        m.start(RunKind.QUICK, 1, by="maya")


def test_a_run_never_touches_the_saved_report(tmp_path):
    saved = tmp_path / "evaluation.json"
    saved.write_text('{"saved": true}')
    m = manager(tmp_path, fake_script(tmp_path, WRITES_REPORT))
    wait(m.start(RunKind.QUICK, 5, by="maya"))
    assert saved.read_text() == '{"saved": true}'


def test_full_run_uses_every_gate_and_quick_does_not(tmp_path):
    m = manager(tmp_path, fake_script(tmp_path, WRITES_REPORT))
    full = wait(m.start(RunKind.FULL, 1, by="maya")).result["argv"]
    quick = wait(m.start(RunKind.QUICK, 1, by="maya")).result["argv"]
    for flag in ("--codescan", "--connectors", "--dashboard", "--agents", "--cross-dataset"):
        assert flag in full and flag not in quick
    assert full[full.index("--n") + 1] == "20000"


def test_the_script_runs_under_this_interpreter(tmp_path):
    assert manager(tmp_path, None)._python == sys.executable


# --------------------------------------------------------------------------- #
# The API
# --------------------------------------------------------------------------- #


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def build_app(m: RunManager | None):
    tokens = TokenRegistry(
        {
            ANALYST: Identity("maya@acme.example", "acme", Role.ANALYST),
            VIEWER: Identity("omar@acme.example", "acme", Role.VIEWER),
        }
    )
    return create_app({"acme": SimpleNamespace(tenant_id="acme")}, tokens, runs=m)


@pytest.fixture
def api(tmp_path):
    m = manager(tmp_path, fake_script(tmp_path, WRITES_REPORT))
    with TestClient(build_app(m)) as client:
        yield client, m


def test_a_viewer_can_read_runs_but_not_start_or_cancel_them(api):
    client, _ = api
    assert client.get("/api/runs", headers=auth(VIEWER)).status_code == 200
    started = client.post("/api/runs", json={"kind": "eval_quick"}, headers=auth(VIEWER))
    assert started.status_code == 403
    assert client.post("/api/runs/abc/cancel", headers=auth(VIEWER)).status_code == 403
    assert client.get("/api/runs").status_code == 401


def test_start_poll_and_read_the_result(api):
    client, m = api
    started = client.post(
        "/api/runs", json={"kind": "eval_quick", "seed": 9}, headers=auth(ANALYST)
    )
    assert started.status_code == 202 and started.json()["started_by"] == "maya@acme.example"
    wait(m.get(started.json()["id"]))
    state = client.get("/api/runs", headers=auth(VIEWER)).json()
    assert state["enabled"] and state["busy"] is None
    assert state["jobs"]["eval_quick"]["state"] == "done"
    assert state["jobs"]["eval_quick"]["result"]["seed"] == 9


def test_a_second_start_while_busy_is_409(tmp_path):
    m = manager(tmp_path, fake_script(tmp_path, "import time; time.sleep(30)"))
    with TestClient(build_app(m)) as client:
        first = client.post("/api/runs", json={"kind": "eval_full"}, headers=auth(ANALYST)).json()
        again = client.post("/api/runs", json={"kind": "retrain"}, headers=auth(ANALYST))
        assert again.status_code == 409
        cancelled = client.post(f"/api/runs/{first['id']}/cancel", headers=auth(ANALYST))
        assert cancelled.status_code == 200
        wait(m.get(first["id"]))


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "rm -rf /"},
        {"kind": "retrain", "seed": -5},
        {"kind": "retrain", "seed": "x"},
        {"kind": "retrain", "extra": 1},
        {},
    ],
)
def test_malformed_run_requests_are_422(api, body):
    client, _ = api
    assert client.post("/api/runs", json=body, headers=auth(ANALYST)).status_code == 422


def test_cancelling_an_unknown_run_is_404(api):
    client, _ = api
    assert client.post("/api/runs/nope/cancel", headers=auth(ANALYST)).status_code == 404


def test_runs_are_off_when_no_manager_is_configured():
    client = TestClient(build_app(None))
    assert client.get("/api/runs", headers=auth(ANALYST)).json()["enabled"] is False
    off = client.post("/api/runs", json={"kind": "retrain"}, headers=auth(ANALYST))
    assert off.status_code == 404
