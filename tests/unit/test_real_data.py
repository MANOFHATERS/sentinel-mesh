"""The real-data side: flows, the ATT&CK catalogue, a GitHub repository.

Fixtures here are *test fixtures*: a generated CSV in UNSW-NB15's column layout, a small
STIX bundle, an in-memory zip. They exist so these tests need no download and no network;
none of them is presented anywhere as real data.
"""

from __future__ import annotations

import csv
import io
import json
import random
import stat
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from sentinel.dashboard.app import create_app
from sentinel.dashboard.auth import Identity, Role, TokenRegistry
from sentinel.dashboard.runs import RunManager
from sentinel.real.attack import (
    AttackError,
    build_attack_kb,
    documents_from_stix,
    load_bundle,
    search_view,
)
from sentinel.real.network import RealDataError, analyse_flows, detect_format, find_dataset
from sentinel.real.repos import (
    RepoError,
    RepoRef,
    fetch_zip,
    parse_github_url,
    scan_repository,
    snapshot_from_zip,
)
from sentinel.real.service import RealData

ANALYST = "analyst-token-" + "a" * 30
VIEWER = "viewer-token-" + "v" * 30

UNSW_HEADER = [
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
]


def write_unsw(path: Path, rows: int = 1500, seed: int = 3) -> Path:
    """A generated CSV with UNSW-NB15's columns: attacks are short, chatty, small-payload."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(UNSW_HEADER)
        for i in range(rows):
            attack = rng.random() < 0.4
            cat = rng.choice(["Generic", "Exploits", "DoS", "Fuzzers"]) if attack else "Normal"
            dur = rng.expovariate(40) if attack else rng.expovariate(2)
            sp = rng.randint(2, 6) if attack else rng.randint(5, 40)
            dp = rng.randint(0, 3) if attack else rng.randint(4, 40)
            sb = rng.randint(40, 300) if attack else rng.randint(300, 9000)
            db = rng.randint(0, 200) if attack else rng.randint(200, 20000)
            row = [
                i + 1,
                f"{dur:.6f}",
                "udp" if attack else "tcp",
                "-",
                "INT" if attack else "FIN",
                sp,
                dp,
                sb,
                db,
                f"{(sp + dp) / max(dur, 1e-6):.3f}",
                254,
                252 if not attack else 0,
                1000.0,
                500.0,
                0,
                0,
                1.0,
                1.0,
                0.0,
                0.0,
                255,
                0,
                0,
                255,
                0.0,
                0.0,
                0.0,
                sb // max(sp, 1),
                db // max(dp, 1),
                0,
                0,
                rng.randint(1, 30),
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                "" if not attack else cat,
                1 if attack else 0,
            ]
            writer.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# Network flows
# --------------------------------------------------------------------------- #


def test_the_format_is_read_from_the_header(tmp_path):
    unsw = write_unsw(tmp_path / "flows.csv", rows=10)
    assert detect_format(unsw) == "unsw"
    cic = tmp_path / "cic.csv"
    cic.write_text(" Destination Port, Flow Duration, Label\n80,10,BENIGN\n", encoding="utf-8")
    assert detect_format(cic) == "cic"
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n", encoding="utf-8")
    assert detect_format(other) is None
    assert detect_format(tmp_path / "missing.csv") is None


def test_find_dataset_looks_in_the_conventional_folders(tmp_path):
    assert find_dataset(tmp_path) is None
    folder = tmp_path / "data" / "raw" / "unsw-nb15"
    folder.mkdir(parents=True)
    (folder / "notes.csv").write_text("x,y\n1,2\n")
    assert find_dataset(tmp_path) is None  # a CSV that is not flow data is ignored
    write_unsw(folder / "UNSW_NB15_training-set.csv", rows=10)
    assert find_dataset(tmp_path).name == "UNSW_NB15_training-set.csv"


def test_analysis_reports_real_metrics_on_a_held_out_split(tmp_path):
    path = write_unsw(tmp_path / "flows.csv")
    report = analyse_flows(path, limit=1200, seed=5)
    assert report["format"] == "unsw" and report["rows_in_file"] == 1500
    assert report["rows_sampled"] + report["rows_rejected"] == 1200
    m = report["metrics"]
    for key in ("roc_auc", "agreement", "recall", "precision", "alert_reduction"):
        assert m[key] is not None and 0.0 <= m[key] <= 1.0
    assert m["roc_auc"] > 0.7  # the fixture's attacks are separable; a real run may not be
    c = report["confusion"]
    assert sum(c.values()) == report["test_flows"]
    assert report["per_family"] and all(
        0 <= f["recall"] <= 1 for f in report["per_family"].values()
    )
    assert 0 < len(report["samples"]) <= 24
    assert {"decision", "true_label", "correct"} <= set(report["samples"][0])
    assert report["notes"]


def test_analysis_is_reproducible_for_a_seed_and_differs_across_seeds(tmp_path):
    path = write_unsw(tmp_path / "flows.csv")
    a = analyse_flows(path, limit=1200, seed=1)
    b = analyse_flows(path, limit=1200, seed=1)
    c = analyse_flows(path, limit=1200, seed=2)
    for report in (a, b, c):
        report.pop("seconds")
    assert a == b
    assert a["metrics"] != c["metrics"] or a["confusion"] != c["confusion"]


def test_analysis_refuses_what_it_cannot_evaluate(tmp_path):
    with pytest.raises(RealDataError, match="not a UNSW"):
        analyse_flows(tmp_path / "nope.csv")
    junk = tmp_path / "junk.csv"
    junk.write_text("a,b\n1,2\n")
    with pytest.raises(RealDataError, match="not a UNSW"):
        analyse_flows(junk)
    small = write_unsw(tmp_path / "small.csv", rows=150)
    with pytest.raises(RealDataError, match="usable rows"):
        analyse_flows(small, limit=1000)
    with pytest.raises(RealDataError, match="limit"):
        analyse_flows(write_unsw(tmp_path / "f.csv", rows=50), limit=10)


# --------------------------------------------------------------------------- #
# ATT&CK
# --------------------------------------------------------------------------- #


def technique(
    tid, name, *, revoked=False, deprecated=False, detection="", phases=("initial-access",)
):
    obj = {
        "type": "attack-pattern",
        "id": f"attack-pattern--{tid}",
        "name": name,
        "description": (
            f"{name} lets an adversary abuse trust. See [the report](https://example.org/x)."
            " (Citation: Some Vendor 2020) Second line with <code>markup</code>."
        ),
        "external_references": [
            {
                "source_name": "mitre-attack",
                "external_id": tid,
                "url": f"https://attack.mitre.org/techniques/{tid}",
            },
            {"source_name": "other", "external_id": "IGNORED"},
        ],
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": p} for p in phases]
        + [{"kill_chain_name": "someone-else", "phase_name": "ignored"}],
        "x_mitre_platforms": ["Windows", "Linux"],
    }
    if revoked:
        obj["revoked"] = True
    if deprecated:
        obj["x_mitre_deprecated"] = True
    if detection:
        obj["x_mitre_detection"] = detection
    return obj


def bundle(*objects):
    return {"type": "bundle", "id": "bundle--1", "objects": list(objects)}


def sample_bundle():
    return bundle(
        technique("T1566", "Phishing", detection="Monitor mail gateways for suspicious links."),
        technique("T1566.001", "Spearphishing Attachment"),
        technique("T1059", "Command and Scripting Interpreter", phases=("execution",)),
        technique("T1110", "Brute Force", phases=("credential-access",)),
        technique("T1003", "OS Credential Dumping", revoked=True),
        technique("T1004", "Old Thing", deprecated=True),
        technique("T1999.001", "Orphan Sub-technique"),
        {"type": "intrusion-set", "name": "not a technique"},
        {"type": "attack-pattern", "name": "no id", "description": "x", "external_references": []},
    )


def test_stix_becomes_documents_with_clean_prose_and_only_live_techniques():
    docs = {d.doc_id: d for d in documents_from_stix(sample_bundle())}
    assert set(docs) == {
        "T1566",
        "T1566.001",
        "T1059",
        "T1110",
    }  # revoked, deprecated, orphan, junk gone
    phishing = docs["T1566"]
    text = phishing.full_text
    assert "Citation" not in text and "<code>" not in text and "example.org" not in text
    assert "the report" in text and "markup" in text
    assert dict(phishing.sections)["Detection"].startswith("Monitor mail")
    assert phishing.tactics == ("initial-access",)  # the other kill chain is ignored
    assert phishing.platforms == ("windows", "linux")
    assert docs["T1566.001"].parent_id == "T1566"


def test_a_missing_or_wrong_bundle_is_a_clear_error(tmp_path):
    with pytest.raises(AttackError, match="fetch_real_data"):
        load_bundle(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(AttackError, match="not readable"):
        load_bundle(bad)
    notbundle = tmp_path / "x.json"
    notbundle.write_text('{"type": "other"}')
    with pytest.raises(AttackError, match="not a STIX bundle"):
        load_bundle(notbundle)
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps(bundle()))
    with pytest.raises(AttackError, match="no ATT&CK techniques"):
        build_attack_kb(empty)


def test_the_real_catalogue_is_searchable_and_hits_link_back_to_mitre(tmp_path):
    path = tmp_path / "attack.json"
    path.write_text(json.dumps(sample_bundle()))
    kb = build_attack_kb(path)
    hits = search_view(kb, "phishing email attachment")
    assert hits and hits[0]["doc_id"].startswith("T1566")
    assert all(h["kind"] == "technique" for h in hits)
    sub = (
        next(h for h in hits if h["doc_id"] == "T1566.001")
        if any(h["doc_id"] == "T1566.001" for h in hits)
        else hits[0]
    )
    assert sub["url"].startswith("https://attack.mitre.org/techniques/T1566")
    if "." in sub["doc_id"]:
        assert sub["url"].endswith("T1566/001/")
    assert search_view(kb, "zzzzqqqq nothing") == []


# --------------------------------------------------------------------------- #
# GitHub repositories
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/psf/requests", RepoRef("psf", "requests")),
        ("https://github.com/psf/requests/", RepoRef("psf", "requests")),
        ("https://github.com/psf/requests.git", RepoRef("psf", "requests")),
        ("  https://github.com/pallets/flask/tree/main  ", RepoRef("pallets", "flask", "main")),
        ("https://github.com/o/r/tree/release/1.x", RepoRef("o", "r", "release/1.x")),
    ],
)
def test_github_urls_are_parsed(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "psf/requests",
        "http://github.com/psf/requests",
        "https://gitlab.com/psf/requests",
        "https://github.com.evil.test/psf/requests",
        "https://evil.test/https://github.com/a/b",
        "https://user@github.com/psf/requests",
        "https://github.com/psf",
        "https://github.com/",
        "https://github.com/psf/requests/issues/1",
        "https://github.com/../etc",
        "https://github.com/a/b/tree/../../x",
        "https://github.com/a b/c",
        "file:///etc/passwd",
        "https://github.com/psf/requests?x=1",
        "javascript:alert(1)",
    ],
)
def test_anything_else_is_refused(url):
    with pytest.raises(RepoError):
        parse_github_url(url)


def test_the_archive_url_is_codeload_only():
    assert RepoRef("a", "b").archive_url == "https://codeload.github.com/a/b/zip/HEAD"
    assert (
        RepoRef("a", "b", "dev/x").archive_url
        == "https://codeload.github.com/a/b/zip/refs/heads/dev/x"
    )


def make_zip(files: dict[str, str | bytes], *, symlinks=()) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        for name, target in symlinks:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, target)
    return buffer.getvalue()


VULNERABLE = """\
import os
import sqlite3


def find_user(name):
    conn = sqlite3.connect("app.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name = '" + name + "'")
    return cur.fetchall()


def ping(host):
    os.system("ping -c 1 " + host)


raise SystemExit("this line must never run: the scanner only parses")
"""


def test_only_python_files_are_read_and_the_top_directory_is_stripped():
    data = make_zip(
        {
            "repo-main/app/main.py": "x = 1\n",
            "repo-main/README.md": "# hi",
            "repo-main/app/data.bin.py": b"\x00\x01binary",
            "repo-main/node_modules/lib/x.py": "y = 2\n",
            "repo-main/.git/hooks/pre.py": "z = 3\n",
            "repo-main/venv/lib/site.py": "w = 4\n",
        }
    )
    snap = snapshot_from_zip(data, root="o/r")
    assert [f.path for f in snap.files] == ["app/main.py"]
    assert ("app/data.bin.py", "binary content") in snap.skipped


def test_symlinks_oversize_files_and_the_file_budget_are_refused():
    big = "x = 1\n" * 100_000  # > 400 KB
    data = make_zip(
        {"r/ok.py": "a = 1\n", "r/big.py": big}, symlinks=[("r/link.py", "/etc/passwd")]
    )
    snap = snapshot_from_zip(data, root="o/r")
    assert [f.path for f in snap.files] == ["ok.py"]
    reasons = dict(snap.skipped)
    assert "symlink" in reasons["link.py"] and "KB" in reasons["big.py"]
    many = make_zip({f"r/m{i}.py": "a = 1\n" for i in range(320)})
    snap = snapshot_from_zip(many, root="o/r")
    assert len(snap.files) == 300 and sum(1 for _, why in snap.skipped if "budget" in why) == 20


def test_an_archive_with_no_python_or_not_a_zip_is_an_error():
    with pytest.raises(RepoError, match="no Python files"):
        snapshot_from_zip(make_zip({"r/readme.md": "x"}), root="o/r")
    with pytest.raises(RepoError, match="not a zip"):
        snapshot_from_zip(b"this is not a zip", root="o/r")


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_download_refuses_redirects_missing_repos_and_oversize():
    ref = RepoRef("o", "r")
    seen = []

    def ok(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=b"zipbytes")

    assert fetch_zip(ref, client=client_for(ok)) == b"zipbytes"
    assert seen == ["https://codeload.github.com/o/r/zip/HEAD"]
    with pytest.raises(RepoError, match="not found"):
        fetch_zip(ref, client=client_for(lambda r: httpx.Response(404)))

    def redirect(r):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    with pytest.raises(RepoError, match="answered 302"):
        fetch_zip(ref, client=client_for(redirect))

    def declared(r):
        return httpx.Response(200, headers={"content-length": str(60_000_000)}, content=b"x")

    with pytest.raises(RepoError, match="larger than"):
        fetch_zip(ref, client=client_for(declared))

    def boom(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(RepoError, match="could not reach GitHub"):
        fetch_zip(ref, client=client_for(boom))


def test_scanning_a_repository_finds_defects_without_running_its_code():
    data = make_zip({"r-main/app.py": VULNERABLE, "r-main/util.py": "def f(x):\n    return x\n"})
    report = scan_repository(
        "https://github.com/o/r", client=client_for(lambda r: httpx.Response(200, content=data))
    )
    assert report["repository"] == "o/r" and report["files_scanned"] == 2
    rules = {f["rule_id"] for f in report["findings"]}
    assert any("sql" in rule for rule in rules) and any(
        "command" in rule or "os-system" in rule for rule in rules
    )
    finding = report["findings"][0]
    assert finding["link"].startswith("https://github.com/o/r/blob/HEAD/app.py#L")
    assert report["validated_patches"] >= 1 and report["patches"][0]["diff"].startswith("---")
    assert any("answer key" in note for note in report["notes"])  # the limitation is stated


def test_scan_reports_the_branch_and_an_empty_result_cleanly():
    data = make_zip({"r-dev/clean.py": "def f(x):\n    return x + 1\n"})
    report = scan_repository(
        "https://github.com/o/r/tree/dev",
        client=client_for(lambda r: httpx.Response(200, content=data)),
    )
    assert report["ref"] == "dev" and report["finding_total"] == 0 and report["findings"] == []


# --------------------------------------------------------------------------- #
# The service and the API
# --------------------------------------------------------------------------- #


def test_status_says_what_is_and_is_not_on_this_machine(tmp_path):
    real = RealData(tmp_path)
    status = real.status()
    assert status["network"]["available"] is False and status["attack"]["available"] is False
    assert "fetch_real_data" in status["fetch_command"]
    folder = tmp_path / "data" / "raw" / "unsw-nb15"
    folder.mkdir(parents=True)
    write_unsw(folder / "UNSW_NB15_training-set.csv", rows=20)
    (tmp_path / "data" / "real").mkdir(parents=True)
    (tmp_path / "data" / "real" / "enterprise-attack.json").write_text(json.dumps(sample_bundle()))
    status = real.status()
    assert status["network"]["available"] and status["network"]["format"] == "unsw"
    assert status["attack"]["available"] and status["attack"]["built"] is False


def test_kb_search_without_the_file_says_how_to_get_it(tmp_path):
    result = RealData(tmp_path).kb_search("phishing")
    assert result["available"] is False and "fetch_real_data" in result["fetch_command"]


def test_kb_search_compares_the_real_catalogue_with_the_curated_one(tmp_path):
    (tmp_path / "data" / "real").mkdir(parents=True)
    (tmp_path / "data" / "real" / "enterprise-attack.json").write_text(json.dumps(sample_bundle()))
    real = RealData(
        tmp_path, curated_kb=build_attack_kb(tmp_path / "data" / "real" / "enterprise-attack.json")
    )
    result = real.kb_search("brute force password guessing")
    assert result["available"] and result["real_documents"] == 4
    assert result["real"] and result["real"][0]["doc_id"] == "T1110"
    assert real.status()["attack"]["built"] is True


def app_with(handlers=None, real=None):
    tokens = TokenRegistry(
        {
            ANALYST: Identity("maya@acme.example", "acme", Role.ANALYST),
            VIEWER: Identity("omar@acme.example", "acme", Role.VIEWER),
        }
    )
    runs = RunManager(
        workdir=Path("."),
        retrain=lambda s: {},
        read_evaluation=lambda p: {},
        script=Path("missing.py"),
        handlers=handlers or {},
    )
    return create_app({"acme": SimpleNamespace(tenant_id="acme")}, tokens, runs=runs, real=real)


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def wait_done(client, kind, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        state = client.get("/api/runs", headers=auth(ANALYST)).json()
        if state["busy"] is None and kind in state["jobs"]:
            return state["jobs"][kind]
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_a_real_scan_is_started_with_a_validated_url_and_returns_the_handler_result():
    seen = {}

    def scan(seed, params):
        seen.update(params)
        return {"repository": "o/r", "findings": []}

    with TestClient(app_with({"real_scan": scan})) as client:
        started = client.post(
            "/api/runs",
            json={"kind": "real_scan", "url": "https://github.com/o/r"},
            headers=auth(ANALYST),
        )
        assert started.status_code == 202
        job = wait_done(client, "real_scan")
    assert job["state"] == "done" and job["result"]["repository"] == "o/r"
    assert seen == {"url": "https://github.com/o/r"}


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://evil.test/a/b",
        "http://github.com/a/b",
        "https://github.com/a/b/../../c",
        "file:///etc/passwd",
    ],
)
def test_a_bad_repository_url_is_refused_before_anything_runs(url):
    with TestClient(app_with({"real_scan": lambda s, p: {}})) as client:
        body = {"kind": "real_scan"}
        if url is not None:
            body["url"] = url
        assert client.post("/api/runs", json=body, headers=auth(ANALYST)).status_code == 400
        assert client.get("/api/runs", headers=auth(ANALYST)).json()["busy"] is None


def test_real_runs_are_analyst_only_and_unavailable_without_a_handler():
    with TestClient(app_with({"real_scan": lambda s, p: {}})) as client:
        denied = client.post(
            "/api/runs",
            json={"kind": "real_scan", "url": "https://github.com/o/r"},
            headers=auth(VIEWER),
        )
        assert denied.status_code == 403
    with TestClient(app_with({})) as client:
        off = client.post("/api/runs", json={"kind": "real_network"}, headers=auth(ANALYST))
        assert off.status_code == 404


def test_the_network_limit_is_bounded_and_passed_through():
    seen = {}
    with TestClient(app_with({"real_network": lambda s, p: seen.update(p) or {"ok": 1}})) as client:
        too_small = client.post(
            "/api/runs", json={"kind": "real_network", "limit": 10}, headers=auth(ANALYST)
        )
        assert too_small.status_code == 422
        assert (
            client.post(
                "/api/runs", json={"kind": "real_network", "limit": 5000}, headers=auth(ANALYST)
            ).status_code
            == 202
        )
        wait_done(client, "real_network")
    assert seen == {"limit": 5000}


def test_a_failing_real_run_is_reported_not_raised():
    def boom(seed, params):
        raise FileNotFoundError("no real flow dataset found")

    with TestClient(app_with({"real_network": boom})) as client:
        client.post("/api/runs", json={"kind": "real_network"}, headers=auth(ANALYST))
        job = wait_done(client, "real_network")
    assert job["state"] == "failed" and "no real flow dataset" in job["error"]


def test_real_endpoints_need_a_token_and_degrade_when_disabled(tmp_path):
    with TestClient(app_with(real=RealData(tmp_path))) as client:
        assert client.get("/api/real/status").status_code == 401
        assert client.get("/api/real/kb/search?q=phishing").status_code == 401
        status = client.get("/api/real/status", headers=auth(VIEWER)).json()
        assert status["enabled"] and status["attack"]["available"] is False
        search = client.get("/api/real/kb/search?q=phishing", headers=auth(VIEWER)).json()
        assert search["available"] is False
        assert client.get("/api/real/kb/search?q=a", headers=auth(VIEWER)).status_code == 422
    with TestClient(app_with()) as client:
        assert client.get("/api/real/status", headers=auth(VIEWER)).json() == {"enabled": False}
        assert client.get("/api/real/kb/search?q=phishing", headers=auth(VIEWER)).status_code == 404
