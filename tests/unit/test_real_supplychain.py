"""The real supply chain: parsing, fetching (cached), the graph, and the launchable real cases.

Fixtures are tiny hand-made snapshots and a mock HTTP transport in the shapes deps.dev and OSV.dev
return, so these tests need no network. None of it is presented as real data.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime

import httpx
import pytest

from sentinel.dashboard import views
from sentinel.dashboard.workspace import MeshModels, Workspace, workspace_features
from sentinel.graph.schema import EdgeKind, NodeKind
from sentinel.graph.synthetic import label_ground_truth
from sentinel.real.supplychain import (
    Fetcher,
    Project,
    RealAdvisory,
    SupplyChainError,
    build_snapshot,
    canonical,
    graph_from_snapshot,
    issue_clusters,
    load_snapshot,
    normalise,
    parse_requirements,
    real_advisories,
)
from tests.unit.test_real_data import bundle, technique, write_unsw

GENERATED = "2026-09-30T00:00:00+00:00"


def vuln(vid, aliases=(), *, summary="A flaw", severity="HIGH", withdrawn=False, fixed=("9.9",)):
    return {
        "aliases": sorted(aliases),
        "summary": summary,
        "published": "2020-01-01T00:00:00Z",
        "withdrawn": withdrawn,
        "severity": severity,
        "cwes": ["CWE-89"],
        "fixed": list(fixed),
    }


def snapshot():
    """Six pinned packages in one project: a stale, badly vulnerable library at the bottom, two
    packages built on it, one known-malware release, and two that require each other."""
    old_vulns = ["GHSA-aaaa", "PYSEC-2020-1", "GHSA-bbbb", "GHSA-cccc", "GHSA-dddd", "GHSA-gone"]
    return {
        "generated_at": GENERATED,
        "sources": {"vulnerabilities": "test"},
        "projects": [
            {
                "repo": "o/app",
                "lockfile": "requirements.txt",
                "primary": True,
                "pins": {
                    "cyca": "1.0",
                    "cycb": "1.0",
                    "evil": "0.1",
                    "midlib": "2.0",
                    "oldlib": "1.0",
                    "toplib": "3.0",
                },
            }
        ],
        "packages": {
            "pypi:oldlib@1.0": {
                "name": "oldlib",
                "version": "1.0",
                "published_at": "2019-01-01T00:00:00Z",
                "requires": [],
                "vulns": old_vulns,
            },
            "pypi:midlib@2.0": {
                "name": "midlib",
                "version": "2.0",
                "published_at": "2026-06-01T00:00:00Z",
                "requires": ["oldlib"],
                "vulns": [],
            },
            "pypi:toplib@3.0": {
                "name": "toplib",
                "version": "3.0",
                "published_at": "2026-08-01T00:00:00Z",
                "requires": ["midlib", "oldlib"],
                "vulns": [],
            },
            "pypi:evil@0.1": {
                "name": "evil",
                "version": "0.1",
                "published_at": "2026-01-01T00:00:00Z",
                "requires": [],
                "vulns": ["MAL-2026-1"],
            },
            "pypi:cyca@1.0": {
                "name": "cyca",
                "version": "1.0",
                "published_at": "2026-01-01T00:00:00Z",
                "requires": ["cycb"],
                "vulns": [],
            },
            "pypi:cycb@1.0": {
                "name": "cycb",
                "version": "1.0",
                "published_at": "2026-01-01T00:00:00Z",
                "requires": ["cyca"],
                "vulns": [],
            },
        },
        "vulns": {
            "GHSA-aaaa": vuln("GHSA-aaaa", ["CVE-2020-0001", "PYSEC-2020-1"], severity="CRITICAL"),
            "PYSEC-2020-1": vuln("PYSEC-2020-1", ["CVE-2020-0001", "GHSA-aaaa"], severity=None),
            "GHSA-bbbb": vuln("GHSA-bbbb", ["CVE-2020-0002"], severity="MODERATE"),
            "GHSA-cccc": vuln("GHSA-cccc", [], severity="LOW"),
            "GHSA-dddd": vuln("GHSA-dddd", ["CVE-2020-0004"]),
            "GHSA-gone": vuln("GHSA-gone", ["CVE-2020-9999"], withdrawn=True),
            "MAL-2026-1": vuln("MAL-2026-1", [], summary="Malicious code in evil"),
        },
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_names_are_normalised_the_way_pypi_does():
    assert normalise("Django") == "django" and normalise("Flask_Cors") == "flask-cors"
    assert normalise("zope.interface") == "zope-interface" and normalise("a--b__c") == "a-b-c"


def test_a_lockfile_yields_exact_pins_and_ignores_everything_else():
    text = """
    # a comment
    Django==3.2.0   # inline comment
    requests[security]==2.28.1 ; python_version >= "3.7"
    -r other.txt
    flask>=2.0
    Pillow == 9.4.0
    git+https://github.com/x/y.git@abc#egg=y
    --hash=sha256:abc
    zope.interface==5.5.0 \\
    """
    assert parse_requirements(text) == {
        "django": "3.2.0",
        "requests": "2.28.1",
        "pillow": "9.4.0",
        "zope-interface": "5.5.0",
    }


# --------------------------------------------------------------------------- #
# Deduplicating advisories
# --------------------------------------------------------------------------- #


def test_records_that_share_an_alias_are_one_issue_and_a_cve_is_preferred():
    v = snapshot()["vulns"]
    ids = ["GHSA-aaaa", "PYSEC-2020-1", "GHSA-bbbb", "GHSA-cccc"]
    clusters = issue_clusters(v, ids)
    assert sorted(map(sorted, clusters)) == [
        ["GHSA-aaaa", "PYSEC-2020-1"],
        ["GHSA-bbbb"],
        ["GHSA-cccc"],
    ]
    merged = next(c for c in clusters if "GHSA-aaaa" in c)
    assert canonical(v, merged) == "CVE-2020-0001"
    assert canonical(v, ["GHSA-cccc"]) == "GHSA-cccc"  # no CVE alias: keep the record id


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #


def test_features_are_real_counts_and_ages_not_generated():
    graph, facts = graph_from_snapshot(snapshot())
    old = graph.node("pypi:oldlib@1.0")
    assert old.cve_exposure_count == 4, (
        "five live records make four distinct issues; a withdrawn one is ignored"
    )
    assert (
        old.days_since_last_update
        == (datetime.fromisoformat(GENERATED) - datetime(2019, 1, 1, tzinfo=UTC)).days
    )
    assert old.is_unmaintained and old.breach_history == 0
    assert graph.node("pypi:evil@0.1").breach_history == 1  # OSV lists this release as malware
    assert graph.node("pypi:evil@0.1").cve_exposure_count == 0  # malware is not counted as a CVE
    assert not graph.node("pypi:midlib@2.0").is_unmaintained
    assert [i["id"] for i in facts["issues"]["pypi:oldlib@1.0"]].count("CVE-2020-0001") == 1


def test_edges_point_the_way_risk_flows_and_projects_hang_off_top_level_packages():
    graph, facts = graph_from_snapshot(snapshot())
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("pypi:oldlib@1.0", "pypi:midlib@2.0") in edges  # dependency -> dependent
    assert ("pypi:oldlib@1.0", "pypi:toplib@3.0") in edges and (
        "pypi:midlib@2.0",
        "pypi:toplib@3.0",
    ) in edges
    assert ("pypi:toplib@3.0", "org:o/app") in edges and ("pypi:evil@0.1", "org:o/app") in edges
    assert (
        "pypi:oldlib@1.0",
        "org:o/app",
    ) not in edges  # required by others: not a direct dependency
    assert {e.kind for e in graph.edges} == {EdgeKind.DEPENDENCY}
    assert graph.node("org:o/app").kind is NodeKind.ORGANIZATION
    assert facts["primary"] == "o/app"


def test_depth_counts_hops_below_the_project():
    graph, _ = graph_from_snapshot(snapshot())
    assert graph.node("pypi:toplib@3.0").sbom_depth == 1
    assert graph.node("pypi:midlib@2.0").sbom_depth == 2
    assert graph.node("pypi:oldlib@1.0").sbom_depth == 2  # nearest route: via toplib


def test_a_dependency_cycle_is_cut_once_and_counted():
    graph, facts = graph_from_snapshot(snapshot())
    assert facts["cycle_edges_cut"] == 1
    both = [
        (e.source, e.target)
        for e in graph.edges
        if {e.source, e.target} == {"pypi:cyca@1.0", "pypi:cycb@1.0"}
    ]
    assert len(both) == 1
    graph.assert_acyclic()


def test_the_ground_truth_rule_finds_the_stale_vulnerable_library_and_what_stands_on_it():
    graph, _ = graph_from_snapshot(snapshot())
    truth = label_ground_truth(graph)
    assert truth.intrinsic == {"pypi:oldlib@1.0"}
    assert truth.risk_scores["pypi:toplib@3.0"] > 0  # inherited exposure reaches its dependents
    assert truth.risk_scores["pypi:oldlib@1.0"] >= 1.0


def test_a_real_advisory_scopes_a_review_and_changes_nothing():
    graph, facts = graph_from_snapshot(snapshot())
    advisories = real_advisories(graph, facts)
    assert list(advisories) == ["CVE-2020-0001"]  # the most severe issue on the worst package
    advisory = advisories["CVE-2020-0001"]
    assert isinstance(advisory, RealAdvisory) and advisory.package_id == "pypi:oldlib@1.0"
    assert "4 known vulnerabilities" in advisory.summary and "Fixed in 9.9" in advisory.summary
    assert advisory.apply(graph) is graph, "the vulnerabilities are already in the graph's features"
    assert real_advisories(graph, facts) == advisories  # deterministic


def test_a_missing_snapshot_says_how_to_build_it(tmp_path):
    with pytest.raises(SupplyChainError, match="build_real_supply_chain"):
        load_snapshot(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(SupplyChainError, match="not readable"):
        load_snapshot(bad)


# --------------------------------------------------------------------------- #
# Fetching, cached, against a mock of deps.dev and OSV.dev
# --------------------------------------------------------------------------- #


def mock_services(counter):
    def handler(request: httpx.Request) -> httpx.Response:
        counter.append(f"{request.method} {request.url}")
        url = str(request.url)
        if url.endswith("/requirements.txt"):
            return httpx.Response(200, text="OldLib==1.0\nTopLib==3.0\n")
        if "api.deps.dev" in url and url.endswith(":requirements"):
            deps = [{"projectName": "OldLib", "environmentMarker": ""}] if "toplib" in url else []
            deps.append({"projectName": "test-only", "environmentMarker": "extra == 'test'"})
            return httpx.Response(200, json={"pypi": {"dependencies": deps}})
        if "api.deps.dev" in url:
            return httpx.Response(
                200, json={"publishedAt": "2020-01-01T00:00:00Z", "advisoryKeys": []}
            )
        if url.endswith("/querybatch"):
            queries = json.loads(request.content)["queries"]
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"vulns": [{"id": "GHSA-x", "modified": "2020"}]}
                        if q["package"]["name"] == "oldlib"
                        else {}
                        for q in queries
                    ]
                },
            )
        if "/vulns/GHSA-x" in url:
            return httpx.Response(
                200,
                json={
                    "id": "GHSA-x",
                    "aliases": ["CVE-2020-1"],
                    "summary": "Bad",
                    "published": "2020-02-01T00:00:00Z",
                    "database_specific": {"severity": "HIGH", "cwe_ids": ["CWE-79"]},
                    "affected": [{"ranges": [{"events": [{"introduced": "0"}, {"fixed": "1.1"}]}]}],
                },
            )
        return httpx.Response(404)

    return handler


def test_build_snapshot_reads_lockfiles_dependencies_and_vulnerabilities(tmp_path):
    calls: list[str] = []
    client = httpx.Client(transport=httpx.MockTransport(mock_services(calls)))
    fetcher = Fetcher(tmp_path / "cache.json", client=client)
    snap = build_snapshot(
        fetcher,
        (Project("o/app", "requirements.txt", primary=True),),
        now=datetime(2026, 9, 30, tzinfo=UTC),
    )
    assert snap["projects"][0]["pins"] == {"oldlib": "1.0", "toplib": "3.0"}
    top = snap["packages"]["pypi:toplib@3.0"]
    assert top["requires"] == ["oldlib"], "an optional 'extra' dependency is not a real requirement"
    assert snap["packages"]["pypi:oldlib@1.0"]["vulns"] == ["GHSA-x"]
    assert snap["vulns"]["GHSA-x"] == {
        "aliases": ["CVE-2020-1"],
        "summary": "Bad",
        "published": "2020-02-01T00:00:00Z",
        "withdrawn": False,
        "severity": "HIGH",
        "cwes": ["CWE-79"],
        "fixed": ["1.1"],
    }
    graph, _facts = graph_from_snapshot(snap)
    assert graph.node("pypi:oldlib@1.0").cve_exposure_count == 1
    assert ("pypi:oldlib@1.0", "pypi:toplib@3.0") in {(e.source, e.target) for e in graph.edges}


def test_every_response_is_cached_so_a_rebuild_is_offline(tmp_path):
    calls: list[str] = []
    cache = tmp_path / "cache.json"
    project = (Project("o/app", "requirements.txt"),)
    first = Fetcher(cache, client=httpx.Client(transport=httpx.MockTransport(mock_services(calls))))
    build_snapshot(first, project, now=datetime(2026, 9, 30, tzinfo=UTC))
    api_calls = [c for c in calls if "api." in c]
    assert api_calls, "the first build asks the APIs"
    calls.clear()
    second = Fetcher(
        cache, client=httpx.Client(transport=httpx.MockTransport(mock_services(calls)))
    )
    build_snapshot(second, project, now=datetime(2026, 9, 30, tzinfo=UTC))
    assert [c for c in calls if "api." in c] == [], "nothing is asked twice"


def test_a_flaky_service_is_retried_and_a_dead_one_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("sentinel.real.supplychain.time.sleep", lambda _s: None)
    attempts = {"n": 0}

    def flaky(request):
        attempts["n"] += 1
        return httpx.Response(500) if attempts["n"] < 3 else httpx.Response(200, json={"ok": True})

    fetcher = Fetcher(
        tmp_path / "c.json", client=httpx.Client(transport=httpx.MockTransport(flaky))
    )
    assert fetcher.get("https://api.example.test/x") == {"ok": True} and attempts["n"] == 3
    dead = Fetcher(
        tmp_path / "d.json",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
    )
    with pytest.raises(SupplyChainError, match="kept failing"):
        dead.get("https://api.example.test/y")
    gone = Fetcher(
        tmp_path / "e.json",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
    )
    assert gone.get("https://api.example.test/z") is None  # not found is an answer, not an error


# --------------------------------------------------------------------------- #
# The real workspace with a real-shaped supply chain and repository
# --------------------------------------------------------------------------- #


VULNERABLE_APP = """\
import os
import sqlite3


def find(name):
    conn = sqlite3.connect("app.db")
    return conn.cursor().execute("SELECT * FROM users WHERE name = '" + name + "'").fetchall()


def ping(host):
    os.system("ping -c 1 " + host)
"""


@pytest.fixture(scope="module")
def full_real_models(mesh_models, tmp_path_factory):
    folder = tmp_path_factory.mktemp("full-real")
    dataset = write_unsw(folder / "UNSW_NB15_training-set.csv", rows=1600)
    attack = folder / "enterprise-attack.json"
    from sentinel.agents.triage import TECHNIQUE_BY_FAMILY

    attack.write_text(
        json.dumps(
            bundle(
                *[technique(t, f"Technique {t}") for t in sorted(set(TECHNIQUE_BY_FAMILY.values()))]
            )
        )
    )
    supply = folder / "supply-chain.json"
    supply.write_text(json.dumps(snapshot()))
    cache = folder / "repos"
    cache.mkdir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("app-main/app.py", VULNERABLE_APP)
    (cache / "o__app.zip").write_bytes(buffer.getvalue())
    return MeshModels.build_real(
        mesh_models,
        dataset_path=dataset,
        attack_path=attack,
        seed=11,
        limit=1400,
        supply_chain_path=supply,
        repo_cache_dir=cache,
    )


@pytest.fixture
def workspace(full_real_models, tmp_path):
    ws = Workspace(full_real_models, tenant_id="real", workdir=tmp_path / "real")
    yield ws
    ws.close()


def test_a_real_workspace_with_real_data_behind_every_page_offers_every_page(full_real_models):
    assert workspace_features(full_real_models) == {
        "scenarios": True,
        "supply_chain": True,
        "code_scan": True,
        "models": True,
        "kb": True,
    }
    assert full_real_models.repository == "o/app" and full_real_models.graph.n_nodes == 7
    assert full_real_models.real_supply["truth"]["intrinsic"] == 1
    assert full_real_models.graph_evaluation["test_nodes"] > 0


def test_the_real_cases_are_built_from_what_the_real_data_contains(workspace):
    cards = {c["name"]: c for c in views.scenarios_view(workspace)}
    assert any(n.startswith("campaign-") for n in cards)
    assert "advisory-cve-2020-0001" in cards and "code-scan" in cards
    assert "o/app" in cards["code-scan"]["title"]
    assert all(c["agents"] and c["walkthrough"] and not c["launched"] for c in cards.values())


def test_a_campaign_takes_the_next_real_flows_each_time_it_is_launched(workspace):
    name = next(
        n for n in (c["name"] for c in views.scenarios_view(workspace)) if n.startswith("campaign-")
    )
    first = workspace.launch_real(name, launched_by="maya")
    second = workspace.launch_real(name, launched_by="maya")
    assert len(first) == 5 and len(second) == 5 and not set(first) & set(second)
    card = next(c for c in views.scenarios_view(workspace) if c["name"] == name)
    assert card["total"] == 10 and card["launched"] is False, "a campaign stays launchable"


def test_a_real_advisory_review_runs_over_the_real_graph_and_waits_for_approval(workspace):
    opened = workspace.launch_real("advisory-cve-2020-0001", launched_by="maya")
    assert len(opened) == 1
    graph = views.graph_view(workspace, scope="advisory:CVE-2020-0001", k=10)
    assert graph["advisory"]["advisory_id"] == "CVE-2020-0001"
    assert "pypi:oldlib@1.0" in {n["id"] for n in graph["nodes"]}
    card = next(c for c in views.scenarios_view(workspace) if c["name"] == "advisory-cve-2020-0001")
    assert card["launched"] and card["total"] == 1
    with pytest.raises(Exception, match="already"):
        workspace.launch_real("advisory-cve-2020-0001", launched_by="maya")


def test_the_code_scan_reviews_the_real_project_and_names_it(workspace):
    workspace.launch_real("code-scan", launched_by="maya")
    view = views.code_scan_view(workspace)
    assert view["repository"] == "o/app" and view["files_scanned"] == 1
    assert {f["rule_id"] for f in view["findings"]} >= {"python.sql-injection"}
    assert view["draft"] is not None and "app.py" in view["draft"]["files"]
    with pytest.raises(Exception, match="already been scanned"):
        workspace.launch_real("code-scan", launched_by="maya")


def test_unknown_real_cases_and_synthetic_workspaces_are_refused(workspace, mesh_models, tmp_path):
    with pytest.raises(Exception, match="no real scenario"):
        workspace.launch_real("campaign-nonsense", launched_by="maya")
    synthetic = Workspace(mesh_models, tenant_id="acme", workdir=tmp_path / "acme")
    try:
        with pytest.raises(Exception, match="no real scenarios"):
            synthetic.launch_real("code-scan", launched_by="maya")
    finally:
        synthetic.close()


def test_real_launches_survive_a_restart(full_real_models, tmp_path):
    first = Workspace(full_real_models, tenant_id="real", workdir=tmp_path / "w")
    first.launch_real("advisory-cve-2020-0001", launched_by="maya")
    first.launch_real("code-scan", launched_by="maya")
    first.close()
    again = Workspace(full_real_models, tenant_id="real", workdir=tmp_path / "w")
    try:
        cards = {c["name"]: c for c in views.scenarios_view(again)}
        assert cards["advisory-cve-2020-0001"]["launched"] and cards["code-scan"]["launched"]
        assert cards["code-scan"]["total"] == 1
    finally:
        again.close()
