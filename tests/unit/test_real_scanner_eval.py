"""The scanner's measurement against real answer keys: the maths, the selection and the plumbing.

Mock GitHub responses and in-memory archives stand in for the network; none of it is real data.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from sentinel.dashboard.app import create_app
from sentinel.dashboard.auth import Identity, Role, TokenRegistry
from sentinel.real.scanner_eval import (
    ScannerEvalError,
    advisory_cases,
    evaluate_advisories,
    evaluate_paired_repo,
    load_result,
    wilson,
)
from sentinel.real.service import RealData

VULNERABLE = """\
import sqlite3


def find(name):
    conn = sqlite3.connect("x.db")
    return conn.cursor().execute("SELECT * FROM t WHERE n = '" + name + "'").fetchall()
"""
FIXED = """\
import sqlite3


def find(name):
    conn = sqlite3.connect("x.db")
    return conn.cursor().execute("SELECT * FROM t WHERE n = ?", (name,)).fetchall()
"""
LOGIC_FLAW_BEFORE = "def check(user):\n    return user.is_admin or True\n"
LOGIC_FLAW_AFTER = "def check(user):\n    return user.is_admin\n"


def test_wilson_interval_is_sane():
    assert wilson(0, 0) == (0.0, 0.0)
    low, high = wilson(4, 25)
    assert 0.05 < low < 0.16 < high < 0.36  # 16% observed, wide because 25 is small
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(50, 100)[0] < 0.5 < wilson(50, 100)[1]
    narrow, wide = wilson(500, 1000), wilson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def osv(vid, cwes, refs=(), summary="A flaw"):
    return {
        "id": vid,
        "summary": summary,
        "database_specific": {"cwe_ids": list(cwes)},
        "references": [{"type": t, "url": u} for t, u in refs],
    }


def test_only_advisories_the_scanner_covers_and_that_link_a_fix_commit_become_cases():
    fix = ("FIX", "https://github.com/o/r/commit/abc1234")
    cache = {
        "GET https://api.osv.dev/v1/vulns/GHSA-1": osv(
            "GHSA-1", ["CWE-89"], [("WEB", "https://x.test"), fix]
        ),
        "GET https://api.osv.dev/v1/vulns/GHSA-2": osv("GHSA-2", ["CWE-123456"], [fix]),  # no rule
        "GET https://api.osv.dev/v1/vulns/GHSA-3": osv(
            "GHSA-3", ["CWE-89"], [("WEB", "https://x.test")]
        ),
        "GET https://api.osv.dev/v1/vulns/GHSA-4": None,
        "POST https://api.osv.dev/v1/querybatch abc": {"results": []},
    }
    cases = advisory_cases(cache)
    assert [c["advisory"] for c in cases] == ["GHSA-1"]
    case = cases[0]
    assert (case["owner"], case["repo"], case["sha"], case["cwe"]) == (
        "o",
        "r",
        "abc1234",
        "CWE-89",
    )
    assert "python.sql-injection" in case["rules"]


def github_mock(before: str, after: str, *, files=("pkg/db.py",), parents=True):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://api.github.com/repos/o/r/commits/"):
            body = {
                "sha": "f" * 40,
                "parents": [{"sha": "e" * 40}] if parents else [],
                "files": [{"filename": f, "status": "modified"} for f in files]
                + [
                    {"filename": "tests/test_db.py", "status": "modified"},
                    {"filename": "README.md", "status": "modified"},
                    {"filename": "pkg/new.py", "status": "added"},
                ],
            }
            return httpx.Response(200, json=body)
        if "raw.githubusercontent.com" in url:
            return httpx.Response(200, text=before if "/" + "e" * 40 + "/" in url else after)
        return httpx.Response(404)

    return handler


CASE = {
    "advisory": "GHSA-1",
    "summary": "SQL injection",
    "cwe": "CWE-89",
    "rules": ["python.sql-injection"],
    "owner": "o",
    "repo": "r",
    "sha": "abc1234",
}


def run(before, after, tmp_path, **kw):
    client = httpx.Client(transport=httpx.MockTransport(github_mock(before, after, **kw)))
    return evaluate_advisories([CASE], client=client, cache_path=tmp_path / "cache.json", pause=0)


def test_a_fix_the_scanner_notices_is_caught_before_and_clean_after(tmp_path):
    result = run(VULNERABLE, FIXED, tmp_path)
    row = result["rows"][0]
    assert row["before_hit"] is True and row["after_hit"] is False
    assert (result["cases"], result["hit"], result["flagged_before_and_clean_after"]) == (1, 1, 1)
    assert row["files"] == ["pkg/db.py"], (
        "tests, docs and added files are not part of the answer key"
    )
    assert result["recall"] == 1.0 and result["fixed_version_flagged"] == 0
    assert result["by_cwe"] == {"CWE-89": {"cases": 1, "hit": 1}}


def test_a_flaw_the_rules_cannot_see_is_a_miss_and_is_counted_as_one(tmp_path):
    result = run(LOGIC_FLAW_BEFORE, LOGIC_FLAW_AFTER, tmp_path)
    assert result["rows"][0]["before_hit"] is False and result["recall"] == 0.0


def test_a_fix_that_still_looks_dangerous_is_counted_as_a_false_alarm(tmp_path):
    result = run(VULNERABLE, VULNERABLE, tmp_path)
    assert result["rows"][0]["after_hit"] is True and result["fixed_version_flagged"] == 1


def test_a_commit_with_no_parent_or_no_python_files_is_skipped_not_scored(tmp_path):
    assert run(VULNERABLE, FIXED, tmp_path, parents=False)["cases"] == 0
    assert run(VULNERABLE, FIXED, tmp_path, files=())["cases"] == 0


def test_answers_are_cached_so_a_rerun_makes_no_requests(tmp_path):
    calls = []

    def counting(request):
        calls.append(str(request.url))
        return github_mock(VULNERABLE, FIXED)(request)

    client = httpx.Client(transport=httpx.MockTransport(counting))
    evaluate_advisories([CASE], client=client, cache_path=tmp_path / "c.json", pause=0)
    made = len(calls)
    assert made > 0
    evaluate_advisories([CASE], client=client, cache_path=tmp_path / "c.json", pause=0)
    assert len(calls) == made


def test_hitting_the_github_rate_limit_says_how_to_lift_it(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    with pytest.raises(ScannerEvalError, match="GITHUB_TOKEN"):
        evaluate_advisories([CASE], client=client, cache_path=tmp_path / "c.json", pause=0)
    assert (tmp_path / "c.json").is_file(), "finished work is kept even when a run stops early"


def make_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_paired_trees_score_each_vulnerable_file_against_its_fixed_namesake():
    data = make_zip(
        {
            "vulpy-main/bad/db.py": VULNERABLE,
            "vulpy-main/good/db.py": FIXED,
            "vulpy-main/bad/auth.py": LOGIC_FLAW_BEFORE,
            "vulpy-main/good/auth.py": LOGIC_FLAW_AFTER,
            "vulpy-main/bad/only_here.py": VULNERABLE,  # no fixed namesake: not a pair
            "vulpy-main/utils/x.py": VULNERABLE,  # outside both trees
        }
    )
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=data))
    )
    result = evaluate_paired_repo("o/vulpy", client=client)
    assert result["pairs"] == 2
    assert (result["bad_flagged"], result["good_flagged"]) == (1, 0)
    assert result["recall"] == 0.5 and result["false_alarm_rate"] == 0.0
    by_file = {f["file"]: f for f in result["files"]}
    assert (
        by_file["db.py"]["bad"]["rules"] == ["python.sql-injection"]
        and by_file["auth.py"]["bad"]["findings"] == 0
    )
    assert result["recall_ci"][0] < 0.5 < result["recall_ci"][1]


def test_a_missing_result_says_how_to_produce_it(tmp_path):
    with pytest.raises(ScannerEvalError, match="build_scanner_eval"):
        load_result(tmp_path / "nope.json")


def test_the_dashboard_serves_the_saved_measurement_or_says_it_is_missing(tmp_path):
    tokens = TokenRegistry({"t" * 30: Identity("v@acme.example", "acme", Role.VIEWER)})
    headers = {"Authorization": "Bearer " + "t" * 30}

    def client_for(root: Path):
        app = create_app({"acme": SimpleNamespace(tenant_id="acme")}, tokens, real=RealData(root))
        return TestClient(app)

    with client_for(tmp_path) as client:
        assert client.get("/api/real/scanner-quality").status_code == 401
        missing = client.get("/api/real/scanner-quality", headers=headers).json()
        assert missing == {"available": False, "command": "python scripts/build_scanner_eval.py"}
    (tmp_path / "data" / "real").mkdir(parents=True)
    (tmp_path / "data" / "real" / "scanner-eval.json").write_text(
        json.dumps({"generated_at": "2026-09-30", "paired": {}, "advisories": {}})
    )
    with client_for(tmp_path) as client:
        found = client.get("/api/real/scanner-quality", headers=headers).json()
        assert found["available"] is True and found["generated_at"] == "2026-09-30"
