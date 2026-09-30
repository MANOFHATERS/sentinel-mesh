"""Runs the front end's own unit tests (``tests/js``) under ``node --test``.

The dashboard's JavaScript has no build step and no dependencies, so Node's built-in
test runner is enough. Skipped, loudly, when Node is not installed: the Python suite
must stay runnable on a machine with no JavaScript toolchain.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "sentinel" / "dashboard" / "static"
JS_TESTS = sorted((ROOT / "tests" / "js").glob("*.test.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_front_end_unit_tests_pass():
    assert JS_TESTS, "no front-end tests found"
    result = subprocess.run(
        ["node", "--test", "--test-reporter=tap", *map(str, JS_TESTS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "# fail 0" in result.stdout


def test_the_sso_button_is_not_an_anchor_the_href_filter_would_strip():
    # dom.js drops any href that is not "#/..." or absolute http(s), so a relative
    # "/auth/login" anchor renders with no href and the button silently does nothing.
    source = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert 'href: "/auth/login"' not in source
    assert 'window.location.assign("/auth/login")' in source


def test_the_live_feed_is_analyst_only_and_leaves_the_approval_queue_guarded():
    source = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    # only an analyst is given the control, and it drives the analyst-only replay route
    assert "session.can_act ? (live.button" in source
    assert '"/api/feed/replay", { count: LIVE_BATCH }' in source
    # while live, only pages without approve/reject controls refresh under the pointer
    assert 'name === "overview" || name === "incidents"' in source
    assert '"queue"' not in source.split("const liveView")[1].split("\n")[0]


def test_live_run_controls_are_analyst_only_and_the_saved_report_stays_the_default():
    source = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert "const canRun = session.can_act;" in source
    assert 'api.post("/api/runs"' in source
    # the pages still open on the saved / start-up results; a live run is added beside them
    assert "`Saved report (" in source and '"Trained at server start-up"' in source
    assert "liveEvaluationBlock(runs.jobs[kind], e)" in source
