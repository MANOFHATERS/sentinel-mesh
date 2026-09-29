"""Dashboard view helpers that do not need a live workspace (Part 5)."""

from __future__ import annotations

import json

from sentinel.audit.log import Finding, FindingKind, VerificationResult
from sentinel.dashboard.views import _chain_view, evaluation_view


def result(ok: bool, *kinds: FindingKind, rows: int = 0) -> VerificationResult:
    return VerificationResult(
        ok=ok,
        rows_checked=rows,
        head_seq=rows or None,
        head_hash=None,
        findings=tuple(Finding(kind, None, kind.value) for kind in kinds),
    )


class TestChainStatus:
    def test_verified(self):
        assert _chain_view(result(True, rows=12), fresh=False)["status"] == "verified"

    def test_an_empty_log_in_a_fresh_workspace_is_empty_not_broken(self):
        assert _chain_view(result(False, FindingKind.EMPTY), fresh=True)["status"] == "empty"

    def test_an_empty_log_after_runs_is_truncation(self):
        # The registry says incidents exist, the log says nothing happened: that is
        # the erasure the EMPTY finding exists to catch.
        assert _chain_view(result(False, FindingKind.EMPTY), fresh=False)["status"] == "broken"

    def test_any_other_finding_is_broken_even_when_fresh(self):
        view = _chain_view(result(False, FindingKind.EMPTY, FindingKind.BROKEN_LINK),
                           fresh=True)
        assert view["status"] == "broken"
        assert len(view["findings"]) == 2


class TestEvaluationView:
    def test_missing_artifact_is_reported_not_invented(self, tmp_path):
        view = evaluation_view(tmp_path / "absent.json")
        assert view == {"available": False, "path": str(tmp_path / "absent.json"),
                        "gates": [], "headline": {}}

    def test_gates_are_found_at_any_depth_and_only_booleans_count(self, tmp_path):
        path = tmp_path / "evaluation.json"
        path.write_text(json.dumps({
            "f03_auc_pass": True,
            "test": {"roc_auc": 0.99},
            "agents": {"f08_approval_gate_pass": False, "nested": {"x_pass": True}},
            "not_a_gate_pass": "yes",
        }))
        view = evaluation_view(path)
        assert view["gates"] == [
            {"gate": "f03_auc_pass", "passed": True},
            {"gate": "agents.f08_approval_gate_pass", "passed": False},
            {"gate": "agents.nested.x_pass", "passed": True},
        ]
        assert view["passed"] == 2
        assert view["headline"]["roc_auc"] == 0.99
