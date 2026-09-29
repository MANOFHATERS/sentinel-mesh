"""The Git connector against a live GitHub emulator (F-07, Section 5.4, Section 5.7).

The guarantee under test is Section 5.4's *"opened as draft PRs for human merge, never
auto-merged"*, now against an API where the same token that opens a PR could merge
it. The emulator implements the merge endpoint and counts calls to it, so "never
merged" is asserted as a zero, not inferred from a missing method.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime

import pytest

from sentinel.connectors.base import (
    Capability,
    ConnectorError,
    Credential,
    EgressDenied,
    LeastPrivilegeError,
    Secret,
)
from sentinel.connectors.github import GitHubConnector
from sentinel.connectors.http import HttpError
from sentinel.connectors.sandbox import GitHubEmulator, LiveServer
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import ActionRequest, ActionType, AgentName, RiskTier
from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.patch import PullRequestDraft
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.seeded import FIXTURE_DIR

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
TOKEN = "sandbox-token-github"
BRANCH = "sentinel/code-scan/abc123"


@pytest.fixture(scope="module")
def fixture_scan():
    snapshot = RepoSnapshot.from_dir(FIXTURE_DIR)
    analyzer = AstAnalyzer()
    return snapshot, analyzer, analyzer.scan(snapshot)


@pytest.fixture
def draft(fixture_scan):
    _snapshot, _analyzer, scan = fixture_scan
    return PullRequestDraft(
        branch=BRANCH, title="Fix 15 findings", body="Review every hunk.",
        patches=scan.valid_patches,
    )


def _emulator(fixture_scan, **kwargs):
    snapshot, _a, _s = fixture_scan
    return GitHubEmulator(owner="acme", repo="billing", token=TOKEN,
                          files={f.path: f.text for f in snapshot.files}, **kwargs)


@pytest.fixture
def github(fixture_scan):
    emulator = _emulator(fixture_scan)
    with LiveServer(emulator) as server:
        yield emulator, server


def _connector(server, *, scopes=("contents:write", "pull_requests:write", "issues:write"),
               caps=(Capability.PR_OPEN_DRAFT, Capability.ISSUE_OPEN), token=TOKEN):
    return GitHubConnector(
        owner="acme", repo="billing",
        credential=Credential(Secret(token), frozenset(scopes)),
        capabilities=caps, base_url=server.url, allow_insecure_loopback=True,
        sleep=lambda _s: None,
    )


def _approved(target=BRANCH, *, approve=True):
    action = ActionRequest.propose(
        alert_id="scan-1", tenant_id="acme", proposed_by=AgentName.CODE_SCAN,
        action_type=ActionType.OPEN_PATCH_PR, target=target, rationale="fixture",
        risk_tier=RiskTier.RECOMMEND, created_at=NOW,
    )
    return action.approve(approver="dev@acme", at=NOW) if approve else action


class TestOpensADraft:
    def test_opens_exactly_one_draft_pull_request(self, github, draft):
        emulator, server = github
        outcome = _connector(server).open_draft(_approved(), draft)
        assert outcome.succeeded
        assert len(emulator.pulls) == 1
        pull = emulator.pulls[0]
        assert pull["draft"] is True
        assert pull["head"]["ref"] == BRANCH and pull["base"]["ref"] == "main"
        assert pull["maintainer_can_modify"] is False
        assert outcome.reference == pull["html_url"]

    def test_never_merges_and_never_edits_the_pull(self, github, draft):
        emulator, server = github
        _connector(server).open_draft(_approved(), draft)
        assert emulator.merges == 0 and emulator.pull_edits == 0
        methods = {r.method for r in emulator.requests}
        assert methods <= {"GET", "POST"}

    def test_the_pushed_files_are_the_composed_patches_and_rescan_clean(
        self, github, draft, fixture_scan
    ):
        emulator, server = github
        snapshot, analyzer, scan = fixture_scan
        _connector(server).open_draft(_approved(), draft)
        for file in snapshot.files:
            pushed = emulator.file_at(BRANCH, file.path)
            patches = [p for p in scan.valid_patches if p.path == file.path]
            ast.parse(pushed)
            before = analyzer.rule_counts(file)
            after = analyzer.rule_counts(file.with_text(pushed))
            assert sum(after.values()) == sum(before.values()) - len(patches), file.path

    def test_the_default_branch_is_untouched(self, github, draft, fixture_scan):
        emulator, server = github
        snapshot, _a, _s = fixture_scan
        _connector(server).open_draft(_approved(), draft)
        for file in snapshot.files:
            assert emulator.file_at("main", file.path) == file.text

    def test_the_commit_binds_the_action_and_the_approver(self, github, draft):
        emulator, server = github
        action = _approved()
        _connector(server).open_draft(action, draft)
        head = emulator.refs[f"heads/{BRANCH}"]
        message = emulator.commits[head]["message"]
        assert f"Sentinel-Action: {action.action_id}" in message
        assert "Approved-by: dev@acme" in message

    def test_the_branch_appears_in_one_ref_creation(self, github, draft):
        emulator, server = github
        _connector(server).open_draft(_approved(), draft)
        assert len(emulator.calls("POST", r"/git/refs$")) == 1
        assert len(emulator.calls("POST", r"/git/commits$")) == 1
        assert len(emulator.calls("POST", r"/git/trees$")) == 1

    def test_the_request_carries_the_api_version_and_bearer_token(self, github, draft):
        emulator, server = github
        _connector(server).open_draft(_approved(), draft)
        first = emulator.requests[0]
        assert first.headers["authorization"] == f"Bearer {TOKEN}"
        assert first.headers["x-github-api-version"] == "2022-11-28"


class TestIdempotence:
    def test_a_second_call_opens_nothing_new(self, github, draft):
        emulator, server = github
        connector = _connector(server)
        action = _approved()
        connector.open_draft(action, draft)
        again = connector.open_draft(action, draft)
        assert again.succeeded and "already open" in again.detail
        assert len(emulator.pulls) == 1
        assert len(emulator.calls("POST", r"/git/refs$")) == 1

    def test_branch_pushed_but_pr_missing_resumes_at_the_pr(self, github, draft):
        emulator, server = github
        connector = _connector(server)
        action = _approved()
        emulator.inject(502, times=3)  # fail the first attempt's repo lookup
        with pytest.raises(HttpError):
            connector.open_draft(action, draft)
        connector.open_draft(action, draft)
        emulator.pulls.clear()  # simulate a crash after the ref, before the PR
        outcome = connector.open_draft(action, draft)
        assert outcome.succeeded and len(emulator.pulls) == 1
        assert len(emulator.calls("POST", r"/git/refs$")) == 1

    def test_a_strangers_branch_of_the_same_name_is_not_overwritten(self, github, draft):
        emulator, server = github
        main = emulator.refs["heads/main"]
        emulator.refs[f"heads/{BRANCH}"] = main  # someone else's branch
        with pytest.raises(GuardrailViolation, match="not created by this action"):
            _connector(server).open_draft(_approved(), draft)
        assert emulator.pulls == []


class TestRefusals:
    def test_unapproved_action_makes_no_request_at_all(self, github, draft):
        emulator, server = github
        with pytest.raises(GuardrailViolation, match="approval required"):
            _connector(server).open_draft(_approved(approve=False), draft)
        assert emulator.requests == []

    def test_approval_for_a_different_branch_is_refused(self, github, draft):
        emulator, server = github
        with pytest.raises(GuardrailViolation, match="approval is for branch"):
            _connector(server).open_draft(_approved("sentinel/other"), draft)
        assert emulator.requests == []

    def test_wrong_action_type_is_refused(self, github, draft):
        _emu, server = github
        block = ActionRequest.propose(
            alert_id="a", tenant_id="acme", proposed_by=AgentName.CONTAINMENT,
            action_type=ActionType.BLOCK_IP, target="203.0.113.1", rationale="r",
            risk_tier=RiskTier.AUTONOMOUS, created_at=NOW,
        )
        with pytest.raises(GuardrailViolation):
            _connector(server).open_draft(block, draft)

    def test_a_moved_default_branch_refuses_and_creates_nothing(self, github, draft):
        emulator, server = github
        emulator.push_to_default("app.py", "print('rewritten upstream')\n")
        with pytest.raises(ConnectorError, match="differs from the version that was scanned"):
            _connector(server).open_draft(_approved(), draft)
        assert f"heads/{BRANCH}" not in emulator.refs and emulator.pulls == []
        assert emulator.calls("POST", r"/git/blobs$") == []

    def test_a_crlf_file_is_refused(self, github, draft, fixture_scan):
        emulator, server = github
        snapshot, _a, _s = fixture_scan
        app = next(f for f in snapshot.files if f.path == "app.py")
        emulator.push_to_default("app.py", app.text.replace("\n", "\r\n"))
        with pytest.raises(ConnectorError, match="CRLF"):
            _connector(server).open_draft(_approved(), draft)

    def test_a_repository_without_draft_support_fails_loudly(self, fixture_scan, draft):
        emulator = _emulator(fixture_scan, supports_drafts=False)
        with LiveServer(emulator) as server, pytest.raises(HttpError, match="422"):
            _connector(server).open_draft(_approved(), draft)
        assert emulator.pulls == []

    def test_a_repository_that_ignores_the_draft_flag_is_reported_not_accepted(
        self, fixture_scan, draft
    ):
        emulator = _emulator(fixture_scan, ignore_draft_flag=True)
        with LiveServer(emulator) as server, pytest.raises(ConnectorError, match="not a draft"):
            _connector(server).open_draft(_approved(), draft)
        assert emulator.merges == 0

    def test_a_bad_token_is_a_401(self, github, draft):
        _emu, server = github
        with pytest.raises(HttpError, match="401"):
            _connector(server, token="wrong").open_draft(_approved(), draft)


class TestLeastPrivilege:
    def test_the_connector_cannot_reach_the_merge_endpoint(self, github):
        emulator, server = github
        connector = _connector(server)
        with pytest.raises(EgressDenied):
            connector.http.request("PUT", "/repos/acme/billing/pulls/1/merge")
        with pytest.raises(EgressDenied):
            connector.http.request("PATCH", "/repos/acme/billing/pulls/1")
        with pytest.raises(EgressDenied):
            connector.http.request("DELETE", "/repos/acme/billing/git/refs/heads/main")
        assert emulator.requests == []

    def test_no_route_other_than_get_and_post_exists(self, github):
        _emu, server = github
        routes = _connector(server).http.policy.routes
        assert {route.method for route in routes} == {"GET", "POST"}

    def test_the_connector_cannot_reach_another_repository(self, github):
        _emu, server = github
        with pytest.raises(EgressDenied):
            _connector(server).http.request("GET", "/repos/acme/other")

    def test_a_broader_declared_token_is_refused_at_construction(self, github):
        _emu, server = github
        with pytest.raises(LeastPrivilegeError, match="administration:write"):
            _connector(server, scopes=("contents:write", "pull_requests:write",
                                       "issues:write", "administration:write"))

    def test_a_token_missing_a_scope_is_refused_at_construction(self, github):
        _emu, server = github
        with pytest.raises(LeastPrivilegeError, match="missing"):
            _connector(server, scopes=("contents:write",), caps=(Capability.PR_OPEN_DRAFT,))

    def test_issue_capability_is_not_granted_by_default(self, github):
        _emu, server = github
        connector = _connector(server, scopes=("contents:write", "pull_requests:write"),
                               caps=(Capability.PR_OPEN_DRAFT,))
        assert not any(r.name.startswith("issues") for r in connector.http.policy.routes)

    def test_an_admin_token_reported_by_github_is_refused_before_any_write(
        self, fixture_scan, draft
    ):
        emulator = _emulator(fixture_scan, reported_scopes="repo, admin:org, workflow")
        with LiveServer(emulator) as server, pytest.raises(LeastPrivilegeError, match="admin:org"):
            _connector(server).open_draft(_approved(), draft)
        assert all(r.method == "GET" for r in emulator.requests)
        assert emulator.pulls == []

    def test_a_classic_token_with_acceptable_scopes_proceeds(self, fixture_scan, draft):
        emulator = _emulator(fixture_scan, reported_scopes="public_repo")
        with LiveServer(emulator) as server:
            assert _connector(server).open_draft(_approved(), draft).succeeded

    @pytest.mark.parametrize(("owner", "repo"), [("ac me", "r"), ("acme", ".."), ("-x", "r")])
    def test_invalid_repository_coordinates_are_refused(self, owner, repo):
        with pytest.raises(ValueError):
            GitHubConnector(owner=owner, repo=repo,
                            credential=Credential(Secret("t"), frozenset(
                                {"contents:write", "pull_requests:write"})))

    def test_unsupported_capability_is_refused(self):
        with pytest.raises(LeastPrivilegeError):
            GitHubConnector(owner="a", repo="b", capabilities=(Capability.HOST_ISOLATE,),
                            credential=Credential(Secret("t"), frozenset()))


class TestIssues:
    def _supply_chain_action(self):
        action = ActionRequest.propose(
            alert_id="sc-1", tenant_id="acme", proposed_by=AgentName.SUPPLY_CHAIN,
            action_type=ActionType.OPEN_PATCH_PR, target="pkg:pypi/leftpad",
            rationale="Unmaintained for 4 years with 3 open CVEs.",
            risk_tier=RiskTier.RECOMMEND, created_at=NOW,
        )
        return action.approve(approver="vciso@acme", at=NOW)

    def test_execute_opens_a_labelled_issue(self, github):
        emulator, server = github
        outcome = _connector(server).execute(self._supply_chain_action())
        assert outcome.succeeded and len(emulator.issues) == 1
        issue = emulator.issues[0]
        assert issue["labels"] == ["sentinel-mesh"]
        assert "pkg:pypi/leftpad" in issue["title"]
        assert emulator.pulls == []

    def test_execute_is_idempotent(self, github):
        emulator, server = github
        connector = _connector(server)
        action = self._supply_chain_action()
        connector.execute(action)
        again = connector.execute(action)
        assert "already open" in again.detail and len(emulator.issues) == 1

    def test_execute_without_the_issue_capability_is_refused(self, github):
        emulator, server = github
        connector = _connector(server, scopes=("contents:write", "pull_requests:write"),
                               caps=(Capability.PR_OPEN_DRAFT,))
        with pytest.raises(GuardrailViolation, match=r"issue.open"):
            connector.execute(self._supply_chain_action())
        assert emulator.requests == []
