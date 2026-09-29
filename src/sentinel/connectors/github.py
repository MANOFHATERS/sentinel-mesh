"""The Git connector: a draft pull request on GitHub, and nothing more (F-07, Section 5.4).

PRD Section 5.4: *"Patches are opened as draft PRs for human merge, never
auto-merged."* The Part 3 stand-in guaranteed that by having no ``merge`` method.
This connector talks to a real API, where the same token that opens a pull request
could merge it, so the guarantee has to live somewhere a code change cannot reach:

*   **The egress policy has no merge route.** Every route is ``GET`` or ``POST`` on
    the git data and pulls/issues endpoints. There is no ``PUT``, ``PATCH`` or
    ``DELETE`` in the allowlist at all, so merging, closing, force-pushing a ref,
    editing a PR out of draft or changing a repository setting are each a
    :class:`~sentinel.connectors.base.EgressDenied`, not a discipline.
*   **The request says ``draft: true`` and the response is checked.** A repository
    whose plan does not support drafts answers ``422``; one that silently ignored the
    flag would return ``draft: false``, and the connector reports that as a failure
    rather than calling it done.
*   **The approval is bound to the branch.** The action's ``target`` is the draft's
    branch, and the connector refuses a draft whose branch differs from what the
    human approved.

Why the git data API rather than the contents API
--------------------------------------------------
``PUT /contents/{path}`` commits one file per call, so a three-file patch would land
as three commits with the branch briefly in states nobody reviewed, and it is also a
``PUT`` route that would widen the allowlist. The data API builds blobs, one tree
and one commit, and only then points a new ref at it: the branch appears atomically,
fully formed, or not at all.

What the connector verifies before it writes anything
------------------------------------------------------
Every touched file is fetched **at the base commit** and compared byte-for-byte with
the text the scan validated its patches against. If the default branch moved since
the scan, the diffs were validated against a file that no longer exists, and the
connector refuses — the failure is a visible ``ACTION_FAILED`` naming the file, not a
pull request with a hunk applied at the wrong offset. A file with CRLF line endings
is refused for the same reason: the scan normalises to LF, and pushing the result
would rewrite every line of the file.
"""

from __future__ import annotations

import ast
import base64
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final

from sentinel.connectors.base import (
    Capability,
    ConnectorError,
    Credential,
    ExecutionOutcome,
    LeastPrivilegeError,
    check_scopes,
    require_executable,
)
from sentinel.connectors.http import (
    CallRecord,
    EgressPolicy,
    HttpResponse,
    RetryPolicy,
    Route,
    ScopedHttpClient,
    Transport,
    UrllibTransport,
    quote_segment,
)
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import ActionRequest, ActionType
from sentinel.scan.patch import PatchError, PullRequestDraft, compose_patches, normalise_source

__all__ = [
    "FORBIDDEN_CLASSIC_SCOPES",
    "GITHUB_API",
    "SCOPES_FOR_CAPABILITY",
    "GitHubConnector",
]

GITHUB_API: Final[str] = "https://api.github.com"

#: Fine-grained token permissions each capability needs. ``metadata:read`` is
#: implied by every fine-grained token and is permitted without being required.
SCOPES_FOR_CAPABILITY: Final[dict[Capability, frozenset[str]]] = {
    Capability.PR_OPEN_DRAFT: frozenset({"contents:write", "pull_requests:write"}),
    Capability.ISSUE_OPEN: frozenset({"issues:write"}),
}

#: Classic OAuth scopes that are refused if GitHub reports the token holds them. A
#: classic token cannot be narrowed to one repository, so these are the ones whose
#: presence means the token is an administrator's, whatever it was declared as.
FORBIDDEN_CLASSIC_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "admin:org",
        "write:org",
        "admin:enterprise",
        "delete_repo",
        "admin:repo_hook",
        "admin:org_hook",
        "admin:public_key",
        "admin:gpg_key",
        "workflow",
        "user",
        "write:packages",
        "delete:packages",
        "site_admin",
    }
)

_OWNER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_SHA: Final[str] = r"[0-9a-f]{40}"
_ACTION_TRAILER: Final[str] = "Sentinel-Action"
#: GitHub rejects PR and issue bodies above 65,536 characters.
_MAX_BODY: Final[int] = 60_000


@dataclass(frozen=True, slots=True)
class _FileChange:
    path: str
    mode: str
    before: str
    after: str


class GitHubConnector:
    """Opens draft pull requests (and, optionally, tracking issues) on one repository."""

    name = "github"

    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        credential: Credential,
        capabilities: Iterable[Capability] = (Capability.PR_OPEN_DRAFT,),
        base_url: str = GITHUB_API,
        base_branch: str | None = None,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        allow_insecure_loopback: bool = False,
        issue_labels: tuple[str, ...] = ("sentinel-mesh",),
    ) -> None:
        if not _OWNER.match(owner):
            raise ValueError(f"invalid GitHub owner {owner!r}")
        if not _REPO.match(repo) or repo in {".", ".."}:
            raise ValueError(f"invalid GitHub repository name {repo!r}")
        caps = frozenset(capabilities)
        unsupported = caps - SCOPES_FOR_CAPABILITY.keys()
        if unsupported:
            raise LeastPrivilegeError(
                f"github: cannot hold {sorted(c.value for c in unsupported)}; this "
                "connector only opens draft pull requests and issues"
            )
        if not caps:
            raise LeastPrivilegeError("github: a connector with no capabilities does nothing")
        required = frozenset().union(*(SCOPES_FOR_CAPABILITY[c] for c in caps))
        check_scopes(
            connector=self.name,
            declared=credential.scopes,
            required=required,
            permitted_extra=("metadata:read",),
        )
        self.owner = owner
        self.repo = repo
        self.base_branch = base_branch
        self.issue_labels = issue_labels
        self._capabilities = caps
        self.http = ScopedHttpClient(
            connector=self.name,
            policy=EgressPolicy(
                base_url=base_url,
                routes=self._routes(caps),
                allow_insecure_loopback=allow_insecure_loopback,
            ),
            transport=transport or UrllibTransport(),
            auth=lambda: {"Authorization": f"Bearer {credential.secret.reveal()}"},
            retry=retry or RetryPolicy(),
            base_headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            **({"sleep": sleep} if sleep is not None else {}),
        )

    # --- the connector protocol ------------------------------------------------ #

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def observe(self, callback: Callable[[CallRecord], None] | None) -> None:
        self.http.on_call = callback

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        """Open a remediation *issue* for an ``OPEN_PATCH_PR`` that carries no diff.

        The Supply-Chain Agent proposes ``OPEN_PATCH_PR`` against a dependency node,
        and on the sprint's synthetic graph there is no manifest to edit — so the
        honest real-world effect is a tracking issue a maintainer can act on, which
        needs ``issues:write`` and nothing that touches code.
        """
        require_executable(action, connector=self.name)
        if action.action_type is not ActionType.OPEN_PATCH_PR:
            raise GuardrailViolation(
                f"github: cannot execute {action.action_type.value}; it opens pull "
                "requests and issues only"
            )
        if Capability.ISSUE_OPEN not in self._capabilities:
            raise GuardrailViolation(
                "github: this action carries no diff, so it would become an issue, and "
                "this connector was not granted issue.open"
            )
        self._repository()
        marker = _marker(action.action_id)
        existing = self.http.request(
            "GET",
            self._path("issues"),
            query={"state": "open", "labels": ",".join(self.issue_labels), "per_page": "100"},
        ).json()
        for issue in existing or []:
            if marker in str(issue.get("body") or ""):
                return ExecutionOutcome(
                    succeeded=True,
                    detail=f"remediation issue #{issue.get('number')} already open",
                    reference=issue.get("html_url"),
                )
        body = _clip(
            f"{action.rationale}\n\n"
            f"Proposed by `{action.proposed_by.value}`, approved by "
            f"`{action.approved_by or 'n/a (unattended tier)'}`.\n\n"
            + "\n".join(f"- `{item.ref}`" for item in action.evidence[:16])
            + f"\n\n{marker}\n"
        )
        created = self.http.request(
            "POST",
            self._path("issues"),
            json_body={
                "title": f"Supply-chain remediation: {action.target}"[:256],
                "body": body,
                "labels": list(self.issue_labels),
            },
            expect=(201,),
        ).json()
        return ExecutionOutcome(
            succeeded=True,
            detail=f"opened remediation issue #{created.get('number')} for {action.target}",
            reference=created.get("html_url"),
        )

    def open_draft(self, action: ActionRequest, draft: PullRequestDraft) -> ExecutionOutcome:
        """Push ``draft`` as a new branch and open it as a **draft** pull request."""
        require_executable(action, connector=self.name)
        if action.action_type is not ActionType.OPEN_PATCH_PR:
            raise GuardrailViolation(
                f"github: this connector opens pull requests; it was handed "
                f"{action.action_type.value}"
            )
        if Capability.PR_OPEN_DRAFT not in self._capabilities:
            raise GuardrailViolation("github: this connector was not granted pr.open_draft")
        if action.target != draft.branch:
            raise GuardrailViolation(
                f"github: the approval is for branch {action.target!r} but the draft is "
                f"for {draft.branch!r}; refusing to push something other than what was "
                "approved"
            )

        repository = self._repository()
        base = self.base_branch or str(repository.get("default_branch") or "main")
        base_sha = self._ref_sha(base)
        if base_sha is None:
            raise ConnectorError(f"github: base branch {base!r} does not exist")

        marker = _marker(action.action_id)
        head_sha = self._ref_sha(draft.branch)
        if head_sha is not None:
            existing = self._open_pull(draft.branch)
            if existing is not None:
                return ExecutionOutcome(
                    succeeded=True,
                    detail=f"draft PR #{existing.get('number')} already open on {draft.branch}",
                    reference=existing.get("html_url"),
                )
            commit = self._get(self._path("git", "commits", head_sha))
            if marker not in str(commit.get("message") or ""):
                raise GuardrailViolation(
                    f"github: branch {draft.branch!r} already exists and was not created by "
                    "this action; refusing to push over someone else's work"
                )
        else:
            changes = self._file_changes(draft, base_sha)
            head_sha = self._commit(changes, base_sha, draft, action, marker)
            created = self.http.request(
                "POST",
                self._path("git", "refs"),
                json_body={"ref": f"refs/heads/{draft.branch}", "sha": head_sha},
                expect=(201, 422),
            )
            if created.status == 422:
                # Lost a race with a concurrent run of the same action, or a stranger.
                racing = self._ref_sha(draft.branch)
                if racing is None:
                    raise ConnectorError(
                        f"github: creating {draft.branch!r} reported a conflict, but the "
                        "branch does not exist"
                    )
                if racing != head_sha:
                    commit = self._get(self._path("git", "commits", racing))
                    if marker not in str(commit.get("message") or ""):
                        raise GuardrailViolation(
                            f"github: branch {draft.branch!r} was created concurrently by "
                            "something other than this action"
                        )

        pull = self._open_pull(draft.branch)
        if pull is None:
            pull = self.http.request(
                "POST",
                self._path("pulls"),
                json_body={
                    "title": draft.title[:256],
                    "head": draft.branch,
                    "base": base,
                    "body": _clip(f"{draft.body}\n\n{marker}\n"),
                    "draft": True,
                    "maintainer_can_modify": False,
                },
                expect=(201,),
            ).json()
        if pull.get("draft") is not True:
            raise ConnectorError(
                f"github: pull request #{pull.get('number')} was opened but is not a draft; "
                "the repository ignored the draft flag. It must be converted or closed by "
                "a human — this connector has no route that could do either"
            )
        return ExecutionOutcome(
            succeeded=True,
            detail=(
                f"opened draft PR #{pull.get('number')} on {draft.branch} with "
                f"{len(draft.patches)} patch(es) across {len(draft.files_touched)} file(s)"
            ),
            reference=pull.get("html_url"),
        )

    # --- internals --------------------------------------------------------------- #

    def _routes(self, caps: frozenset[Capability]) -> tuple[Route, ...]:
        prefix = f"/repos/{re.escape(self.owner)}/{re.escape(self.repo)}"
        routes = [Route("repo.get", "GET", prefix)]
        if Capability.PR_OPEN_DRAFT in caps:
            branch = r"[A-Za-z0-9._%-]+(?:/[A-Za-z0-9._%-]+)*"
            routes += [
                Route("ref.get", "GET", rf"{prefix}/git/ref/heads/{branch}"),
                Route("commit.get", "GET", rf"{prefix}/git/commits/{_SHA}"),
                Route("tree.get", "GET", rf"{prefix}/git/trees/{_SHA}", query={"recursive"}),
                Route("blob.get", "GET", rf"{prefix}/git/blobs/{_SHA}"),
                Route("blob.create", "POST", rf"{prefix}/git/blobs"),
                Route("tree.create", "POST", rf"{prefix}/git/trees"),
                Route("commit.create", "POST", rf"{prefix}/git/commits"),
                Route("ref.create", "POST", rf"{prefix}/git/refs"),
                Route("pulls.list", "GET", rf"{prefix}/pulls", query={"head", "state", "base"}),
                Route("pulls.create", "POST", rf"{prefix}/pulls"),
            ]
        if Capability.ISSUE_OPEN in caps:
            routes += [
                Route(
                    "issues.list",
                    "GET",
                    rf"{prefix}/issues",
                    query={"state", "labels", "per_page"},
                ),
                Route("issues.create", "POST", rf"{prefix}/issues"),
            ]
        # Every route is GET or POST. Asserted, because the day someone adds a PUT
        # here is the day the "no merge" guarantee needs a new argument.
        assert all(route.method in {"GET", "POST"} for route in routes)
        return tuple(routes)

    def _path(self, *segments: str) -> str:
        return "/".join(
            ["", "repos", quote_segment(self.owner), quote_segment(self.repo)]
            + [quote_segment(s) for s in segments]
        )

    def _get(self, path: str, query: dict[str, str] | None = None) -> dict[str, Any]:
        body = self.http.request("GET", path, query=query).json()
        if not isinstance(body, dict):
            raise ConnectorError(f"github: expected an object from GET {path}")
        return body

    def _repository(self) -> dict[str, Any]:
        response = self.http.request("GET", self._path())
        self._check_reported_scopes(response)
        body = response.json()
        if not isinstance(body, dict):
            raise ConnectorError("github: repository lookup returned no object")
        return body

    def _check_reported_scopes(self, response: HttpResponse) -> None:
        """Hold a classic token to the forbidden list, from GitHub's own report.

        Runs on the first call, which is a read, so a token that turns out to be an
        administrator's is refused before this connector has written anything.
        Fine-grained tokens send no ``X-OAuth-Scopes`` header and are held to their
        declaration by :func:`~sentinel.connectors.base.check_scopes` instead.
        """
        header = response.headers.get("x-oauth-scopes")
        if header is None:
            return
        reported = {scope.strip() for scope in header.split(",") if scope.strip()}
        forbidden = reported & FORBIDDEN_CLASSIC_SCOPES
        if forbidden:
            raise LeastPrivilegeError(
                f"github: the token GitHub authenticated holds {sorted(forbidden)}; "
                "refusing to act with an administrator's credential (PRD Section 5.7)"
            )

    def _ref_sha(self, branch: str) -> str | None:
        path = self._path("git", "ref", "heads") + "/" + "/".join(
            quote_segment(part) for part in branch.split("/")
        )
        response = self.http.request("GET", path, expect=(200, 404))
        if response.status == 404:
            return None
        body = response.json()
        sha = (body or {}).get("object", {}).get("sha")
        if not isinstance(sha, str) or not re.fullmatch(_SHA, sha):
            raise ConnectorError(f"github: malformed ref response for {branch!r}")
        return sha

    def _open_pull(self, branch: str) -> dict[str, Any] | None:
        pulls = self.http.request(
            "GET",
            self._path("pulls"),
            query={"head": f"{self.owner}:{branch}", "state": "open"},
        ).json()
        for pull in pulls or []:
            if (pull.get("head") or {}).get("ref") == branch:
                return pull
        return None

    def _file_changes(self, draft: PullRequestDraft, base_sha: str) -> list[_FileChange]:
        commit = self._get(self._path("git", "commits", base_sha))
        tree_sha = str((commit.get("tree") or {}).get("sha") or "")
        tree = self._get(self._path("git", "trees", tree_sha), query={"recursive": "1"})
        entries = {
            str(item.get("path")): item
            for item in tree.get("tree") or []
            if isinstance(item, dict)
        }
        changes: list[_FileChange] = []
        for path in draft.files_touched:
            entry = entries.get(path)
            if entry is None:
                hint = " (the tree listing was truncated)" if tree.get("truncated") else ""
                raise ConnectorError(f"github: {path} is not in the base tree{hint}")
            if entry.get("type") != "blob" or entry.get("mode") not in {"100644", "100755"}:
                raise GuardrailViolation(
                    f"github: {path} is a {entry.get('type')} with mode {entry.get('mode')}, "
                    "not a regular file; refusing to write through it"
                )
            blob = self._get(self._path("git", "blobs", str(entry.get("sha"))))
            if blob.get("encoding") != "base64":
                raise ConnectorError(f"github: unexpected blob encoding for {path}")
            try:
                remote = base64.b64decode(str(blob.get("content") or "")).decode("utf-8")
            except (ValueError, UnicodeDecodeError) as exc:
                raise ConnectorError(f"github: {path} is not UTF-8 text: {exc}") from exc
            if remote != normalise_source(remote):
                raise ConnectorError(
                    f"github: {path} uses CRLF line endings or lacks a final newline; the "
                    "validated diffs assume LF and pushing them would rewrite every line"
                )
            patches = [p for p in draft.patches if p.path == path]
            if any(p.before != remote for p in patches):
                raise ConnectorError(
                    f"github: {path} on the base branch differs from the version that was "
                    "scanned; the patches were validated against a file that no longer "
                    "exists, so a fresh scan is needed"
                )
            try:
                after = compose_patches(remote, [p.diff for p in patches])
            except PatchError as exc:
                raise ConnectorError(f"github: patches to {path} do not compose: {exc}") from exc
            if path.endswith(".py"):
                try:
                    ast.parse(after, filename=path)
                except SyntaxError as exc:
                    raise ConnectorError(
                        f"github: the composed {path} does not parse ({exc.msg}); refusing "
                        "to push it"
                    ) from exc
            changes.append(_FileChange(path, str(entry.get("mode")), remote, after))
        return changes

    def _commit(
        self,
        changes: list[_FileChange],
        base_sha: str,
        draft: PullRequestDraft,
        action: ActionRequest,
        marker: str,
    ) -> str:
        tree_items = []
        for change in changes:
            blob = self.http.request(
                "POST",
                self._path("git", "blobs"),
                json_body={"content": change.after, "encoding": "utf-8"},
                expect=(201,),
            ).json()
            tree_items.append(
                {"path": change.path, "mode": change.mode, "type": "blob", "sha": blob["sha"]}
            )
        base_commit = self._get(self._path("git", "commits", base_sha))
        tree = self.http.request(
            "POST",
            self._path("git", "trees"),
            json_body={"base_tree": base_commit["tree"]["sha"], "tree": tree_items},
            expect=(201,),
        ).json()
        message = (
            f"{draft.title}\n\n"
            f"{len(draft.patches)} mechanical patch(es) from the Sentinel Mesh Code-Scan "
            "Agent. Opened as a draft for human review; never merged automatically.\n\n"
            f"{_ACTION_TRAILER}: {action.action_id}\n"
            f"Approved-by: {action.approved_by or 'n/a (unattended tier)'}\n"
            f"{marker}\n"
        )
        commit = self.http.request(
            "POST",
            self._path("git", "commits"),
            json_body={"message": message, "tree": tree["sha"], "parents": [base_sha]},
            expect=(201,),
        ).json()
        sha = commit.get("sha")
        if not isinstance(sha, str) or not re.fullmatch(_SHA, sha):
            raise ConnectorError("github: commit creation returned no sha")
        return sha


def _marker(action_id: str) -> str:
    """A hidden, greppable marker binding a remote object to the action that made it."""
    return f"<!-- sentinel-action: {action_id} -->"


def _clip(body: str) -> str:
    if len(body) <= _MAX_BODY:
        return body
    return body[: _MAX_BODY - 80] + "\n\n_(truncated by Sentinel Mesh to fit GitHub's limit)_\n"

