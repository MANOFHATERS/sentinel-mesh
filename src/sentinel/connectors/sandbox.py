"""Local emulators of the connector APIs, served over real sockets.

Why emulators and not mocks
---------------------------
A mock of ``GitHubConnector.open_draft`` tests nothing about GitHub; a mock of the
transport tests the connector against the author's memory of the API. These are
HTTP servers on ``127.0.0.1`` that implement the documented request and response
shapes of the endpoints each connector uses, and they are **strict**: a missing auth
header is a ``401``, a malformed body is a ``4xx``, a tree that names a blob nobody
created is a ``422``. The connectors reach them through
:class:`~sentinel.connectors.http.UrllibTransport` — the same transport, the same
redirect handling and the same size limits they would use in production — so a test
that passes here has exercised every byte the connector puts on the wire.

What they are not: proof that the real services agree. They implement the APIs as
documented (GitHub REST ``2022-11-28``, Wazuh server API 4.x, SCIM 2.0 RFC 7644,
Slack incoming webhooks), and the ledger says in so many words that no request has
been sent to a real instance.

They also implement the endpoints the connectors must **never** call —
``PUT .../pulls/{n}/merge``, ``PATCH .../pulls/{n}`` — so a test can assert those
counters stayed at zero rather than inferring it from a missing method.

Every emulator records every request (method, path, query, headers, body) and can
be told to fail the next N requests with a given status, which is how retries,
``Retry-After`` and redirect refusal are tested against a live socket.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import urllib.parse
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sentinel.connectors.base import Capability, Credential, Secret
from sentinel.connectors.github import GitHubConnector
from sentinel.connectors.journal import BlastRadiusLimiter, ExecutionJournal
from sentinel.connectors.notify import (
    DELIVERY_HEADER,
    SIGNATURE_HEADER,
    LocalEnrichmentConnector,
    SignedWebhookConnector,
    SlackWebhookConnector,
    verify_signature,
)
from sentinel.connectors.router import ConnectorRouter
from sentinel.connectors.scim import PATCH_OP_SCHEMA, ScimIdentityConnector
from sentinel.connectors.targets import TargetPolicy
from sentinel.connectors.wazuh import WazuhConnector
from sentinel.core.clock import Clock, SystemClock

__all__ = [
    "Fault",
    "GitHubEmulator",
    "LiveServer",
    "RecordedRequest",
    "Sandbox",
    "ScimEmulator",
    "SlackEmulator",
    "WazuhEmulator",
    "WebhookReceiver",
]

Reply = tuple[int, dict[str, str], bytes]


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8")) if self.body else None


@dataclass(slots=True)
class Fault:
    """Answer the next ``times`` requests with this, instead of the real handler."""

    status: int
    times: int = 1
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b'{"message":"injected fault"}'


def _json(status: int, body: Any, headers: Mapping[str, str] | None = None) -> Reply:
    merged = {"Content-Type": "application/json"}
    merged.update(headers or {})
    return status, merged, json.dumps(body).encode("utf-8")


class EmulatedService:
    """Base class: request recording, fault injection, and dispatch."""

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.faults: deque[Fault] = deque()
        self.lock = threading.RLock()

    def inject(self, status: int, *, times: int = 1, headers: dict[str, str] | None = None,
               body: bytes = b'{"message":"injected fault"}') -> None:
        if times < 1:
            raise ValueError("a fault must fire at least once")
        with self.lock:
            self.faults.append(Fault(status, times, dict(headers or {}), body))

    def calls(self, method: str | None = None, pattern: str | None = None) -> list[RecordedRequest]:
        with self.lock:
            return [
                r
                for r in self.requests
                if (method is None or r.method == method)
                and (pattern is None or re.search(pattern, r.path))
            ]

    def handle(self, request: RecordedRequest) -> Reply:
        with self.lock:
            self.requests.append(request)
            if self.faults:
                fault = self.faults[0]
                fault.times -= 1
                if fault.times <= 0:
                    self.faults.popleft()
                headers = {"Content-Type": "application/json", **fault.headers}
                return fault.status, headers, fault.body
            return self.route(request)

    def route(self, request: RecordedRequest) -> Reply:  # pragma: no cover - abstract
        raise NotImplementedError


class _Handler(BaseHTTPRequestHandler):
    server_version = "sentinel-sandbox/1"
    protocol_version = "HTTP/1.1"

    def _serve(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        request = RecordedRequest(
            method=self.command,
            path=parsed.path,
            query={k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()},
            headers={k.lower(): v for k, v in self.headers.items()},
            body=body,
        )
        try:
            status, headers, payload = self.server.service.handle(request)  # type: ignore[attr-defined]
        except Exception as exc:  # an emulator bug must surface as a 500, not a hang
            status, headers, payload = _json(500, {"message": f"emulator error: {exc}"})
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload and self.command != "HEAD":
            self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _serve

    def log_message(self, format: str, *args: object) -> None:
        return


class LiveServer:
    """A threaded HTTP server on an ephemeral loopback port."""

    def __init__(self, service: EmulatedService) -> None:
        self.service = service
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.service = service  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> LiveServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> LiveServer:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #


def _git_blob_sha(content: bytes) -> str:
    """The real git object id of a blob, so the emulator's shas look like GitHub's."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def _object_sha(kind: str, value: Any) -> str:
    material = json.dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha1(kind.encode() + b"\0" + material).hexdigest()


class GitHubEmulator(EmulatedService):
    """GitHub REST API (2022-11-28): repos, git data, pulls, issues — and merge."""

    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        files: Mapping[str, str],
        token: str,
        default_branch: str = "main",
        reported_scopes: str | None = None,
        supports_drafts: bool = True,
        ignore_draft_flag: bool = False,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.repo = repo
        self.token = token
        self.default_branch = default_branch
        self.reported_scopes = reported_scopes
        self.supports_drafts = supports_drafts
        self.ignore_draft_flag = ignore_draft_flag
        self.blobs: dict[str, bytes] = {}
        self.trees: dict[str, dict[str, dict[str, str]]] = {}
        self.commits: dict[str, dict[str, Any]] = {}
        self.refs: dict[str, str] = {}
        self.pulls: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []
        self.merges = 0
        self.pull_edits = 0
        entries = {}
        for path, text in files.items():
            sha = self._store_blob(text.encode("utf-8"))
            entries[path] = {"path": path, "mode": "100644", "type": "blob", "sha": sha}
        tree_sha = self._store_tree(entries)
        root = self._store_commit(tree_sha, [], "initial import")
        self.refs[f"heads/{default_branch}"] = root

    # --- state helpers, also used by tests ------------------------------------- #

    def _store_blob(self, content: bytes) -> str:
        sha = _git_blob_sha(content)
        self.blobs[sha] = content
        return sha

    def _store_tree(self, entries: dict[str, dict[str, str]]) -> str:
        sha = _object_sha("tree", entries)
        self.trees[sha] = entries
        return sha

    def _store_commit(self, tree: str, parents: list[str], message: str) -> str:
        sha = _object_sha("commit", {"tree": tree, "parents": parents, "message": message})
        self.commits[sha] = {"tree": tree, "parents": parents, "message": message}
        return sha

    def file_at(self, branch: str, path: str) -> str | None:
        with self.lock:
            commit = self.commits[self.refs[f"heads/{branch}"]]
            entry = self.trees[commit["tree"]].get(path)
            return None if entry is None else self.blobs[entry["sha"]].decode("utf-8")

    def push_to_default(self, path: str, text: str, message: str = "upstream change") -> None:
        """Simulate someone else moving the default branch after the scan."""
        with self.lock:
            ref = f"heads/{self.default_branch}"
            base = self.commits[self.refs[ref]]
            entries = dict(self.trees[base["tree"]])
            sha = self._store_blob(text.encode("utf-8"))
            entries[path] = {"path": path, "mode": "100644", "type": "blob", "sha": sha}
            self.refs[ref] = self._store_commit(self._store_tree(entries), [self.refs[ref]],
                                                message)

    # --- routing ---------------------------------------------------------------- #

    def route(self, request: RecordedRequest) -> Reply:
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return _json(401, {"message": "Bad credentials"})
        prefix = f"/repos/{self.owner}/{self.repo}"
        path = urllib.parse.unquote(request.path)
        if not path.startswith(prefix):
            return _json(404, {"message": "Not Found"})
        rest = path[len(prefix):]
        method = request.method
        headers = {"X-OAuth-Scopes": self.reported_scopes} if self.reported_scopes else {}

        if method == "GET" and rest == "":
            return _json(200, {"full_name": f"{self.owner}/{self.repo}",
                               "default_branch": self.default_branch, "private": True}, headers)
        if method == "GET" and rest.startswith("/git/ref/heads/"):
            ref = rest[len("/git/ref/"):]
            sha = self.refs.get(ref)
            if sha is None:
                return _json(404, {"message": "Not Found"})
            return _json(200, {"ref": f"refs/{ref}", "object": {"sha": sha, "type": "commit"}})
        if method == "GET" and (m := re.fullmatch(r"/git/commits/([0-9a-f]{40})", rest)):
            commit = self.commits.get(m.group(1))
            if commit is None:
                return _json(404, {"message": "Not Found"})
            return _json(200, {"sha": m.group(1), "tree": {"sha": commit["tree"]},
                               "message": commit["message"],
                               "parents": [{"sha": p} for p in commit["parents"]]})
        if method == "GET" and (m := re.fullmatch(r"/git/trees/([0-9a-f]{40})", rest)):
            tree = self.trees.get(m.group(1))
            if tree is None:
                return _json(404, {"message": "Not Found"})
            return _json(200, {"sha": m.group(1), "truncated": False,
                               "tree": [dict(e, size=len(self.blobs[e["sha"]]))
                                        for e in tree.values()]})
        if method == "GET" and (m := re.fullmatch(r"/git/blobs/([0-9a-f]{40})", rest)):
            blob = self.blobs.get(m.group(1))
            if blob is None:
                return _json(404, {"message": "Not Found"})
            encoded = base64.b64encode(blob).decode("ascii")
            # GitHub wraps base64 at 60 columns; the connector must cope.
            wrapped = "\n".join(encoded[i:i + 60] for i in range(0, len(encoded), 60))
            return _json(200, {"sha": m.group(1), "encoding": "base64", "content": wrapped,
                               "size": len(blob)})
        if method == "POST" and rest == "/git/blobs":
            body = request.json() or {}
            if body.get("encoding") not in {"utf-8", "base64"} or "content" not in body:
                return _json(422, {"message": "Invalid request"})
            raw = (body["content"].encode("utf-8") if body["encoding"] == "utf-8"
                   else base64.b64decode(body["content"]))
            return _json(201, {"sha": self._store_blob(raw)})
        if method == "POST" and rest == "/git/trees":
            body = request.json() or {}
            base = body.get("base_tree")
            entries = dict(self.trees.get(base, {})) if base else {}
            if base and base not in self.trees:
                return _json(422, {"message": "base_tree not found"})
            for item in body.get("tree") or []:
                if item.get("type") != "blob" or item.get("sha") not in self.blobs:
                    return _json(422, {"message": f"tree entry {item.get('path')} is invalid"})
                if item.get("mode") not in {"100644", "100755"}:
                    return _json(422, {"message": "invalid mode"})
                entries[item["path"]] = {"path": item["path"], "mode": item["mode"],
                                         "type": "blob", "sha": item["sha"]}
            return _json(201, {"sha": self._store_tree(entries)})
        if method == "POST" and rest == "/git/commits":
            body = request.json() or {}
            if body.get("tree") not in self.trees:
                return _json(422, {"message": "tree not found"})
            if any(p not in self.commits for p in body.get("parents") or []):
                return _json(422, {"message": "parent not found"})
            if not body.get("message"):
                return _json(422, {"message": "message required"})
            return _json(201, {"sha": self._store_commit(body["tree"], body.get("parents") or [],
                                                         body["message"])})
        if method == "POST" and rest == "/git/refs":
            body = request.json() or {}
            ref = str(body.get("ref") or "")
            if not ref.startswith("refs/heads/") or body.get("sha") not in self.commits:
                return _json(422, {"message": "Invalid request"})
            key = ref[len("refs/"):]
            if key in self.refs:
                return _json(422, {"message": "Reference already exists"})
            self.refs[key] = body["sha"]
            return _json(201, {"ref": ref, "object": {"sha": body["sha"]}})
        if method == "GET" and rest == "/pulls":
            head = request.query.get("head", "")
            state = request.query.get("state", "open")
            branch = head.split(":", 1)[1] if ":" in head else head
            found = [p for p in self.pulls
                     if (not head or p["head"]["ref"] == branch)
                     and (state == "all" or p["state"] == state)]
            return _json(200, found)
        if method == "POST" and rest == "/pulls":
            body = request.json() or {}
            head, base = body.get("head"), body.get("base")
            if f"heads/{head}" not in self.refs or f"heads/{base}" not in self.refs:
                return _json(422, {"message": "Validation Failed"})
            if head == base:
                return _json(422, {"message": "head and base are the same"})
            if body.get("draft") and not self.supports_drafts:
                return _json(422, {"message": "Draft pull requests are not supported in "
                                              "this repository."})
            number = len(self.pulls) + len(self.issues) + 1
            pull = {
                "number": number,
                "html_url": f"https://github.example/{self.owner}/{self.repo}/pull/{number}",
                "state": "open",
                "draft": bool(body.get("draft")) and not self.ignore_draft_flag,
                "title": body.get("title"),
                "body": body.get("body"),
                "head": {"ref": head},
                "base": {"ref": base},
                "merged": False,
                "maintainer_can_modify": body.get("maintainer_can_modify", True),
            }
            self.pulls.append(pull)
            return _json(201, pull)
        if method == "PUT" and (m := re.fullmatch(r"/pulls/(\d+)/merge", rest)):
            # Implemented so a test can assert it was never reached.
            self.merges += 1
            for pull in self.pulls:
                if pull["number"] == int(m.group(1)):
                    pull["merged"] = True
                    pull["state"] = "closed"
            return _json(200, {"merged": True})
        if method == "PATCH" and re.fullmatch(r"/pulls/\d+", rest):
            self.pull_edits += 1
            return _json(200, {})
        if method == "GET" and rest == "/issues":
            return _json(200, [i for i in self.issues if i["state"] == request.query.get(
                "state", "open")])
        if method == "POST" and rest == "/issues":
            body = request.json() or {}
            if not body.get("title"):
                return _json(422, {"message": "title required"})
            number = len(self.pulls) + len(self.issues) + 1
            issue = {"number": number, "state": "open", "title": body["title"],
                     "body": body.get("body"), "labels": body.get("labels") or [],
                     "html_url": f"https://github.example/{self.owner}/{self.repo}/issues/{number}"}
            self.issues.append(issue)
            return _json(201, issue)
        return _json(404, {"message": "Not Found"})


# --------------------------------------------------------------------------- #
# Wazuh
# --------------------------------------------------------------------------- #


class WazuhEmulator(EmulatedService):
    """Wazuh server API 4.x: authenticate, agents, active-response."""

    def __init__(self, *, username: str, password: str, agents: Iterable[Mapping[str, str]],
                 commands: Iterable[str] = ("!firewall-drop", "sentinel-isolate")) -> None:
        super().__init__()
        self.username = username
        self.password = password
        self.agents: dict[str, dict[str, str]] = {a["id"]: dict(a) for a in agents}
        self.commands = frozenset(commands)
        self.tokens: set[str] = set()
        self.executed: list[dict[str, Any]] = []
        self.authentications = 0

    def enroll(self, agent_id: str, name: str, ip: str, status: str = "active") -> None:
        with self.lock:
            self.agents[agent_id] = {"id": agent_id, "name": name, "ip": ip, "status": status}

    def revoke_tokens(self) -> None:
        with self.lock:
            self.tokens.clear()

    def isolated_hosts(self) -> set[str]:
        return {e["agent"]["ip"] for e in self.executed if e["command"] == "sentinel-isolate"}

    def blocked_addresses(self) -> set[str]:
        return {e["alert"]["data"]["srcip"] for e in self.executed
                if e["command"] == "!firewall-drop"}

    def route(self, request: RecordedRequest) -> Reply:
        if request.method == "POST" and request.path == "/security/user/authenticate":
            expected = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            if request.headers.get("authorization") != f"Basic {expected}":
                return _json(401, {"title": "Unauthorized", "error": 401})
            token = secrets.token_hex(16)
            self.tokens.add(token)
            self.authentications += 1
            return _json(200, {"data": {"token": token}, "error": 0})
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[len("Bearer "):] not in self.tokens:
            return _json(401, {"title": "Unauthorized", "detail": "Invalid token", "error": 401})

        if request.method == "GET" and request.path == "/agents":
            items = list(self.agents.values())
            if "ip" in request.query:
                items = [a for a in items if a["ip"] == request.query["ip"]]
            if "name" in request.query:
                items = [a for a in items if a["name"] == request.query["name"]]
            limit = int(request.query.get("limit", "500"))
            return _json(200, {"data": {"affected_items": items[:limit],
                                        "total_affected_items": len(items),
                                        "failed_items": [], "total_failed_items": 0},
                               "message": "All selected agents information was returned",
                               "error": 0})
        if request.method == "PUT" and request.path == "/active-response":
            body = request.json() or {}
            command = body.get("command")
            if command not in self.commands or not isinstance(body.get("arguments", []), list):
                return _json(400, {"title": "Bad Request", "detail": f"unknown command {command}",
                                   "error": 1652})
            affected, failed = [], []
            for agent_id in request.query.get("agents_list", "").split(","):
                agent = self.agents.get(agent_id)
                if agent is None or agent["status"] != "active":
                    failed.append({"error": {"code": 1707, "message": "Agent not active"},
                                   "id": [agent_id]})
                    continue
                affected.append(agent_id)
                self.executed.append({"command": command, "agent": dict(agent),
                                      "alert": body.get("alert") or {}})
            return _json(200, {"data": {"affected_items": affected,
                                        "total_affected_items": len(affected),
                                        "failed_items": failed,
                                        "total_failed_items": len(failed)},
                               "message": "AR command was sent to all agents" if not failed
                               else "AR command was not sent to some agents",
                               "error": 0 if not failed else 2})
        return _json(404, {"title": "Not Found", "error": 404})


# --------------------------------------------------------------------------- #
# SCIM
# --------------------------------------------------------------------------- #

_FILTER = re.compile(r'^userName eq "((?:[^"\\]|\\.)*)"$')


class ScimEmulator(EmulatedService):
    """SCIM 2.0 (RFC 7644) /Users search and PATCH."""

    def __init__(self, *, token: str, users: Iterable[Mapping[str, Any]]) -> None:
        super().__init__()
        self.token = token
        self.users: dict[str, dict[str, Any]] = {u["id"]: dict(u) for u in users}
        self.patches = 0

    def route(self, request: RecordedRequest) -> Reply:
        scim = {"Content-Type": "application/scim+json"}
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return _json(401, {"schemas": ["urn:ietf:params:scim:api:messages:2.0:Error"],
                               "status": "401"}, scim)
        if request.method == "GET" and request.path == "/Users":
            match = _FILTER.match(request.query.get("filter", ""))
            if match is None:
                return _json(400, {"scimType": "invalidFilter", "status": "400"}, scim)
            name = re.sub(r"\\(.)", r"\1", match.group(1))
            found = [u for u in self.users.values() if u["userName"].lower() == name.lower()]
            return _json(200, {"schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                               "totalResults": len(found), "Resources": found}, scim)
        if request.method == "PATCH" and (m := re.fullmatch(r"/Users/([^/]+)", request.path)):
            user = self.users.get(urllib.parse.unquote(m.group(1)))
            if user is None:
                return _json(404, {"status": "404"}, scim)
            if request.headers.get("content-type") != "application/scim+json":
                return _json(415, {"status": "415"}, scim)
            body = request.json() or {}
            if body.get("schemas") != [PATCH_OP_SCHEMA]:
                return _json(400, {"scimType": "invalidSyntax", "status": "400"}, scim)
            for op in body.get("Operations") or []:
                if op.get("op") != "replace" or op.get("path") != "active":
                    return _json(400, {"scimType": "mutability", "status": "400"}, scim)
                user["active"] = bool(op.get("value"))
            self.patches += 1
            return _json(200, user, scim)
        return _json(404, {"status": "404"}, scim)


# --------------------------------------------------------------------------- #
# Slack and generic webhooks
# --------------------------------------------------------------------------- #


class SlackEmulator(EmulatedService):
    """A Slack incoming webhook: 200 ``ok``, or Slack's documented error strings."""

    def __init__(self, *, path: str) -> None:
        super().__init__()
        self.path = path
        self.messages: list[dict[str, Any]] = []

    def route(self, request: RecordedRequest) -> Reply:
        if request.method != "POST" or request.path != self.path:
            return 404, {"Content-Type": "text/plain"}, b"no_service"
        try:
            body = request.json()
        except json.JSONDecodeError:
            return 400, {"Content-Type": "text/plain"}, b"invalid_payload"
        if not isinstance(body, dict) or not body.get("text"):
            return 400, {"Content-Type": "text/plain"}, b"no_text"
        self.messages.append(body)
        return 200, {"Content-Type": "text/plain"}, b"ok"


class WebhookReceiver(EmulatedService):
    """A receiver that verifies Sentinel's signature before accepting a delivery."""

    def __init__(self, *, path: str, secret: Secret, clock: Clock | None = None) -> None:
        super().__init__()
        self.path = path
        self.secret = secret
        self.clock = clock or SystemClock()
        self.deliveries: list[dict[str, Any]] = []
        self.rejected = 0

    def route(self, request: RecordedRequest) -> Reply:
        if request.method != "POST" or request.path != self.path:
            return _json(404, {"message": "not found"})
        header = request.headers.get(SIGNATURE_HEADER.lower(), "")
        if not verify_signature(self.secret, header, request.body, now=self.clock.now()):
            self.rejected += 1
            return _json(401, {"message": "bad signature"})
        delivery = request.headers.get(DELIVERY_HEADER.lower())
        if any(d["delivery"] == delivery for d in self.deliveries):
            return _json(200, {"duplicate": True})
        self.deliveries.append({"delivery": delivery, "body": request.json()})
        return _json(202, {"accepted": True})


# --------------------------------------------------------------------------- #
# The whole sandbox
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Sandbox:
    """Every emulator on its own loopback port, and a router wired to all of them.

    Credentials are generated per sandbox with :mod:`secrets` and never leave the
    process except to ``127.0.0.1``.
    """

    repo_files: Mapping[str, str] = field(default_factory=dict)
    owner: str = "acme"
    repo: str = "billing"
    hosts: Iterable[str] = ()
    users: Iterable[str] = ("alice@acme.example", "bob@acme.example")
    firewall_agent: str = "001"
    clock: Clock = field(default_factory=SystemClock)
    github: GitHubEmulator = field(init=False)
    wazuh: WazuhEmulator = field(init=False)
    scim: ScimEmulator = field(init=False)
    slack: SlackEmulator = field(init=False)
    webhook: WebhookReceiver = field(init=False)
    _servers: dict[str, LiveServer] = field(init=False, default_factory=dict)
    _secrets: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._secrets = {
            "github": secrets.token_urlsafe(24),
            "wazuh": secrets.token_urlsafe(24),
            "scim": secrets.token_urlsafe(24),
            "webhook": secrets.token_urlsafe(24),
            "slack": secrets.token_hex(12),
        }
        self.github = GitHubEmulator(owner=self.owner, repo=self.repo,
                                     files=dict(self.repo_files) or {"README.md": "hello\n"},
                                     token=self._secrets["github"])
        agents = [{"id": self.firewall_agent, "name": "perimeter-fw", "ip": "10.255.0.1",
                   "status": "active"}]
        for index, host in enumerate(dict.fromkeys(self.hosts), start=2):
            agents.append({"id": f"{index:03d}", "name": f"host-{index:03d}", "ip": host,
                           "status": "active"})
        self.wazuh = WazuhEmulator(username="sentinel-ar", password=self._secrets["wazuh"],
                                   agents=agents)
        self.scim = ScimEmulator(token=self._secrets["scim"], users=[
            {"id": f"u-{i}", "userName": name, "active": True}
            for i, name in enumerate(self.users, start=1)
        ])
        self.slack = SlackEmulator(path=f"/services/T0SANDBOX/B0SANDBOX/{self._secrets['slack']}")
        self.webhook = WebhookReceiver(path="/hooks/sentinel",
                                       secret=Secret(self._secrets["webhook"]), clock=self.clock)

    def start(self) -> Sandbox:
        for name, service in (("github", self.github), ("wazuh", self.wazuh),
                              ("scim", self.scim), ("slack", self.slack),
                              ("webhook", self.webhook)):
            self._servers[name] = LiveServer(service).start()
        return self

    def stop(self) -> None:
        for server in self._servers.values():
            server.stop()
        self._servers.clear()

    def __enter__(self) -> Sandbox:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def url(self, name: str) -> str:
        return self._servers[name].url

    # --- connectors wired to the sandbox ----------------------------------------- #

    def github_connector(self, *, issues: bool = True, **overrides: Any) -> GitHubConnector:
        caps = [Capability.PR_OPEN_DRAFT] + ([Capability.ISSUE_OPEN] if issues else [])
        scopes = {"contents:write", "pull_requests:write"} | ({"issues:write"} if issues else set())
        kwargs: dict[str, Any] = {
            "owner": self.owner,
            "repo": self.repo,
            "credential": Credential(Secret(self._secrets["github"]), frozenset(scopes)),
            "capabilities": caps,
            "base_url": self.url("github"),
            "allow_insecure_loopback": True,
            "sleep": lambda _s: None,
        }
        kwargs.update(overrides)
        return GitHubConnector(**kwargs)

    def wazuh_connector(self, **overrides: Any) -> WazuhConnector:
        kwargs: dict[str, Any] = {
            "base_url": self.url("wazuh"),
            "credential": Credential(Secret(self._secrets["wazuh"]),
                                     frozenset({"agent:read", "active-response:command"}),
                                     username="sentinel-ar"),
            "firewall_agents": (self.firewall_agent,),
            "allow_insecure_loopback": True,
            "sleep": lambda _s: None,
        }
        kwargs.update(overrides)
        return WazuhConnector(**kwargs)

    def scim_connector(self, **overrides: Any) -> ScimIdentityConnector:
        kwargs: dict[str, Any] = {
            "base_url": self.url("scim"),
            "credential": Credential(Secret(self._secrets["scim"]),
                                     frozenset({"users:read", "users:write"})),
            "allow_insecure_loopback": True,
            "sleep": lambda _s: None,
        }
        kwargs.update(overrides)
        return ScimIdentityConnector(**kwargs)

    def slack_connector(self, **overrides: Any) -> SlackWebhookConnector:
        kwargs: dict[str, Any] = {
            "webhook_url": Secret(self.url("slack") + self.slack.path),
            "allow_insecure_loopback": True,
            "sleep": lambda _s: None,
        }
        kwargs.update(overrides)
        return SlackWebhookConnector(**kwargs)

    def webhook_connector(self, **overrides: Any) -> SignedWebhookConnector:
        kwargs: dict[str, Any] = {
            "url": self.url("webhook") + self.webhook.path,
            "signing_secret": Secret(self._secrets["webhook"]),
            "clock": self.clock,
            "allow_insecure_loopback": True,
            "sleep": lambda _s: None,
        }
        kwargs.update(overrides)
        return SignedWebhookConnector(**kwargs)

    def router(
        self,
        *,
        tenant_id: str = "acme",
        audit: Any = None,
        journal: ExecutionJournal | None = None,
        limiter: BlastRadiusLimiter | None = None,
        targets: TargetPolicy | None = None,
        notify: str = "slack",
    ) -> ConnectorRouter:
        github = self.github_connector()
        notifier = self.slack_connector() if notify == "slack" else self.webhook_connector()
        return ConnectorRouter(
            tenant_id=tenant_id,
            connectors=(
                self.wazuh_connector(),
                self.scim_connector(),
                notifier,
                github,
                LocalEnrichmentConnector(),
            ),
            draft_opener=github,
            targets=targets,
            journal=journal,
            limiter=limiter,
            audit=audit,
        )

