"""Scan a real public GitHub repository with the same analyzer the demo uses.

The Code-Scan Agent's demo runs on a hand-written fixture with planted defects. This
points the same hermetic AST analyzer at a real repository someone else wrote.

What keeps that safe, since the repository is attacker-supplied input:

*   **Only github.com.** The URL must be ``https://github.com/<owner>/<repo>`` (optionally
    ``/tree/<branch>``). The download goes to ``codeload.github.com`` for that repository
    and nowhere else, with redirects off, so the URL cannot be aimed at an internal host.
*   **Nothing is executed or installed.** The archive is read in memory; Python files are
    parsed to syntax trees and never imported. Setup scripts, hooks and tests do not run.
*   **Bounded.** A byte cap on the download, a cap on entries, on files, on lines and per
    file; symlinks and skipped directories (``.git``, ``node_modules``, ``venv`` ...) are
    ignored, so a zip bomb or a giant monorepo costs a refusal, not the server.
*   **Findings are untrusted text.** Excerpts come from the repository and are rendered as
    text by the dashboard, never as markup.

What it cannot tell you: how many real bugs it missed, or how many of its findings are
false alarms, because a real repository has no answer key. The demo fixture does, which is
why its recall and false-positive numbers can be stated and these cannot.
"""

from __future__ import annotations

import io
import re
import stat
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Final

import httpx

from sentinel.core.errors import SentinelError
from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.repo import SKIP_DIRECTORIES, RepoSnapshot, SourceFile

__all__ = [
    "RepoError",
    "RepoRef",
    "fetch_zip",
    "parse_github_url",
    "scan_repository",
    "snapshot_from_zip",
]

MAX_DOWNLOAD_BYTES: Final[int] = 25_000_000
MAX_ENTRIES: Final[int] = 20_000
MAX_FILES: Final[int] = 300
MAX_FILE_BYTES: Final[int] = 400_000
MAX_TOTAL_BYTES: Final[int] = 40_000_000
MAX_LINES: Final[int] = 90_000
MAX_FINDINGS_SHOWN: Final[int] = 300
MAX_PATCHES_SHOWN: Final[int] = 25

_NAME = r"[A-Za-z0-9_.-]{1,100}"
_URL: Final[re.Pattern[str]] = re.compile(
    rf"^https://github\.com/({_NAME})/({_NAME}?)(?:\.git)?(?:/tree/([A-Za-z0-9_./-]{{1,120}}))?/?$"
)


class RepoError(SentinelError):
    """The repository could not be fetched or is refused."""


@dataclass(frozen=True, slots=True)
class RepoRef:
    owner: str
    repo: str
    branch: str | None = None

    @property
    def name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def archive_url(self) -> str:
        ref = "HEAD" if self.branch is None else f"refs/heads/{self.branch}"
        return f"https://codeload.github.com/{self.owner}/{self.repo}/zip/{ref}"


def parse_github_url(url: str) -> RepoRef:
    text = (url or "").strip()
    match = _URL.match(text)
    if match is None:
        raise RepoError("give a public repository as https://github.com/<owner>/<repo>")
    owner, repo, branch = match.groups()
    if owner in {".", ".."} or repo in {".", ".."} or (branch and ".." in branch.split("/")):
        raise RepoError("that is not a valid repository address")
    return RepoRef(owner, repo, branch)


def fetch_zip(ref: RepoRef, *, client: httpx.Client | None = None) -> bytes:
    """Download the archive, refusing anything over the size cap and any redirect."""
    own = client is None
    http = client or httpx.Client(timeout=30.0, follow_redirects=False)
    try:
        with http.stream("GET", ref.archive_url) as response:
            if response.status_code == 404:
                raise RepoError(f"{ref.name} was not found (or it is private)")
            if response.status_code != 200:
                raise RepoError(f"GitHub answered {response.status_code} for {ref.name}")
            declared = int(response.headers.get("content-length", "0") or 0)
            if declared > MAX_DOWNLOAD_BYTES:
                raise RepoError(f"{ref.name} is larger than {MAX_DOWNLOAD_BYTES // 1_000_000} MB")
            buffer = bytearray()
            for chunk in response.iter_bytes():
                buffer.extend(chunk)
                if len(buffer) > MAX_DOWNLOAD_BYTES:
                    raise RepoError(
                        f"{ref.name} is larger than {MAX_DOWNLOAD_BYTES // 1_000_000} MB"
                    )
            return bytes(buffer)
    except httpx.HTTPError as exc:
        raise RepoError(f"could not reach GitHub: {type(exc).__name__}") from exc
    finally:
        if own:
            http.close()


def snapshot_from_zip(data: bytes, *, root: str) -> RepoSnapshot:
    """The Python sources in an archive, bounded and with symlinks refused."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise RepoError("the download was not a zip archive") from exc
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise RepoError(
            f"the archive has {len(infos):,} entries; refusing more than {MAX_ENTRIES:,}"
        )

    candidates: list[tuple[str, zipfile.ZipInfo]] = []
    skipped: list[tuple[str, str]] = []
    for info in infos:
        if info.is_dir():
            continue
        parts = info.filename.split("/")[1:]  # drop the "<repo>-<ref>/" prefix
        if not parts or not parts[-1].endswith(".py"):
            continue
        path = "/".join(parts)
        if any(p in SKIP_DIRECTORIES for p in parts[:-1]) or ".." in parts:
            continue
        if stat.S_ISLNK(info.external_attr >> 16):
            skipped.append((path, "symlink; not followed"))
        elif info.file_size > MAX_FILE_BYTES:
            skipped.append((path, f"larger than {MAX_FILE_BYTES // 1000} KB"))
        else:
            candidates.append((path, info))
    candidates.sort(key=lambda item: (item[0].count("/"), item[0]))  # shallow files first

    files: list[SourceFile] = []
    total_bytes = 0
    total_lines = 0
    for path, info in candidates:
        if (
            len(files) >= MAX_FILES
            or total_lines >= MAX_LINES
            or total_bytes + info.file_size > MAX_TOTAL_BYTES
        ):
            skipped.append((path, "over the scan budget"))
            continue
        raw = archive.read(info)
        if b"\x00" in raw:
            skipped.append((path, "binary content"))
            continue
        text = raw.decode("utf-8", errors="replace")
        total_bytes += len(raw)
        total_lines += text.count("\n")
        files.append(SourceFile.of(path, text))
    if not files:
        raise RepoError("no Python files found (this scanner reads Python only)")
    return RepoSnapshot(root=root, files=tuple(files), skipped=tuple(skipped))


def scan_repository(url: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetch, snapshot and scan. Returns a JSON-able report."""
    started = time.perf_counter()
    ref = parse_github_url(url)
    snapshot = snapshot_from_zip(fetch_zip(ref, client=client), root=ref.name)
    return report_of(ref, snapshot, started=started)


def report_of(ref: RepoRef, snapshot: RepoSnapshot, *, started: float) -> dict[str, Any]:
    result = AstAnalyzer().scan(snapshot)
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    findings = sorted(
        result.findings, key=lambda f: (order.get(f.severity.value, 9), f.path, f.line)
    )
    by_severity: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    for finding in result.findings:
        by_severity[finding.severity.value] = by_severity.get(finding.severity.value, 0) + 1
        by_rule[finding.rule_id] = by_rule.get(finding.rule_id, 0) + 1
    patched = result.patched_refs
    return {
        "repository": ref.name,
        "ref": ref.branch or "default branch",
        "url": f"https://github.com/{ref.name}",
        "files_scanned": result.files_scanned,
        "lines_scanned": result.lines_scanned,
        "skipped": [{"path": p, "reason": r} for p, r in snapshot.skipped[:15]],
        "skipped_total": len(snapshot.skipped),
        "parse_errors": len(result.parse_errors),
        "finding_total": len(result.findings),
        "by_severity": by_severity,
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
        "validated_patches": len(result.valid_patches),
        "rejected_patches": len(result.rejected_patches),
        "findings": [
            {
                "rule_id": f.rule_id,
                "cwe": f.cwe,
                "title": f.title,
                "path": f.path,
                "line": f.line,
                "severity": f.severity.value,
                "confidence": f.confidence.value,
                "message": f.message,
                "remediation": f.remediation,
                "excerpt": f.excerpt,
                "tainted": f.is_tainted,
                "patched": f.ref in patched,
                "link": f"https://github.com/{ref.name}/blob/HEAD/{f.path}#L{f.line}",
            }
            for f in findings[:MAX_FINDINGS_SHOWN]
        ],
        "patches": [
            {"path": p.path, "rule_id": p.rule_id, "diff": p.diff, "checks": list(p.checks)}
            for p in result.valid_patches[:MAX_PATCHES_SHOWN]
        ],
        "seconds": round(time.perf_counter() - started, 1),
        "notes": [
            "A real repository has no answer key, so how many real bugs were missed, and "
            "how many of these findings are false alarms, is not known.",
            "Python only, 14 rules. Nothing in the repository was executed.",
            "Patches are drafts proposed by the analyzer; none has been applied or pushed.",
        ],
    }
