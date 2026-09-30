"""How good is the code scanner on real code? Measured against real answer keys.

On the demo's hand-written fixture the scanner scores 18/18 with no false positives. That number
is real but weak evidence: the fixture was written alongside the rules. This measures the same
analyzer against two answer keys nobody wrote for it.

**Paired directories.** A deliberately vulnerable teaching project (``fportantier/vulpy``) keeps a
``bad/`` tree and a ``good/`` tree with the same file names, the second being the fixed version of
the first. A file in ``bad/`` should be flagged; its namesake in ``good/`` should not.

**Real advisories, before and after the fix.** Real PyPI advisories (OSV.dev / GitHub Advisory
Database) name a weakness type (a CWE) and link the commit that fixed it. For each one whose CWE is
one the scanner has a rule for, the changed Python files are fetched as they were *just before* the
fix and *at* the fix, and both are scanned. Vulnerable code the scanner flags with a rule for the
advisory's own CWE is a hit; the fixed version being flagged again is a false alarm.

What this cannot claim: the advisories are flaws in *libraries* (Django's ORM, pip, aiohttp), which
are harder targets than application code and often not the pattern the rules look for, so a low
hit rate here is expected and is not the scanner's score on application code. The samples are small
and the intervals show it. And a rule firing in the right file is credited to the scanner without
checking it is the exact line.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx

from sentinel.core.errors import SentinelError
from sentinel.real.repos import fetch_zip, parse_github_url, snapshot_from_zip
from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.rules import RULES

__all__ = [
    "ScannerEvalError",
    "advisory_cases",
    "evaluate_advisories",
    "evaluate_paired_repo",
    "load_result",
    "wilson",
]

RESULT_PATH: Final[Path] = Path("data/real/scanner-eval.json")
_COMMIT: Final[re.Pattern[str]] = re.compile(
    r"^https://github\.com/([^/]+)/([^/]+)/commit/([0-9a-f]{7,40})"
)
_MAX_FILES_PER_COMMIT: Final[int] = 4
_MAX_FILE_BYTES: Final[int] = 400_000


class ScannerEvalError(SentinelError):
    """The evaluation could not be run or its result is missing."""


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% Wilson score interval for a proportion. ``(0, 0)`` for an empty sample."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _cwe_of_rules() -> dict[str, set[str]]:
    by_rule: dict[str, set[str]] = {}
    for rule in RULES:
        by_rule.setdefault(rule.cwe, set()).add(rule.rule_id)
    return by_rule


# --------------------------------------------------------------------------- #
# Paired directories
# --------------------------------------------------------------------------- #


def evaluate_paired_repo(
    repo: str, *, bad: str = "bad/", good: str = "good/", client: httpx.Client | None = None
) -> dict[str, Any]:
    ref = parse_github_url(f"https://github.com/{repo}")
    snapshot = snapshot_from_zip(fetch_zip(ref, client=client), root=ref.name)
    analyzer = AstAnalyzer()
    files_by_name: dict[str, dict[str, Any]] = {}
    for file in snapshot.files:
        side = (
            "bad" if file.path.startswith(bad) else "good" if file.path.startswith(good) else None
        )
        if side is None:
            continue
        name = file.path.split("/", 1)[1]
        result = analyzer.scan(RepoSnapshot.of_texts({file.path: file.text}))
        entry = files_by_name.setdefault(name, {})
        entry[side] = {
            "findings": len(result.findings),
            "rules": sorted({f.rule_id for f in result.findings}),
            "lines": file.line_count,
        }
    pairs = {n: e for n, e in files_by_name.items() if "bad" in e and "good" in e}
    vulnerable = len(pairs)
    flagged_bad = sum(1 for e in pairs.values() if e["bad"]["findings"] > 0)
    flagged_good = sum(1 for e in pairs.values() if e["good"]["findings"] > 0)
    return {
        "repository": repo,
        "url": f"https://github.com/{repo}",
        "pairs": vulnerable,
        "bad_flagged": flagged_bad,
        "good_flagged": flagged_good,
        "bad_findings": sum(e["bad"]["findings"] for e in pairs.values()),
        "good_findings": sum(e["good"]["findings"] for e in pairs.values()),
        "recall": flagged_bad / vulnerable if vulnerable else None,
        "recall_ci": wilson(flagged_bad, vulnerable),
        "false_alarm_rate": flagged_good / vulnerable if vulnerable else None,
        "false_alarm_ci": wilson(flagged_good, vulnerable),
        "files": [
            {"file": n, "bad": e["bad"], "good": e["good"]} for n, e in sorted(pairs.items())
        ],
        "note": (
            "File level: a file counts as caught if it has any finding. Not every file pair "
            "changes a flaw the rules cover, so the recall here is a floor."
        ),
    }


# --------------------------------------------------------------------------- #
# Real advisories, before and after
# --------------------------------------------------------------------------- #


def advisory_cases(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Advisories from an OSV cache whose CWE the scanner covers and that link a fix commit."""
    covered = _cwe_of_rules()
    cases: list[dict[str, Any]] = []
    for key, record in sorted(cache.items()):
        if "/v1/vulns/" not in key or not record:
            continue
        cwes = [
            c for c in (record.get("database_specific") or {}).get("cwe_ids", []) if c in covered
        ]
        if not cwes:
            continue
        for ref in record.get("references", []):
            match = _COMMIT.match(ref.get("url", ""))
            if match and ref.get("type") in ("FIX", "WEB"):
                owner, repo, sha = match.groups()
                cases.append(
                    {
                        "advisory": record["id"],
                        "summary": (record.get("summary") or "")[:110],
                        "cwe": cwes[0],
                        "rules": sorted(covered[cwes[0]]),
                        "owner": owner,
                        "repo": repo,
                        "sha": sha,
                    }
                )
                break
    return cases


class _GitHub:
    def __init__(self, client: httpx.Client, cache: dict[str, Any], pause: float) -> None:
        self._client = client
        self._cache = cache
        self._pause = pause
        token = os.environ.get("GITHUB_TOKEN")
        self._headers = {"Accept": "application/vnd.github+json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def commit(self, owner: str, repo: str, sha: str) -> dict[str, Any] | None:
        return self._json(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}")

    def raw(self, owner: str, repo: str, sha: str, path: str) -> str | None:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"
        if url in self._cache:
            return self._cache[url]
        response = self._client.get(url)
        text = (
            response.text
            if response.status_code == 200 and len(response.content) <= _MAX_FILE_BYTES
            else None
        )
        self._cache[url] = text
        time.sleep(self._pause)
        return text

    def _json(self, url: str) -> dict[str, Any] | None:
        if url in self._cache:
            return self._cache[url]
        response = self._client.get(url, headers=self._headers)
        if response.status_code == 403:
            raise ScannerEvalError(
                "GitHub's unauthenticated API limit was reached; set GITHUB_TOKEN (a free "
                "personal token with no scopes) and re-run. Finished cases are cached."
            )
        data = response.json() if response.status_code == 200 else None
        self._cache[url] = data
        time.sleep(self._pause)
        return data


def evaluate_advisories(
    cases: list[dict[str, Any]],
    *,
    client: httpx.Client | None = None,
    cache_path: Path | None = None,
    pause: float = 0.3,
    limit: int | None = None,
) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    if cache_path and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    github = _GitHub(client or httpx.Client(timeout=30.0, follow_redirects=True), cache, pause)
    analyzer = AstAnalyzer()
    rows: list[dict[str, Any]] = []
    try:
        for case in cases[: limit or len(cases)]:
            row = _evaluate_case(case, github, analyzer)
            if row is not None:
                rows.append(row)
    finally:
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
    n = len(rows)
    hit = sum(1 for r in rows if r["before_hit"])
    still = sum(1 for r in rows if r["after_hit"])
    fixed_clean = sum(1 for r in rows if r["before_hit"] and not r["after_hit"])
    by_cwe: dict[str, dict[str, int]] = {}
    for r in rows:
        tally = by_cwe.setdefault(r["cwe"], {"cases": 0, "hit": 0})
        tally["cases"] += 1
        tally["hit"] += bool(r["before_hit"])
    return {
        "cases": n,
        "hit": hit,
        "recall": hit / n if n else None,
        "recall_ci": wilson(hit, n),
        "fixed_version_flagged": still,
        "fixed_version_flagged_rate": still / n if n else None,
        "fixed_version_flagged_ci": wilson(still, n),
        "flagged_before_and_clean_after": fixed_clean,
        "by_cwe": dict(sorted(by_cwe.items())),
        "rows": rows,
        "note": (
            "These are flaws inside widely used libraries (Django, pip, aiohttp, Pillow ...), "
            "which are harder than application code and often not the pattern the rules look "
            "for. A low recall here is expected; it is not the scanner's score on application "
            "code. A hit means a rule for the advisory's own CWE fired in a file the fix changed."
        ),
    }


def _evaluate_case(
    case: Mapping[str, Any], github: _GitHub, analyzer: AstAnalyzer
) -> dict[str, Any] | None:
    commit = github.commit(case["owner"], case["repo"], case["sha"])
    if not commit or not commit.get("parents"):
        return None
    parent = commit["parents"][0]["sha"]
    changed = [
        f["filename"]
        for f in commit.get("files", [])
        if f["filename"].endswith(".py")
        and f.get("status") in ("modified", "renamed")
        and "test" not in f["filename"].lower()
    ][:_MAX_FILES_PER_COMMIT]
    if not changed:
        return None
    wanted = set(case["rules"])
    before_hit = after_hit = False
    before_rules: set[str] = set()
    after_rules: set[str] = set()
    scanned: list[str] = []
    for path in changed:
        before = github.raw(case["owner"], case["repo"], parent, path)
        after = github.raw(case["owner"], case["repo"], commit["sha"], path)
        if before is None or after is None:
            continue
        scanned.append(path)
        for text, sink in ((before, before_rules), (after, after_rules)):
            result = analyzer.scan(RepoSnapshot.of_texts({path: text}))
            sink.update(f.rule_id for f in result.findings)
    if not scanned:
        return None
    before_hit = bool(before_rules & wanted)
    after_hit = bool(after_rules & wanted)
    return {
        "advisory": case["advisory"],
        "summary": case["summary"],
        "package": f"{case['owner']}/{case['repo']}",
        "cwe": case["cwe"],
        "files": scanned,
        "before_hit": before_hit,
        "after_hit": after_hit,
        "before_any": sorted(before_rules),
        "after_any": sorted(after_rules),
        "commit": f"https://github.com/{case['owner']}/{case['repo']}/commit/{case['sha']}",
    }


def load_result(path: Path | str = RESULT_PATH) -> dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        raise ScannerEvalError(f"{file} not found; run scripts/build_scanner_eval.py")
    return json.loads(file.read_text(encoding="utf-8"))
