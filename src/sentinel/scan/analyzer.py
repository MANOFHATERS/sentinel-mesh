"""The scanner: one AST walk, every rule, and a validated patch per finding (F-07).

:class:`StaticAnalyzer` is a Protocol so a Semgrep-backed implementation is a
drop-in — that seam is not decorative, and ``test_agents_codescan.py`` drives the
Code-Scan Agent through a stub analyzer to prove the agent depends on the protocol
rather than on :class:`AstAnalyzer`.

Ordering is part of the contract
--------------------------------
Findings come back sorted by ``(path, line, column, rule_id)``. Two scans of the
same tree therefore produce byte-identical output, which is what makes a scan
result diffable between commits and what lets
:func:`~sentinel.scan.findings.finding_ref` identify the same finding across runs.
An analyzer whose output order depends on dictionary iteration produces a "new"
finding set on every run and makes triage impossible.

Syntax errors are findings, not crashes
---------------------------------------
A file that does not parse is reported as a finding against the scanner's own
pseudo-rule rather than raising. A repository with one unparseable file is the
normal case (a Python 2 leftover, a template with placeholder syntax), and a
scanner that aborts the whole run over it scans nothing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from sentinel.core.schemas import Severity
from sentinel.scan.findings import CodeFinding, FindingConfidence, SourceSpan
from sentinel.scan.patch import (
    PatchProposal,
    SourceEdit,
    apply_edits,
    apply_unified_diff,
    unified_diff,
)
from sentinel.scan.repo import RepoSnapshot, SourceFile
from sentinel.scan.rules import RULES, Match, Rule, RuleContext
from sentinel.scan.symbols import ImportTable
from sentinel.scan.taint import analyse_taint

__all__ = [
    "PARSE_ERROR_RULE",
    "PATCH_CHECKS",
    "AstAnalyzer",
    "ScanResult",
    "StaticAnalyzer",
    "build_patch",
    "validate_patch",
]

PARSE_ERROR_RULE: Final[str] = "python.unparseable-source"
"""Reported when a file does not parse. Not in ``RULES``: it is a scanner fact."""

PATCH_CHECKS: Final[tuple[str, ...]] = (
    "patched source parses",
    "diff reconstructs the patched source",
    "one instance of the rule is removed",
    "no rule gains an instance",
)
"""The four gates in :func:`validate_patch`, in order. See :mod:`sentinel.scan.patch`."""


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Everything one scan produced."""

    findings: tuple[CodeFinding, ...]
    patches: tuple[PatchProposal, ...]
    files_scanned: int
    lines_scanned: int
    #: Files that could not be parsed, as ``(path, message)``.
    parse_errors: tuple[tuple[str, str], ...] = field(default=())

    @property
    def valid_patches(self) -> tuple[PatchProposal, ...]:
        return tuple(patch for patch in self.patches if patch.is_valid)

    @property
    def rejected_patches(self) -> tuple[PatchProposal, ...]:
        return tuple(patch for patch in self.patches if not patch.is_valid)

    @property
    def patched_refs(self) -> frozenset[str]:
        return frozenset(patch.finding_ref for patch in self.valid_patches)

    def findings_by_severity(self, minimum: Severity) -> tuple[CodeFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity >= minimum)

    def unpatched(self) -> tuple[CodeFinding, ...]:
        """Findings with no valid patch — the ones the PR body has to name."""
        patched = self.patched_refs
        return tuple(finding for finding in self.findings if finding.ref not in patched)

    def summary(self) -> str:
        by_rule: dict[str, int] = {}
        for finding in self.findings:
            by_rule[finding.rule_id] = by_rule.get(finding.rule_id, 0) + 1
        lines = [
            f"{len(self.findings)} finding(s) across {self.files_scanned} file(s), "
            f"{self.lines_scanned:,} lines; {len(self.valid_patches)} validated patch(es)"
        ]
        for rule_id, count in sorted(by_rule.items()):
            lines.append(f"  {rule_id}: {count}")
        for path, message in self.parse_errors:
            lines.append(f"  ! {path} did not parse: {message}")
        return "\n".join(lines)


@runtime_checkable
class StaticAnalyzer(Protocol):
    """Anything that can find vulnerabilities in a snapshot and draft patches."""

    @property
    def name(self) -> str: ...

    def scan(self, snapshot: RepoSnapshot) -> ScanResult: ...


@dataclass(slots=True)
class AstAnalyzer:
    """The hermetic AST analyzer. See :mod:`sentinel.scan.rules` for why not Semgrep."""

    rules: tuple[Rule, ...] = RULES
    #: Patches are drafted from this confidence up. ``LOW`` findings — a dangerous
    #: construct on a literal — are reported without one, because a code change
    #: proposed on a guess is how a reviewer learns to approve without reading.
    min_patch_confidence: FindingConfidence = FindingConfidence.MEDIUM
    version: str = "ast-analyzer-1"

    @property
    def name(self) -> str:
        return self.version

    def scan(self, snapshot: RepoSnapshot) -> ScanResult:
        findings: list[CodeFinding] = []
        patches: list[PatchProposal] = []
        parse_errors: list[tuple[str, str]] = []
        lines = 0

        for file in snapshot.files:
            lines += file.line_count
            try:
                tree = ast.parse(file.text, filename=file.path)
            except SyntaxError as exc:
                parse_errors.append((file.path, f"{exc.msg} at line {exc.lineno}"))
                findings.append(_parse_error_finding(file, exc))
                continue
            file_findings, file_patches = self._scan_file(file, tree)
            findings.extend(file_findings)
            patches.extend(file_patches)

        findings.sort(key=lambda f: (f.path, f.span.line, f.span.col, f.rule_id))
        patches.sort(key=lambda p: (p.path, p.rule_id, p.finding_ref))
        return ScanResult(
            findings=tuple(findings),
            patches=tuple(patches),
            files_scanned=len(snapshot.files),
            lines_scanned=lines,
            parse_errors=tuple(parse_errors),
        )

    # --- internals ------------------------------------------------------------ #

    def _scan_file(
        self, file: SourceFile, tree: ast.Module
    ) -> tuple[list[CodeFinding], list[PatchProposal]]:
        ctx = self._context(file, tree)
        by_type = _dispatch_table(self.rules)

        findings: list[CodeFinding] = []
        matches: list[tuple[Rule, Match]] = []
        for node in ast.walk(tree):
            for rule in by_type.get(type(node), ()):
                match = rule.matcher(node, ctx)
                if match is None:
                    continue
                finding = _finding_of(rule, match, ctx, file)
                # A rule can legitimately match the same node twice if two of its
                # patterns overlap; a duplicate ref would then produce two patches
                # for one line, which would conflict in apply_edits.
                if any(existing.ref == finding.ref for existing in findings):
                    continue
                findings.append(finding)
                matches.append((rule, match))

        patches: list[PatchProposal] = []
        for (rule, match), finding in zip(matches, findings, strict=True):
            if finding.confidence.rank < self.min_patch_confidence.rank:
                continue
            if not rule.has_fix:
                continue
            proposal = build_patch(rule, match, ctx, file, analyzer=self)
            if proposal is not None:
                patches.append(proposal)
        return findings, patches

    def _context(self, file: SourceFile, tree: ast.Module) -> RuleContext:
        imports = ImportTable.of(tree)
        return RuleContext(
            path=file.path,
            source=file.text,
            tree=tree,
            imports=imports,
            taint=analyse_taint(tree, imports),
        )

    def rule_counts(self, file: SourceFile) -> dict[str, int]:
        """How many times each rule fires in ``file``.

        Counts rather than a boolean or a set of locations, and that choice is the
        whole reason patch validation works on real files.

        The obvious check — "does this rule still fire in this file?" — is wrong
        whenever a file contains two instances of the same bug, which is the normal
        case: ``app.py`` in the F-07 fixture has two SQL injections, so fixing
        either one leaves the rule firing and every patch is rejected. Measured on
        the fixture before this was changed: **8 of 16 patches rejected**, each with
        the message "still fires after the patch", all of them correct fixes.

        The other obvious check — "is the finding at line N gone?" — is wrong in the
        opposite direction, because a fix that inserts an import shifts every line
        below it, so a still-broken line simply moves and the check passes.

        A count is immune to both. One fewer instance of the rule means one instance
        was fixed; no rule gaining an instance means nothing new was introduced.
        """
        try:
            tree = ast.parse(file.text, filename=file.path)
        except SyntaxError:
            # An unparseable patched file is caught by the parse check before this
            # runs; reporting a sentinel here keeps the count comparison meaningful
            # if the order ever changes.
            return {PARSE_ERROR_RULE: 1}
        ctx = self._context(file, tree)
        counts: dict[str, int] = {}
        by_type = _dispatch_table(self.rules)
        seen: set[tuple[str, int, int]] = set()
        for node in ast.walk(tree):
            for rule in by_type.get(type(node), ()):
                if rule.matcher(node, ctx) is None:
                    continue
                span = ctx.span_of(node)
                key = (rule.rule_id, span.line, span.col)
                if key in seen:
                    continue
                seen.add(key)
                counts[rule.rule_id] = counts.get(rule.rule_id, 0) + 1
        return counts


def _dispatch_table(rules: tuple[Rule, ...]) -> dict[type[ast.AST], tuple[Rule, ...]]:
    grouped: dict[type[ast.AST], list[Rule]] = {}
    for rule in rules:
        for node_type in rule.node_types:
            grouped.setdefault(node_type, []).append(rule)
    return {node_type: tuple(items) for node_type, items in grouped.items()}


def _finding_of(
    rule: Rule, match: Match, ctx: RuleContext, file: SourceFile
) -> CodeFinding:
    span = ctx.span_of(match.node)
    message = rule.message if not match.detail else f"{rule.message} — {match.detail}"
    return CodeFinding(
        rule_id=rule.rule_id,
        cwe=rule.cwe,
        title=rule.title,
        path=file.path,
        span=span,
        severity=rule.severity,
        confidence=match.confidence,
        message=message,
        remediation=rule.remediation,
        excerpt=ctx.line_text(span.line),
        taint_path=match.taint_path,
        cve_hints=rule.cve_hints,
    )


def _parse_error_finding(file: SourceFile, exc: SyntaxError) -> CodeFinding:
    line = exc.lineno or 1
    return CodeFinding(
        rule_id=PARSE_ERROR_RULE,
        cwe="CWE-1164",
        title="Source file could not be parsed",
        path=file.path,
        span=SourceSpan(line=line, col=0, end_line=line, end_col=0),
        severity=Severity.INFO,
        confidence=FindingConfidence.HIGH,
        message=f"the scanner could not analyse this file: {exc.msg}",
        remediation="Fix the syntax error, or exclude the file from scanning",
        excerpt=file.line(line),
    )


# --------------------------------------------------------------------------- #
# Patch construction and validation
# --------------------------------------------------------------------------- #


def build_patch(
    rule: Rule,
    match: Match,
    ctx: RuleContext,
    file: SourceFile,
    *,
    analyzer: AstAnalyzer,
) -> PatchProposal | None:
    """Build and validate the patch for one match. ``None`` when the fix declined.

    ``None`` and a *rejected* proposal mean different things and the distinction is
    reported. ``None`` is the rule declining — ``_fix_weak_hash`` on a
    ``from hashlib import md5`` call site knows the edit it would need is wider than
    one finding should make. A rejected proposal is a fix that was built and then
    failed validation, which is a defect in the fix and must be visible.
    """
    if rule.fix is None:
        return None
    try:
        edits = rule.fix(match, ctx)
    except Exception as exc:  # a broken fix must not take the scan down with it
        return PatchProposal(
            finding_ref=_ref_for(rule, match, ctx, file),
            path=file.path,
            rule_id=rule.rule_id,
            diff="",
            before=file.text,
            after=file.text,
            notes=(),
            rejection=f"the fix raised {type(exc).__name__}: {exc}",
        )
    if not edits:
        return None
    return validate_patch(
        rule=rule,
        edits=edits,
        ctx=ctx,
        file=file,
        finding_ref=_ref_for(rule, match, ctx, file),
        analyzer=analyzer,
    )


def _ref_for(rule: Rule, match: Match, ctx: RuleContext, file: SourceFile) -> str:
    from sentinel.scan.findings import finding_ref

    return finding_ref(
        path=file.path, rule_id=rule.rule_id, line=ctx.span_of(match.node).line
    )


def validate_patch(
    *,
    rule: Rule,
    edits: tuple[SourceEdit, ...],
    ctx: RuleContext,
    file: SourceFile,
    finding_ref: str,
    analyzer: AstAnalyzer,
) -> PatchProposal:
    """Run the four gates from :data:`PATCH_CHECKS`. Never raises."""
    before = file.text
    notes = tuple(dict.fromkeys(edit.note for edit in edits))

    def rejected(reason: str, *, after: str = before, diff: str = "") -> PatchProposal:
        return PatchProposal(
            finding_ref=finding_ref,
            path=file.path,
            rule_id=rule.rule_id,
            diff=diff,
            before=before,
            after=after,
            notes=notes,
            rejection=reason,
        )

    try:
        after = apply_edits(before, edits)
    except Exception as exc:
        return rejected(f"edits did not apply: {type(exc).__name__}: {exc}")
    if after == before:
        return rejected("the fix produced no change")

    passed: list[str] = []

    try:
        ast.parse(after, filename=file.path)
    except SyntaxError as exc:
        return rejected(
            f"patched source does not parse: {exc.msg} at line {exc.lineno}", after=after
        )
    passed.append(PATCH_CHECKS[0])

    diff = unified_diff(path=file.path, before=before, after=after)
    if not diff:
        return rejected("the patch produced an empty diff", after=after)
    try:
        replayed = apply_unified_diff(before, diff)
    except Exception as exc:
        return rejected(f"the diff does not apply: {exc}", after=after, diff=diff)
    if replayed != after:
        return rejected(
            "replaying the diff does not reproduce the patched source, so the diff "
            "and the validated text disagree",
            after=after,
            diff=diff,
        )
    passed.append(PATCH_CHECKS[1])

    patched_file = file.with_text(after)
    before_counts = analyzer.rule_counts(file)
    after_counts = analyzer.rule_counts(patched_file)

    expected = before_counts.get(rule.rule_id, 0)
    remaining = after_counts.get(rule.rule_id, 0)
    if remaining >= expected:
        return rejected(
            f"{rule.rule_id} still fires {remaining} time(s) after the patch "
            f"(was {expected}), so the fix does not remove an instance",
            after=after,
            diff=diff,
        )
    passed.append(PATCH_CHECKS[2])

    introduced = sorted(
        f"{rule_id} ({before_counts.get(rule_id, 0)} -> {count})"
        for rule_id, count in after_counts.items()
        if count > before_counts.get(rule_id, 0)
    )
    if introduced:
        return rejected(
            f"the patch introduces new findings: {introduced}", after=after, diff=diff
        )
    passed.append(PATCH_CHECKS[3])

    return PatchProposal(
        finding_ref=finding_ref,
        path=file.path,
        rule_id=rule.rule_id,
        diff=diff,
        before=before,
        after=after,
        notes=notes,
        checks=tuple(passed),
    )
