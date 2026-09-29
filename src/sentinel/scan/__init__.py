"""Static analysis and mechanical patching (PRD F-07, Section 5.5).

The Code-Scan Agent's engine room. PRD Section 5.6 names Semgrep and this is a
deliberate hermetic substitution, for the reasons set out in
:mod:`sentinel.scan.rules`:

==============================  ===============================================
:mod:`~sentinel.scan.repo`      an immutable in-memory view of a source tree
:mod:`~sentinel.scan.symbols`   what a name in a file actually refers to
:mod:`~sentinel.scan.taint`     does attacker data reach this line?
:mod:`~sentinel.scan.rules`     the vulnerability patterns and their fixes
:mod:`~sentinel.scan.patch`     source edits, unified diffs, and a diff *applier*
:mod:`~sentinel.scan.findings`  what a finding is, and how it is cited
:mod:`~sentinel.scan.analyzer`  one AST walk, every rule, a validated patch each
:mod:`~sentinel.scan.seeded`    the F-07 fixture's ground truth, read from itself
==============================  ===============================================

:class:`~sentinel.scan.analyzer.StaticAnalyzer` is a Protocol, so a Semgrep-backed
analyzer is a drop-in and the agent is tested through a stub to prove the seam is
real. Nothing here imports the agent layer: a scanner that needed an orchestrator to
run would be a scanner nobody could run in CI.
"""

from sentinel.scan.analyzer import AstAnalyzer, ScanResult, StaticAnalyzer
from sentinel.scan.findings import CodeFinding, FindingConfidence, ScanError, SourceSpan
from sentinel.scan.patch import PatchProposal, PullRequestDraft
from sentinel.scan.repo import RepoSnapshot, SourceFile
from sentinel.scan.rules import RULES, Rule

__all__ = [
    "RULES",
    "AstAnalyzer",
    "CodeFinding",
    "FindingConfidence",
    "PatchProposal",
    "PullRequestDraft",
    "RepoSnapshot",
    "Rule",
    "ScanError",
    "ScanResult",
    "SourceFile",
    "SourceSpan",
    "StaticAnalyzer",
]
