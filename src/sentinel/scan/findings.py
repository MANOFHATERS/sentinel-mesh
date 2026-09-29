"""What a static-analysis run produces (PRD F-07, Section 5.4).

F-07's acceptance criterion is *"at least 3 seeded vulnerabilities detected and a
syntactically valid patch PR opened for each"*, so the unit of work is a
**finding** that carries enough to be cited, patched and approved:

*   *where* — file, line, column, and the exact source span, so a patch can be
    built by span replacement rather than by re-parsing a formatted string;
*   *what* — a rule id and a CWE, because "CWE-89" is the identifier a
    vulnerability-management process already speaks, and an identifier is
    checkable in a way a prose description is not;
*   *why it is exploitable* — whether the dangerous value is attacker-influenced,
    which is the difference between a true finding and the noise that makes
    developers turn scanners off;
*   *the evidence* — a typed :class:`~sentinel.core.schemas.Evidence` so a finding
    can appear in an :class:`~sentinel.core.schemas.InvestigationReport` whose
    validator rejects any claim citing a ref that does not resolve.

Source excerpts are untrusted text
----------------------------------
PRD Section 5.1 lists *"code comments"* alongside alert payloads as
attacker-influenced text, and a scanner reading a pull request from an outside
contributor is a very direct instance of that. So :meth:`CodeFinding.as_evidence`
emits the offending line through the schema's untrusted wrapper, and
:mod:`sentinel.agents.codescan` fences it with
:meth:`~sentinel.agents.prompts.AgentPrompt.with_untrusted` like any other
attacker text. A scanner that interpolates the line it just flagged into a prompt
has handed the attacker the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import Evidence, EvidenceKind, Severity

__all__ = [
    "CodeFinding",
    "FindingConfidence",
    "ScanError",
    "SourceSpan",
    "finding_ref",
]


class ScanError(SentinelError):
    """The scanner could not analyse what it was given."""


class FindingConfidence(StrEnum):
    """How sure the analyzer is that this is really exploitable.

    Ordered, and the ordering is load-bearing: :mod:`sentinel.agents.codescan`
    gates a drafted patch on confidence, and ``LOW`` findings are reported without
    one. A scanner that proposes a code change on a guess trains reviewers to
    rubber-stamp, which is worse than reporting nothing.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return _CONFIDENCE_RANK[self]

    @property
    def score(self) -> float:
        """A number in [0, 1] for the contracts that need a ``Confidence``."""
        return _CONFIDENCE_SCORE[self]


_CONFIDENCE_RANK: Final[dict[FindingConfidence, int]] = {
    FindingConfidence.LOW: 0,
    FindingConfidence.MEDIUM: 1,
    FindingConfidence.HIGH: 2,
}

_CONFIDENCE_SCORE: Final[dict[FindingConfidence, float]] = {
    FindingConfidence.LOW: 0.45,
    FindingConfidence.MEDIUM: 0.7,
    FindingConfidence.HIGH: 0.9,
}


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A half-open source range, 1-based lines and 0-based columns.

    These are exactly ``ast`` node coordinates, kept in ``ast``'s own convention
    rather than normalised to offsets. Converting at the boundary would mean every
    rule that reports a span has an opportunity to convert it wrongly, and an
    off-by-one in a *patch* coordinate is a silent code corruption rather than a
    visible error.
    """

    line: int
    col: int
    end_line: int
    end_col: int

    def __post_init__(self) -> None:
        if self.line < 1 or self.end_line < 1:
            raise ScanError(f"source lines are 1-based, got {self.line}..{self.end_line}")
        if self.col < 0 or self.end_col < 0:
            raise ScanError("source columns cannot be negative")
        if (self.end_line, self.end_col) < (self.line, self.col):
            raise ScanError(
                f"span ends before it starts: {self.line}:{self.col} -> "
                f"{self.end_line}:{self.end_col}"
            )

    @property
    def single_line(self) -> bool:
        return self.line == self.end_line

    def text_of(self, source: str) -> str:
        """The source this span covers. Line endings normalised to ``\\n``."""
        lines = source.splitlines(keepends=True)
        if self.end_line > len(lines):
            raise ScanError(
                f"span ends on line {self.end_line} but the source has {len(lines)}"
            )
        if self.single_line:
            return lines[self.line - 1][self.col : self.end_col]
        first = lines[self.line - 1][self.col :]
        middle = lines[self.line : self.end_line - 1]
        last = lines[self.end_line - 1][: self.end_col]
        return "".join([first, *middle, last])


def finding_ref(*, path: str, rule_id: str, line: int) -> str:
    """The citation ref for a finding.

    ``code://<path>#<rule>@<line>``. Stable under a re-scan of the same tree,
    which is what lets an approval granted for a finding still name the same
    finding after the scan is repeated — and unstable under an *edit*, which is
    correct: a moved line is a different finding and should be re-approved.
    """
    return f"code://{path}#{rule_id}@{line}"


@dataclass(frozen=True, slots=True)
class CodeFinding:
    """One vulnerability, located and explained."""

    rule_id: str
    cwe: str
    title: str
    path: str
    span: SourceSpan
    severity: Severity
    confidence: FindingConfidence
    message: str
    remediation: str
    #: The offending source line, verbatim. Attacker-influenced text.
    excerpt: str
    #: How the dangerous value reaches the sink, when the analyzer traced one.
    taint_path: tuple[str, ...] = ()
    #: Free-text search terms for the CVE lookup. Not CVE ids: the analyzer does
    #: not get to assert a CVE, it asks the knowledge base and cites what answers.
    cve_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ScanError("a finding must name the rule that produced it")
        if not self.path:
            raise ScanError(f"{self.rule_id}: a finding must name a file")

    @property
    def line(self) -> int:
        return self.span.line

    @property
    def ref(self) -> str:
        return finding_ref(path=self.path, rule_id=self.rule_id, line=self.line)

    @property
    def is_tainted(self) -> bool:
        """True when the analyzer traced attacker-influenced data into the sink."""
        return bool(self.taint_path)

    def describe(self) -> str:
        head = (
            f"{self.path}:{self.line}:{self.col_for_display} "
            f"[{self.rule_id}] {self.cwe} {self.severity.value} "
            f"({self.confidence.value} confidence)"
        )
        lines = [head, f"  {self.message}"]
        if self.taint_path:
            lines.append(f"  taint: {' -> '.join(self.taint_path)}")
        lines.append(f"  fix: {self.remediation}")
        return "\n".join(lines)

    @property
    def col_for_display(self) -> int:
        """1-based column, because every editor and every compiler reports it that way."""
        return self.span.col + 1

    def as_evidence(self) -> Evidence:
        """The finding as a citable :class:`~sentinel.core.schemas.Evidence`."""
        return Evidence(
            kind=EvidenceKind.CODE_FINDING,
            ref=self.ref,
            # Truncated hard: a minified or generated line can be megabytes, and
            # Evidence.excerpt is bounded. The line number in the ref is what makes
            # the full context recoverable.
            excerpt=f"{self.path}:{self.line}: {self.excerpt.strip()[:400]}",
            relevance=self.confidence.score,
        )
