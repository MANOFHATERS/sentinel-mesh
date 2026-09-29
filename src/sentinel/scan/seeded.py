"""Ground truth for the F-07 fixture, read out of the fixture itself.

F-07 is graded on *"at least 3 seeded vulnerabilities detected"*, which requires
knowing which ones were seeded. The obvious way to record that is a manifest file
listing paths and line numbers, and it is the wrong way: the moment anyone edits the
fixture, the manifest's line numbers point at the wrong lines and the measurement
quietly becomes a measurement of something else. There is no test that can catch
that, because both halves are "correct" in isolation.

So the ground truth is a marker comment on the vulnerable line:

.. code-block:: python

    cursor.execute(f"SELECT ... '{term}%'")   # SEEDED: python.sql-injection
    cursor.execute("SELECT ... = ?", (id,))   # SAFE: python.sql-injection

An edit that moves the line moves its own ground truth with it, and a marker naming
a rule that does not exist fails :meth:`SeedManifest.validate` rather than silently
scoring zero for a typo.

``SAFE`` is the half that matters
--------------------------------
A ``SEEDED`` marker measures recall, and recall alone is satisfiable by a scanner
that flags every line in the file. ``SAFE`` markers are the correct construction for
the *same rule*, placed next to the defective one, and a scanner that flags them has
told a developer that parameterised SQL is a SQL injection. The first such finding
costs the tool its credibility and the second costs it its installation, so
``scripts/evaluate.py --codescan`` gates on the false-positive count as hard as on
recall.

Markers are also **not** how the scanner works. Nothing in
:mod:`sentinel.scan.analyzer` reads them; a test asserts the analyzer's findings are
unchanged when every marker comment is stripped, so the fixture cannot accidentally
become a scanner that looks for its own answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from sentinel.scan.findings import CodeFinding, FindingConfidence, ScanError
from sentinel.scan.repo import RepoSnapshot

__all__ = [
    "ACTIONABLE_CONFIDENCE",
    "FIXTURE_DIR",
    "MARKER_PATTERN",
    "Expectation",
    "ScanScore",
    "SeedExpectation",
    "SeedManifest",
    "load_seed_manifest",
    "score_scan",
    "strip_markers",
]

FIXTURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "vulnerable_app"
)
"""The seeded fixture shipped with the repository."""

MARKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"#\s*(?P<kind>SEEDED|SAFE|INFO)\s*:\s*(?P<rules>[\w.\-\s]+?)\s*$"
)
"""``# SEEDED: rule.a rule.b``, ``# SAFE: rule.a`` or ``# INFO: rule.a``, at end of line.

Anchored at end of line so a marker cannot be smuggled into the middle of a source
line and change what a *different* construct on the same line is graded as.

The marker goes on the line :mod:`ast` reports for the construct, which for a call
spread over several lines is the line the *callee* is on — so a multi-line call
carries its marker after the opening parenthesis:

.. code-block:: python

    cursor.execute(  # SEEDED: python.sql-injection
        "SELECT ... = '%s'" % role
    )

Placing it on the closing parenthesis instead is the natural thing to do and it is
wrong: the scan would report line 76 while the ground truth claimed line 78, and the
finding would score as both a miss and an unexplained extra. Three of the fixture's
seventeen seeded cases were written that way first, and this is what they taught.
"""

#: The confidence at or above which a finding is *actionable* — a patch is drafted
#: and a reviewer is expected to look. Mirrors
#: :attr:`~sentinel.scan.analyzer.AstAnalyzer.min_patch_confidence`; a test asserts
#: they agree, because an ``INFO`` control graded against a different threshold than
#: the analyzer patches at would pass while the scanner sent a pull request.
ACTIONABLE_CONFIDENCE: Final[FindingConfidence] = FindingConfidence.MEDIUM


class Expectation(StrEnum):
    """What a marker claims about its line.

    Three states rather than two, because two cannot express the honest answer for
    ``os.system("logrotate -f /etc/logrotate.conf")``. The construct genuinely is
    ``os.system``, so a scanner that says nothing about it is hiding something; the
    argument is a literal, so calling it a vulnerability is crying wolf. It is
    *informational*: listed, never patched, never blocking. Forcing that line to be
    either ``SEEDED`` or ``SAFE`` would have meant either inflating recall with a
    finding nobody should act on, or recording a correct low-confidence report as a
    false positive and tuning it away.
    """

    #: Must be reported, and at :data:`ACTIONABLE_CONFIDENCE` or above.
    SEEDED = "SEEDED"
    #: Must not be reported at all.
    SAFE = "SAFE"
    #: May be reported, but only *below* :data:`ACTIONABLE_CONFIDENCE`.
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class SeedExpectation:
    """One marker: what a named rule must do on one line."""

    path: str
    line: int
    rule_id: str
    expectation: Expectation

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.path, self.line, self.rule_id)

    @property
    def expected(self) -> bool:
        """True for ``SEEDED``. Kept as the readable spelling at call sites."""
        return self.expectation is Expectation.SEEDED

    def describe(self) -> str:
        verb = {
            Expectation.SEEDED: "must be flagged (actionably) by",
            Expectation.SAFE: "must NOT be flagged by",
            Expectation.INFO: "must be flagged only informationally by",
        }[self.expectation]
        return f"{self.path}:{self.line} {verb} {self.rule_id}"


@dataclass(frozen=True, slots=True)
class SeedManifest:
    """Every expectation the fixture states about itself."""

    expectations: tuple[SeedExpectation, ...]

    @property
    def seeded(self) -> tuple[SeedExpectation, ...]:
        return tuple(item for item in self.expectations if item.expected)

    @property
    def controls(self) -> tuple[SeedExpectation, ...]:
        """``SAFE`` lines: the correct construction for a rule that also has a defect."""
        return tuple(
            item for item in self.expectations if item.expectation is Expectation.SAFE
        )

    @property
    def informational(self) -> tuple[SeedExpectation, ...]:
        return tuple(
            item for item in self.expectations if item.expectation is Expectation.INFO
        )

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.rule_id for item in self.expectations))

    def validate(self, known_rule_ids: frozenset[str]) -> None:
        """Refuse a marker naming a rule that does not exist."""
        unknown = sorted(set(self.rule_ids) - known_rule_ids)
        if unknown:
            raise ScanError(
                f"the fixture claims rules that do not exist: {unknown}. A marker "
                "typo would otherwise score as an undetected vulnerability."
            )
        if not self.seeded:
            raise ScanError("the fixture seeds no vulnerabilities; F-07 cannot be measured")
        if not self.controls:
            raise ScanError(
                "the fixture has no SAFE controls, so recall could be satisfied by a "
                "scanner that flags every line"
            )


def load_seed_manifest(snapshot: RepoSnapshot) -> SeedManifest:
    """Parse every marker in ``snapshot``.

    A marker may name several rules (``# SEEDED: a b``) for a line where two
    constructs genuinely coincide — ``app.run(debug=True, host="0.0.0.0")`` is one
    line and two defects.
    """
    found: list[SeedExpectation] = []
    for file in snapshot.files:
        for number, text in enumerate(file.text.splitlines(), start=1):
            match = MARKER_PATTERN.search(text)
            if match is None:
                continue
            expectation = Expectation(match.group("kind"))
            for rule_id in match.group("rules").split():
                found.append(
                    SeedExpectation(
                        path=file.path,
                        line=number,
                        rule_id=rule_id,
                        expectation=expectation,
                    )
                )
    deduplicated = {item.key: item for item in found}
    if len(deduplicated) != len(found):
        raise ScanError("the fixture states the same expectation twice")
    return SeedManifest(expectations=tuple(sorted(found, key=lambda item: item.key)))


def strip_markers(snapshot: RepoSnapshot) -> RepoSnapshot:
    """The same snapshot with every marker comment removed.

    Used by ``test_scan_seeded.py`` to prove the analyzer's findings do not depend on
    the markers. Without that test, a rule could come to key on the comment — by
    accident, through a text-matching helper — and the fixture would be grading the
    scanner on its ability to read the answer key.
    """
    stripped = {}
    for file in snapshot.files:
        lines = [
            MARKER_PATTERN.sub("", line).rstrip() for line in file.text.splitlines()
        ]
        stripped[file.path] = "\n".join(lines) + "\n"
    return RepoSnapshot.of_texts(stripped, root=snapshot.root)


@dataclass(frozen=True, slots=True)
class ScanScore:
    """How a scan did against the fixture's own claims."""

    detected: tuple[SeedExpectation, ...]
    missed: tuple[SeedExpectation, ...]
    #: ``SAFE``-marked lines the scanner flagged anyway. The number that decides
    #: whether anyone would run this in CI.
    false_positives: tuple[SeedExpectation, ...]
    #: Findings on lines carrying no marker at all. Not errors — the fixture does not
    #: claim to have marked every defect in it — but reported, because a jump here
    #: after a rule change is how an over-eager new rule announces itself.
    unmarked: tuple[CodeFinding, ...] = field(default=())
    #: ``INFO`` lines the scanner reported *actionably*, i.e. at
    #: :data:`ACTIONABLE_CONFIDENCE` or above. A gate failure: it means a construct
    #: the fixture says is informational would arrive as a pull request.
    over_escalated: tuple[SeedExpectation, ...] = field(default=())

    @property
    def n_seeded(self) -> int:
        return len(self.detected) + len(self.missed)

    @property
    def recall(self) -> float:
        return len(self.detected) / self.n_seeded if self.n_seeded else 0.0

    @property
    def control_precision(self) -> float:
        """Share of ``SAFE`` controls the scanner correctly left alone."""
        total = len(self.false_positives) + self.n_controls_passed
        return self.n_controls_passed / total if total else 1.0

    n_controls_passed: int = 0

    @property
    def clean(self) -> bool:
        """True when the scan matched every claim the fixture makes about itself."""
        return not self.missed and not self.false_positives and not self.over_escalated

    def describe(self) -> str:
        lines = [
            f"detected {len(self.detected)}/{self.n_seeded} seeded "
            f"(recall {self.recall:.3f})",
            f"false positives on SAFE controls: {len(self.false_positives)}",
            f"INFO lines escalated to actionable: {len(self.over_escalated)}",
            f"findings on unmarked lines: {len(self.unmarked)}",
        ]
        for item in self.missed:
            lines.append(f"  MISSED  {item.describe()}")
        for item in self.false_positives:
            lines.append(f"  FALSE+  {item.describe()}")
        for item in self.over_escalated:
            lines.append(f"  LOUD    {item.describe()}")
        return "\n".join(lines)


def score_scan(manifest: SeedManifest, findings: tuple[CodeFinding, ...]) -> ScanScore:
    """Grade ``findings`` against ``manifest``.

    A ``SEEDED`` expectation is satisfied by a finding for the same rule on the same
    line **at :data:`ACTIONABLE_CONFIDENCE` or above**. The confidence floor is part
    of the claim: F-07 asks for a patch per seeded vulnerability, patches are only
    drafted from that confidence up, and a "detection" that can never produce one
    would inflate recall while leaving the acceptance criterion unmet.

    Exact-line matching rather than a tolerance window, because a window would let a
    rule that reports the wrong line pass. See :data:`MARKER_PATTERN` for where a
    marker goes on a multi-line construct, and what happens when it goes elsewhere.
    """
    best: dict[tuple[str, int, str], FindingConfidence] = {}
    for finding in findings:
        key = (finding.path, finding.line, finding.rule_id)
        current = best.get(key)
        if current is None or finding.confidence.rank > current.rank:
            best[key] = finding.confidence
    marked_lines: set[tuple[str, int]] = {
        (item.path, item.line) for item in manifest.expectations
    }

    detected: list[SeedExpectation] = []
    missed: list[SeedExpectation] = []
    false_positives: list[SeedExpectation] = []
    over_escalated: list[SeedExpectation] = []
    controls_passed = 0

    for expectation in manifest.expectations:
        confidence = best.get(expectation.key)
        actionable = (
            confidence is not None
            and confidence.rank >= ACTIONABLE_CONFIDENCE.rank
        )
        if expectation.expectation is Expectation.SEEDED:
            (detected if actionable else missed).append(expectation)
        elif expectation.expectation is Expectation.SAFE:
            if confidence is None:
                controls_passed += 1
            else:
                false_positives.append(expectation)
        else:  # INFO
            if actionable:
                over_escalated.append(expectation)
            else:
                controls_passed += 1

    unmarked = tuple(
        finding
        for finding in findings
        if (finding.path, finding.line) not in marked_lines
    )
    return ScanScore(
        detected=tuple(detected),
        missed=tuple(missed),
        false_positives=tuple(false_positives),
        unmarked=unmarked,
        over_escalated=tuple(over_escalated),
        n_controls_passed=controls_passed,
    )
