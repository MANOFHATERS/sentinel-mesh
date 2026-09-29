"""The F-07 fixture and its self-describing ground truth.

Two things are being protected here.

The first is the measurement itself: that the fixture's markers name real rules, that
the scanner finds every seeded defect, and that it leaves every ``SAFE`` control
alone. That last number is the one that decides whether anyone would run this in CI,
so it is gated as hard as recall.

The second is subtler and is why :func:`strip_markers` exists: that the analyzer's
findings do not depend on the marker comments. Without that test a rule could come to
key on the comment — through a text-matching helper, or a regex that happens to see
it — and the fixture would be grading the scanner on its ability to read the answer
key. The test strips every marker and asserts the finding set is identical.
"""

from __future__ import annotations

import pytest

from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.findings import FindingConfidence, ScanError
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.rules import rule_ids
from sentinel.scan.seeded import (
    ACTIONABLE_CONFIDENCE,
    FIXTURE_DIR,
    Expectation,
    SeedExpectation,
    SeedManifest,
    load_seed_manifest,
    score_scan,
    strip_markers,
)

#: F-07's own bar: "at least 3 seeded vulnerabilities detected and a syntactically
#: valid patch PR opened for each".
F07_MINIMUM = 3


@pytest.fixture(scope="module")
def snapshot() -> RepoSnapshot:
    return RepoSnapshot.from_dir(FIXTURE_DIR)


@pytest.fixture(scope="module")
def manifest(snapshot: RepoSnapshot) -> SeedManifest:
    return load_seed_manifest(snapshot)


@pytest.fixture(scope="module")
def result(snapshot: RepoSnapshot):
    return AstAnalyzer().scan(snapshot)


class TestFixtureIntegrity:
    def test_the_fixture_exists_and_has_files(self, snapshot: RepoSnapshot):
        assert FIXTURE_DIR.is_dir()
        assert len(snapshot) >= 4

    def test_the_fixture_is_not_importable_by_the_package(self):
        # It lives under data/, never under src/, so it can never be imported or run.
        assert "data" in FIXTURE_DIR.parts
        assert "src" not in FIXTURE_DIR.parts

    def test_the_fixture_warns_about_itself(self):
        readme = (FIXTURE_DIR / "README.md").read_text(encoding="utf-8")
        assert "Do not deploy" in readme

    def test_every_marker_names_a_real_rule(self, manifest: SeedManifest):
        manifest.validate(frozenset(rule_ids()))

    def test_a_marker_naming_an_unknown_rule_is_refused(self):
        bogus = SeedManifest(
            expectations=(
                SeedExpectation(
                    path="a.py",
                    line=1,
                    rule_id="python.does-not-exist",
                    expectation=Expectation.SEEDED,
                ),
                SeedExpectation(
                    path="a.py", line=2, rule_id="python.weak-hash",
                    expectation=Expectation.SAFE,
                ),
            )
        )
        with pytest.raises(ScanError, match="do not exist"):
            bogus.validate(frozenset(rule_ids()))

    def test_a_fixture_with_no_controls_is_refused(self):
        # Recall alone is satisfiable by a scanner that flags every line.
        only_seeded = SeedManifest(
            expectations=(
                SeedExpectation(
                    path="a.py", line=1, rule_id="python.weak-hash",
                    expectation=Expectation.SEEDED,
                ),
            )
        )
        with pytest.raises(ScanError, match="no SAFE controls"):
            only_seeded.validate(frozenset(rule_ids()))

    def test_a_fixture_with_no_seeds_is_refused(self):
        only_safe = SeedManifest(
            expectations=(
                SeedExpectation(
                    path="a.py", line=1, rule_id="python.weak-hash",
                    expectation=Expectation.SAFE,
                ),
            )
        )
        with pytest.raises(ScanError, match="seeds no vulnerabilities"):
            only_safe.validate(frozenset(rule_ids()))

    def test_the_fixture_covers_a_broad_range_of_rules(self, manifest: SeedManifest):
        # F-07 needs three; a fixture that only exercises three rules would leave the
        # rest of the catalogue unmeasured.
        seeded_rules = {item.rule_id for item in manifest.seeded}
        assert len(seeded_rules) >= 10

    def test_every_seeded_rule_has_a_safe_control(self, manifest: SeedManifest):
        # A rule with a seeded case and no control has an unmeasured false-positive
        # rate, which is the half of the measurement that matters.
        seeded = {item.rule_id for item in manifest.seeded}
        controlled = {item.rule_id for item in manifest.controls} | {
            item.rule_id for item in manifest.informational
        }
        assert not seeded - controlled, sorted(seeded - controlled)

    def test_the_actionable_threshold_matches_the_analyzer(self):
        # An INFO control graded against a different threshold than the analyzer
        # drafts patches at would pass while the scanner sent a pull request.
        assert ACTIONABLE_CONFIDENCE is AstAnalyzer().min_patch_confidence


class TestF07AcceptanceCriteria:
    def test_every_seeded_vulnerability_is_detected(self, manifest, result):
        score = score_scan(manifest, result.findings)
        assert not score.missed, score.describe()
        assert score.recall == 1.0

    def test_no_safe_control_is_flagged(self, manifest, result):
        score = score_scan(manifest, result.findings)
        assert not score.false_positives, score.describe()

    def test_no_informational_finding_is_escalated_to_actionable(self, manifest, result):
        score = score_scan(manifest, result.findings)
        assert not score.over_escalated, score.describe()

    def test_the_scan_is_clean_against_every_claim(self, manifest, result):
        assert score_scan(manifest, result.findings).clean

    def test_f07_detection_bar_is_cleared_with_room(self, manifest, result):
        score = score_scan(manifest, result.findings)
        assert len(score.detected) >= F07_MINIMUM

    def test_f07_patch_bar_is_cleared(self, result):
        assert len(result.valid_patches) >= F07_MINIMUM

    def test_no_patch_was_built_and_then_rejected(self, result):
        # A rejected proposal is a defect in a fix, not an acceptable outcome. Rules
        # that cannot fix something decline *before* building anything.
        assert not result.rejected_patches, [
            (p.rule_id, p.rejection) for p in result.rejected_patches
        ]

    def test_every_finding_on_a_marked_line_is_accounted_for(self, manifest, result):
        assert not score_scan(manifest, result.findings).unmarked

    def test_the_fixture_parses_entirely(self, result):
        assert not result.parse_errors

    def test_nothing_in_the_fixture_was_skipped(self, snapshot: RepoSnapshot):
        assert not snapshot.skipped


class TestMarkersAreNotTheMechanism:
    def test_stripping_every_marker_changes_nothing(self, snapshot: RepoSnapshot):
        # The test this module exists for. If a rule ever keyed on the comment, the
        # fixture would be grading the scanner on reading its own answer key.
        from sentinel.scan.seeded import MARKER_PATTERN

        stripped = strip_markers(snapshot)
        # No *marker* survives. The word itself still appears in the fixture's own
        # docstrings, which is prose about the mechanism rather than an instance of it.
        for file in stripped.files:
            for line in file.text.splitlines():
                assert MARKER_PATTERN.search(line) is None, line

        analyzer = AstAnalyzer()
        with_markers = analyzer.scan(snapshot)
        without = analyzer.scan(stripped)
        assert [
            (f.path, f.line, f.rule_id, f.confidence) for f in with_markers.findings
        ] == [(f.path, f.line, f.rule_id, f.confidence) for f in without.findings]

    def test_stripping_markers_keeps_the_same_patch_count(self, snapshot: RepoSnapshot):
        analyzer = AstAnalyzer()
        assert len(analyzer.scan(strip_markers(snapshot)).valid_patches) == len(
            analyzer.scan(snapshot).valid_patches
        )

    def test_the_stripped_fixture_still_parses(self, snapshot: RepoSnapshot):
        assert not AstAnalyzer().scan(strip_markers(snapshot)).parse_errors


class TestManifestParsing:
    def test_a_marker_may_name_several_rules(self):
        source = 'app.run(debug=True, host="0.0.0.0")  # SEEDED: a.b c.d\n'
        manifest = load_seed_manifest(RepoSnapshot.of_texts({"a.py": source}))
        assert {item.rule_id for item in manifest.expectations} == {"a.b", "c.d"}

    def test_the_marker_must_be_at_end_of_line(self):
        # Anchored so a marker cannot be smuggled mid-line and change what a
        # different construct on the same line is graded as.
        source = 'x = 1  # SEEDED: a.b then more text\ny = 2\n'
        manifest = load_seed_manifest(RepoSnapshot.of_texts({"a.py": source}))
        assert {item.rule_id for item in manifest.expectations} == {
            "a.b",
            "then",
            "more",
            "text",
        }

    @pytest.mark.parametrize(
        ("kind", "expectation"),
        [
            ("SEEDED", Expectation.SEEDED),
            ("SAFE", Expectation.SAFE),
            ("INFO", Expectation.INFO),
        ],
    )
    def test_all_three_marker_kinds_parse(self, kind: str, expectation):
        source = f"x = 1  # {kind}: a.b\n"
        manifest = load_seed_manifest(RepoSnapshot.of_texts({"a.py": source}))
        assert manifest.expectations[0].expectation is expectation

    def test_a_duplicate_expectation_is_refused(self):
        source = "x = 1  # SEEDED: a.b a.b\n"
        with pytest.raises(ScanError, match="same expectation twice"):
            load_seed_manifest(RepoSnapshot.of_texts({"a.py": source}))

    def test_an_unmarked_file_yields_nothing(self):
        manifest = load_seed_manifest(RepoSnapshot.of_texts({"a.py": "x = 1\n"}))
        assert manifest.expectations == ()

    def test_expectations_are_sorted(self, manifest: SeedManifest):
        keys = [item.key for item in manifest.expectations]
        assert keys == sorted(keys)


class TestScoring:
    def _score(self, expectation: Expectation, confidence: FindingConfidence | None):
        from sentinel.core.schemas import Severity
        from sentinel.scan.findings import CodeFinding, SourceSpan

        manifest = SeedManifest(
            expectations=(
                SeedExpectation(
                    path="a.py", line=1, rule_id="python.weak-hash",
                    expectation=expectation,
                ),
            )
        )
        findings: tuple[CodeFinding, ...] = ()
        if confidence is not None:
            findings = (
                CodeFinding(
                    rule_id="python.weak-hash",
                    cwe="CWE-327",
                    title="t",
                    path="a.py",
                    span=SourceSpan(line=1, col=0, end_line=1, end_col=1),
                    severity=Severity.MEDIUM,
                    confidence=confidence,
                    message="m",
                    remediation="r",
                    excerpt="x",
                ),
            )
        return score_scan(manifest, findings)

    def test_a_seeded_line_needs_an_actionable_finding(self):
        # A "detection" too weak to produce a patch would inflate recall while
        # leaving F-07's actual criterion unmet.
        assert self._score(Expectation.SEEDED, FindingConfidence.HIGH).detected
        assert self._score(Expectation.SEEDED, FindingConfidence.LOW).missed
        assert self._score(Expectation.SEEDED, None).missed

    def test_a_safe_line_must_not_be_flagged_at_any_confidence(self):
        assert self._score(Expectation.SAFE, None).n_controls_passed == 1
        assert self._score(Expectation.SAFE, FindingConfidence.LOW).false_positives
        assert self._score(Expectation.SAFE, FindingConfidence.HIGH).false_positives

    def test_an_info_line_may_be_flagged_below_the_threshold_only(self):
        assert self._score(Expectation.INFO, FindingConfidence.LOW).n_controls_passed == 1
        assert self._score(Expectation.INFO, None).n_controls_passed == 1
        assert self._score(Expectation.INFO, FindingConfidence.HIGH).over_escalated

    def test_recall_is_zero_when_nothing_is_seeded(self):
        assert score_scan(SeedManifest(expectations=()), ()).recall == 0.0

    def test_describe_names_every_failure(self, manifest, result):
        text = score_scan(manifest, result.findings).describe()
        assert "recall" in text and "false positives" in text
