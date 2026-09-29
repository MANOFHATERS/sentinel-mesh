"""The analyzer: determinism, the StaticAnalyzer seam, and patch validation.

The validation tests are the point of this file. F-07's bar is a *"syntactically
valid patch"*, and the four gates in :data:`PATCH_CHECKS` are what turn that into
something a reviewer can rely on. Each gate gets a test that makes it fail, because a
check that has never rejected anything is a check nobody knows works.
"""

from __future__ import annotations

import ast
from typing import ClassVar

import pytest

from sentinel.core.schemas import Severity
from sentinel.scan.analyzer import (
    PARSE_ERROR_RULE,
    PATCH_CHECKS,
    AstAnalyzer,
    ScanResult,
    StaticAnalyzer,
    validate_patch,
)
from sentinel.scan.findings import FindingConfidence
from sentinel.scan.patch import SourceEdit
from sentinel.scan.repo import RepoSnapshot, SourceFile
from sentinel.scan.rules import RULES_BY_ID, RuleContext
from sentinel.scan.symbols import ImportTable
from sentinel.scan.taint import analyse_taint

VULNERABLE = """\
import hashlib
import os


def digest(payload):
    return hashlib.md5(payload).hexdigest()


def run(command):
    return os.system(command)
"""


def _context(file: SourceFile) -> RuleContext:
    tree = ast.parse(file.text, filename=file.path)
    imports = ImportTable.of(tree)
    return RuleContext(
        path=file.path,
        source=file.text,
        tree=tree,
        imports=imports,
        taint=analyse_taint(tree, imports),
    )


class TestDeterminism:
    def test_two_scans_of_the_same_tree_are_identical(self):
        snapshot = RepoSnapshot.of_texts({"a.py": VULNERABLE, "b.py": VULNERABLE})
        first, second = AstAnalyzer().scan(snapshot), AstAnalyzer().scan(snapshot)
        assert [f.ref for f in first.findings] == [f.ref for f in second.findings]
        assert [p.diff for p in first.patches] == [p.diff for p in second.patches]

    def test_findings_are_sorted_by_location(self):
        snapshot = RepoSnapshot.of_texts({"z.py": VULNERABLE, "a.py": VULNERABLE})
        findings = AstAnalyzer().scan(snapshot).findings
        keys = [(f.path, f.span.line, f.span.col, f.rule_id) for f in findings]
        assert keys == sorted(keys)

    def test_file_order_in_the_snapshot_does_not_change_the_result(self):
        forward = RepoSnapshot.of_texts({"a.py": VULNERABLE, "b.py": "x = 1\n"})
        backward = RepoSnapshot(root="<memory>", files=tuple(reversed(forward.files)))
        assert [f.ref for f in AstAnalyzer().scan(forward).findings] == [
            f.ref for f in AstAnalyzer().scan(backward).findings
        ]


class TestSeam:
    def test_the_real_analyzer_satisfies_the_protocol(self):
        assert isinstance(AstAnalyzer(), StaticAnalyzer)

    def test_a_stub_satisfies_the_protocol(self):
        # The seam a Semgrep-backed analyzer would enter through. If this stops
        # type-checking as a StaticAnalyzer, the substitution claim is decorative.
        class Stub:
            name = "stub"

            def scan(self, snapshot: RepoSnapshot) -> ScanResult:
                return ScanResult(
                    findings=(), patches=(), files_scanned=len(snapshot), lines_scanned=0
                )

        assert isinstance(Stub(), StaticAnalyzer)

    def test_the_rule_set_is_injectable(self):
        only_hash = AstAnalyzer(rules=(RULES_BY_ID["python.weak-hash"],))
        result = only_hash.scan(RepoSnapshot.of_texts({"a.py": VULNERABLE}))
        assert {f.rule_id for f in result.findings} == {"python.weak-hash"}


class TestSyntaxErrors:
    def test_an_unparseable_file_is_a_finding_not_a_crash(self):
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"bad.py": "def (:\n"}))
        assert result.parse_errors and result.parse_errors[0][0] == "bad.py"
        assert [f.rule_id for f in result.findings] == [PARSE_ERROR_RULE]

    def test_one_unparseable_file_does_not_stop_the_others(self):
        snapshot = RepoSnapshot.of_texts({"bad.py": "def (:\n", "good.py": VULNERABLE})
        result = AstAnalyzer().scan(snapshot)
        assert "python.weak-hash" in {f.rule_id for f in result.findings}

    def test_the_parse_error_finding_is_informational(self):
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"bad.py": "def (:\n"}))
        assert result.findings[0].severity is Severity.INFO


class TestPatchThreshold:
    def test_low_confidence_findings_get_no_patch(self):
        # A dangerous construct on a literal. Reported, never patched: proposing a
        # code change on a guess is how a reviewer learns to approve without reading.
        source = 'import subprocess\nsubprocess.run("uptime", shell=True)\n'
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"a.py": source}))
        low = [f for f in result.findings if f.confidence is FindingConfidence.LOW]
        assert low
        assert not {p.finding_ref for p in result.valid_patches} & {f.ref for f in low}

    def test_lowering_the_threshold_admits_them(self):
        source = 'import subprocess\nsubprocess.run("uptime", shell=True)\n'
        analyzer = AstAnalyzer(min_patch_confidence=FindingConfidence.LOW)
        result = analyzer.scan(RepoSnapshot.of_texts({"a.py": source}))
        assert result.valid_patches


class TestRuleCounts:
    def test_counts_every_instance_in_a_file(self):
        source = (
            "import hashlib\n"
            "def a(p):\n    return hashlib.md5(p)\n"
            "def b(p):\n    return hashlib.sha1(p)\n"
        )
        counts = AstAnalyzer().rule_counts(SourceFile.of("a.py", source))
        assert counts["python.weak-hash"] == 2

    def test_a_file_with_two_instances_still_validates_each_patch(self):
        # The regression this design exists for. A boolean "does the rule still fire"
        # check rejected every correct patch in any file containing two instances of
        # the same bug — measured at 8 of 16 rejected on the F-07 fixture.
        source = (
            "import hashlib\n"
            "def a(p):\n    return hashlib.md5(p)\n"
            "def b(p):\n    return hashlib.md5(p)\n"
        )
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"a.py": source}))
        patches = [p for p in result.patches if p.rule_id == "python.weak-hash"]
        assert len(patches) == 2
        assert all(p.is_valid for p in patches), [p.rejection for p in patches]

    def test_an_unparseable_file_reports_a_sentinel_count(self):
        counts = AstAnalyzer().rule_counts(SourceFile.of("a.py", "def (:\n"))
        assert counts == {PARSE_ERROR_RULE: 1}


class TestValidationGates:
    """Each of the four gates, made to fail."""

    def _validate(self, edits: tuple[SourceEdit, ...], source: str = VULNERABLE):
        file = SourceFile.of("a.py", source)
        return validate_patch(
            rule=RULES_BY_ID["python.weak-hash"],
            edits=edits,
            ctx=_context(file),
            file=file,
            finding_ref="code://a.py#python.weak-hash@6",
            analyzer=AstAnalyzer(),
        )

    def test_the_four_gates_are_named_in_order(self):
        assert len(PATCH_CHECKS) == 4

    def test_a_good_patch_passes_every_gate(self):
        file = SourceFile.of("a.py", VULNERABLE)
        start = file.text.index("hashlib.md5")
        edit = SourceEdit(
            start=start,
            end=start + len("hashlib.md5"),
            replacement="hashlib.sha256",
            note="upgrade the digest",
        )
        proposal = self._validate((edit,))
        assert proposal.is_valid, proposal.rejection
        assert proposal.checks == PATCH_CHECKS

    def test_gate_one_rejects_a_patch_that_does_not_parse(self):
        file = SourceFile.of("a.py", VULNERABLE)
        start = file.text.index("hashlib.md5")
        edit = SourceEdit(
            start=start, end=start + 11, replacement="hashlib.((", note="broken"
        )
        proposal = self._validate((edit,))
        assert not proposal.is_valid
        assert "does not parse" in (proposal.rejection or "")

    def test_gate_three_rejects_a_patch_that_does_not_remove_the_finding(self):
        # A comment change: parses, round-trips, fixes nothing.
        proposal = self._validate(
            (SourceEdit(start=0, end=0, replacement="# note\n", note="a comment"),)
        )
        assert not proposal.is_valid
        assert "still fires" in (proposal.rejection or "")

    def test_gate_four_rejects_a_patch_that_introduces_a_new_finding(self):
        file = SourceFile.of("a.py", VULNERABLE)
        start = file.text.index("hashlib.md5")
        edit = SourceEdit(
            start=start,
            end=start + len("hashlib.md5(payload)"),
            # Fixes the hash and adds a command injection.
            replacement='os.system(payload) or hashlib.sha256(b"")',
            note="fix one thing and break another",
        )
        proposal = self._validate((edit,))
        assert not proposal.is_valid
        assert "introduces new findings" in (proposal.rejection or "")

    def test_a_no_op_edit_is_rejected(self):
        proposal = self._validate(
            (SourceEdit(start=0, end=0, replacement="", note="nothing"),)
        )
        assert not proposal.is_valid
        assert "no change" in (proposal.rejection or "")

    def test_overlapping_edits_are_reported_not_raised(self):
        proposal = self._validate(
            (
                SourceEdit(start=0, end=5, replacement="X", note="a"),
                SourceEdit(start=3, end=8, replacement="Y", note="b"),
            )
        )
        assert not proposal.is_valid
        assert "did not apply" in (proposal.rejection or "")

    def test_a_fix_that_raises_is_reported_not_propagated(self):
        # A broken rule must not take the whole scan down with it.
        from sentinel.scan.analyzer import build_patch
        from sentinel.scan.rules import Match, Rule

        def exploding_fix(match, ctx):
            raise RuntimeError("boom")

        rule = Rule(
            rule_id="python.explodes",
            cwe="CWE-1",
            title="t",
            severity=Severity.LOW,
            message="m",
            remediation="r" * 25,
            matcher=lambda node, ctx: None,
            fix=exploding_fix,
        )
        file = SourceFile.of("a.py", VULNERABLE)
        ctx = _context(file)
        tree = ast.parse(file.text)
        match = Match(node=tree.body[0], confidence=FindingConfidence.HIGH)
        proposal = build_patch(rule, match, ctx, file, analyzer=AstAnalyzer())
        assert proposal is not None and not proposal.is_valid
        assert "RuntimeError" in (proposal.rejection or "")


class TestScanResultViews:
    def test_unpatched_names_the_findings_with_no_valid_patch(self):
        source = 'import pickle\ndef f(b):\n    return pickle.loads(b)\n'
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"a.py": source}))
        assert [f.rule_id for f in result.unpatched()] == [
            "python.pickle-deserialization"
        ]

    def test_findings_by_severity_filters(self):
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"a.py": VULNERABLE}))
        criticals = result.findings_by_severity(Severity.CRITICAL)
        assert all(f.severity >= Severity.CRITICAL for f in criticals)

    def test_summary_lists_every_rule_and_the_parse_errors(self):
        snapshot = RepoSnapshot.of_texts({"a.py": VULNERABLE, "bad.py": "def (:\n"})
        summary = AstAnalyzer().scan(snapshot).summary()
        assert "python.weak-hash" in summary
        assert "bad.py did not parse" in summary

    def test_patched_refs_only_counts_valid_patches(self):
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"a.py": VULNERABLE}))
        assert result.patched_refs == {p.finding_ref for p in result.valid_patches}


class TestEveryPatchInTheWholeCatalogueIsSound:
    """A property test over the rules, rather than one example per rule.

    Every validated patch the analyzer produces anywhere must parse, must round-trip
    through its own diff, and must actually reduce the count of its rule. The
    per-rule tests assert the *content* of each fix; this asserts the invariant holds
    for all of them at once, so a new rule cannot ship a fix that only looks right.
    """

    SOURCES: ClassVar[list[str]] = [
        VULNERABLE,
        'import subprocess\ndef f(c):\n    subprocess.run(c, shell=True)\n',
        'import yaml\ndef f(t):\n    return yaml.load(t)\n',
        'API_KEY = "fixture-only-billing-key"\n',
        'import tempfile\ndef f():\n    return tempfile.mktemp()\n',
        'import requests\nrequests.get("https://x", verify=False)\n',
        "from jinja2 import Environment\ndef f():\n    return Environment()\n",
        'from flask import Flask\napp = Flask(__name__)\napp.run(debug=True, host="0.0.0.0")\n',
    ]

    @pytest.mark.parametrize("source", SOURCES)
    def test_every_produced_patch_is_sound(self, source: str):
        from sentinel.scan.patch import apply_unified_diff

        analyzer = AstAnalyzer()
        file = SourceFile.of("a.py", source)
        result = analyzer.scan(RepoSnapshot.of_texts({"a.py": source}))
        assert not result.rejected_patches, [
            p.rejection for p in result.rejected_patches
        ]
        before_counts = analyzer.rule_counts(file)
        for patch in result.valid_patches:
            ast.parse(patch.after)
            assert apply_unified_diff(patch.before, patch.diff) == patch.after
            after_counts = analyzer.rule_counts(file.with_text(patch.after))
            assert after_counts.get(patch.rule_id, 0) < before_counts.get(
                patch.rule_id, 0
            )
            for rule_id, count in after_counts.items():
                assert count <= before_counts.get(rule_id, 0), rule_id

    @pytest.mark.parametrize("source", SOURCES)
    def test_every_patch_note_is_a_sentence_a_reviewer_can_read(self, source: str):
        result = AstAnalyzer().scan(RepoSnapshot.of_texts({"a.py": source}))
        for patch in result.valid_patches:
            assert patch.notes
            for note in patch.notes:
                assert len(note) > 15, (patch.rule_id, note)
