"""Composing several validated patches to one file into a single commit (Part 4).

The Code-Scan Agent validates every patch against the *original* file, which is the
right unit for review and the wrong unit for a commit: the fixture has five patches in
``app.py`` and applying them one after another fails on line offsets. The Git
connector therefore composes them, and a composition that is only *usually* right
would push a silently half-applied security fix. So the tests here are of three
kinds: exact semantics on hand-written diffs, a randomized equivalence against
sequential application, and the real fixture — where the composed file must lose
exactly the findings that were patched and gain none.
"""

from __future__ import annotations

import ast
import random

import pytest

from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.patch import (
    PatchError,
    apply_unified_diff,
    compose_patches,
    unified_diff,
)
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.seeded import FIXTURE_DIR

BASE = "".join(f"line {i}\n" for i in range(1, 31))


def _replace(text: str, line: int, new: str) -> str:
    lines = text.splitlines(keepends=True)
    lines[line - 1] = new
    return "".join(lines)


def _diff(before: str, after: str, path: str = "f.py") -> str:
    return unified_diff(path=path, before=before, after=after)


class TestSingleDiff:
    def test_one_diff_composes_to_exactly_what_apply_produces(self):
        after = _replace(BASE, 7, "changed 7\n")
        diff = _diff(BASE, after)
        assert compose_patches(BASE, [diff]) == apply_unified_diff(BASE, diff) == after

    def test_no_diffs_is_the_identity(self):
        assert compose_patches(BASE, []) == BASE

    def test_pure_insertion_at_the_top(self):
        after = "import shlex\n" + BASE
        assert compose_patches(BASE, [_diff(BASE, after)]) == after

    def test_pure_deletion(self):
        after = BASE.replace("line 12\n", "")
        assert compose_patches(BASE, [_diff(BASE, after)]) == after

    def test_multi_line_replacement(self):
        after = BASE.replace("line 4\nline 5\n", "four\nfive\nfive-and-a-half\n")
        assert compose_patches(BASE, [_diff(BASE, after)]) == after


class TestSeveralDiffs:
    def test_two_distant_changes_both_land(self):
        a = _diff(BASE, _replace(BASE, 3, "three\n"))
        b = _diff(BASE, _replace(BASE, 25, "twenty-five\n"))
        out = compose_patches(BASE, [a, b])
        assert "three\n" in out and "twenty-five\n" in out
        assert out.count("\n") == BASE.count("\n")

    def test_changes_sharing_context_compose(self):
        # Lines 10 and 12 share context lines; the hunks overlap as text but the
        # changes do not, and that is the pair a real file most often needs.
        a = _diff(BASE, _replace(BASE, 10, "ten\n"))
        b = _diff(BASE, _replace(BASE, 12, "twelve\n"))
        out = compose_patches(BASE, [a, b])
        assert out == _replace(_replace(BASE, 10, "ten\n"), 12, "twelve\n")

    def test_order_of_diffs_does_not_matter(self):
        a = _diff(BASE, _replace(BASE, 5, "five\n"))
        b = _diff(BASE, _replace(BASE, 20, "twenty\n"))
        assert compose_patches(BASE, [a, b]) == compose_patches(BASE, [b, a])

    def test_identical_insertions_are_merged_not_duplicated(self):
        # Two fixes that each add "import shlex" want one import.
        a = _diff(BASE, "import shlex\n" + _replace(BASE, 8, "eight\n"))
        b = _diff(BASE, "import shlex\n" + _replace(BASE, 22, "twenty-two\n"))
        out = compose_patches(BASE, [a, b])
        assert out.count("import shlex\n") == 1
        assert "eight\n" in out and "twenty-two\n" in out

    def test_different_insertions_at_one_point_both_land_in_diff_order(self):
        a = _diff(BASE, "import os\n" + BASE)
        b = _diff(BASE, "import shlex\n" + BASE)
        assert compose_patches(BASE, [a, b]).startswith("import os\nimport shlex\nline 1\n")

    def test_insertion_before_a_replaced_line_composes(self):
        lines = BASE.splitlines(keepends=True)
        inserted = "".join([*lines[:4], "new\n", *lines[4:]])
        a = _diff(BASE, inserted)
        b = _diff(BASE, _replace(BASE, 5, "five\n"))
        out = compose_patches(BASE, [a, b])
        assert "line 4\nnew\nfive\nline 6\n" in out

    def test_overlapping_different_replacements_are_refused(self):
        a = _diff(BASE, BASE.replace("line 4\nline 5\n", "A\nB\n"))
        b = _diff(BASE, BASE.replace("line 5\nline 6\n", "C\nD\n"))
        with pytest.raises(PatchError, match="overlapping"):
            compose_patches(BASE, [a, b])

    def test_insertion_inside_a_replaced_range_is_refused(self):
        lines = BASE.splitlines(keepends=True)
        a = _diff(BASE, BASE.replace("line 4\nline 5\nline 6\n", "X\n"))
        b = _diff(BASE, "".join([*lines[:4], "wedge\n", *lines[4:]]))
        with pytest.raises(PatchError, match="overlapping"):
            compose_patches(BASE, [a, b])

    def test_a_diff_against_a_different_file_is_refused(self):
        other = BASE.replace("line 9\n", "nine\n")
        diff = _diff(other, _replace(other, 9, "NINE\n"))
        with pytest.raises(PatchError, match="does not match"):
            compose_patches(BASE, [diff])

    def test_a_change_past_the_end_is_refused(self):
        diff = "@@ -40,1 +40,1 @@\n-line 40\n+forty\n"
        with pytest.raises(PatchError, match="past the end"):
            compose_patches(BASE, [diff])

    def test_garbage_in_a_hunk_is_refused(self):
        with pytest.raises(PatchError, match="unrecognised"):
            compose_patches(BASE, ["@@ -1,1 +1,1 @@\n?line 1\n"])


class TestSameLineMerge:
    """Two fixes on one line — found on the fixture, not anticipated."""

    LINE = 'app.run(debug=True, host="0.0.0.0", port=8080)\n'
    TEXT = "def serve():\n    " + LINE + "\n"

    def test_disjoint_edits_on_one_line_both_land(self):
        debug = self.TEXT.replace("debug=True", "debug=False")
        bind = self.TEXT.replace('host="0.0.0.0"', 'host="127.0.0.1"')
        out = compose_patches(self.TEXT, [_diff(self.TEXT, debug), _diff(self.TEXT, bind)])
        assert 'app.run(debug=False, host="127.0.0.1", port=8080)' in out
        ast.parse(out)

    def test_merge_is_symmetric(self):
        debug = _diff(self.TEXT, self.TEXT.replace("debug=True", "debug=False"))
        bind = _diff(self.TEXT, self.TEXT.replace('"0.0.0.0"', '"127.0.0.1"'))
        assert compose_patches(self.TEXT, [debug, bind]) == compose_patches(
            self.TEXT, [bind, debug]
        )

    def test_competing_edits_of_the_same_characters_are_refused(self):
        a = _diff(self.TEXT, self.TEXT.replace('"0.0.0.0"', '"127.0.0.1"'))
        b = _diff(self.TEXT, self.TEXT.replace('"0.0.0.0"', '"10.0.0.1"'))
        with pytest.raises(PatchError, match="overlapping places"):
            compose_patches(self.TEXT, [a, b])

    def test_identical_whole_line_rewrites_are_one_change(self):
        fixed = self.TEXT.replace("debug=True", "debug=False")
        assert compose_patches(self.TEXT, [_diff(self.TEXT, fixed)] * 2) == fixed


class TestRandomizedEquivalence:
    """Non-overlapping single-line edits: composing == applying them in sequence."""

    @pytest.mark.parametrize("seed", range(40))
    def test_compose_matches_sequential_application(self, seed):
        rng = random.Random(seed)
        n = rng.randint(20, 80)
        base = "".join(f"row {i} {rng.random():.6f}\n" for i in range(n))
        # Lines at least 2 apart, so every edit is its own change.
        picks = sorted(rng.sample(range(0, n, 2), k=rng.randint(2, min(8, n // 2))))
        expected = base.splitlines(keepends=True)
        diffs = []
        for line in picks:
            kind = rng.choice(["replace", "delete", "insert"])
            single = base.splitlines(keepends=True)
            if kind == "replace":
                single[line] = f"edited {line}\n"
                expected[line] = f"edited {line}\n"
            elif kind == "delete":
                single[line] = ""
                expected[line] = ""
            else:
                single[line] = f"inserted {line}\n" + single[line]
                expected[line] = f"inserted {line}\n" + expected[line]
            diffs.append(_diff(base, "".join(single)))
        rng.shuffle(diffs)
        assert compose_patches(base, diffs) == "".join(expected)


class TestTheFixture:
    """The real patches the Code-Scan Agent produces for ``data/vulnerable_app``."""

    @pytest.fixture(scope="class")
    def scan(self):
        snapshot = RepoSnapshot.from_dir(FIXTURE_DIR)
        analyzer = AstAnalyzer()
        return snapshot, analyzer, analyzer.scan(snapshot)

    def test_several_files_carry_more_than_one_patch(self, scan):
        _snapshot, _analyzer, result = scan
        paths = [patch.path for patch in result.valid_patches]
        assert max(paths.count(p) for p in set(paths)) >= 3

    def test_every_file_composes_parses_and_loses_exactly_the_patched_findings(self, scan):
        snapshot, analyzer, result = scan
        for file in snapshot.files:
            patches = [p for p in result.valid_patches if p.path == file.path]
            if not patches:
                continue
            composed = compose_patches(file.text, [p.diff for p in patches])
            ast.parse(composed)
            before = analyzer.rule_counts(file)
            after = analyzer.rule_counts(file.with_text(composed))
            for rule_id in {p.rule_id for p in patches}:
                fixed = sum(1 for p in patches if p.rule_id == rule_id)
                assert after.get(rule_id, 0) == before[rule_id] - fixed, (file.path, rule_id)
            introduced = {r for r, n in after.items() if n > before.get(r, 0)}
            assert not introduced, (file.path, introduced)

    def test_the_same_line_double_defect_is_fully_fixed(self, scan):
        snapshot, _analyzer, result = scan
        config = next(f for f in snapshot.files if f.path == "config.py")
        composed = compose_patches(
            config.text, [p.diff for p in result.valid_patches if p.path == "config.py"]
        )
        assert 'app.run(debug=False, host="127.0.0.1", port=8080)  # SEEDED' in composed

    def test_each_patch_alone_composes_to_its_own_after(self, scan):
        _snapshot, _analyzer, result = scan
        for patch in result.valid_patches:
            assert compose_patches(patch.before, [patch.diff]) == patch.after
