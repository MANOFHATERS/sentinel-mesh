"""Source edits, unified diffs, and the applier that makes "valid patch" a measurement.

The applier gets the most attention here, because it is the component that turns
F-07's *"syntactically valid patch"* from a claim into a check. It is written as an
independent replay rather than as a reuse of the edit machinery, so these tests
include the cases where a diff should be *refused*: a patch that silently applies to
the wrong content is how a security fix lands on the wrong line.
"""

from __future__ import annotations

import ast

import pytest

from sentinel.scan.patch import (
    CONTEXT_LINES,
    PatchError,
    PatchProposal,
    PullRequestDraft,
    SourceEdit,
    apply_edits,
    apply_unified_diff,
    ensure_import_edit,
    normalise_source,
    offset_of,
    unified_diff,
)


def _roundtrip(before: str, after: str) -> None:
    diff = unified_diff(path="f.py", before=before, after=after)
    assert apply_unified_diff(before, diff) == after


class TestNormaliseSource:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a\r\nb\r\n", "a\nb\n"),
            ("a\rb\r", "a\nb\n"),
            ("a\nb", "a\nb\n"),
            ("a\nb\n", "a\nb\n"),
            ("", ""),
        ],
    )
    def test_normalises_endings_and_guarantees_a_final_newline(self, raw, expected):
        assert normalise_source(raw) == expected

    def test_normalisation_is_idempotent(self):
        once = normalise_source("a\r\nb")
        assert normalise_source(once) == once


class TestOffsetOf:
    def test_start_of_file(self):
        assert offset_of("abc\ndef\n", 1, 0) == 0

    def test_second_line(self):
        assert offset_of("abc\ndef\n", 2, 0) == 4

    def test_column_within_a_line(self):
        assert offset_of("abc\ndef\n", 2, 2) == 6

    def test_one_past_the_last_line_is_end_of_file(self):
        source = "abc\n"
        assert offset_of(source, 2, 0) == len(source)

    def test_columns_are_utf8_byte_offsets_like_ast_reports_them(self):
        # CPython's ast reports col_offset in UTF-8 bytes. On a line with a
        # multi-byte character, treating it as a character index puts a splice
        # boundary mid-character and corrupts the file.
        source = 'x = "café"\ny = 1\n'
        node = ast.parse(source).body[0]
        assert isinstance(node, ast.Assign)
        value = node.value
        start = offset_of(source, value.lineno, value.col_offset)
        end = offset_of(source, value.end_lineno, value.end_col_offset)
        assert source[start:end] == '"café"'

    @pytest.mark.parametrize(("line", "col"), [(0, 0), (9, 0), (1, 99)])
    def test_out_of_range_is_refused(self, line, col):
        with pytest.raises(PatchError):
            offset_of("abc\n", line, col)


class TestApplyEdits:
    def test_no_edits_is_the_identity(self):
        assert apply_edits("abc", ()) == "abc"

    def test_single_replacement(self):
        edit = SourceEdit(start=0, end=1, replacement="X", note="n")
        assert apply_edits("abc", (edit,)) == "Xbc"

    def test_edits_apply_right_to_left_so_offsets_stay_valid(self):
        source = "aaaa"
        edits = (
            SourceEdit(start=0, end=1, replacement="LONGER", note="n"),
            SourceEdit(start=3, end=4, replacement="Z", note="n"),
        )
        assert apply_edits(source, edits) == "LONGERaaZ"

    def test_insertion_is_a_zero_width_edit(self):
        edit = SourceEdit(start=1, end=1, replacement="X", note="n")
        assert apply_edits("ab", (edit,)) == "aXb"

    def test_overlapping_edits_are_refused_rather_than_merged(self):
        edits = (
            SourceEdit(start=0, end=3, replacement="X", note="n"),
            SourceEdit(start=2, end=5, replacement="Y", note="n"),
        )
        with pytest.raises(PatchError, match="overlap"):
            apply_edits("abcdef", edits)

    def test_adjacent_edits_are_allowed(self):
        edits = (
            SourceEdit(start=0, end=2, replacement="X", note="n"),
            SourceEdit(start=2, end=4, replacement="Y", note="n"),
        )
        assert apply_edits("abcd", edits) == "XY"

    def test_an_edit_past_the_end_is_refused(self):
        with pytest.raises(PatchError, match="source is"):
            apply_edits("ab", (SourceEdit(start=0, end=9, replacement="X", note="n"),))

    def test_an_edit_must_explain_itself(self):
        with pytest.raises(PatchError, match="note"):
            SourceEdit(start=0, end=1, replacement="X", note="   ")

    def test_an_inverted_range_is_refused(self):
        with pytest.raises(PatchError, match="invalid edit range"):
            SourceEdit(start=5, end=1, replacement="X", note="n")


class TestEnsureImportEdit:
    def _apply(self, source: str, module: str) -> str:
        tree = ast.parse(source)
        edit = ensure_import_edit(source, tree, module)
        return source if edit is None else apply_edits(source, (edit,))

    def test_adds_after_the_last_top_level_import(self):
        source = "import os\nimport sys\n\nx = 1\n"
        assert self._apply(source, "shlex") == "import os\nimport sys\nimport shlex\n\nx = 1\n"

    def test_adds_after_the_module_docstring(self):
        source = '"""Doc."""\n\nx = 1\n'
        assert self._apply(source, "os").startswith('"""Doc."""\nimport os\n')

    def test_adds_at_the_top_when_there_is_nothing_else(self):
        assert self._apply("x = 1\n", "os") == "import os\nx = 1\n"

    def test_already_imported_is_a_no_op(self):
        source = "import os\nx = 1\n"
        assert ensure_import_edit(source, ast.parse(source), "os") is None

    def test_a_submodule_import_counts_as_importing_the_package(self):
        source = "import os.path\nx = 1\n"
        assert ensure_import_edit(source, ast.parse(source), "os") is None

    def test_an_aliased_import_does_not_count(self):
        # ``import os as operating_system`` does not bind the name ``os``, so a fix
        # emitting ``os.path.basename(...)`` would raise NameError without this.
        source = "import os as operating_system\nx = 1\n"
        assert ensure_import_edit(source, ast.parse(source), "os") is not None

    def test_a_from_import_does_not_count(self):
        source = "from os import path\nx = 1\n"
        assert ensure_import_edit(source, ast.parse(source), "os") is not None

    def test_a_function_local_import_does_not_count(self):
        # The fix's call site is at module scope or in a different function, so
        # reusing a function-local binding would NameError on the vulnerable path.
        source = "def f():\n    import os\n    return os\n"
        assert ensure_import_edit(source, ast.parse(source), "os") is not None

    def test_an_import_after_real_code_is_not_part_of_the_prologue(self):
        source = "x = 1\nimport sys\n"
        patched = self._apply(source, "os")
        assert patched.startswith("import os\nx = 1\n")

    def test_the_result_always_parses(self):
        for source in ('"""D."""\n', "import os\n", "x = 1\n", "\n", "from a import b\n"):
            ast.parse(self._apply(source, "shlex"))


class TestUnifiedDiff:
    def test_identical_inputs_produce_no_diff(self):
        assert unified_diff(path="f.py", before="a\n", after="a\n") == ""

    def test_headers_name_the_path(self):
        diff = unified_diff(path="pkg/f.py", before="a\n", after="b\n")
        assert diff.startswith("--- a/pkg/f.py\n+++ b/pkg/f.py\n")

    def test_context_is_the_documented_width(self):
        before = "".join(f"{i}\n" for i in range(20))
        after = before.replace("10\n", "TEN\n")
        diff = unified_diff(path="f.py", before=before, after=after)
        context = [line for line in diff.splitlines() if line.startswith(" ")]
        assert len(context) == CONTEXT_LINES * 2

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            ("a\nb\nc\n", "a\nB\nc\n"),
            ("a\nb\nc\n", "x\na\nb\nc\n"),
            ("a\nb\nc\n", "a\nb\nc\nd\n"),
            ("a\nb\nc\n", ""),
            ("", "a\n"),
            ("a\n\nb\n", "a\n\nc\n"),
            ("\n\n\n", "x\n\n\n"),
            ("a\nb\nc\nd\ne\n", "a\nc\ne\n"),
        ],
    )
    def test_every_shape_round_trips(self, before, after):
        _roundtrip(before, after)

    def test_multiple_distant_hunks_round_trip(self):
        before = "".join(f"{i}\n" for i in range(60))
        after = "".join(
            ("X\n" if i in (2, 25, 55) else f"{i}\n") for i in range(60)
        )
        diff = unified_diff(path="f.py", before=before, after=after)
        assert diff.count("@@ -") == 3
        assert apply_unified_diff(before, diff) == after


class TestApplyUnifiedDiffRefusals:
    """The half that makes the check worth having."""

    def test_an_empty_diff_is_the_identity(self):
        assert apply_unified_diff("a\n", "") == "a\n"

    def test_a_context_mismatch_is_refused(self):
        diff = unified_diff(path="f.py", before="a\nb\nc\n", after="a\nB\nc\n")
        with pytest.raises(PatchError, match="context mismatch"):
            apply_unified_diff("x\ny\nz\n", diff)

    def test_a_removal_that_does_not_match_is_refused(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-nope\n+yes\n"
        with pytest.raises(PatchError, match="does not apply"):
            apply_unified_diff("actual\n", diff)

    def test_a_hunk_past_the_end_of_the_file_is_refused(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -50,1 +50,1 @@\n-x\n+y\n"
        with pytest.raises(PatchError, match="the file has"):
            apply_unified_diff("a\n", diff)

    def test_out_of_order_hunks_are_refused(self):
        diff = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -3,1 +3,1 @@\n-c\n+C\n"
            "@@ -1,1 +1,1 @@\n-a\n+A\n"
        )
        with pytest.raises(PatchError, match="backwards"):
            apply_unified_diff("a\nb\nc\n", diff)

    def test_garbage_before_the_first_hunk_is_refused(self):
        with pytest.raises(PatchError, match="before the first hunk"):
            apply_unified_diff("a\n", "this is not a diff\n")

    def test_an_unrecognised_body_line_is_refused(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n?what\n"
        with pytest.raises(PatchError):
            apply_unified_diff("a\n", diff)

    def test_git_metadata_lines_are_tolerated(self):
        body = unified_diff(path="f.py", before="a\n", after="b\n")
        decorated = "diff --git a/f.py b/f.py\nindex 000..111 100644\n" + body
        assert apply_unified_diff("a\n", decorated) == "b\n"

    def test_a_pure_insertion_hunk_is_placed_correctly(self):
        # ``@@ -0,0 +1,1 @@`` means "insert before line 1", and the off-by-one here
        # is the one that breaks every added import.
        diff = "--- a/f.py\n+++ b/f.py\n@@ -0,0 +1,1 @@\n+first\n"
        assert apply_unified_diff("a\n", diff) == "first\na\n"


class TestPatchProposal:
    def _proposal(self, **overrides) -> PatchProposal:
        defaults = dict(
            finding_ref="code://f.py#python.weak-hash@3",
            path="f.py",
            rule_id="python.weak-hash",
            diff=unified_diff(path="f.py", before="a\n", after="b\n"),
            before="a\n",
            after="b\n",
            notes=("did a thing",),
            checks=("one", "two"),
        )
        return PatchProposal(**{**defaults, **overrides})

    def test_a_clean_proposal_is_valid(self):
        assert self._proposal().is_valid

    def test_a_rejected_proposal_is_not_valid(self):
        assert not self._proposal(rejection="nope").is_valid

    def test_an_empty_diff_is_not_valid(self):
        assert not self._proposal(diff="").is_valid

    def test_changed_lines_excludes_the_file_headers(self):
        assert self._proposal().changed_lines == 2

    def test_describe_states_the_rejection(self):
        assert "rejected: nope" in self._proposal(rejection="nope").describe()


class TestPullRequestDraft:
    def _draft(self, **overrides) -> PullRequestDraft:
        patch = PatchProposal(
            finding_ref="code://f.py#r@1",
            path="f.py",
            rule_id="r",
            diff=unified_diff(path="f.py", before="a\n", after="b\n"),
            before="a\n",
            after="b\n",
            notes=("n",),
        )
        defaults = dict(branch="sentinel/code-scan/abc", title="t", body="b",
                        patches=(patch,))
        return PullRequestDraft(**{**defaults, **overrides})

    def test_a_draft_needs_patches(self):
        with pytest.raises(PatchError, match="empty pull request"):
            self._draft(patches=())

    def test_an_unvalidated_patch_cannot_be_drafted(self):
        bad = PatchProposal(
            finding_ref="code://f.py#r@1",
            path="f.py",
            rule_id="r",
            diff="",
            before="a\n",
            after="a\n",
            notes=(),
            rejection="did not fix it",
        )
        with pytest.raises(PatchError, match="unvalidated patch"):
            self._draft(patches=(bad,))

    @pytest.mark.parametrize(
        "branch",
        ["../escape", "a b", "-leading-dash", "", "x" * 200, "br;rm -rf /", "a\nb"],
    )
    def test_unsafe_branch_names_are_refused(self, branch: str):
        # Branch names reach a git ref. A rule id or a path that carried a shell
        # metacharacter or a traversal into the name must not get that far.
        with pytest.raises(PatchError, match="unsafe branch name"):
            self._draft(branch=branch)

    def test_there_is_no_way_to_construct_a_non_draft(self):
        # PRD Section 5.4: never auto-merged. The guarantee is that the capability
        # does not exist, so this asserts the absence rather than a flag's value.
        assert not hasattr(self._draft(), "merge")
        assert "draft" not in PullRequestDraft.__dataclass_fields__

    def test_files_touched_deduplicates_and_keeps_order(self):
        assert self._draft().files_touched == ("f.py",)
