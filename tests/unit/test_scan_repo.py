"""Reading a tree as a snapshot, including every refusal the walk makes.

The refusals matter more than the reads. In the case F-07 is actually about —
scanning a pull request from an outside contributor — the tree is attacker-supplied,
so a symlink the walk follows is an exfiltration path the moment a finding quotes the
line it matched.
"""

from __future__ import annotations

import pytest

from sentinel.scan.findings import ScanError
from sentinel.scan.repo import MAX_FILE_BYTES, SKIP_DIRECTORIES, RepoSnapshot, SourceFile


class TestSourceFile:
    def test_normalises_and_reports_that_it_did(self):
        file = SourceFile.of("a.py", "x = 1\r\ny = 2")
        assert file.text == "x = 1\ny = 2\n"
        assert file.normalised

    def test_already_normal_text_is_not_flagged(self):
        assert not SourceFile.of("a.py", "x = 1\n").normalised

    def test_line_count_counts_newlines(self):
        assert SourceFile.of("a.py", "a\nb\nc\n").line_count == 3

    def test_line_is_one_based_and_out_of_range_is_empty(self):
        file = SourceFile.of("a.py", "a\nb\n")
        assert file.line(1) == "a"
        assert file.line(2) == "b"
        assert file.line(99) == ""
        assert file.line(0) == ""

    def test_with_text_keeps_the_path_and_the_normalisation_flag(self):
        original = SourceFile.of("a.py", "x = 1\r\n")
        replaced = original.with_text("y = 2\n")
        assert replaced.path == "a.py"
        assert replaced.normalised is original.normalised

    @pytest.mark.parametrize("path", ["/abs/a.py", "../escape.py", "a/../../b.py"])
    def test_a_non_relative_path_is_refused(self, path: str):
        with pytest.raises(ScanError, match="repo-relative"):
            SourceFile(path=path, text="x = 1\n")

    def test_an_empty_path_is_refused(self):
        with pytest.raises(ScanError, match="needs a path"):
            SourceFile(path="", text="x\n")


class TestOfTexts:
    def test_files_are_sorted_by_path(self):
        snapshot = RepoSnapshot.of_texts({"z.py": "z\n", "a.py": "a\n"})
        assert [f.path for f in snapshot.files] == ["a.py", "z.py"]

    def test_file_lookup(self):
        snapshot = RepoSnapshot.of_texts({"a.py": "x = 1\n"})
        assert snapshot.file("a.py") is not None
        assert snapshot.file("missing.py") is None

    def test_total_lines_sums(self):
        snapshot = RepoSnapshot.of_texts({"a.py": "1\n2\n", "b.py": "3\n"})
        assert snapshot.total_lines == 3

    def test_duplicate_paths_are_refused(self):
        file = SourceFile.of("a.py", "x\n")
        with pytest.raises(ScanError, match="duplicate path"):
            RepoSnapshot(root="r", files=(file, file))

    def test_with_file_replaces_in_place_and_keeps_order(self):
        snapshot = RepoSnapshot.of_texts({"a.py": "1\n", "b.py": "2\n"})
        updated = snapshot.with_file(SourceFile.of("a.py", "NEW\n"))
        assert [f.path for f in updated.files] == ["a.py", "b.py"]
        assert updated.file("a.py").text == "NEW\n"

    def test_with_file_appends_an_unknown_path(self):
        snapshot = RepoSnapshot.of_texts({"a.py": "1\n"})
        assert len(snapshot.with_file(SourceFile.of("c.py", "3\n"))) == 2

    def test_the_original_is_untouched_by_with_file(self):
        # Immutability is what lets patch validation re-scan without the original
        # ever changing under the finding it is validating.
        snapshot = RepoSnapshot.of_texts({"a.py": "1\n"})
        snapshot.with_file(SourceFile.of("a.py", "NEW\n"))
        assert snapshot.file("a.py").text == "1\n"


class TestFromDir:
    def test_reads_python_files_recursively(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "a.py").write_text("a = 1\n")
        (tmp_path / "pkg" / "b.py").write_text("b = 2\n")
        (tmp_path / "notes.txt").write_text("ignored\n")
        snapshot = RepoSnapshot.from_dir(tmp_path)
        assert [f.path for f in snapshot.files] == ["a.py", "pkg/b.py"]

    def test_skip_directories_are_pruned(self, tmp_path):
        for name in sorted(SKIP_DIRECTORIES)[:4]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "x.py").write_text("x = 1\n")
        (tmp_path / "real.py").write_text("real = 1\n")
        assert [f.path for f in RepoSnapshot.from_dir(tmp_path).files] == ["real.py"]

    def test_a_git_directory_is_never_scanned(self, tmp_path):
        # .git holds every historical version of every file, so scanning it
        # multiplies the finding count by the length of the history.
        (tmp_path / ".git" / "objects").mkdir(parents=True)
        (tmp_path / ".git" / "objects" / "old.py").write_text("import os\n")
        (tmp_path / "a.py").write_text("a = 1\n")
        assert [f.path for f in RepoSnapshot.from_dir(tmp_path).files] == ["a.py"]

    def test_a_symlinked_file_is_skipped_and_reported(self, tmp_path):
        secret = tmp_path.parent / "outside.py"
        secret.write_text("SECRET = 'do not read'\n")
        (tmp_path / "link.py").symlink_to(secret)
        (tmp_path / "a.py").write_text("a = 1\n")
        snapshot = RepoSnapshot.from_dir(tmp_path)
        assert [f.path for f in snapshot.files] == ["a.py"]
        assert any(path == "link.py" for path, _ in snapshot.skipped)
        assert "do not read" not in "".join(f.text for f in snapshot.files)

    def test_a_symlinked_directory_is_not_followed(self, tmp_path):
        outside = tmp_path.parent / "outside_dir"
        outside.mkdir(exist_ok=True)
        (outside / "hidden.py").write_text("HIDDEN = 1\n")
        (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
        (tmp_path / "a.py").write_text("a = 1\n")
        assert [f.path for f in RepoSnapshot.from_dir(tmp_path).files] == ["a.py"]

    def test_an_oversized_file_is_skipped_and_reported(self, tmp_path):
        (tmp_path / "big.py").write_text("x = 1\n" * 10)
        (tmp_path / "small.py").write_text("a = 1\n")
        snapshot = RepoSnapshot.from_dir(tmp_path, max_bytes=10)
        # The small file is still read: the cap skips the offending file, not the scan.
        assert [f.path for f in snapshot.files] == ["small.py"]
        reasons = dict(snapshot.skipped)
        assert "exceeds" in reasons["big.py"]

    def test_non_utf8_is_skipped_and_reported(self, tmp_path):
        (tmp_path / "bin.py").write_bytes(b"\xff\xfe\x00invalid")
        (tmp_path / "a.py").write_text("a = 1\n")
        snapshot = RepoSnapshot.from_dir(tmp_path)
        assert [f.path for f in snapshot.files] == ["a.py"]
        assert any("UTF-8" in reason for _, reason in snapshot.skipped)

    def test_skipped_entries_are_sorted_for_reproducibility(self, tmp_path):
        for name in ("z", "a", "m"):
            (tmp_path / f"{name}.py").write_bytes(b"\xff\xfe")
        skipped = RepoSnapshot.from_dir(tmp_path).skipped
        assert [path for path, _ in skipped] == sorted(path for path, _ in skipped)

    def test_a_missing_directory_is_refused(self, tmp_path):
        with pytest.raises(ScanError, match="not a directory"):
            RepoSnapshot.from_dir(tmp_path / "nope")

    def test_a_file_path_is_refused(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("a = 1\n")
        with pytest.raises(ScanError, match="not a directory"):
            RepoSnapshot.from_dir(target)

    def test_the_default_cap_is_documented_and_generous(self):
        assert MAX_FILE_BYTES >= 100_000

    def test_line_endings_are_normalised_on_read(self, tmp_path):
        (tmp_path / "crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
        file = RepoSnapshot.from_dir(tmp_path).file("crlf.py")
        assert file is not None
        assert "\r" not in file.text
        assert file.normalised
