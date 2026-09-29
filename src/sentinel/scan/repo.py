"""Reading a repository as something a scanner can be pointed at (PRD F-07).

A :class:`RepoSnapshot` is an in-memory, immutable set of source files. In-memory
because the scanner is asked the same question repeatedly — *does this rule still
fire after my patch?* — and re-reading from disk to answer it would mean either
writing the candidate patch to the working tree (editing a customer's checkout to
test a hypothesis) or maintaining a shadow directory. Immutable because
:meth:`SourceFile.with_text` returning a new file is what makes
:func:`~sentinel.scan.analyzer.validate_patch` able to re-scan the patched text
without the original ever changing.

Traversal is deliberately narrow
--------------------------------
Only ``*.py``, only under the root, no symlinks followed, and a byte cap per file.
Each is a refusal rather than a convenience:

*   **No symlinks.** A repository is attacker-supplied input in the case F-07 is
    about — scanning a pull request from an outside contributor — and a symlink to
    ``/etc/shadow`` turns a scanner into an exfiltration tool the moment a finding
    quotes the line it matched.
*   **Root containment.** Every resolved path is checked to be inside the root, so
    a ``..`` component in an explicitly listed path cannot escape.
*   **A size cap.** A generated or minified file can be tens of megabytes on one
    line; parsing it is slow and every finding in it is unreviewable anyway.
*   **Skipped directories.** ``.git`` first of all: it holds every historical
    version of every file, so scanning it multiplies the finding count by the
    length of the history and reports each one against an unreadable path.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from sentinel.scan.findings import ScanError
from sentinel.scan.patch import normalise_source

__all__ = [
    "MAX_FILE_BYTES",
    "SKIP_DIRECTORIES",
    "RepoSnapshot",
    "SourceFile",
]

MAX_FILE_BYTES: Final[int] = 1_000_000
"""Files larger than this are skipped and reported, not parsed."""

SKIP_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        "dist",
        "build",
        ".tox",
        ".eggs",
    }
)


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One source file: a repo-relative path and its normalised text."""

    path: str
    text: str
    #: True when :func:`~sentinel.scan.patch.normalise_source` changed the bytes.
    #: Surfaced rather than hidden: a diff generated against normalised text does
    #: not apply cleanly to a CRLF working copy, and the PR body needs to say so.
    normalised: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ScanError("a source file needs a path")
        if self.path.startswith("/") or ".." in Path(self.path).parts:
            raise ScanError(f"{self.path!r} is not a repo-relative path")

    @property
    def line_count(self) -> int:
        return self.text.count("\n")

    def line(self, number: int) -> str:
        lines = self.text.splitlines()
        return lines[number - 1] if 1 <= number <= len(lines) else ""

    def with_text(self, text: str) -> SourceFile:
        return SourceFile(path=self.path, text=text, normalised=self.normalised)

    @classmethod
    def of(cls, path: str, text: str) -> SourceFile:
        normalised = normalise_source(text)
        return cls(path=path, text=normalised, normalised=normalised != text)


@dataclass(frozen=True, slots=True)
class RepoSnapshot:
    """An immutable view of the Python sources in one tree."""

    root: str
    files: tuple[SourceFile, ...]
    #: ``(path, reason)`` for files the walk deliberately did not read.
    skipped: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        seen = {file.path for file in self.files}
        if len(seen) != len(self.files):
            raise ScanError("duplicate path in snapshot; findings would be ambiguous")

    def __len__(self) -> int:
        return len(self.files)

    def file(self, path: str) -> SourceFile | None:
        for candidate in self.files:
            if candidate.path == path:
                return candidate
        return None

    @property
    def total_lines(self) -> int:
        return sum(file.line_count for file in self.files)

    def with_file(self, file: SourceFile) -> RepoSnapshot:
        """Replace one file, keeping order. The seam patch validation re-scans through."""
        replaced = False
        files: list[SourceFile] = []
        for existing in self.files:
            if existing.path == file.path:
                files.append(file)
                replaced = True
            else:
                files.append(existing)
        if not replaced:
            files.append(file)
        return RepoSnapshot(root=self.root, files=tuple(files), skipped=self.skipped)

    # --- construction --------------------------------------------------------- #

    @classmethod
    def of_texts(cls, texts: dict[str, str], *, root: str = "<memory>") -> RepoSnapshot:
        """A snapshot from literal sources. How every test builds one."""
        return cls(
            root=root,
            files=tuple(
                SourceFile.of(path, text) for path, text in sorted(texts.items())
            ),
        )

    @classmethod
    def from_dir(
        cls,
        directory: Path | str,
        *,
        patterns: Sequence[str] = ("*.py",),
        max_bytes: int = MAX_FILE_BYTES,
    ) -> RepoSnapshot:
        """Walk ``directory`` and read every matching file."""
        base = Path(directory)
        if not base.is_dir():
            raise ScanError(f"{base} is not a directory")
        resolved_base = base.resolve()

        files: list[SourceFile] = []
        skipped: list[tuple[str, str]] = []
        for candidate in _walk(base, patterns):
            relative = candidate.relative_to(base).as_posix()
            if candidate.is_symlink():
                skipped.append((relative, "symlink; not followed"))
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                skipped.append((relative, f"unreadable: {exc.strerror or exc}"))
                continue
            if not resolved.is_relative_to(resolved_base):
                skipped.append((relative, "resolves outside the repository root"))
                continue
            size = candidate.stat().st_size
            if size > max_bytes:
                skipped.append((relative, f"{size:,} bytes exceeds the {max_bytes:,} cap"))
                continue
            try:
                # Bytes, then an explicit decode. ``read_text`` applies universal
                # newline translation, which converts CRLF to LF *before*
                # ``normalise_source`` can notice it did — so a CRLF checkout was
                # reported as needing no normalisation, and the one caveat the pull
                # request body exists to state (that these diffs assume LF and will
                # not apply cleanly) could never fire.
                text = candidate.read_bytes().decode("utf-8")
            except UnicodeDecodeError as exc:
                skipped.append((relative, f"not valid UTF-8: {exc.reason}"))
                continue
            except OSError as exc:
                skipped.append((relative, f"unreadable: {exc.strerror or exc}"))
                continue
            files.append(SourceFile.of(relative, text))

        files.sort(key=lambda file: file.path)
        skipped.sort()
        return cls(root=str(base), files=tuple(files), skipped=tuple(skipped))


def _walk(base: Path, patterns: Sequence[str]) -> Iterable[Path]:
    """Yield matching files, pruning :data:`SKIP_DIRECTORIES` as it goes.

    Pruning during the walk rather than filtering afterwards: ``rglob`` on a tree
    containing ``node_modules`` or a virtualenv spends most of its time inside the
    directories whose results are then thrown away.
    """
    stack = [base]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIRECTORIES or entry.is_symlink():
                    continue
                stack.append(entry)
            elif any(entry.match(pattern) for pattern in patterns):
                yield entry
