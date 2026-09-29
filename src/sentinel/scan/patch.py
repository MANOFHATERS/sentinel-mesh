"""Turning a finding into a reviewable patch (PRD F-07).

F-07 asks for *"a syntactically valid patch PR"* per seeded vulnerability. Taken
literally that is a low bar — ``ast.parse`` on the result clears it — and clearing
it literally would produce exactly the tool developers already ignore. So a patch
here is only accepted once four things hold, checked in
:func:`validate_patch`:

1.  **The patched file parses.** F-07's literal requirement.
2.  **The diff reconstructs the patched file.** :func:`apply_unified_diff` replays
    the diff against the original and the result must equal the text the edits
    produced. This is not redundant: the diff is the artifact that reaches the pull
    request, and a diff generated from one string while a different string was
    validated is a patch that passes review and breaks the build. The applier
    verifies every context and removal line, so a stale or mis-offset hunk fails
    here rather than in CI.
3.  **The finding is gone.** The scanner re-runs on the patched source and the
    rule must no longer fire at that location. A "fix" that does not remove the
    finding is a formatting change with a security label on it.
4.  **Nothing new appeared.** The re-scan must not introduce a finding the original
    did not have. Mechanical rewrites can do this — replacing ``shell=True`` with a
    ``shlex.split`` call adds an import and a call, and both are things rules look
    at.

Why edits are character offsets
-------------------------------
Every fix is expressed as a set of ``(start, end, replacement)`` splices over the
original text, applied right-to-left. The alternative — unparsing a modified AST
with :func:`ast.unparse` — reformats the entire file: comments vanish, string
quoting normalises, and line breaks move. The resulting diff touches every line,
which makes it unreviewable, and a reviewer who cannot see the change is a reviewer
who approves it. Splicing keeps the diff to the lines that actually changed.

Why the patch is a draft, always
--------------------------------
PRD Section 5.4: *"Patches are opened as draft PRs for human merge, never
auto-merged."* :class:`PullRequestDraft` has no non-draft state to construct, and
``ActionType.OPEN_PATCH_PR`` is classified destructive in
:mod:`sentinel.core.schemas`, so the Human Approval Gate stands in front of it.
The patches this module writes are mechanical rewrites of one expression; they are
right often enough to be worth reviewing and not often enough to be worth trusting,
and the design says so structurally rather than in a comment.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Final

from sentinel.core.errors import SentinelError

__all__ = [
    "CONTEXT_LINES",
    "PatchError",
    "PatchProposal",
    "PullRequestDraft",
    "SourceEdit",
    "apply_edits",
    "apply_unified_diff",
    "ensure_import_edit",
    "normalise_source",
    "offset_of",
    "unified_diff",
]

CONTEXT_LINES: Final[int] = 3
"""Context lines per hunk. Three is what git defaults to and what reviewers read."""


class PatchError(SentinelError):
    """A patch could not be built, applied, or verified."""


def normalise_source(text: str) -> str:
    """Normalise line endings to ``\\n`` and guarantee a trailing newline.

    Both halves earn their place. Mixed line endings make a diff's context lines
    compare unequal for reasons invisible on screen, and a file with no final
    newline produces a hunk that needs the ``\\ No newline at end of file``
    convention to round-trip — a convention worth not having to implement twice
    (once in the generator, once in the applier) when normalising at the boundary
    removes the case entirely. The normalisation is reported by
    :class:`~sentinel.scan.repo.SourceFile`, so a file that was changed on load is
    not silently presented as untouched.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    if unified and not unified.endswith("\n"):
        unified += "\n"
    return unified


def offset_of(source: str, line: int, col: int) -> int:
    """Character offset of 1-based ``line`` / 0-based ``col`` (``ast`` coordinates).

    ``col`` is a *byte* offset into the UTF-8 encoding of the line in CPython's
    ``ast``, not a character offset, and the two differ on any line containing a
    non-ASCII character. Ignoring that puts a patch's splice boundary in the middle
    of a multi-byte character for any file with an accented identifier or a unicode
    string literal, so the conversion is done here rather than assumed away.
    """
    if line < 1:
        raise PatchError(f"lines are 1-based, got {line}")
    lines = source.splitlines(keepends=True)
    if line > len(lines):
        if line == len(lines) + 1 and col == 0:
            return len(source)
        raise PatchError(f"line {line} is past the end of a {len(lines)}-line file")
    base = sum(len(item) for item in lines[: line - 1])
    target = lines[line - 1]
    encoded = target.encode("utf-8")
    if col > len(encoded):
        raise PatchError(f"column {col} is past the end of line {line}")
    return base + len(encoded[:col].decode("utf-8", errors="strict"))


@dataclass(frozen=True, slots=True)
class SourceEdit:
    """One splice: replace ``source[start:end]`` with ``replacement``.

    ``note`` is what the pull-request body says this edit did, in one clause. It is
    required rather than optional because an unexplained hunk in a security patch is
    the thing a reviewer stalls on, and the author of the rule is the only party who
    knows the reason.
    """

    start: int
    end: int
    replacement: str
    note: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise PatchError(f"invalid edit range [{self.start}, {self.end})")
        if not self.note.strip():
            raise PatchError("every edit must carry a note explaining it")

    @property
    def is_insertion(self) -> bool:
        return self.start == self.end


def apply_edits(source: str, edits: tuple[SourceEdit, ...]) -> str:
    """Apply ``edits`` to ``source``. Overlapping edits are refused.

    Refused rather than resolved by precedence: two rules editing the same
    expression have made independent decisions about it, and combining them
    silently produces a third rewrite that neither rule proposed and no test covers.
    :class:`~sentinel.agents.codescan.CodeScanAgent` groups edits per finding and
    proposes one patch per finding, so a conflict here means two fixes were merged
    that should have stayed separate pull requests.
    """
    if not edits:
        return source
    ordered = sorted(edits, key=lambda edit: (edit.start, edit.end))
    for previous, current in pairwise(ordered):
        if current.start < previous.end:
            raise PatchError(
                f"edits overlap: [{previous.start}, {previous.end}) and "
                f"[{current.start}, {current.end}). Propose them as separate patches."
            )
    if ordered[-1].end > len(source):
        raise PatchError(
            f"edit ends at {ordered[-1].end} but the source is {len(source)} characters"
        )
    out = source
    for edit in reversed(ordered):
        out = out[: edit.start] + edit.replacement + out[edit.end :]
    return out


def ensure_import_edit(source: str, tree: ast.Module, module: str) -> SourceEdit | None:
    """An edit adding ``import <module>`` if the module is not already imported.

    Inserted after the last *top-level* import, which is where a reviewer expects to
    find it and where it cannot land inside a conditional or a function body. A
    module imported only inside a function does not count as imported: the fix's
    call site is elsewhere, and reusing a function-local binding would produce a
    ``NameError`` that only fires on the vulnerable path.
    """
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None and (
                    alias.name == module or alias.name.startswith(f"{module}.")
                ):
                    return None
    # Walk the prologue — the module docstring and the run of top-level imports —
    # and insert after it. The loop stops at the first statement that is neither,
    # rather than scanning the whole body, because an import appearing after real
    # code is not part of the prologue and inserting after *it* would put the new
    # import below a use of the module in between.
    insert_line = 1
    for index, node in enumerate(tree.body):
        is_prologue = isinstance(node, ast.Import | ast.ImportFrom) or (
            index == 0 and _is_docstring(node)
        )
        if not is_prologue:
            break
        insert_line = (node.end_lineno or node.lineno) + 1
    offset = offset_of(source, min(insert_line, len(source.splitlines()) + 1), 0)
    return SourceEdit(
        start=offset,
        end=offset,
        replacement=f"import {module}\n",
        note=f"import {module}, required by the fix",
    )


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


# --------------------------------------------------------------------------- #
# Unified diff
# --------------------------------------------------------------------------- #


def unified_diff(*, path: str, before: str, after: str) -> str:
    """A git-style unified diff. Empty string when nothing changed."""
    if before == after:
        return ""
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=CONTEXT_LINES,
        )
    )
    return "".join(lines)


_HUNK_HEADER: Final[re.Pattern[str]] = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def apply_unified_diff(before: str, diff: str) -> str:
    """Replay ``diff`` against ``before``, verifying every context line.

    This is the check that makes "a syntactically valid patch" a measurement rather
    than a claim. It is written as an independent replay rather than reusing the
    edits: if the diff and the edits shared an implementation, agreement between
    them would prove nothing. A context or removal line that does not match raises
    :class:`PatchError` naming the line, which is exactly the failure a stale patch
    produces when a human applies it.
    """
    if not diff.strip():
        return before

    source_lines = before.splitlines(keepends=True)
    out: list[str] = []
    cursor = 0
    lines = diff.splitlines(keepends=True)
    index = 0
    seen_hunk = False

    while index < len(lines):
        raw = lines[index]
        if raw.startswith(("--- ", "+++ ", "diff ", "index ")):
            index += 1
            continue
        match = _HUNK_HEADER.match(raw)
        if match is None:
            if not seen_hunk:
                raise PatchError(f"unexpected line before the first hunk: {raw!r}")
            raise PatchError(f"unexpected line outside a hunk: {raw!r}")
        seen_hunk = True
        index += 1

        old_start = int(match.group("old_start"))
        old_count = 1 if match.group("old_count") is None else int(match.group("old_count"))
        # A zero-length old side means "insert after line old_start", so the target
        # index is old_start rather than old_start - 1. Getting this backwards
        # off-by-ones every pure insertion, which is what an added import is.
        target = old_start if old_count == 0 else old_start - 1
        if target < cursor:
            raise PatchError(
                f"hunk at -{old_start} moves backwards past line {cursor + 1}; "
                "hunks must be in ascending order"
            )
        if target > len(source_lines):
            raise PatchError(
                f"hunk starts at line {old_start} but the file has "
                f"{len(source_lines)} lines"
            )
        out.extend(source_lines[cursor:target])
        cursor = target

        # The hunk ends when the old side is fully consumed *and* the next line is
        # not another addition. Both halves are needed: a hunk whose additions come
        # after its last context line (an appended import) still has lines to read
        # once ``consumed == old_count``, and a pure-insertion hunk
        # (``old_count == 0``) never consumes anything at all.
        consumed = 0
        while index < len(lines):
            body = lines[index]
            if _HUNK_HEADER.match(body):
                break
            if consumed >= old_count and not body.startswith("+"):
                break
            marker, content = body[:1], body[1:]
            if marker == "+":
                out.append(content)
            elif marker == "-":
                if cursor >= len(source_lines):
                    raise PatchError(
                        f"diff removes a line past the end of the file: {content!r}"
                    )
                if source_lines[cursor] != content:
                    raise PatchError(
                        f"diff does not apply at line {cursor + 1}: expected "
                        f"{content!r}, found {source_lines[cursor]!r}"
                    )
                cursor += 1
                consumed += 1
            elif marker == " " or body in {"\n", ""}:
                # A bare "\n" in a diff body is a context line for an empty source
                # line; difflib emits it without the leading space.
                expected = content if marker == " " else "\n"
                if cursor >= len(source_lines):
                    raise PatchError("diff context runs past the end of the file")
                if source_lines[cursor] != expected:
                    raise PatchError(
                        f"context mismatch at line {cursor + 1}: expected "
                        f"{expected!r}, found {source_lines[cursor]!r}"
                    )
                out.append(source_lines[cursor])
                cursor += 1
                consumed += 1
            elif marker == "\\":
                pass  # "\ No newline at end of file"; normalise_source removes the case
            else:
                raise PatchError(f"unrecognised diff line: {body!r}")
            index += 1

    out.extend(source_lines[cursor:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PatchProposal:
    """A validated patch for one finding, ready to appear in a draft PR."""

    finding_ref: str
    path: str
    rule_id: str
    diff: str
    before: str
    after: str
    notes: tuple[str, ...]
    #: Every check in :func:`validate_patch` that passed.
    checks: tuple[str, ...] = ()
    #: Why a check failed, when one did. An invalid proposal is kept rather than
    #: discarded so :mod:`sentinel.agents.codescan` can report *that the fix was
    #: attempted and rejected*, which is information a reviewer wants and a silent
    #: drop destroys.
    rejection: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.rejection is None and bool(self.diff)

    @property
    def changed_lines(self) -> int:
        return sum(
            1
            for line in self.diff.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )

    def describe(self) -> str:
        status = "valid" if self.is_valid else f"rejected: {self.rejection}"
        return (
            f"patch for {self.rule_id} in {self.path} ({self.changed_lines} lines, "
            f"{status})"
        )


@dataclass(frozen=True, slots=True)
class PullRequestDraft:
    """A draft pull request. There is no non-draft variant, by construction.

    PRD Section 5.4 requires patches to be *"opened as draft PRs for human merge,
    never auto-merged"*. Modelling that as a ``draft: bool`` field would make
    ``draft=False`` one keyword argument away; leaving the state out of the type
    means the only way to merge is a human in a browser, which is the requirement.
    """

    branch: str
    title: str
    body: str
    patches: tuple[PatchProposal, ...]
    #: The findings that had no mechanical fix, named so the PR says what it leaves.
    unpatched_refs: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.patches:
            raise PatchError("a draft PR with no patches is an empty pull request")
        if any(not patch.is_valid for patch in self.patches):
            raise PatchError(
                "refusing to draft a PR containing an unvalidated patch; the "
                "rejected proposal should be reported as a finding instead"
            )
        if not _BRANCH_NAME.match(self.branch):
            raise PatchError(f"unsafe branch name {self.branch!r}")

    @property
    def combined_diff(self) -> str:
        return "".join(patch.diff for patch in self.patches)

    @property
    def files_touched(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(patch.path for patch in self.patches))


#: Branch names are interpolated into git refs. Restricting the alphabet here means
#: a rule id or a file path that reached the name cannot carry a shell metacharacter
#: or a ``../`` into the connector layer.
_BRANCH_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,98}$")
