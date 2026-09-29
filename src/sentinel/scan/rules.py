"""The vulnerability rules, and the mechanical fix for each (PRD F-07).

Why this is not Semgrep
-----------------------
PRD Section 5.6 names Semgrep, and this is a deliberate substitution in the same
spirit as :mod:`sentinel.ml.nn` standing in for PyTorch and
:mod:`sentinel.agents.runtime` standing in for LangGraph. Three reasons, and the
third is the one that decides it:

*   **Hermetic.** Semgrep pulls a large dependency tree and, for its useful rule
    packs, a registry fetch. Part 1's property — the whole repository installs,
    tests and evaluates in one command with no network — is worth more than a rule
    count.
*   **The fix is the deliverable, not the finding.** F-07 is graded on *"a
    syntactically valid patch PR"*, and Semgrep's ``autofix`` is a textual pattern
    rewrite. Producing the patches this module produces — a parameterised SQL query
    built from an f-string, an import inserted in the right place — needs the AST
    positions and the taint result, which means owning the analysis anyway.
*   **Taint is the difference between a finding and noise.** The rules here fire at
    ``HIGH`` confidence only when :mod:`sentinel.scan.taint` traced attacker data
    into the sink. Semgrep's taint mode can do this; expressing it as a pattern
    language, for one language, over rules that then need the same traces to build
    a patch, would be a layer of indirection over the analysis rather than a
    replacement for it.

:class:`~sentinel.scan.analyzer.StaticAnalyzer` is a Protocol, so a Semgrep-backed
analyzer remains a drop-in — and ``test_scan_analyzer.py`` drives the agent through
a stub analyzer to prove the seam is real rather than decorative.

Every rule earns its place three times
--------------------------------------
Each entry states a CWE (the identifier a vulnerability-management process already
speaks), a *reason the pattern is exploitable* rather than merely unusual, and
either a mechanical fix or an explicit admission that there is not one. That last
part matters: :data:`RULES` contains rules with ``fix=None``, and they are reported
without a patch. A scanner that invents a rewrite for
``pickle.loads(attacker_bytes)`` — where the only real fix is to stop using pickle
— produces a patch that passes review and fixes nothing.

Confidence, and the helper-function case
----------------------------------------
``HIGH`` needs a traced taint path. ``MEDIUM`` is a dangerous sink reached by a
value the analyzer cannot prove is attacker-controlled but also cannot prove is
not — the ``def run(cmd): os.system(cmd)`` case, where the source is in a caller
the intraprocedural analysis does not follow. ``LOW`` is a dangerous construct on a
literal, which is usually a deploy script and occasionally a real bug. The Code-Scan
Agent drafts patches from ``MEDIUM`` up and reports ``LOW`` without one.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from sentinel.core.schemas import Severity
from sentinel.scan.findings import FindingConfidence, ScanError, SourceSpan
from sentinel.scan.patch import SourceEdit, ensure_import_edit, offset_of
from sentinel.scan.symbols import ImportTable, keyword_of, positional_or_keyword
from sentinel.scan.taint import TaintFact

__all__ = [
    "RULES",
    "RULES_BY_ID",
    "Match",
    "Rule",
    "RuleContext",
    "rule_ids",
]


@dataclass(slots=True)
class RuleContext:
    """Everything a rule may read about the file it is looking at."""

    path: str
    source: str
    tree: ast.Module
    imports: ImportTable
    taint: dict[int, TaintFact]

    # --- taint ------------------------------------------------------------- #

    def taint_for(self, node: ast.AST | None, rule_id: str) -> TaintFact | None:
        """The live taint on ``node`` for ``rule_id``, or ``None``.

        "Live for this rule" rather than "tainted": ``os.path.basename`` clears a
        path traversal and does nothing for a command injection, so the question
        only has an answer once the rule asking is known. See
        :mod:`sentinel.scan.taint`.
        """
        if node is None:
            return None
        fact = self.taint.get(id(node))
        if fact is None or not fact.is_live_for(rule_id):
            return None
        return fact

    def sanitised_for(self, node: ast.AST | None, rule_id: str) -> bool:
        """True when ``node`` carries taint that a sanitizer already neutralised.

        This is a *stronger* statement than "not tainted", and the difference is
        what stops the scanner reporting correct code. "Not tainted" covers the
        value whose source the intraprocedural analysis could not see — a helper's
        parameter — where a report is warranted. "Sanitised" means the analyzer
        watched ``shlex.quote`` run on this exact value, which is positive evidence
        the developer handled it. Continuing to report that is the behaviour that
        gets a scanner uninstalled, so the rules suppress on it.
        """
        if node is None:
            return False
        fact = self.taint.get(id(node))
        return fact is not None and not fact.is_live_for(rule_id)

    # --- source ------------------------------------------------------------ #

    def span_of(self, node: ast.AST) -> SourceSpan:
        line = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        if line is None or col is None:
            raise ScanError(f"{type(node).__name__} carries no source position")
        return SourceSpan(
            line=line,
            col=col,
            end_line=getattr(node, "end_lineno", None) or line,
            end_col=getattr(node, "end_col_offset", None) or col,
        )

    def offsets(self, node: ast.AST) -> tuple[int, int]:
        span = self.span_of(node)
        return (
            offset_of(self.source, span.line, span.col),
            offset_of(self.source, span.end_line, span.end_col),
        )

    def text(self, node: ast.AST) -> str:
        start, end = self.offsets(node)
        return self.source[start:end]

    def line_text(self, lineno: int) -> str:
        lines = self.source.splitlines()
        return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""

    def replace(self, node: ast.AST, replacement: str, note: str) -> SourceEdit:
        start, end = self.offsets(node)
        return SourceEdit(start=start, end=end, replacement=replacement, note=note)

    def require_import(self, module: str) -> SourceEdit | None:
        return ensure_import_edit(self.source, self.tree, module)


@dataclass(frozen=True, slots=True)
class Match:
    """A rule fired. ``node`` is what the finding points at."""

    node: ast.AST
    confidence: FindingConfidence
    detail: str = ""
    taint: TaintFact | None = None
    #: The node a fix should rewrite, when it is not ``node`` itself.
    fix_target: ast.AST | None = None

    @property
    def taint_path(self) -> tuple[str, ...]:
        return () if self.taint is None else self.taint.path


Matcher = Callable[[ast.AST, RuleContext], "Match | None"]
Fixer = Callable[[Match, RuleContext], "tuple[SourceEdit, ...] | None"]


@dataclass(frozen=True, slots=True)
class Rule:
    """One vulnerability pattern, its rationale, and its fix."""

    rule_id: str
    cwe: str
    title: str
    severity: Severity
    message: str
    remediation: str
    matcher: Matcher = field(repr=False)
    #: ``None`` when no mechanical rewrite is safe. See the module docstring.
    fix: Fixer | None = field(default=None, repr=False)
    #: AST node types this rule inspects. Dispatch is keyed on these so a scan is
    #: one walk of the tree rather than one walk per rule.
    node_types: tuple[type[ast.AST], ...] = (ast.Call,)
    cve_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.rule_id.startswith("python."):
            raise ScanError(f"rule id {self.rule_id!r} must be namespaced by language")
        if not re.fullmatch(r"CWE-\d{1,4}", self.cwe):
            raise ScanError(f"{self.rule_id}: {self.cwe!r} is not a CWE identifier")
        if not self.node_types:
            raise ScanError(f"{self.rule_id}: a rule must inspect at least one node type")

    @property
    def has_fix(self) -> bool:
        return self.fix is not None


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _confidence_for_sink(
    argument: ast.expr | None, ctx: RuleContext, rule_id: str
) -> tuple[FindingConfidence | None, TaintFact | None]:
    """Grade a dangerous sink by what it is being handed.

    Four outcomes, and the two middle ones are what make the scanner usable on real
    code:

    ``HIGH``
        A traced path from an attacker-controlled source into this argument.
    ``None`` — suppress entirely
        The analyzer watched a sanitizer run on this value *for this rule*. That is
        evidence of a fix, not absence of evidence, so reporting it would be telling
        a developer their ``shlex.quote`` call is a command injection.
    ``MEDIUM``
        The value came from somewhere this intraprocedural analysis does not follow
        — which describes every utility function taking a command as a parameter.
        Dropping these would make the scanner blind to the most common way the bug
        is actually written.
    ``LOW``
        A literal. ``os.system("systemctl reload nginx")`` is a deploy script, and
        the construct is still worth listing, so it is reported without a patch.
    """
    fact = ctx.taint_for(argument, rule_id)
    if fact is not None:
        return FindingConfidence.HIGH, fact
    if ctx.sanitised_for(argument, rule_id):
        return None, None
    if argument is None:
        return FindingConfidence.LOW, None
    if _is_static_string(argument):
        return FindingConfidence.LOW, None
    return FindingConfidence.MEDIUM, None


def _is_static_string(node: ast.expr) -> bool:
    """True when ``node`` is a string built only from literals."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str | bytes)
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(value, ast.Constant) for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Mod):
        return _is_static_string(node.left) and _is_static_string(node.right)
    if isinstance(node, ast.Tuple | ast.List):
        return all(_is_static_string(element) for element in node.elts)
    return False


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_false(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _string_literal(text: str) -> str:
    """A valid Python string literal for ``text``, preferring double quotes.

    ``repr`` would do, but it prefers single quotes, so every generated query in a
    double-quoted codebase would flip quoting style and the diff would look like a
    reformat. A patch that looks like a reformat gets skimmed.
    """
    if '"' not in text and "\\" not in text and "\n" not in text:
        return f'"{text}"'
    return repr(text)


def _env_var_name(identifier: str) -> str:
    """A conventional environment-variable name for a Python identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", identifier).strip("_").upper()
    return cleaned or "SECRET"


# --------------------------------------------------------------------------- #
# CWE-78: OS command injection via shell=True
# --------------------------------------------------------------------------- #

_SHELL_CALLS: Final[frozenset[str]] = frozenset(
    {
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "asyncio.create_subprocess_shell",
    }
)


def _match_shell_true(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if ctx.imports.resolve(node.func) not in _SHELL_CALLS:
        return None
    shell = keyword_of(node, "shell")
    if shell is None or not _is_true(shell.value):
        return None
    command = positional_or_keyword(node, index=0, name="args")
    confidence, fact = _confidence_for_sink(command, ctx, "python.os-command-injection")
    if confidence is None:
        return None
    return Match(
        node=node,
        confidence=confidence,
        detail=(
            "the command is assembled from attacker-controlled data"
            if fact is not None
            else "the command is not a literal, so its contents are not verifiable here"
        ),
        taint=fact,
        fix_target=shell,
    )


def _fix_shell_true(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    shell = match.fix_target
    if not isinstance(call, ast.Call) or not isinstance(shell, ast.keyword):
        return None
    command = positional_or_keyword(call, index=0, name="args")
    if command is None:
        return None
    edits: list[SourceEdit] = [
        ctx.replace(
            shell,
            "shell=False",
            "shell=False so the command is exec'd directly instead of by /bin/sh, "
            "which is what makes metacharacters in the arguments inert",
        )
    ]
    if isinstance(command, ast.List | ast.Tuple):
        # Already an argv list; dropping the shell is the entire fix.
        return tuple(edits)
    # A string command with shell=False would be treated as a single program name,
    # so it has to be split. shlex.split applies POSIX word rules, which is the
    # closest mechanical equivalent to what the shell was doing.
    edits.append(
        ctx.replace(
            command,
            f"shlex.split({ctx.text(command)})",
            "split the command string into an argv list; shlex.split applies the "
            "same word rules the shell did, without the shell's metacharacters",
        )
    )
    shlex_import = ctx.require_import("shlex")
    if shlex_import is not None:
        edits.append(shlex_import)
    return tuple(edits)


# --------------------------------------------------------------------------- #
# CWE-78: os.system / os.popen
# --------------------------------------------------------------------------- #

_SYSTEM_CALLS: Final[frozenset[str]] = frozenset(
    {"os.system", "os.popen", "commands.getoutput", "commands.getstatusoutput"}
)


def _match_os_system(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if ctx.imports.resolve(node.func) not in _SYSTEM_CALLS:
        return None
    command = positional_or_keyword(node, index=0, name="command")
    confidence, fact = _confidence_for_sink(command, ctx, "python.os-system-injection")
    if confidence is None:
        return None
    return Match(
        node=node,
        confidence=confidence,
        detail=(
            "the command string is attacker-influenced"
            if fact is not None
            else "the command string is built at runtime"
        ),
        taint=fact,
        fix_target=command,
    )


def _fix_os_system(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    command = match.fix_target
    if not isinstance(call, ast.Call) or not isinstance(command, ast.expr):
        return None
    resolved = ctx.imports.resolve(call.func)
    if resolved != "os.system":
        # os.popen returns a file object and the callers read from it; rewriting it
        # to subprocess.run changes the return type, so the call sites would break.
        # A patch that does not compile at the call site is worse than a finding.
        return None
    edits = [
        ctx.replace(
            call,
            f"subprocess.run(shlex.split({ctx.text(command)}), check=False).returncode",
            "replace os.system with subprocess.run over an argv list, so no shell "
            "interprets the string; .returncode preserves os.system's int result "
            "shape (though not its exit-status encoding — see the PR body)",
        )
    ]
    for module in ("shlex", "subprocess"):
        required = ctx.require_import(module)
        if required is not None:
            edits.append(required)
    return tuple(edits)


# --------------------------------------------------------------------------- #
# CWE-89: SQL injection
# --------------------------------------------------------------------------- #

_EXECUTE_METHODS: Final[frozenset[str]] = frozenset(
    {"execute", "executemany", "executescript"}
)

#: ``%``-style conversions a query template uses. Matched so they can be turned into
#: placeholders; anything more exotic means the fix declines rather than guesses.
_PERCENT_CONVERSION: Final[re.Pattern[str]] = re.compile(r"%[sdrfi]")


def _match_sql_injection(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in _EXECUTE_METHODS:
        return None
    query = positional_or_keyword(node, index=0, name="sql")
    if query is None or _is_static_string(query):
        return None
    if not _is_dynamic_string(query):
        # A variable holding a query is common and legitimate (a module constant, a
        # query built by an ORM). Only a *string built here* is evidence of the bug,
        # unless the taint analysis traced attacker data into the variable.
        fact = ctx.taint_for(query, "python.sql-injection")
        if fact is None:
            return None
        return Match(
            node=node,
            confidence=FindingConfidence.HIGH,
            detail="the query variable carries attacker-controlled data",
            taint=fact,
            fix_target=query,
        )
    confidence, fact = _confidence_for_sink(query, ctx, "python.sql-injection")
    if confidence is None or confidence is FindingConfidence.LOW:
        return None
    return Match(
        node=node,
        confidence=confidence,
        detail=(
            "attacker-controlled data is concatenated into the statement"
            if fact is not None
            else "the statement is assembled by string formatting, so any value "
            "reaching it becomes SQL"
        ),
        taint=fact,
        fix_target=query,
    )


def _is_dynamic_string(node: ast.expr) -> bool:
    """True when ``node`` is a string *built* by formatting or concatenation."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(value, ast.FormattedValue) for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Mod):
        return not _is_static_string(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in {"format", "join"}
    return False


def _fix_sql_injection(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    query = match.fix_target
    if not isinstance(call, ast.Call) or not isinstance(query, ast.expr):
        return None
    if len(call.args) > 1 or keyword_of(call, "parameters") is not None:
        # A second argument already exists, so the call is partly parameterised and
        # merging a generated tuple into it would silently change the binding order.
        return None
    parameterised = _parameterise(query, ctx)
    if parameterised is None:
        return None
    raw_template, params = parameterised
    if not params:
        return None
    # Done here, on the fully assembled template, rather than inside each branch of
    # _parameterise. A concatenation assembles its template from sub-results and the
    # quotes around a placeholder can straddle the join:
    # ``"... a = '" + n + "'"`` puts the opening quote in the left fragment and the
    # closing quote in the right, so neither sub-result contains ``'?'`` to act on and
    # the patch emitted ``a = '?'`` — a query that runs, matches the literal string
    # "?", returns nothing, and looks fixed.
    template = _bind_quoted_placeholders(raw_template)
    if template is None:
        return None
    joined = ", ".join(params)
    tuple_text = f"({joined},)" if len(params) == 1 else f"({joined})"
    return (
        ctx.replace(
            query,
            f"{_string_literal(template)}, {tuple_text}",
            "bind the values as query parameters instead of interpolating them; the "
            "driver then sends them out-of-band and they can never be parsed as SQL",
        ),
    )


def _parameterise(node: ast.expr, ctx: RuleContext) -> tuple[str, list[str]] | None:
    """Rewrite a built query string as a ``?``-placeholder template plus params.

    Returns ``None`` when the shape is not one this can rewrite correctly. Declining
    is the important half: a half-understood query turned into a template with the
    placeholders in the wrong order is a data-corruption bug wearing a security fix.
    """
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        params: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                if value.conversion not in (-1, 115) or value.format_spec is not None:
                    # ``{x!r}`` and ``{x:>10}`` change the rendered text, so the
                    # placeholder is not an equivalent substitution.
                    return None
                parts.append("?")
                params.append(ctx.text(value.value))
            else:
                return None
        return "".join(parts), params

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _parameterise(node.left, ctx)
        right = _parameterise(node.right, ctx)
        if left is None or right is None:
            return None
        return left[0] + right[0], [*left[1], *right[1]]

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, []

    if isinstance(node, ast.Name | ast.Attribute | ast.Subscript | ast.Call):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "format":
                return _parameterise_format(node, ctx)
            return None
        return "?", [ctx.text(node)]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if not isinstance(node.left, ast.Constant) or not isinstance(node.left.value, str):
            return None
        template = node.left.value
        placeholders = _PERCENT_CONVERSION.findall(template)
        if not placeholders or "%%" in template:
            return None
        arguments = (
            list(node.right.elts)
            if isinstance(node.right, ast.Tuple)
            else [node.right]
        )
        if len(arguments) != len(placeholders):
            return None
        return (
            _PERCENT_CONVERSION.sub("?", template),
            [ctx.text(argument) for argument in arguments],
        )

    return None


def _parameterise_format(node: ast.Call, ctx: RuleContext) -> tuple[str, list[str]] | None:
    """``"... {} ...".format(a, b)`` with positional, unnamed, unformatted slots only."""
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    if not isinstance(receiver, ast.Constant) or not isinstance(receiver.value, str):
        return None
    if node.keywords:
        return None
    template = receiver.value
    if template.count("{}") != len(node.args) or "{" in template.replace("{}", ""):
        return None
    return (
        template.replace("{}", "?"),
        [ctx.text(argument) for argument in node.args],
    )


def _bind_quoted_placeholders(template: str) -> str | None:
    """Remove the SQL quotes around each placeholder, or decline if it cannot be done.

    Two jobs, and the second is the one that keeps a wrong patch from shipping.

    **Stripping.** The original interpolated a bare value *inside* SQL quotes, while a
    bound parameter supplies its own quoting. Leaving the quotes turns every parameter
    into the two-character literal ``?``: the query runs, matches nothing, and looks
    fixed.

    **Declining.** When a placeholder shares its quoted region with anything else, no
    amount of quote removal is correct. The case that taught this is a ``LIKE``
    pattern::

        cursor.execute(f"... WHERE name LIKE '{term}%'")

    The template is ``... LIKE '?%'``. Stripping only the quotes immediately around
    the ``?`` leaves ``LIKE '?%'`` — a search for the literal string ``?%``. The
    genuine fix moves the wildcard into the bound value (``term + "%"``), which is a
    change to the *value* expression rather than to the query, and inventing it here
    would be guessing at intent. So the rule reports the finding and declines the
    patch, which is what ``fix=None`` means everywhere else in this module.

    Returns the rewritten template, or ``None`` to decline.
    """
    out: list[str] = []
    index = 0
    length = len(template)
    while index < length:
        character = template[index]
        if character not in "'\"":
            out.append(character)
            index += 1
            continue
        closing = template.find(character, index + 1)
        if closing == -1:
            # An unbalanced quote means the template is not something this can reason
            # about, and a half-understood query is exactly what not to rewrite.
            return None
        region = template[index + 1 : closing]
        if "?" in region:
            if region != "?":
                return None
            out.append("?")
        else:
            out.append(template[index : closing + 1])
        index = closing + 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# CWE-502: unsafe deserialization
# --------------------------------------------------------------------------- #

_SAFE_YAML_LOADERS: Final[frozenset[str]] = frozenset(
    {"yaml.SafeLoader", "yaml.CSafeLoader", "yaml.BaseLoader", "yaml.CBaseLoader"}
)


def _match_yaml_load(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if ctx.imports.resolve(node.func) != "yaml.load":
        return None
    loader = positional_or_keyword(node, index=1, name="Loader")
    if loader is not None and ctx.imports.resolve(loader) in _SAFE_YAML_LOADERS:
        return None
    stream = positional_or_keyword(node, index=0, name="stream")
    confidence, fact = _confidence_for_sink(stream, ctx, "python.yaml-unsafe-load")
    # Unlike a command sink, the danger here is the *loader*, not the input, so a
    # sanitised stream does not make yaml.load safe. MEDIUM is the floor.
    confidence = confidence or FindingConfidence.MEDIUM
    return Match(
        node=node,
        # yaml.load's default loader constructs arbitrary Python objects, so the
        # construct is dangerous even where the input cannot be traced. Never LOW.
        confidence=max(confidence, FindingConfidence.MEDIUM, key=lambda c: c.rank),
        detail="yaml.load without a safe loader instantiates arbitrary Python types",
        taint=fact,
        fix_target=loader,
    )


def _fix_yaml_load(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    if not isinstance(call, ast.Call):
        return None
    stream = positional_or_keyword(call, index=0, name="stream")
    if stream is None:
        return None
    other = [
        ctx.text(kw) for kw in call.keywords if kw.arg not in {"Loader", None}
    ]
    arguments = ", ".join([ctx.text(stream), *other])
    return (
        ctx.replace(
            call,
            f"yaml.safe_load({arguments})",
            "yaml.safe_load builds only plain Python data, so a document cannot "
            "name a class to instantiate or a function to call",
        ),
    )


def _match_pickle_loads(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    resolved = ctx.imports.resolve(node.func)
    if resolved not in {"pickle.loads", "pickle.load", "dill.loads", "cPickle.loads"}:
        return None
    payload = positional_or_keyword(node, index=0, name="data")
    confidence, fact = _confidence_for_sink(payload, ctx, "python.pickle-deserialization")
    # As with yaml.load: pickle is unsafe by construction, not by its input.
    confidence = confidence or FindingConfidence.MEDIUM
    return Match(
        node=node,
        confidence=max(confidence, FindingConfidence.MEDIUM, key=lambda c: c.rank),
        detail=(
            f"{resolved} executes ``__reduce__`` on the incoming stream, so the "
            "payload chooses what code runs"
        ),
        taint=fact,
    )


# --------------------------------------------------------------------------- #
# CWE-327: weak hash
# --------------------------------------------------------------------------- #

_WEAK_HASHES: Final[dict[str, str]] = {
    "hashlib.md5": "sha256",
    "hashlib.sha1": "sha256",
    "hashlib.new": "",  # handled below: the algorithm is a string argument
}


def _match_weak_hash(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    resolved = ctx.imports.resolve(node.func)
    if resolved is None:
        return None
    if resolved == "hashlib.new":
        algorithm = positional_or_keyword(node, index=0, name="name")
        if not isinstance(algorithm, ast.Constant) or not isinstance(algorithm.value, str):
            return None
        if algorithm.value.lower().replace("-", "") not in {"md5", "sha1"}:
            return None
    elif resolved not in _WEAK_HASHES:
        return None

    # Python 3.9 added ``usedforsecurity=False`` precisely so a non-security use — a
    # cache key, an ETag — can be spelled unambiguously. Honouring it is what keeps
    # this rule from being the one developers add a blanket suppression for.
    flag = keyword_of(node, "usedforsecurity")
    if flag is not None and _is_false(flag.value):
        return None
    return Match(
        node=node,
        confidence=FindingConfidence.MEDIUM,
        detail=(
            "MD5 and SHA-1 have practical collision attacks, so a digest used for "
            "integrity or authentication can be forged"
        ),
        fix_target=node.func,
    )


def _fix_weak_hash(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    target = match.fix_target
    if not isinstance(call, ast.Call):
        return None
    resolved = ctx.imports.resolve(call.func)
    if resolved == "hashlib.new":
        algorithm = positional_or_keyword(call, index=0, name="name")
        if algorithm is None:
            return None
        return (
            ctx.replace(
                algorithm,
                '"sha256"',
                "SHA-256 has no known collision attack and is a drop-in for the "
                "same hashlib interface",
            ),
        )
    if not isinstance(target, ast.Attribute):
        # ``from hashlib import md5`` binds a bare name, and rewriting the call site
        # alone would leave it undefined. Changing the import as well is a wider
        # edit than one finding should make, so this reports without a patch.
        return None
    return (
        ctx.replace(
            target,
            f"{ctx.text(target.value)}.sha256",
            "SHA-256 has no known collision attack and exposes the same interface; "
            "the digest length changes, so any fixed-width storage needs widening",
        ),
    )


# --------------------------------------------------------------------------- #
# CWE-798: hardcoded credentials
# --------------------------------------------------------------------------- #

_SECRET_NAME: Final[re.Pattern[str]] = re.compile(
    r"(?:^|_)(?:password|passwd|pwd|secret|token|api_?key|access_?key|"
    r"private_?key|credential|auth|salt|signing_?key)s?(?:_|$)",
    re.IGNORECASE,
)

#: Values that name a secret without being one. A scanner that flags
#: ``PASSWORD = ""`` or ``TOKEN = "changeme"`` in an example config trains the team
#: to ignore CWE-798, which is the finding class where a single true positive is
#: worth more than every other rule here combined.
_PLACEHOLDER_SECRETS: Final[frozenset[str]] = frozenset(
    {
        "",
        "changeme",
        "change_me",
        "your_password_here",
        "xxx",
        "todo",
        "none",
        "null",
        "example",
        "placeholder",
        "redacted",
        "secret",
        "password",
        "dummy",
        "test",
        "fake",
        "sample",
    }
)

MIN_SECRET_LENGTH: Final[int] = 6


def _match_hardcoded_secret(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Assign | ast.AnnAssign):
        return None
    value = node.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    literal = value.value
    if len(literal) < MIN_SECRET_LENGTH:
        return None
    if literal.strip().lower() in _PLACEHOLDER_SECRETS:
        return None
    if _looks_like_reference(literal):
        return None
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        name = _assigned_name(target)
        if name is None or not _SECRET_NAME.search(name):
            continue
        return Match(
            node=node,
            confidence=FindingConfidence.HIGH,
            detail=(
                f"{name} is assigned a literal value, so the credential is in version "
                "control and in every copy of the repository and its history"
            ),
            fix_target=target,
        )
    return None


def _looks_like_reference(literal: str) -> bool:
    """True for values that are plainly a *name* rather than a secret."""
    stripped = literal.strip()
    if stripped.startswith(("${", "%(", "{{", "$")):
        return True
    # A path or a URL naming where the secret lives is not the secret.
    return stripped.startswith(("/", "./", "http://", "https://", "file://"))


def _assigned_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
        return target.slice.value if isinstance(target.slice.value, str) else None
    return None


def _fix_hardcoded_secret(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    assignment = match.node
    target = match.fix_target
    if not isinstance(assignment, ast.Assign | ast.AnnAssign) or target is None:
        return None
    name = _assigned_name(target)
    if name is None or assignment.value is None:
        return None
    variable = _env_var_name(name)
    edits = [
        ctx.replace(
            assignment.value,
            f'os.environ["{variable}"]',
            f"read the value from the {variable} environment variable; the literal "
            "stays in git history and must be rotated regardless of this patch",
        )
    ]
    os_import = ctx.require_import("os")
    if os_import is not None:
        edits.append(os_import)
    return tuple(edits)


# --------------------------------------------------------------------------- #
# CWE-489 / CWE-1327: debug mode and interface binding
# --------------------------------------------------------------------------- #


def _match_debug_true(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "run":
        return None
    debug = keyword_of(node, "debug")
    if debug is None or not _is_true(debug.value):
        return None
    return Match(
        node=node,
        confidence=FindingConfidence.HIGH,
        detail=(
            "Flask's debug mode serves the Werkzeug debugger, which exposes an "
            "interactive Python console on any unhandled exception"
        ),
        fix_target=debug,
    )


def _fix_debug_true(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    debug = match.fix_target
    if not isinstance(debug, ast.keyword):
        return None
    return (
        ctx.replace(
            debug,
            "debug=False",
            "disable the debugger; it offers a remote code-execution console to "
            "anyone who can trigger an exception",
        ),
    )


_ANY_INTERFACE: Final[frozenset[str]] = frozenset({"0.0.0.0", "::", "[::]"})


def _match_bind_all_interfaces(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in {"run", "bind"}:
        return None
    host = keyword_of(node, "host")
    if host is None:
        host = keyword_of(node, "hostname")
    if host is None or not isinstance(host.value, ast.Constant):
        return None
    if host.value.value not in _ANY_INTERFACE:
        return None
    return Match(
        node=node,
        # HIGH, with Severity.LOW on the rule. The two fields answer different
        # questions and conflating them is a real mistake: confidence is "is this
        # construct really here and really reachable", which for a literal
        # ``host="0.0.0.0"`` is certain, while severity is "how much does it
        # matter", which is little. Grading it LOW confidence would put it below
        # the patch threshold, so the one-word fix would never be drafted for a
        # finding the analyzer is completely sure about.
        confidence=FindingConfidence.HIGH,
        detail=(
            "binding every interface publishes the service on any network the host "
            "is attached to, including ones the deployment did not intend"
        ),
        fix_target=host,
    )


def _fix_bind_all_interfaces(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    host = match.fix_target
    if not isinstance(host, ast.keyword):
        return None
    return (
        ctx.replace(
            host,
            f'{host.arg}="127.0.0.1"',
            "bind the loopback interface; expose the service deliberately through a "
            "reverse proxy or an explicit configuration value instead",
        ),
    )


# --------------------------------------------------------------------------- #
# CWE-22: path traversal
# --------------------------------------------------------------------------- #

_FILE_OPENERS: Final[frozenset[str]] = frozenset(
    {
        "open",
        "io.open",
        "os.remove",
        "os.unlink",
        "os.rename",
        "shutil.copy",
        "shutil.copyfile",
        "shutil.move",
        "shutil.rmtree",
        "pathlib.Path",
    }
)


def _match_path_traversal(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if ctx.imports.resolve(node.func) not in _FILE_OPENERS:
        return None
    path = positional_or_keyword(node, index=0, name="file")
    if path is None:
        path = positional_or_keyword(node, index=0, name="path")
    if path is None:
        return None
    tainted = _tainted_component(path, ctx)
    if tainted is None:
        return None
    fact = ctx.taint_for(tainted, "python.path-traversal")
    return Match(
        node=node,
        confidence=FindingConfidence.HIGH,
        detail=(
            "the path contains attacker-controlled text, so '../' in it escapes the "
            "intended directory"
        ),
        taint=fact,
        fix_target=tainted,
    )


def _tainted_component(node: ast.expr, ctx: RuleContext) -> ast.expr | None:
    """The innermost attacker-controlled sub-expression of a path expression.

    The innermost one, because that is where the fix belongs: wrapping the whole
    ``os.path.join(BASE, name)`` in ``basename`` would collapse the base directory
    away, while wrapping ``name`` leaves the join intact and neutralises the
    traversal. Getting this backwards produces a patch that breaks every upload.

    The descent stops at any node already sanitised for this rule. Without that
    stop, ``open(os.path.basename(path))`` — a correct fix — reports the ``path``
    *inside* the ``basename`` call as the tainted component, because the inner name
    still carries live taint while only the wrapping call is cleared. The measured
    consequence was a patch proposing ``os.path.basename(os.path.basename(path))``,
    which validation then rejected for not removing the finding. The rejection was
    right and the cause was here.
    """
    if ctx.sanitised_for(node, "python.path-traversal"):
        return None
    tainted = ctx.taint_for(node, "python.path-traversal") is not None

    # A name, attribute access or subscript is an **atomic** value expression and
    # must be taken whole. Descending into one splits it: for
    # ``request.args["file"]`` the subscript's child ``request.args`` is also
    # tainted, and returning that child produced the patch
    # ``os.path.basename(request.args)["file"]`` — valid Python, accepted by every
    # check except the one that matters, and it calls basename on a MultiDict.
    if isinstance(node, ast.Name | ast.Attribute | ast.Subscript):
        return node if tainted else None

    # A call is where composition happens, so its arguments are worth descending
    # into: ``os.path.join(BASE, name)`` is itself tainted because the join
    # propagates, and wrapping the whole join in basename would discard BASE.
    if isinstance(node, ast.Call):
        for argument in [*node.args, *(kw.value for kw in node.keywords)]:
            found = _tainted_component(argument, ctx)
            if found is not None:
                return found
        return node if tainted else None

    # Concatenations, f-strings, tuples: composition, so descend.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            found = _tainted_component(child, ctx)
            if found is not None:
                return found
    return node if tainted and isinstance(node, ast.expr) else None


def _fix_path_traversal(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    tainted = match.fix_target
    if not isinstance(tainted, ast.expr) or not isinstance(call, ast.Call):
        return None
    path_argument = positional_or_keyword(call, index=0, name="file") or (
        positional_or_keyword(call, index=0, name="path")
    )
    if path_argument is tainted:
        # The whole path *is* the tainted value — ``open(path, "rb")`` where ``path``
        # was composed on an earlier line. Wrapping it in ``basename`` would discard
        # the directory the file is supposed to be read from, so the patch would
        # remove the finding and break the endpoint. The fix belongs at the
        # composition site, which is a different statement than the finding, and
        # locating it needs def-use chains that :mod:`sentinel.scan.taint`
        # deliberately does not build. So this reports without a patch, and the
        # remediation text says where the edit goes.
        return None
    edits = [
        ctx.replace(
            tainted,
            f"os.path.basename({ctx.text(tainted)})",
            "reduce the attacker-controlled component to a bare filename, so any "
            "directory traversal in it is discarded before the join",
        )
    ]
    os_import = ctx.require_import("os")
    if os_import is not None:
        edits.append(os_import)
    return tuple(edits)


# --------------------------------------------------------------------------- #
# CWE-95: eval / exec
# --------------------------------------------------------------------------- #


def _match_eval(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    resolved = ctx.imports.resolve(node.func)
    if resolved not in {"eval", "exec", "execfile"}:
        return None
    code = positional_or_keyword(node, index=0, name="source")
    confidence, fact = _confidence_for_sink(code, ctx, "python.code-injection-eval")
    if confidence is None:
        # ast.literal_eval already ran on it, so the value is data.
        return None
    if confidence is FindingConfidence.LOW:
        confidence = FindingConfidence.MEDIUM
    return Match(
        node=node,
        confidence=confidence,
        detail=f"{resolved} compiles and runs its argument as Python",
        taint=fact,
        fix_target=code,
    )


def _fix_eval(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    code = match.fix_target
    if not isinstance(call, ast.Call) or not isinstance(code, ast.expr):
        return None
    if ctx.imports.resolve(call.func) != "eval":
        # exec has no data-only equivalent: its whole purpose is running statements.
        return None
    if len(call.args) > 1 or call.keywords:
        # globals/locals arguments mean the caller wants an execution environment,
        # which literal_eval does not have.
        return None
    edits = [
        ctx.replace(
            call,
            f"ast.literal_eval({ctx.text(code)})",
            "ast.literal_eval parses Python literals only, so the input becomes "
            "data instead of code; it raises ValueError on anything else",
        )
    ]
    ast_import = ctx.require_import("ast")
    if ast_import is not None:
        edits.append(ast_import)
    return tuple(edits)


# --------------------------------------------------------------------------- #
# CWE-295: TLS verification disabled
# --------------------------------------------------------------------------- #


def _match_tls_disabled(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    verify = keyword_of(node, "verify")
    if verify is None or not _is_false(verify.value):
        return None
    resolved = ctx.imports.resolve(node.func) or ""
    if not (resolved.startswith(("requests.", "httpx.")) or resolved.endswith(
        (".get", ".post", ".put", ".patch", ".delete", ".request", ".head")
    )):
        return None
    return Match(
        node=node,
        confidence=FindingConfidence.HIGH,
        detail=(
            "with certificate verification off, any network position can present "
            "its own certificate and read or rewrite the traffic"
        ),
        fix_target=verify,
    )


def _fix_tls_disabled(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    verify = match.fix_target
    if not isinstance(verify, ast.keyword):
        return None
    return (
        ctx.replace(
            verify,
            "verify=True",
            "verify the server certificate; point verify at a CA bundle path "
            "instead if this endpoint uses a private CA",
        ),
    )


# --------------------------------------------------------------------------- #
# CWE-377: insecure temporary file
# --------------------------------------------------------------------------- #


def _match_insecure_temp(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    if ctx.imports.resolve(node.func) != "tempfile.mktemp":
        return None
    return Match(
        node=node,
        confidence=FindingConfidence.MEDIUM,
        detail=(
            "mktemp returns a name without creating the file, so anything else on "
            "the machine can create it first and win the race"
        ),
    )


def _fix_insecure_temp(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    if not isinstance(call, ast.Call):
        return None
    arguments = ", ".join(
        [*(ctx.text(arg) for arg in call.args), *(ctx.text(kw) for kw in call.keywords)]
    )
    return (
        ctx.replace(
            call,
            f"tempfile.mkstemp({arguments})[1]",
            "mkstemp creates the file atomically with 0600 permissions; it also "
            "returns an open descriptor at index 0 which this call site discards, "
            "so close it if the file is long-lived",
        ),
    )


# --------------------------------------------------------------------------- #
# CWE-79: template autoescaping
# --------------------------------------------------------------------------- #


def _match_autoescape_off(node: ast.AST, ctx: RuleContext) -> Match | None:
    if not isinstance(node, ast.Call):
        return None
    resolved = ctx.imports.resolve(node.func)
    if resolved not in {"jinja2.Environment", "jinja2.environment.Environment"}:
        return None
    autoescape = keyword_of(node, "autoescape")
    if autoescape is not None and not _is_false(autoescape.value):
        return None
    return Match(
        node=node,
        confidence=FindingConfidence.HIGH,
        detail=(
            "Jinja2 does not escape by default, so any value rendered into a "
            "template is injected into the page as markup"
        ),
        fix_target=autoescape,
    )


def _fix_autoescape_off(match: Match, ctx: RuleContext) -> tuple[SourceEdit, ...] | None:
    call = match.node
    autoescape = match.fix_target
    if not isinstance(call, ast.Call):
        return None
    note = (
        "enable autoescaping so rendered values are HTML-escaped; wrap the values "
        "that are deliberately markup in Markup() instead"
    )
    if isinstance(autoescape, ast.keyword):
        return (ctx.replace(autoescape, "autoescape=True", note),)
    # No keyword at all: append one. The insertion point is just before the closing
    # parenthesis, found by scanning back from the call's end rather than assumed,
    # because a trailing comma or a comment can sit between the last argument and it.
    start, end = ctx.offsets(call)
    closing = ctx.source.rfind(")", start, end)
    if closing == -1:
        return None
    has_arguments = bool(call.args or call.keywords)
    prefix = ", " if has_arguments else ""
    return (
        SourceEdit(
            start=closing,
            end=closing,
            replacement=f"{prefix}autoescape=True",
            note=note,
        ),
    )


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #

RULES: Final[tuple[Rule, ...]] = (
    Rule(
        rule_id="python.os-command-injection",
        cwe="CWE-78",
        title="Shell command built from untrusted input",
        severity=Severity.CRITICAL,
        message="subprocess called with shell=True on a command that is not a literal",
        remediation="Pass an argv list and drop shell=True, or quote with shlex.quote",
        matcher=_match_shell_true,
        fix=_fix_shell_true,
        cve_hints=("command injection", "os command injection shell"),
    ),
    Rule(
        rule_id="python.os-system-injection",
        cwe="CWE-78",
        title="os.system on a runtime-assembled command",
        severity=Severity.CRITICAL,
        message="os.system/os.popen passes its argument to a shell",
        remediation=(
            "Use subprocess.run with an argv list, so no shell interprets the string"
        ),
        matcher=_match_os_system,
        fix=_fix_os_system,
        cve_hints=("command injection", "shell metacharacter"),
    ),
    Rule(
        rule_id="python.sql-injection",
        cwe="CWE-89",
        title="SQL statement assembled by string formatting",
        severity=Severity.CRITICAL,
        message="a query is built with an f-string, %, .format() or concatenation",
        remediation=(
            "Bind the values as query parameters so the driver sends them "
            "out-of-band and they can never be parsed as SQL"
        ),
        matcher=_match_sql_injection,
        fix=_fix_sql_injection,
        cve_hints=("sql injection", "injection database query"),
    ),
    Rule(
        rule_id="python.yaml-unsafe-load",
        cwe="CWE-502",
        title="yaml.load without a safe loader",
        severity=Severity.HIGH,
        message="the default YAML loader instantiates arbitrary Python objects",
        remediation=(
            "Use yaml.safe_load, which constructs only plain Python data and cannot "
            "instantiate a class named by the document"
        ),
        matcher=_match_yaml_load,
        fix=_fix_yaml_load,
        cve_hints=("deserialization of untrusted data", "yaml deserialization"),
    ),
    Rule(
        rule_id="python.pickle-deserialization",
        cwe="CWE-502",
        title="Deserializing untrusted data with pickle",
        severity=Severity.HIGH,
        message="pickle executes __reduce__ from the stream it is loading",
        remediation=(
            "Use a data-only format (JSON) or authenticate the payload with an HMAC "
            "before unpickling. There is no safe pickle loader, so this is not "
            "mechanically patchable."
        ),
        matcher=_match_pickle_loads,
        fix=None,
        cve_hints=("deserialization of untrusted data", "insecure deserialization"),
    ),
    Rule(
        rule_id="python.weak-hash",
        cwe="CWE-327",
        title="Broken cryptographic hash",
        severity=Severity.MEDIUM,
        message="MD5/SHA-1 are collision-broken and unsuitable for security use",
        remediation="Use SHA-256, or pass usedforsecurity=False for a non-security digest",
        matcher=_match_weak_hash,
        fix=_fix_weak_hash,
        cve_hints=("weak cryptographic hash", "collision attack"),
    ),
    Rule(
        rule_id="python.hardcoded-credential",
        cwe="CWE-798",
        title="Credential committed to source",
        severity=Severity.HIGH,
        message="a secret-shaped name is assigned a string literal",
        remediation="Read the value from the environment or a secret manager, and rotate it",
        matcher=_match_hardcoded_secret,
        fix=_fix_hardcoded_secret,
        node_types=(ast.Assign, ast.AnnAssign),
        cve_hints=("hardcoded credentials", "use of hard-coded password"),
    ),
    Rule(
        rule_id="python.flask-debug-enabled",
        cwe="CWE-489",
        title="Web framework debug mode enabled",
        severity=Severity.HIGH,
        message="the Werkzeug debugger exposes an interactive console on any traceback",
        remediation="Run with debug=False and use logging in production",
        matcher=_match_debug_true,
        fix=_fix_debug_true,
        cve_hints=("debug mode enabled", "information exposure through debug"),
    ),
    Rule(
        rule_id="python.bind-all-interfaces",
        cwe="CWE-1327",
        title="Service bound to every network interface",
        severity=Severity.LOW,
        message="binding 0.0.0.0 publishes the service on every attached network",
        remediation="Bind 127.0.0.1 and expose deliberately through a proxy",
        matcher=_match_bind_all_interfaces,
        fix=_fix_bind_all_interfaces,
    ),
    Rule(
        rule_id="python.path-traversal",
        cwe="CWE-22",
        title="File path built from untrusted input",
        severity=Severity.HIGH,
        message="an attacker-controlled component in a path allows '../' escape",
        remediation="Reduce the untrusted component to a basename and validate the result",
        matcher=_match_path_traversal,
        fix=_fix_path_traversal,
        cve_hints=("path traversal", "directory traversal"),
    ),
    Rule(
        rule_id="python.code-injection-eval",
        cwe="CWE-95",
        title="eval/exec on a runtime value",
        severity=Severity.CRITICAL,
        message="the argument is compiled and executed as Python",
        remediation="Use ast.literal_eval for data, or a dispatch table for behaviour",
        matcher=_match_eval,
        fix=_fix_eval,
        cve_hints=("code injection", "eval injection"),
    ),
    Rule(
        rule_id="python.tls-verification-disabled",
        cwe="CWE-295",
        title="TLS certificate verification disabled",
        severity=Severity.HIGH,
        message="verify=False accepts any certificate, so the channel is unauthenticated",
        remediation="Leave verification on; point verify at a CA bundle for a private CA",
        matcher=_match_tls_disabled,
        fix=_fix_tls_disabled,
        cve_hints=("improper certificate validation", "tls verification"),
    ),
    Rule(
        rule_id="python.insecure-temp-file",
        cwe="CWE-377",
        title="Predictable temporary file",
        severity=Severity.MEDIUM,
        message="tempfile.mktemp returns a name without creating the file",
        remediation="Use tempfile.mkstemp or NamedTemporaryFile",
        matcher=_match_insecure_temp,
        fix=_fix_insecure_temp,
        cve_hints=("insecure temporary file", "race condition file creation"),
    ),
    Rule(
        rule_id="python.template-autoescape-disabled",
        cwe="CWE-79",
        title="Template autoescaping disabled",
        severity=Severity.HIGH,
        message="Jinja2 renders values as markup unless autoescape is on",
        remediation="Pass autoescape=True and wrap intentional markup in Markup()",
        matcher=_match_autoescape_off,
        fix=_fix_autoescape_off,
        cve_hints=("cross-site scripting", "improper neutralization of script"),
    ),
)

RULES_BY_ID: Final[dict[str, Rule]] = {rule.rule_id: rule for rule in RULES}

if len(RULES_BY_ID) != len(RULES):  # pragma: no cover - import-time invariant
    raise ScanError("duplicate rule id in RULES")


def rule_ids() -> tuple[str, ...]:
    return tuple(RULES_BY_ID)
