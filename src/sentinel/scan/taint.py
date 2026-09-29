"""Does attacker-controlled data reach this line? (PRD F-07)

A scanner that flags ``subprocess.run(cmd, shell=True)`` wherever it appears
reports the deploy script alongside the exploitable endpoint, and the first thing a
team does with such a scanner is stop reading its output. What separates a finding
worth a pull request from noise is whether the dangerous value came from outside,
so this module answers exactly that question and :mod:`sentinel.scan.rules` uses
the answer to set :class:`~sentinel.scan.findings.FindingConfidence`.

The analysis
------------
A forward, flow-sensitive, **may**-taint analysis over one module's AST:

*   **Flow-sensitive**, so an assignment kills the taint on its target. Without
    that, ``x = request.args["q"]`` followed by ``x = "SELECT 1"`` reports the
    literal as attacker-controlled, and a false positive on a *sanitised* value is
    the specific false positive that destroys trust fastest.
*   **May**, so at a branch join the taint sets are unioned. Over-approximating at
    a join costs a reviewable false positive; under-approximating misses the
    vulnerability on the path the attacker takes. For a security analysis only one
    of those errors is acceptable.
*   **Intraprocedural.** Each function is analysed with its enclosing scope's facts
    as the initial state (closures read outer names), and a call to a local helper
    is not followed. :mod:`sentinel.scan.symbols` states that boundary; here the
    consequence is that ``def run_cmd(c): os.system(c)`` is flagged at its
    definition on the argument, not traced from its caller.
*   **Bounded fixed point on loops.** A loop body is re-analysed until the fact set
    stops growing, capped at :data:`LOOP_ITERATION_CAP`. The lattice is a finite
    set of (name, rule) pairs and every transfer function is monotone, so the
    iteration terminates on its own; the cap exists so a pathological input cannot
    turn a scan into a hang.

Sanitizers are per-rule, and that is the subtle part
---------------------------------------------------
``os.path.basename(user_input)`` genuinely fixes a path traversal and does
**nothing** for a command injection — ``basename`` happily returns
``"a; rm -rf /"``. A single boolean "sanitised" flag would therefore have to be
wrong in one direction or the other, and both directions are bad: treat it as
clean and the command injection is missed, treat it as tainted and every correctly
fixed path handler keeps reporting. So a fact records *which rules* a value has
been cleared for, and :meth:`TaintState.fact_for` takes the rule asking.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Final

from sentinel.scan.symbols import ImportTable

__all__ = [
    "LOOP_ITERATION_CAP",
    "SANITIZERS",
    "TAINT_SOURCES",
    "Sanitizer",
    "TaintFact",
    "TaintState",
    "analyse_taint",
]

LOOP_ITERATION_CAP: Final[int] = 8
"""Maximum re-analyses of one loop body. See the module docstring."""


#: Fully-qualified expressions that introduce attacker-controlled data, mapped to
#: the label that appears in a finding's ``taint_path``.
#:
#: ``os.environ`` is deliberately **absent**. Environment variables are set by
#: whoever starts the process, so treating them as attacker input would flag every
#: correct twelve-factor configuration read in the codebase — which is how a
#: scanner teaches a team that its findings do not mean anything.
TAINT_SOURCES: Final[dict[str, str]] = {
    "flask.request.args": "HTTP query string",
    "flask.request.form": "HTTP form body",
    "flask.request.values": "HTTP request (query or form)",
    "flask.request.json": "HTTP JSON body",
    "flask.request.get_json": "HTTP JSON body",
    "flask.request.data": "HTTP raw body",
    "flask.request.cookies": "HTTP cookie",
    "flask.request.headers": "HTTP header",
    "flask.request.files": "HTTP file upload",
    "flask.request.view_args": "URL path parameter",
    "sys.argv": "command line",
    "input": "interactive input",
    "os.getenv": "",  # placeholder, removed below; see _SOURCE_EXCLUSIONS
}

#: Removed from ``TAINT_SOURCES`` after construction. Kept visible in the literal
#: above so the reader can see the call was considered and rejected, rather than
#: wondering whether it was forgotten.
_SOURCE_EXCLUSIONS: Final[frozenset[str]] = frozenset({"os.getenv"})
for _name in _SOURCE_EXCLUSIONS:
    TAINT_SOURCES.pop(_name, None)


@dataclass(frozen=True, slots=True)
class Sanitizer:
    """A call that neutralises taint, and the rules it neutralises it *for*."""

    qualified: str
    label: str
    #: Rule ids this call makes safe. ``None`` means "safe for everything".
    clears: frozenset[str] | None = None

    def clears_rule(self, rule_id: str) -> bool:
        return self.clears is None or rule_id in self.clears


#: Rule ids are duplicated here as plain strings rather than imported from
#: :mod:`sentinel.scan.rules`, which imports this module. ``test_scan_taint.py``
#: asserts every id named here is a real rule, so a rename cannot silently turn a
#: sanitizer into a no-op.
SANITIZERS: Final[tuple[Sanitizer, ...]] = (
    # Numeric coercion cannot carry a payload of any kind.
    Sanitizer("int", "int()"),
    Sanitizer("float", "float()"),
    Sanitizer("bool", "bool()"),
    # Shell quoting makes a string a single argv word, which is precisely the
    # command-injection fix — and does nothing for SQL or a path.
    Sanitizer(
        "shlex.quote",
        "shlex.quote()",
        frozenset({"python.os-command-injection", "python.os-system-injection"}),
    ),
    Sanitizer(
        "pipes.quote",
        "pipes.quote()",
        frozenset({"python.os-command-injection", "python.os-system-injection"}),
    ),
    # basename strips directory traversal and nothing else.
    Sanitizer("os.path.basename", "os.path.basename()", frozenset({"python.path-traversal"})),
    Sanitizer(
        "werkzeug.utils.secure_filename",
        "secure_filename()",
        frozenset({"python.path-traversal"}),
    ),
    Sanitizer(
        "secure_filename", "secure_filename()", frozenset({"python.path-traversal"})
    ),
    # A literal_eval result is data, never code.
    Sanitizer("ast.literal_eval", "ast.literal_eval()"),
)

_SANITIZER_BY_NAME: Final[dict[str, Sanitizer]] = {s.qualified: s for s in SANITIZERS}

#: Calls that pass their argument's taint straight through. A string coercion of a
#: tainted value is still tainted; forgetting this is how ``str(request.args["q"])``
#: launders a payload past an analyzer.
_PASSTHROUGH_CALLS: Final[frozenset[str]] = frozenset(
    {
        "str",
        "bytes",
        "repr",
        "format",
        "list",
        "tuple",
        "set",
        "dict",
        "sorted",
        "reversed",
        "iter",
        "next",
        "urllib.parse.unquote",
        "urllib.parse.unquote_plus",
        "base64.b64decode",
        "json.loads",
        # Path composition carries taint through. Listing these explicitly matters
        # because ``os.path.normpath("../../etc/passwd")`` returns
        # ``"../../etc/passwd"`` — it resolves the path *textually* and removes
        # nothing. A reader who assumes normpath sanitises has the traversal fix
        # exactly backwards, and ``os.path.basename`` is the one that does it
        # (registered as a SANITIZER above, for the traversal rule only).
        "os.path.join",
        "os.path.normpath",
        "os.path.abspath",
        "os.path.realpath",
        "os.path.expanduser",
        "posixpath.join",
        "pathlib.Path",
    }
)

#: String methods that return a derivative of their receiver.
_PASSTHROUGH_METHODS: Final[frozenset[str]] = frozenset(
    {
        "strip",
        "lstrip",
        "rstrip",
        "lower",
        "upper",
        "title",
        "capitalize",
        "replace",
        "split",
        "rsplit",
        "splitlines",
        "partition",
        "rpartition",
        "encode",
        "decode",
        "get",
        "pop",
        "removeprefix",
        "removesuffix",
        "casefold",
        "expandtabs",
        "zfill",
        "ljust",
        "rjust",
        "center",
        "read",
        "readline",
        "readlines",
    }
)

#: Decorators that mark a function as reachable from outside, making its
#: parameters attacker-controlled. Matched on the *attribute* (``app.route``,
#: ``bp.post``) because the object is a local variable whose name is arbitrary.
_ROUTE_DECORATORS: Final[frozenset[str]] = frozenset(
    {"route", "get", "post", "put", "patch", "delete", "websocket", "add_url_rule"}
)


@dataclass(frozen=True, slots=True)
class TaintFact:
    """Why a name is considered attacker-influenced, and what it is clean for."""

    origin: str
    #: Human-readable chain, source first: ``("HTTP query string", "raw", "cmd")``.
    path: tuple[str, ...]
    #: Rule ids a sanitizer has cleared this value for.
    cleared_for: frozenset[str] = frozenset()

    def through(self, step: str) -> TaintFact:
        """Extend the chain by one step, keeping the clearances."""
        if self.path and self.path[-1] == step:
            return self
        return TaintFact(
            origin=self.origin, path=(*self.path, step), cleared_for=self.cleared_for
        )

    def cleared(self, sanitizer: Sanitizer) -> TaintFact:
        """Record that ``sanitizer`` ran on this value."""
        extra = _ALL_RULES_MARKER if sanitizer.clears is None else sanitizer.clears
        return TaintFact(
            origin=self.origin,
            path=(*self.path, sanitizer.label),
            cleared_for=self.cleared_for | extra,
        )

    def is_live_for(self, rule_id: str) -> bool:
        """True when this taint still matters to ``rule_id``."""
        if _ALL_RULES_MARKER & self.cleared_for:
            return False
        return rule_id not in self.cleared_for


#: A sentinel member standing for "cleared for every rule". A frozenset of rule ids
#: cannot express that, and enumerating every rule here would silently stop covering
#: a rule added later — the failure mode being a missed sanitizer, i.e. a false
#: positive on correct code.
_ALL_RULES_MARKER: Final[frozenset[str]] = frozenset({"*"})


@dataclass(slots=True)
class TaintState:
    """Taint facts for one scope, keyed by local name."""

    facts: dict[str, TaintFact] = field(default_factory=dict)

    def copy(self) -> TaintState:
        return TaintState(facts=dict(self.facts))

    def merge(self, other: TaintState) -> None:
        """Union in ``other``'s facts. The join of a may-analysis."""
        for name, fact in other.facts.items():
            existing = self.facts.get(name)
            if existing is None:
                self.facts[name] = fact
            else:
                # Keep the *weaker* clearance set: a value clean on one branch and
                # dirty on the other is dirty.
                self.facts[name] = TaintFact(
                    origin=existing.origin,
                    path=existing.path,
                    cleared_for=existing.cleared_for & fact.cleared_for,
                )

    def set(self, name: str, fact: TaintFact | None) -> None:
        if fact is None:
            self.facts.pop(name, None)
        else:
            self.facts[name] = fact

    def get(self, name: str) -> TaintFact | None:
        return self.facts.get(name)

    def signature(self) -> frozenset[tuple[str, tuple[str, ...], frozenset[str]]]:
        """A comparable snapshot, for the loop fixed-point test."""
        return frozenset(
            (name, fact.path, fact.cleared_for) for name, fact in self.facts.items()
        )


class _ScopeAnalyzer:
    """Analyses one function (or the module) body and records per-node facts."""

    def __init__(self, imports: ImportTable, results: dict[int, TaintFact]) -> None:
        self._imports = imports
        self._results = results

    # --- expressions ------------------------------------------------------- #

    def taint_of(self, node: ast.expr | None, state: TaintState) -> TaintFact | None:
        """The taint of an expression, recording it for later lookup by rules."""
        if node is None:
            return None
        fact = self._taint_of(node, state)
        if fact is not None:
            self._results[id(node)] = fact
        return fact

    def _taint_of(self, node: ast.expr, state: TaintState) -> TaintFact | None:
        qualified = self._imports.resolve(node)
        if qualified is not None:
            source = _source_label(qualified)
            if source is not None:
                return TaintFact(origin=source, path=(source,))

        if isinstance(node, ast.Name):
            return state.get(node.id)

        if isinstance(node, ast.Attribute | ast.Subscript):
            base = node.value
            inherited = self.taint_of(base, state)
            if inherited is not None:
                step = node.attr if isinstance(node, ast.Attribute) else "[...]"
                return inherited.through(step)
            if isinstance(node, ast.Subscript):
                # ``d[user_input]`` is a *lookup keyed by* attacker data; the value
                # that comes back is whatever the dict holds, so it is not tainted.
                # But ``request.args[k]`` is caught above by the qualified match.
                return None
            return None

        if isinstance(node, ast.Call):
            return self._taint_of_call(node, state)

        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    fact = self.taint_of(value.value, state)
                    if fact is not None:
                        return fact.through("f-string")
            return None

        if isinstance(node, ast.FormattedValue):
            fact = self.taint_of(node.value, state)
            return None if fact is None else fact.through("f-string")

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add | ast.Mod):
                step = "concatenation" if isinstance(node.op, ast.Add) else "%-format"
                for side in (node.left, node.right):
                    fact = self.taint_of(side, state)
                    if fact is not None:
                        return fact.through(step)
            return None

        if isinstance(node, ast.IfExp):
            for branch in (node.body, node.orelse):
                fact = self.taint_of(branch, state)
                if fact is not None:
                    return fact
            return None

        if isinstance(node, ast.Tuple | ast.List | ast.Set):
            for element in node.elts:
                fact = self.taint_of(element, state)
                if fact is not None:
                    return fact
            return None

        if isinstance(node, ast.Dict):
            for value in node.values:
                fact = self.taint_of(value, state)
                if fact is not None:
                    return fact
            return None

        if isinstance(node, ast.Starred):
            return self.taint_of(node.value, state)

        if isinstance(node, ast.Await):
            return self.taint_of(node.value, state)

        if isinstance(node, ast.BoolOp):
            for value in node.values:
                fact = self.taint_of(value, state)
                if fact is not None:
                    return fact
            return None

        return None

    def _taint_of_call(self, node: ast.Call, state: TaintState) -> TaintFact | None:
        qualified = self._imports.resolve(node.func)

        # Record every argument's taint before deciding anything about the call
        # itself. A rule asks "is the *argument* to os.system tainted", and
        # ``os.system(...)`` does not return a tainted value, so a version of this
        # that only walked arguments on the paths where the call result is tainted
        # would leave every sink argument unrecorded — which is the whole question
        # the rules need answered. Recording is idempotent, so the specific
        # branches below may re-ask without cost.
        for argument in node.args:
            self.taint_of(argument, state)
        for keyword in node.keywords:
            self.taint_of(keyword.value, state)

        if qualified is not None:
            sanitizer = _SANITIZER_BY_NAME.get(qualified)
            if sanitizer is not None:
                inner = next(
                    (
                        fact
                        for fact in (self.taint_of(arg, state) for arg in node.args)
                        if fact is not None
                    ),
                    None,
                )
                return None if inner is None else inner.cleared(sanitizer)

            if qualified in _PASSTHROUGH_CALLS:
                for argument in node.args:
                    fact = self.taint_of(argument, state)
                    if fact is not None:
                        return fact.through(qualified)
                return None

            source = _source_label(qualified)
            if source is not None:
                return TaintFact(origin=source, path=(source,))

        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            receiver = self.taint_of(node.func.value, state)
            if method in {"format", "join"}:
                # ``"...{}".format(tainted)`` and ``sep.join([tainted])``: the
                # receiver is usually a clean literal and the taint is in the args.
                for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                    fact = self.taint_of(argument, state)
                    if fact is not None:
                        return fact.through(f".{method}()")
                return None if receiver is None else receiver.through(f".{method}()")
            if receiver is not None and method in _PASSTHROUGH_METHODS:
                return receiver.through(f".{method}()")
            if receiver is not None and method in {"execute", "executemany"}:
                # A cursor is not a value; do not propagate.
                return None
            return None

        return None

    # --- statements -------------------------------------------------------- #

    def run_body(self, body: list[ast.stmt], state: TaintState) -> None:
        for statement in body:
            self.run_statement(statement, state)

    def run_statement(self, node: ast.stmt, state: TaintState) -> None:
        if isinstance(node, ast.Assign):
            self._assign(node.targets, node.value, state)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self._assign([node.target], node.value, state)
        elif isinstance(node, ast.AugAssign):
            incoming = self.taint_of(node.value, state)
            if isinstance(node.target, ast.Name):
                existing = state.get(node.target.id)
                merged = incoming or existing
                if merged is not None:
                    state.set(node.target.id, merged.through(f"{node.target.id} +="))
        elif isinstance(node, ast.Expr | ast.Return):
            # A bare expression statement and a return both just evaluate a value.
            # The evaluation is what records taint on the sub-expressions the rules
            # then ask about, which is why a statement with no target still matters.
            self.taint_of(node.value, state)
        elif isinstance(node, ast.If):
            self.taint_of(node.test, state)
            # An ``if`` with an ``else`` is exhaustive: control goes through exactly
            # one branch, so the state after it is the union of the two branch
            # outcomes and *not* the pre-branch state. Including the parent would
            # model a third path that does not exist, and the visible symptom is a
            # false positive: after ``if h: h = "a"`` / ``else: h = "b"`` the
            # variable is a literal on every path, and a join that kept the parent
            # would still call it attacker-controlled.
            self._branches(
                [node.body, node.orelse], state, exhaustive=bool(node.orelse)
            )
        elif isinstance(node, ast.While):
            self.taint_of(node.test, state)
            self._loop(node.body, state)
            self.run_body(node.orelse, state)
        elif isinstance(node, ast.For | ast.AsyncFor):
            iterated = self.taint_of(node.iter, state)
            self._bind_target(node.target, iterated, state, step="iteration")
            self._loop(node.body, state)
            self.run_body(node.orelse, state)
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                fact = self.taint_of(item.context_expr, state)
                if item.optional_vars is not None:
                    self._bind_target(
                        item.optional_vars, fact, state, step="context manager"
                    )
            self.run_body(node.body, state)
        elif isinstance(node, ast.Try | ast.TryStar):
            self.run_body(node.body, state)
            self._branches(
                [handler.body for handler in node.handlers] + [node.orelse], state
            )
            self.run_body(node.finalbody, state)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            self._function(node, state)
        elif isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                self.taint_of(decorator, state)
            # A class body shares the enclosing facts but its assignments are
            # attributes, not locals, so they are analysed in a child state that is
            # discarded rather than merged.
            self.run_body(node.body, state.copy())
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    state.set(target.id, None)
        elif isinstance(node, ast.Assert):
            self.taint_of(node.test, state)
            self.taint_of(node.msg, state)
        elif isinstance(node, ast.Raise):
            self.taint_of(node.exc, state)
        else:
            # Import, Pass, Break, Continue, Global, Nonlocal — no taint transfer.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.taint_of(child, state)

    def _assign(
        self, targets: list[ast.expr], value: ast.expr, state: TaintState
    ) -> None:
        fact = self.taint_of(value, state)
        for target in targets:
            if (
                isinstance(target, ast.Tuple | ast.List)
                and isinstance(value, ast.Tuple | ast.List)
                and len(target.elts) == len(value.elts)
            ):
                for slot, element in zip(target.elts, value.elts, strict=True):
                    self._bind_target(
                        slot, self.taint_of(element, state), state, step="unpack"
                    )
            else:
                self._bind_target(target, fact, state, step="assignment")

    def _bind_target(
        self,
        target: ast.expr,
        fact: TaintFact | None,
        state: TaintState,
        *,
        step: str,
    ) -> None:
        """Write ``fact`` onto ``target``. A ``None`` fact *kills* existing taint."""
        if isinstance(target, ast.Name):
            state.set(target.id, None if fact is None else fact.through(target.id))
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self._bind_target(element, fact, state, step=step)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, fact, state, step=step)
        # Attribute and Subscript targets (``self.x = tainted``) are not tracked:
        # doing so soundly needs object sensitivity. Documented in the module
        # docstring as part of the intraprocedural boundary.

    def _branches(
        self,
        bodies: list[list[ast.stmt]],
        state: TaintState,
        *,
        exhaustive: bool = False,
    ) -> None:
        """Analyse each branch in its own copy, then join the results.

        ``exhaustive`` says whether the branches cover every path through the
        construct. When they do (an ``if``/``else``), the pre-branch state is
        *replaced* by the union of the outcomes. When they do not — an ``if`` with no
        ``else``, or a ``try`` whose body may have run partway before an except
        handler took over — the pre-branch state is one of the paths and is unioned
        in alongside them.
        """
        outcomes: list[TaintState] = []
        for body in bodies:
            if not body:
                continue
            child = state.copy()
            self.run_body(body, child)
            outcomes.append(child)
        if not outcomes:
            return
        if exhaustive:
            joined = outcomes[0]
            for other in outcomes[1:]:
                joined.merge(other)
            state.facts.clear()
            state.facts.update(joined.facts)
            return
        for outcome in outcomes:
            state.merge(outcome)

    def _loop(self, body: list[ast.stmt], state: TaintState) -> None:
        """Re-analyse until the facts stop growing, capped."""
        for _ in range(LOOP_ITERATION_CAP):
            before = state.signature()
            self.run_body(body, state)
            if state.signature() == before:
                return

    def _function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, state: TaintState
    ) -> None:
        for decorator in node.decorator_list:
            self.taint_of(decorator, state)
        inner = state.copy()
        if _is_externally_reachable(node):
            label = "URL path parameter"
            for argument in _all_args(node.args):
                if argument.arg in {"self", "cls"}:
                    continue
                inner.set(
                    argument.arg,
                    TaintFact(origin=label, path=(label, argument.arg)),
                )
        else:
            # A non-route function's parameters are unknown, not clean. Treating
            # them as tainted would flag every internal helper; treating them as
            # clean misses ``def run(cmd): os.system(cmd)``. The compromise is that
            # the *sink* rules raise confidence on a non-literal argument even
            # without a traced source, so the helper is still reported — at MEDIUM
            # rather than HIGH. See rules.py.
            for argument in _all_args(node.args):
                inner.set(argument.arg, None)
        self.run_body(node.body, inner)


def _all_args(args: ast.arguments) -> list[ast.arg]:
    collected = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg is not None:
        collected.append(args.vararg)
    if args.kwarg is not None:
        collected.append(args.kwarg)
    return collected


def _is_externally_reachable(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when a decorator publishes this function as an HTTP endpoint."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr in _ROUTE_DECORATORS:
            return True
    return False


def _source_label(qualified: str) -> str | None:
    """The taint label for a fully-qualified name, longest prefix first."""
    if qualified in TAINT_SOURCES:
        return TAINT_SOURCES[qualified]
    for source, label in TAINT_SOURCES.items():
        if qualified.startswith(f"{source}."):
            return label
    return None


def analyse_taint(tree: ast.Module, imports: ImportTable) -> dict[int, TaintFact]:
    """Taint facts for every expression in ``tree``, keyed by ``id(node)``.

    Keyed by object identity because ``ast`` nodes are unhashable and have no
    stable identifier of their own. The mapping is only valid while ``tree`` is
    alive, which it is: :class:`~sentinel.scan.analyzer.AstAnalyzer` builds the
    tree, analyses it and discards both together.
    """
    results: dict[int, TaintFact] = {}
    analyzer = _ScopeAnalyzer(imports, results)
    analyzer.run_body(list(tree.body), TaintState())
    return results
