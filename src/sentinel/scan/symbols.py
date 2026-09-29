"""Resolving what a name in a Python file actually refers to.

Every rule in :mod:`sentinel.scan.rules` is written against **fully-qualified
dotted names** — ``subprocess.run``, ``yaml.load``, ``hashlib.md5`` — and this
module is what turns the names as they appear in source into those. It exists
because the alternative is the thing that makes most hand-rolled scanners
useless:

.. code-block:: python

    import subprocess as sp
    sp.call(command, shell=True)          # a regex for "subprocess" misses this
    from subprocess import call
    call(command, shell=True)             # and so does an AST rule keyed on Attribute

Both lines are the same vulnerability as ``subprocess.call(cmd, shell=True)``, and
an analyzer that reports one and not the others is worse than no analyzer, because
a developer who sees the obvious spelling flagged reasonably concludes the others
are fine.

What is deliberately *not* done
-------------------------------
No cross-module resolution, no type inference, no following ``__init__.py``
re-exports. A wrapper (``def sh(c): subprocess.call(c, shell=True)``) is caught at
its definition, not at its call sites, and a call through a variable
(``fn = subprocess.call; fn(...)``) is not resolved at all. Those need a call graph
and a points-to analysis, which is a different project; the boundary is stated here
and pinned by ``test_scan_symbols.py`` so it is a known limit rather than an
assumed capability.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

__all__ = [
    "ImportTable",
    "dotted_name",
    "keyword_of",
    "positional_or_keyword",
]


def dotted_name(node: ast.AST) -> str | None:
    """The dotted spelling of an attribute/name chain, or ``None``.

    ``os.path.join`` -> ``"os.path.join"``; ``f(x).y`` -> ``None``, because the
    base is a call and there is no static name to report.
    """
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


@dataclass(slots=True)
class ImportTable:
    """Local name -> fully-qualified module path, built from a module's imports.

    Collected over the whole module rather than per-scope. A function-local
    ``import subprocess`` would strictly only bind inside that function, but a
    security scanner that reported nothing because the import was three lines lower
    than the tracker expected would be broken in the direction that matters. The
    over-approximation can only cause a *rule to fire* on a name that happens to
    match, which surfaces as a reviewable false positive rather than a miss.
    """

    #: ``{"sp": "subprocess", "run": "subprocess.run"}``
    aliases: dict[str, str] = field(default_factory=dict)
    #: Modules imported at all, for rules that care whether a library is in use.
    modules: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, tree: ast.AST) -> ImportTable:
        table = cls()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # ``import os.path`` binds ``os``; ``import os.path as p`` binds ``p``.
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    target = alias.name if alias.asname else bound
                    table.aliases[bound] = target
                    table.modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:  # ``from . import x`` — no absolute path
                    continue
                table.modules.add(node.module)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    table.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        return table

    def resolve(self, node: ast.AST) -> str | None:
        """Fully-qualified name for an attribute/name chain, applying aliases."""
        raw = dotted_name(node)
        if raw is None:
            return None
        head, _, rest = raw.partition(".")
        target = self.aliases.get(head)
        if target is None:
            return raw
        return f"{target}.{rest}" if rest else target

    def imports_module(self, module: str) -> bool:
        """True when ``module`` (or a submodule of it) was imported."""
        return any(name == module or name.startswith(f"{module}.") for name in self.modules)

    def binds(self, qualified: str) -> bool:
        """True when some local name resolves to exactly ``qualified``."""
        return qualified in self.aliases.values()


def keyword_of(call: ast.Call, name: str) -> ast.keyword | None:
    """The named keyword argument of ``call``, or ``None``. ``**kwargs`` is skipped."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def positional_or_keyword(
    call: ast.Call, *, index: int, name: str
) -> ast.expr | None:
    """The argument at ``index``, or the one spelled ``name=``.

    Both spellings, because ``yaml.load(text)`` and ``yaml.load(stream=text)`` are
    the same call and a rule that only understands one of them is a rule an
    attacker can step around by reformatting.
    """
    if index < len(call.args):
        argument = call.args[index]
        if not isinstance(argument, ast.Starred):
            return argument
    kw = keyword_of(call, name)
    return None if kw is None else kw.value
