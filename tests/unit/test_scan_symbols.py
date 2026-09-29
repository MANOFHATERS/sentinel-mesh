"""Name resolution, and the boundary it deliberately stops at.

Every rule in :mod:`sentinel.scan.rules` is written against fully-qualified names,
so a resolution bug is not a resolution bug — it is a rule that silently stops
firing. These tests pin each aliasing form that appears in real code, and they pin
the *limits* too, because an undocumented limit becomes an assumed capability.
"""

from __future__ import annotations

import ast

import pytest

from sentinel.scan.symbols import (
    ImportTable,
    dotted_name,
    keyword_of,
    positional_or_keyword,
)


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _table(source: str) -> ImportTable:
    return ImportTable.of(ast.parse(source))


class TestDottedName:
    def test_plain_name(self):
        assert dotted_name(_expr("subprocess")) == "subprocess"

    def test_attribute_chain(self):
        assert dotted_name(_expr("os.path.join")) == "os.path.join"

    def test_deep_chain(self):
        assert dotted_name(_expr("a.b.c.d.e")) == "a.b.c.d.e"

    @pytest.mark.parametrize(
        "source", ["f(x).y", "d['k'].y", "(a + b).c", "[1, 2].append", "'lit'.format"]
    )
    def test_non_static_base_is_none(self, source: str):
        assert dotted_name(_expr(source)) is None


class TestImportTable:
    def test_plain_import(self):
        assert _table("import subprocess").resolve(_expr("subprocess.run")) == (
            "subprocess.run"
        )

    def test_aliased_import(self):
        table = _table("import subprocess as sp")
        assert table.resolve(_expr("sp.call")) == "subprocess.call"

    def test_from_import(self):
        table = _table("from subprocess import call")
        assert table.resolve(_expr("call")) == "subprocess.call"

    def test_from_import_aliased(self):
        table = _table("from subprocess import call as shell_out")
        assert table.resolve(_expr("shell_out")) == "subprocess.call"

    def test_submodule_import_binds_the_top_package(self):
        # ``import os.path`` binds the name ``os``, not ``os.path``.
        table = _table("import os.path")
        assert table.aliases["os"] == "os"
        assert table.resolve(_expr("os.path.join")) == "os.path.join"

    def test_aliased_submodule_import_binds_the_alias(self):
        table = _table("import os.path as p")
        assert table.resolve(_expr("p.join")) == "os.path.join"

    def test_from_import_of_a_submodule_member(self):
        table = _table("from os.path import basename")
        assert table.resolve(_expr("basename")) == "os.path.basename"

    def test_unimported_name_resolves_to_itself(self):
        # A builtin like ``eval`` has no import, and rules key on the bare name.
        assert _table("").resolve(_expr("eval")) == "eval"

    def test_star_import_is_skipped_not_guessed(self):
        table = _table("from subprocess import *")
        assert table.aliases == {}
        assert table.imports_module("subprocess")

    def test_relative_import_without_a_module_is_skipped(self):
        # ``from . import x`` has no absolute path to record.
        assert _table("from . import helpers").aliases == {}

    def test_a_local_name_shadowing_a_module_resolves_to_the_import(self):
        # The over-approximation documented in ImportTable: a later rebinding is not
        # tracked, so the import wins. It can only cause a false positive.
        table = _table("import subprocess\nsubprocess = None")
        assert table.resolve(_expr("subprocess.run")) == "subprocess.run"

    def test_function_local_import_is_still_collected(self):
        table = _table("def f():\n    import subprocess\n    subprocess.run(x)")
        assert table.resolve(_expr("subprocess.run")) == "subprocess.run"

    def test_imports_module_matches_submodules(self):
        table = _table("import urllib.parse")
        assert table.imports_module("urllib")
        assert table.imports_module("urllib.parse")
        assert not table.imports_module("urllib2")

    def test_binds_reports_exact_qualified_targets(self):
        table = _table("from subprocess import run")
        assert table.binds("subprocess.run")
        assert not table.binds("subprocess.call")


class TestDocumentedLimits:
    """The boundary. Each of these is a *known* miss, not a surprise."""

    def test_call_through_a_variable_is_not_resolved(self):
        table = _table("import subprocess\nfn = subprocess.call")
        # ``fn`` is a local binding to a callable, which needs points-to analysis.
        assert table.resolve(_expr("fn")) == "fn"

    def test_a_wrapper_is_not_followed_to_its_call_sites(self):
        source = "import os\ndef sh(c):\n    os.system(c)\nsh('rm -rf /')"
        table = _table(source)
        # The wrapper's own body resolves; the call to ``sh`` does not become
        # ``os.system``. It is caught at the definition instead — see the rules.
        assert table.resolve(_expr("os.system")) == "os.system"
        assert table.resolve(_expr("sh")) == "sh"


class TestArgumentAccess:
    def test_keyword_of_finds_a_named_argument(self):
        call = _expr("f(a, shell=True)")
        assert isinstance(call, ast.Call)
        keyword = keyword_of(call, "shell")
        assert keyword is not None and keyword.value.value is True

    def test_keyword_of_skips_double_star(self):
        call = _expr("f(a, **kwargs)")
        assert isinstance(call, ast.Call)
        assert keyword_of(call, "shell") is None

    def test_positional_or_keyword_reads_the_position(self):
        call = _expr("yaml.load(text)")
        assert isinstance(call, ast.Call)
        found = positional_or_keyword(call, index=0, name="stream")
        assert isinstance(found, ast.Name) and found.id == "text"

    def test_positional_or_keyword_reads_the_keyword_spelling(self):
        call = _expr("yaml.load(stream=text)")
        assert isinstance(call, ast.Call)
        found = positional_or_keyword(call, index=0, name="stream")
        assert isinstance(found, ast.Name) and found.id == "text"

    def test_star_args_at_the_position_is_not_treated_as_the_argument(self):
        # ``f(*argv)`` does not tell us what argument 0 is.
        call = _expr("yaml.load(*args)")
        assert isinstance(call, ast.Call)
        assert positional_or_keyword(call, index=0, name="stream") is None

    def test_missing_argument_is_none(self):
        call = _expr("yaml.load()")
        assert isinstance(call, ast.Call)
        assert positional_or_keyword(call, index=0, name="stream") is None
