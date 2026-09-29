"""The taint analysis: what propagates, what kills, what a sanitizer clears.

This is the module that decides whether a finding is a vulnerability or noise, so
the tests are organised around the two ways it can be wrong. Missing propagation
means a missed vulnerability; spurious propagation means a false positive on correct
code. Both directions are asserted for every transfer rule.

The per-rule sanitizer semantics get their own class, because the single most
tempting simplification here — one boolean "sanitised" flag — is wrong in a way that
only shows up when two rules disagree about the same call.
"""

from __future__ import annotations

import ast

import pytest

from sentinel.scan.rules import RULES_BY_ID
from sentinel.scan.symbols import ImportTable
from sentinel.scan.taint import (
    LOOP_ITERATION_CAP,
    SANITIZERS,
    TAINT_SOURCES,
    TaintFact,
    TaintState,
    analyse_taint,
)

CMD_RULE = "python.os-system-injection"
SQL_RULE = "python.sql-injection"
PATH_RULE = "python.path-traversal"

PREAMBLE = """\
import os
import shlex
import sys
from flask import Flask, request

app = Flask(__name__)
"""


def _sink_taint(body: str, *, rule_id: str = CMD_RULE, sink: str = "os.system"):
    """Taint on the first argument of the first ``sink`` call in ``body``."""
    tree = ast.parse(PREAMBLE + body)
    imports = ImportTable.of(tree)
    facts = analyse_taint(tree, imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and imports.resolve(node.func) == sink:
            fact = facts.get(id(node.args[0])) if node.args else None
            if fact is None:
                return None
            return fact if fact.is_live_for(rule_id) else None
    raise AssertionError(f"no {sink} call found in the fixture")


def _route(body: str) -> str:
    indented = "\n".join(f"    {line}" for line in body.strip().splitlines())
    return f'@app.route("/x")\ndef handler():\n{indented}\n'


class TestSources:
    @pytest.mark.parametrize(
        "expression",
        [
            'request.args.get("h")',
            'request.args["h"]',
            'request.form["h"]',
            'request.values["h"]',
            "request.json",
            "request.get_json()",
            "request.data",
            'request.cookies["h"]',
            'request.headers["X-H"]',
            'request.view_args["h"]',
            "sys.argv[1]",
            'input("host: ")',
        ],
    )
    def test_every_documented_source_taints(self, expression: str):
        assert _sink_taint(_route(f"os.system({expression})")) is not None

    def test_route_parameters_are_tainted(self):
        body = (
            '@app.route("/u/<name>")\n'
            "def show(name):\n"
            '    os.system("id " + name)\n'
        )
        fact = _sink_taint(body)
        assert fact is not None
        assert fact.origin == "URL path parameter"

    def test_a_non_route_function_parameter_is_not_traced(self):
        # Documented boundary: caught by the sink rule at MEDIUM, not by taint.
        body = "def run(cmd):\n    os.system(cmd)\n"
        assert _sink_taint(body) is None

    def test_os_environ_is_deliberately_not_a_source(self):
        # Flagging config reads would make the scanner cry wolf on every correct
        # twelve-factor app. The exclusion is asserted so it stays a decision.
        assert "os.environ" not in TAINT_SOURCES
        assert "os.getenv" not in TAINT_SOURCES
        body = 'os.system(os.environ["CMD"])\n'
        assert _sink_taint(body) is None

    def test_a_plain_literal_is_not_tainted(self):
        assert _sink_taint('os.system("uptime")\n') is None


class TestPropagation:
    @pytest.mark.parametrize(
        "body",
        [
            'h = request.args["h"]\nos.system("ping " + h)',
            'h = request.args["h"]\nos.system(f"ping {h}")',
            'h = request.args["h"]\nos.system("ping %s" % h)',
            'h = request.args["h"]\nos.system("ping {}".format(h))',
            'h = request.args["h"]\nos.system(" ".join(["ping", h]))',
            'h = request.args["h"]\nos.system(str(h))',
            'h = request.args["h"]\nos.system(h.strip().lower())',
            'h = request.args["h"]\ng = h\nos.system(g)',
            'h = request.args["h"]\nos.system(h if h else "x")',
            'h = request.args["h"]\nos.system("x" if False else h)',
            'h = request.args["h"]\nd = {"c": h}\nos.system(d["c"] or h)',
            'h = request.args["h"]\nc = ""\nc += h\nos.system(c)',
            'for h in request.args.getlist("h"):\n    os.system("ping " + h)',
            'h = request.args["h"]\ntry:\n    pass\nexcept Exception:\n    pass\nos.system(h)',
        ],
    )
    def test_taint_reaches_the_sink(self, body: str):
        assert _sink_taint(_route(body)) is not None

    def test_tuple_unpacking_is_positional(self):
        body = 'a, b = "safe", request.args["h"]\nos.system(b)'
        assert _sink_taint(_route(body)) is not None

    def test_tuple_unpacking_does_not_taint_the_clean_slot(self):
        body = 'a, b = "safe", request.args["h"]\nos.system(a)'
        assert _sink_taint(_route(body)) is None

    def test_the_path_records_how_the_value_arrived(self):
        fact = _sink_taint(_route('h = request.args["h"]\nos.system("ping " + h)'))
        assert fact is not None
        assert fact.path[0] == "HTTP query string"
        assert "concatenation" in fact.path
        # The chain names the variable, which is what makes a finding's taint_path
        # readable to a developer rather than an internal detail.
        assert "h" in fact.path

    def test_a_dict_lookup_keyed_by_untrusted_data_is_not_tainted(self):
        # ``COMMANDS[user_choice]`` is the *dispatch table* remedy, so treating its
        # result as tainted would flag the recommended fix.
        body = (
            'COMMANDS = {"a": "uptime"}\n'
            'choice = request.args["c"]\n'
            "os.system(COMMANDS[choice])"
        )
        assert _sink_taint(_route(body)) is None


class TestKills:
    def test_reassignment_to_a_literal_kills_taint(self):
        body = 'h = request.args["h"]\nh = "localhost"\nos.system("ping " + h)'
        assert _sink_taint(_route(body)) is None

    def test_del_kills_taint(self):
        body = 'h = request.args["h"]\nh = "x"\ndel h\nh = "safe"\nos.system(h)'
        assert _sink_taint(_route(body)) is None

    def test_a_branch_that_cleans_one_path_does_not_clean_the_join(self):
        # May-analysis: tainted on either branch means tainted after.
        body = (
            'h = request.args["h"]\n'
            'if h == "x":\n'
            '    h = "safe"\n'
            'os.system("ping " + h)'
        )
        assert _sink_taint(_route(body)) is not None

    def test_a_branch_that_cleans_every_path_does_clean_the_join(self):
        body = (
            'h = request.args["h"]\n'
            'if h:\n'
            '    h = "a"\n'
            "else:\n"
            '    h = "b"\n'
            'os.system("ping " + h)'
        )
        assert _sink_taint(_route(body)) is None


class TestSanitizers:
    def test_shlex_quote_clears_a_command_injection(self):
        body = 'h = request.args["h"]\nos.system("ping " + shlex.quote(h))'
        assert _sink_taint(_route(body), rule_id=CMD_RULE) is None

    def test_shlex_quote_does_not_clear_a_sql_injection(self):
        # The crux of per-rule sanitizers. shlex.quote adds shell quoting; a SQL
        # parser has never heard of it. A single boolean flag would get this wrong.
        body = 'h = request.args["h"]\nos.system("ping " + shlex.quote(h))'
        assert _sink_taint(_route(body), rule_id=SQL_RULE) is not None

    def test_basename_clears_a_path_traversal_only(self):
        body = 'h = request.args["h"]\nos.system(os.path.basename(h))'
        assert _sink_taint(_route(body), rule_id=PATH_RULE) is None
        assert _sink_taint(_route(body), rule_id=CMD_RULE) is not None

    def test_int_clears_every_rule(self):
        body = 'h = request.args["h"]\nos.system(str(int(h)))'
        for rule_id in (CMD_RULE, SQL_RULE, PATH_RULE):
            assert _sink_taint(_route(body), rule_id=rule_id) is None

    def test_normpath_does_not_sanitise_a_traversal(self):
        # The trap this test exists for: normpath resolves a path textually and
        # removes nothing. ``normpath("../../etc/passwd")`` is unchanged.
        body = 'h = request.args["h"]\nos.system(os.path.normpath(h))'
        assert _sink_taint(_route(body), rule_id=PATH_RULE) is not None

    def test_join_propagates_taint(self):
        body = 'h = request.args["h"]\nos.system(os.path.join("/srv", h))'
        assert _sink_taint(_route(body), rule_id=PATH_RULE) is not None

    def test_merging_a_clean_branch_with_a_dirty_one_keeps_it_dirty(self):
        body = (
            'h = request.args["h"]\n'
            "if h:\n"
            "    h = shlex.quote(h)\n"
            'os.system("ping " + h)'
        )
        assert _sink_taint(_route(body), rule_id=CMD_RULE) is not None

    def test_every_sanitizer_names_real_rules(self):
        # A renamed rule would otherwise turn a sanitizer into a silent no-op, whose
        # symptom is a false positive on code that was correctly fixed.
        for sanitizer in SANITIZERS:
            if sanitizer.clears is None:
                continue
            unknown = sorted(set(sanitizer.clears) - set(RULES_BY_ID))
            assert not unknown, f"{sanitizer.qualified} names unknown rules {unknown}"


class TestLoops:
    def test_accumulation_across_a_loop_is_caught(self):
        body = (
            'cmd = "ping"\n'
            'for h in request.args.getlist("h"):\n'
            '    cmd = cmd + " " + h\n'
            "os.system(cmd)"
        )
        assert _sink_taint(_route(body)) is not None

    def test_a_while_loop_reaches_a_fixed_point(self):
        body = (
            'cmd = "ping"\n'
            "while True:\n"
            '    cmd = cmd + request.args["h"]\n'
            "    break\n"
            "os.system(cmd)"
        )
        assert _sink_taint(_route(body)) is not None

    def test_the_cap_is_a_termination_guard_not_a_correctness_knob(self):
        # The lattice is finite and every transfer is monotone, so the iteration
        # terminates on its own well before the cap. The cap exists so pathological
        # input cannot hang a scan.
        assert LOOP_ITERATION_CAP >= 2


class TestScopes:
    def test_a_nested_function_reads_the_enclosing_scope(self):
        body = (
            '@app.route("/x")\n'
            "def outer():\n"
            '    h = request.args["h"]\n'
            "    def inner():\n"
            '        os.system("ping " + h)\n'
            "    inner()\n"
        )
        assert _sink_taint(body) is not None

    def test_a_class_body_does_not_leak_into_the_enclosing_scope(self):
        body = (
            '@app.route("/x")\n'
            "def outer():\n"
            "    class C:\n"
            '        h = request.args["h"]\n'
            '    os.system("ping " + str(C))\n'
        )
        assert _sink_taint(body) is None

    def test_a_route_handler_does_not_taint_a_sibling_function(self):
        body = (
            '@app.route("/u/<name>")\n'
            "def show(name):\n"
            "    return name\n"
            "\n"
            "def helper(name):\n"
            '    os.system("id " + name)\n'
        )
        assert _sink_taint(body) is None


class TestTaintState:
    def test_merge_takes_the_weaker_clearance(self):
        dirty = TaintFact(origin="src", path=("src",))
        clean = TaintFact(origin="src", path=("src",), cleared_for=frozenset({CMD_RULE}))
        left, right = TaintState({"x": dirty}), TaintState({"x": clean})
        left.merge(right)
        merged = left.get("x")
        assert merged is not None and merged.is_live_for(CMD_RULE)

    def test_merge_keeps_a_clearance_both_sides_agree_on(self):
        clean = TaintFact(origin="s", path=("s",), cleared_for=frozenset({CMD_RULE}))
        left, right = TaintState({"x": clean}), TaintState({"x": clean})
        left.merge(right)
        merged = left.get("x")
        assert merged is not None and not merged.is_live_for(CMD_RULE)

    def test_setting_none_removes_the_fact(self):
        state = TaintState({"x": TaintFact(origin="s", path=("s",))})
        state.set("x", None)
        assert state.get("x") is None

    def test_through_does_not_repeat_the_same_step(self):
        fact = TaintFact(origin="s", path=("s", "x"))
        assert fact.through("x").path == ("s", "x")

    def test_signature_is_comparable(self):
        state = TaintState({"x": TaintFact(origin="s", path=("s",))})
        assert state.signature() == state.copy().signature()
