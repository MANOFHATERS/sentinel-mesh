"""Every rule, in both directions, plus the fix it generates.

Each rule gets three kinds of test and all three are load-bearing:

*   **Fires** on the vulnerable form, including the aliased spellings a regex-based
    scanner misses.
*   **Does not fire** on the correct form. This is the half that decides whether the
    tool is usable: a rule that flags parameterised SQL has told the developer that
    the fix is the bug.
*   **The fix is right.** Not merely that a diff exists — the patched source is
    asserted to parse, and for the interesting rewrites (SQL parameterisation, path
    containment) the exact resulting text is asserted, because "a valid diff that
    does the wrong thing" is the failure mode a patch-generating scanner has.
"""

from __future__ import annotations

import ast

import pytest

from sentinel.core.schemas import Severity
from sentinel.scan.analyzer import AstAnalyzer
from sentinel.scan.findings import FindingConfidence, ScanError
from sentinel.scan.repo import RepoSnapshot
from sentinel.scan.rules import RULES, RULES_BY_ID, Rule, rule_ids

ANALYZER = AstAnalyzer()


def _scan(source: str, path: str = "m.py"):
    return ANALYZER.scan(RepoSnapshot.of_texts({path: source}))


def _rules_fired(source: str) -> set[str]:
    return {finding.rule_id for finding in _scan(source).findings}


def _one(source: str, rule_id: str):
    matches = [f for f in _scan(source).findings if f.rule_id == rule_id]
    assert matches, f"{rule_id} did not fire on:\n{source}"
    assert len(matches) == 1, f"{rule_id} fired {len(matches)} times"
    return matches[0]


def _patch_for(source: str, rule_id: str):
    result = _scan(source)
    patches = [p for p in result.valid_patches if p.rule_id == rule_id]
    assert patches, (
        f"no validated patch for {rule_id}; rejected="
        f"{[(p.rule_id, p.rejection) for p in result.rejected_patches]}"
    )
    return patches[0]


ROUTE = """\
import os
import shlex
import sqlite3
import subprocess
import hashlib
import tempfile
import yaml
import pickle
import requests
from jinja2 import Environment, FileSystemLoader
from flask import Flask, request

app = Flask(__name__)
BASE = "/srv/data"


@app.route("/x")
def handler():
{body}
"""


def _route(body: str) -> str:
    return ROUTE.format(body="\n".join(f"    {line}" for line in body.splitlines()))


# --------------------------------------------------------------------------- #
# Catalogue invariants
# --------------------------------------------------------------------------- #


class TestCatalogue:
    def test_rule_ids_are_unique(self):
        assert len(RULES_BY_ID) == len(RULES)

    def test_every_rule_is_namespaced_by_language(self):
        assert all(rule_id.startswith("python.") for rule_id in rule_ids())

    def test_every_rule_names_a_cwe(self):
        for rule in RULES:
            assert rule.cwe.startswith("CWE-")

    def test_every_rule_explains_its_remediation(self):
        for rule in RULES:
            assert len(rule.remediation) > 20, rule.rule_id

    def test_a_rule_without_a_cwe_is_refused(self):
        with pytest.raises(ScanError, match="CWE"):
            Rule(
                rule_id="python.x",
                cwe="not-a-cwe",
                title="t",
                severity=Severity.LOW,
                message="m",
                remediation="r" * 25,
                matcher=lambda node, ctx: None,
            )

    def test_an_unnamespaced_rule_is_refused(self):
        with pytest.raises(ScanError, match="namespaced"):
            Rule(
                rule_id="bare",
                cwe="CWE-1",
                title="t",
                severity=Severity.LOW,
                message="m",
                remediation="r" * 25,
                matcher=lambda node, ctx: None,
            )

    def test_exactly_the_documented_rules_lack_a_fix(self):
        # pickle has no safe loader, so there is nothing mechanical to write. Any
        # *other* rule losing its fix should be a deliberate, visible change.
        assert {r.rule_id for r in RULES if not r.has_fix} == {
            "python.pickle-deserialization"
        }

    def test_a_clean_file_produces_nothing(self):
        source = (
            "import os\n"
            "import shlex\n"
            "import subprocess\n"
            "\n"
            "def run(command: list[str]) -> int:\n"
            "    return subprocess.run(command, check=False).returncode\n"
        )
        assert _rules_fired(source) == set()


# --------------------------------------------------------------------------- #
# CWE-78
# --------------------------------------------------------------------------- #


class TestCommandInjection:
    RULE = "python.os-command-injection"

    @pytest.mark.parametrize(
        "call",
        [
            'subprocess.run("ping " + h, shell=True)',
            'subprocess.call("ping " + h, shell=True)',
            'subprocess.check_call("ping " + h, shell=True)',
            'subprocess.check_output("ping " + h, shell=True)',
            'subprocess.Popen("ping " + h, shell=True)',
            'subprocess.run(shell=True, args="ping " + h)',
        ],
    )
    def test_fires_on_every_subprocess_entry_point(self, call: str):
        assert self.RULE in _rules_fired(_route(f'h = request.args["h"]\n{call}'))

    @pytest.mark.parametrize(
        ("imports", "call"),
        [
            ("import subprocess as sp", 'sp.call("ping " + h, shell=True)'),
            ("from subprocess import call", 'call("ping " + h, shell=True)'),
            ("from subprocess import run as r", 'r("ping " + h, shell=True)'),
        ],
    )
    def test_fires_through_import_aliases(self, imports: str, call: str):
        source = (
            f"{imports}\nfrom flask import Flask, request\n"
            'app = Flask(__name__)\n\n@app.route("/x")\ndef h2():\n'
            f'    h = request.args["h"]\n    {call}\n'
        )
        assert self.RULE in _rules_fired(source)

    def test_taint_is_recorded_on_the_finding(self):
        finding = _one(
            _route('h = request.args["h"]\nsubprocess.run("p " + h, shell=True)'),
            self.RULE,
        )
        assert finding.confidence is FindingConfidence.HIGH
        assert finding.taint_path[0] == "HTTP query string"

    def test_does_not_fire_on_an_argv_list(self):
        source = _route('h = request.args["h"]\nsubprocess.run(["ping", h])')
        assert self.RULE not in _rules_fired(source)

    def test_does_not_fire_when_shlex_quote_was_applied(self):
        source = _route(
            'h = request.args["h"]\n'
            'cmd = "ping " + shlex.quote(h)\n'
            "subprocess.check_output(cmd, shell=True)"
        )
        assert self.RULE not in _rules_fired(source)

    def test_a_helper_parameter_is_medium_not_dropped(self):
        source = (
            "import subprocess\n"
            "def sh(cmd):\n"
            "    return subprocess.run(cmd, shell=True)\n"
        )
        finding = _one(source, self.RULE)
        assert finding.confidence is FindingConfidence.MEDIUM

    def test_a_literal_command_is_low(self):
        source = 'import subprocess\nsubprocess.run("uptime", shell=True)\n'
        assert _one(source, self.RULE).confidence is FindingConfidence.LOW

    def test_the_fix_drops_the_shell_and_splits_the_command(self):
        patch = _patch_for(
            _route('h = request.args["h"]\nsubprocess.run("ping " + h, shell=True)'),
            self.RULE,
        )
        assert "shell=False" in patch.after
        assert 'shlex.split("ping " + h)' in patch.after
        assert "import shlex" in patch.after
        ast.parse(patch.after)

    def test_the_fix_on_a_list_only_drops_the_shell(self):
        source = _route('h = request.args["h"]\nsubprocess.run([h], shell=True)')
        patch = _patch_for(source, self.RULE)
        assert "shell=False" in patch.after
        assert "shlex.split" not in patch.after


class TestOsSystem:
    RULE = "python.os-system-injection"

    @pytest.mark.parametrize("call", ["os.system(c)", "os.popen(c)"])
    def test_fires_on_both_entry_points(self, call: str):
        assert self.RULE in _rules_fired(_route(f'c = request.args["c"]\n{call}'))

    def test_the_fix_rewrites_os_system_to_subprocess(self):
        patch = _patch_for(_route('c = request.args["c"]\nos.system(c)'), self.RULE)
        assert "subprocess.run(shlex.split(c), check=False).returncode" in patch.after
        ast.parse(patch.after)

    def test_os_popen_is_reported_without_a_patch(self):
        # popen returns a file object and its callers read from it, so rewriting to
        # subprocess.run would change the return type and break the call site.
        result = _scan(_route('c = request.args["c"]\nos.popen(c)'))
        assert any(f.rule_id == self.RULE for f in result.findings)
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]


# --------------------------------------------------------------------------- #
# CWE-89
# --------------------------------------------------------------------------- #


class TestSqlInjection:
    RULE = "python.sql-injection"

    @pytest.mark.parametrize(
        "statement",
        [
            'cur.execute(f"SELECT * FROM t WHERE a = \'{n}\'")',
            'cur.execute("SELECT * FROM t WHERE a = \'%s\'" % n)',
            'cur.execute("SELECT * FROM t WHERE a = \'{}\'".format(n))',
            'cur.execute("SELECT * FROM t WHERE a = \'" + n + "\'")',
            'cur.executemany(f"INSERT INTO t VALUES (\'{n}\')")',
        ],
    )
    def test_fires_on_every_formatting_style(self, statement: str):
        body = f'n = request.args["n"]\ncur = sqlite3.connect(":memory:").cursor()\n{statement}'
        assert self.RULE in _rules_fired(_route(body))

    @pytest.mark.parametrize(
        "statement",
        [
            'cur.execute("SELECT * FROM t WHERE a = ?", (n,))',
            'cur.execute("SELECT * FROM t")',
            'cur.execute(QUERY, (n,))',
        ],
    )
    def test_does_not_fire_on_bound_parameters(self, statement: str):
        body = (
            'n = request.args["n"]\nQUERY = "SELECT 1"\n'
            'cur = sqlite3.connect(":memory:").cursor()\n' + statement
        )
        assert self.RULE not in _rules_fired(_route(body))

    def test_does_not_fire_on_a_module_constant_query(self):
        source = (
            "import sqlite3\n"
            'QUERY = "SELECT id FROM users"\n'
            "def load(cur):\n"
            "    cur.execute(QUERY)\n"
        )
        assert self.RULE not in _rules_fired(source)

    def test_the_fstring_fix_binds_and_strips_the_sql_quotes(self):
        # The quote stripping is the part that matters: leaving them in place turns
        # each parameter into the literal two-character string "?" and the query
        # silently returns nothing while looking fixed.
        body = (
            'n = request.args["n"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            "cur.execute(f\"SELECT * FROM t WHERE a = '{n}'\")"
        )
        patch = _patch_for(_route(body), self.RULE)
        assert 'cur.execute("SELECT * FROM t WHERE a = ?", (n,))' in patch.after
        ast.parse(patch.after)

    def test_the_percent_fix_binds_in_order(self):
        body = (
            'a = request.args["a"]\n'
            'b = request.args["b"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            "cur.execute(\"SELECT * FROM t WHERE a='%s' AND b='%s'\" % (a, b))"
        )
        patch = _patch_for(_route(body), self.RULE)
        assert '"SELECT * FROM t WHERE a=? AND b=?", (a, b)' in patch.after

    def test_the_concatenation_fix_binds_each_fragment(self):
        body = (
            'n = request.args["n"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            'cur.execute("SELECT * FROM t WHERE a = \'" + n + "\'")'
        )
        patch = _patch_for(_route(body), self.RULE)
        assert '"SELECT * FROM t WHERE a = ?", (n,)' in patch.after

    def test_a_conversion_or_format_spec_declines_the_fix(self):
        # ``{n!r}`` and ``{n:>10}`` change the rendered text, so a placeholder is not
        # an equivalent substitution and the rule reports without a patch.
        body = (
            'n = request.args["n"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            'cur.execute(f"SELECT * FROM t WHERE a = {n!r}")'
        )
        result = _scan(_route(body))
        assert any(f.rule_id == self.RULE for f in result.findings)
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]

    def test_a_like_pattern_declines_the_fix(self):
        # ``LIKE '{term}%'`` yields the template ``LIKE '?%'``. Stripping only the
        # quotes around the ``?`` leaves a search for the literal string "?%" — a
        # patch that validates, removes the finding, and breaks the search. The real
        # fix moves the wildcard into the bound value, which is a change to the value
        # expression and not something to guess at, so the rule declines.
        body = (
            'q = request.args["q"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            'cur.execute(f"SELECT * FROM t WHERE name LIKE \'{q}%\'")'
        )
        result = _scan(_route(body))
        assert any(f.rule_id == self.RULE for f in result.findings)
        patches = [p for p in result.valid_patches if p.rule_id == self.RULE]
        assert not patches
        # And nothing was produced-then-rejected either: declining happens before a
        # proposal exists, so no half-built patch is carried around.
        assert not [p for p in result.rejected_patches if p.rule_id == self.RULE]

    @pytest.mark.parametrize(
        "template",
        [
            "SELECT * FROM t WHERE a LIKE '%{n}%'",
            "SELECT * FROM t WHERE a = 'prefix-{n}'",
            "SELECT * FROM t WHERE a = 'x{n}y'",
        ],
    )
    def test_a_placeholder_sharing_a_quoted_region_declines(self, template: str):
        body = (
            'n = request.args["n"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            f'cur.execute(f"{template}")'
        )
        result = _scan(_route(body))
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]

    def test_a_placeholder_beside_a_literal_in_the_same_list_is_still_bound(self):
        # ``IN ('{n}', 'x')`` becomes ``IN (?, 'x')``, which is correct: each quoted
        # region is considered on its own, and this one *is* exactly a placeholder.
        # The neighbouring literal keeps its quotes and is unaffected.
        body = (
            'n = request.args["n"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            'cur.execute(f"SELECT * FROM t WHERE a IN (\'{n}\', \'x\')")'
        )
        patch = _patch_for(_route(body), self.RULE)
        assert "\"SELECT * FROM t WHERE a IN (?, 'x')\", (n,)" in patch.after

    def test_an_unbalanced_quote_declines(self):
        body = (
            'n = request.args["n"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            'cur.execute(f"SELECT * FROM t WHERE a = \'{n}")'
        )
        result = _scan(_route(body))
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]

    def test_an_already_parameterised_call_declines_the_fix(self):
        body = (
            'n = request.args["n"]\n'
            'cur = sqlite3.connect(":memory:").cursor()\n'
            'cur.execute(f"SELECT * FROM t WHERE a = \'{n}\' AND b = ?", (1,))'
        )
        result = _scan(_route(body))
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]


# --------------------------------------------------------------------------- #
# CWE-502
# --------------------------------------------------------------------------- #


class TestUnsafeDeserialization:
    def test_yaml_load_fires_and_is_fixed(self):
        body = 'data = yaml.load(request.data)'
        patch = _patch_for(_route(body), "python.yaml-unsafe-load")
        assert "yaml.safe_load(request.data)" in patch.after
        ast.parse(patch.after)

    @pytest.mark.parametrize(
        "loader", ["yaml.SafeLoader", "yaml.CSafeLoader", "yaml.BaseLoader"]
    )
    def test_a_safe_loader_does_not_fire(self, loader: str):
        body = f"data = yaml.load(request.data, Loader={loader})"
        assert "python.yaml-unsafe-load" not in _rules_fired(_route(body))

    def test_safe_load_does_not_fire(self):
        assert "python.yaml-unsafe-load" not in _rules_fired(
            _route("data = yaml.safe_load(request.data)")
        )

    def test_the_fix_keeps_other_keyword_arguments(self):
        source = "import yaml\ndef f(t):\n    return yaml.load(t, Loader=yaml.Loader)\n"
        patch = _patch_for(source, "python.yaml-unsafe-load")
        assert "yaml.safe_load(t)" in patch.after

    def test_yaml_load_on_a_literal_is_still_medium(self):
        # The danger is the loader, not the input, so this never drops to LOW.
        source = 'import yaml\nyaml.load("a: 1")\n'
        finding = _one(source, "python.yaml-unsafe-load")
        assert finding.confidence.rank >= FindingConfidence.MEDIUM.rank

    def test_pickle_fires_with_no_patch_ever(self):
        body = "obj = pickle.loads(request.data)"
        result = _scan(_route(body))
        assert any(
            f.rule_id == "python.pickle-deserialization" for f in result.findings
        )
        assert not result.valid_patches or all(
            p.rule_id != "python.pickle-deserialization" for p in result.valid_patches
        )

    def test_pickles_remediation_says_it_is_not_mechanical(self):
        rule = RULES_BY_ID["python.pickle-deserialization"]
        assert "not mechanically patchable" in rule.remediation


# --------------------------------------------------------------------------- #
# CWE-327, CWE-798
# --------------------------------------------------------------------------- #


class TestWeakHash:
    RULE = "python.weak-hash"

    @pytest.mark.parametrize("algorithm", ["md5", "sha1"])
    def test_fires_on_broken_algorithms(self, algorithm: str):
        source = f"import hashlib\ndef d(p):\n    return hashlib.{algorithm}(p)\n"
        assert self.RULE in _rules_fired(source)

    @pytest.mark.parametrize("algorithm", ["sha256", "sha512", "blake2b"])
    def test_does_not_fire_on_modern_algorithms(self, algorithm: str):
        source = f"import hashlib\ndef d(p):\n    return hashlib.{algorithm}(p)\n"
        assert self.RULE not in _rules_fired(source)

    def test_usedforsecurity_false_is_honoured(self):
        # Python 3.9 added this keyword precisely so a cache key can be spelled
        # unambiguously. Ignoring it is how a rule earns a blanket suppression.
        source = (
            "import hashlib\ndef k(p):\n"
            "    return hashlib.md5(p, usedforsecurity=False)\n"
        )
        assert self.RULE not in _rules_fired(source)

    def test_hashlib_new_with_a_weak_name_fires_and_is_fixed(self):
        source = 'import hashlib\ndef d(p):\n    return hashlib.new("md5", p)\n'
        patch = _patch_for(source, self.RULE)
        assert 'hashlib.new("sha256", p)' in patch.after

    def test_the_fix_upgrades_the_attribute(self):
        source = "import hashlib\ndef d(p):\n    return hashlib.md5(p).hexdigest()\n"
        patch = _patch_for(source, self.RULE)
        assert "hashlib.sha256(p).hexdigest()" in patch.after
        ast.parse(patch.after)

    def test_a_from_import_call_site_reports_without_a_patch(self):
        # Rewriting the bare name would leave it undefined; changing the import too
        # is a wider edit than one finding should make.
        source = "from hashlib import md5\ndef d(p):\n    return md5(p)\n"
        result = _scan(source)
        assert any(f.rule_id == self.RULE for f in result.findings)
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]


class TestHardcodedCredential:
    RULE = "python.hardcoded-credential"

    @pytest.mark.parametrize(
        "name",
        [
            "DB_PASSWORD",
            "api_key",
            "API_KEY",
            "secret_token",
            "AWS_ACCESS_KEY",
            "private_key",
            "signing_key",
            "passwd",
        ],
    )
    def test_fires_on_secret_shaped_names(self, name: str):
        assert self.RULE in _rules_fired(f'{name} = "Pr0d-Value-9182"\n')

    @pytest.mark.parametrize(
        "value",
        ['""', '"changeme"', '"xxx"', '"placeholder"', '"${VAULT_PATH}"',
         '"/etc/ssl/private/app.pem"', '"https://vault.example.com/k"', '"test"'],
    )
    def test_does_not_fire_on_placeholders_or_references(self, value: str):
        assert self.RULE not in _rules_fired(f"API_KEY = {value}\n")

    def test_does_not_fire_on_an_environment_read(self):
        source = 'import os\nSECRET = os.environ["SECRET"]\n'
        assert self.RULE not in _rules_fired(source)

    def test_does_not_fire_on_an_unrelated_name(self):
        assert self.RULE not in _rules_fired('GREETING = "hello world"\n')

    def test_the_fix_reads_the_environment_and_imports_os(self):
        patch = _patch_for('DB_PASSWORD = "Pr0d-Postgres-2024!"\n', self.RULE)
        assert 'DB_PASSWORD = os.environ["DB_PASSWORD"]' in patch.after
        assert "import os" in patch.after
        ast.parse(patch.after)

    def test_the_fix_says_the_secret_still_needs_rotating(self):
        # The literal is in git history forever; a patch that implies otherwise is
        # worse than no patch.
        patch = _patch_for('API_KEY = "fixture-only-billing"\n', self.RULE)
        assert any("rotate" in note for note in patch.notes)

    def test_an_annotated_assignment_also_fires(self):
        assert self.RULE in _rules_fired('API_KEY: str = "fixture-only-billing"\n')

    def test_a_short_value_does_not_fire(self):
        assert self.RULE not in _rules_fired('PASSWORD = "abc"\n')


# --------------------------------------------------------------------------- #
# Configuration rules
# --------------------------------------------------------------------------- #


class TestConfigurationRules:
    def test_flask_debug_fires_and_is_fixed(self):
        source = "from flask import Flask\napp = Flask(__name__)\napp.run(debug=True)\n"
        patch = _patch_for(source, "python.flask-debug-enabled")
        assert "debug=False" in patch.after

    def test_debug_false_does_not_fire(self):
        source = "from flask import Flask\napp = Flask(__name__)\napp.run(debug=False)\n"
        assert "python.flask-debug-enabled" not in _rules_fired(source)

    def test_bind_all_interfaces_is_high_confidence_and_low_severity(self):
        # Confidence and severity answer different questions. The pattern is certain;
        # the impact is small. Grading confidence LOW would put a one-word fix below
        # the patch threshold for a finding the analyzer is completely sure about.
        source = (
            "from flask import Flask\napp = Flask(__name__)\n"
            'app.run(host="0.0.0.0")\n'
        )
        finding = _one(source, "python.bind-all-interfaces")
        assert finding.confidence is FindingConfidence.HIGH
        assert finding.severity is Severity.LOW
        patch = _patch_for(source, "python.bind-all-interfaces")
        assert 'host="127.0.0.1"' in patch.after

    def test_loopback_does_not_fire(self):
        source = (
            "from flask import Flask\napp = Flask(__name__)\n"
            'app.run(host="127.0.0.1")\n'
        )
        assert "python.bind-all-interfaces" not in _rules_fired(source)

    def test_tls_verification_disabled_fires_and_is_fixed(self):
        source = 'import requests\nrequests.get("https://x", verify=False)\n'
        patch = _patch_for(source, "python.tls-verification-disabled")
        assert "verify=True" in patch.after

    def test_verify_true_does_not_fire(self):
        source = 'import requests\nrequests.get("https://x", verify=True)\n'
        assert "python.tls-verification-disabled" not in _rules_fired(source)

    def test_mktemp_fires_and_is_fixed_to_mkstemp(self):
        source = "import tempfile\ndef p():\n    return tempfile.mktemp()\n"
        patch = _patch_for(source, "python.insecure-temp-file")
        assert "tempfile.mkstemp()[1]" in patch.after
        ast.parse(patch.after)

    def test_the_mktemp_fix_admits_the_descriptor_it_leaves(self):
        source = "import tempfile\ndef p():\n    return tempfile.mktemp()\n"
        patch = _patch_for(source, "python.insecure-temp-file")
        assert any("descriptor" in note for note in patch.notes)

    def test_mkstemp_does_not_fire(self):
        source = "import tempfile\ndef p():\n    return tempfile.mkstemp()\n"
        assert "python.insecure-temp-file" not in _rules_fired(source)

    def test_the_mktemp_fix_keeps_arguments(self):
        source = 'import tempfile\ndef p():\n    return tempfile.mktemp(suffix=".csv")\n'
        patch = _patch_for(source, "python.insecure-temp-file")
        assert 'tempfile.mkstemp(suffix=".csv")[1]' in patch.after


class TestTemplateAutoescape:
    RULE = "python.template-autoescape-disabled"

    def test_fires_when_autoescape_is_absent(self):
        # Jinja2's default is off, so silence is the vulnerable state.
        source = (
            "from jinja2 import Environment, FileSystemLoader\n"
            'def env():\n    return Environment(loader=FileSystemLoader("t"))\n'
        )
        assert self.RULE in _rules_fired(source)

    def test_fires_when_autoescape_is_false(self):
        source = (
            "from jinja2 import Environment\n"
            "def env():\n    return Environment(autoescape=False)\n"
        )
        assert self.RULE in _rules_fired(source)

    def test_does_not_fire_when_autoescape_is_on(self):
        source = (
            "from jinja2 import Environment\n"
            "def env():\n    return Environment(autoescape=True)\n"
        )
        assert self.RULE not in _rules_fired(source)

    def test_the_fix_appends_the_keyword_when_absent(self):
        source = (
            "from jinja2 import Environment, FileSystemLoader\n"
            'def env():\n    return Environment(loader=FileSystemLoader("t"))\n'
        )
        patch = _patch_for(source, self.RULE)
        assert "autoescape=True" in patch.after
        ast.parse(patch.after)

    def test_the_fix_appends_correctly_to_an_empty_call(self):
        source = "from jinja2 import Environment\ndef env():\n    return Environment()\n"
        patch = _patch_for(source, self.RULE)
        assert "Environment(autoescape=True)" in patch.after


# --------------------------------------------------------------------------- #
# CWE-22, CWE-95
# --------------------------------------------------------------------------- #


class TestPathTraversal:
    RULE = "python.path-traversal"

    def test_fires_on_an_inline_join_of_untrusted_input(self):
        body = 'f = open(os.path.join(BASE, request.args["p"]), "rb")'
        assert self.RULE in _rules_fired(_route(body))

    def test_does_not_fire_when_basename_was_applied(self):
        body = 'f = open(os.path.join(BASE, os.path.basename(request.args["p"])), "rb")'
        assert self.RULE not in _rules_fired(_route(body))

    def test_does_not_fire_on_a_constant_path(self):
        body = 'f = open(os.path.join(BASE, "report.csv"), "rb")'
        assert self.RULE not in _rules_fired(_route(body))

    def test_the_fix_wraps_the_untrusted_component_and_keeps_the_base(self):
        # The whole point: wrapping the *join* in basename would collapse BASE away
        # and break every download while removing the finding.
        body = 'f = open(os.path.join(BASE, request.args["p"]), "rb")'
        patch = _patch_for(_route(body), self.RULE)
        assert 'os.path.join(BASE, os.path.basename(request.args["p"]))' in patch.after
        ast.parse(patch.after)

    def test_the_fix_does_not_split_a_subscript(self):
        # A subscript is one value access. Descending into it produced
        # ``os.path.basename(request.args)["p"]`` — valid Python, wrong semantics.
        body = 'f = open(os.path.join(BASE, request.args["p"]), "rb")'
        patch = _patch_for(_route(body), self.RULE)
        assert 'os.path.basename(request.args)["p"]' not in patch.after

    def test_a_path_composed_on_an_earlier_line_reports_without_a_patch(self):
        # The edit belongs at the composition site, which is a different statement
        # than the finding, and locating it needs def-use chains the taint analysis
        # deliberately does not build.
        body = 'p = os.path.join(BASE, request.args["p"])\nf = open(p, "rb")'
        result = _scan(_route(body))
        assert any(f.rule_id == self.RULE for f in result.findings)
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]

    @pytest.mark.parametrize("opener", ["os.remove", "shutil.copy", "io.open"])
    def test_fires_on_other_file_sinks(self, opener: str):
        source = (
            "import io\nimport os\nimport shutil\n"
            "from flask import Flask, request\n"
            'app = Flask(__name__)\nBASE = "/srv"\n'
            '@app.route("/x")\ndef h():\n'
            f'    {opener}(os.path.join(BASE, request.args["p"]))\n'
        )
        assert self.RULE in _rules_fired(source)


class TestCodeInjection:
    RULE = "python.code-injection-eval"

    def test_eval_fires_and_is_fixed_to_literal_eval(self):
        patch = _patch_for(
            _route('v = eval(request.args["e"])'), self.RULE
        )
        assert 'ast.literal_eval(request.args["e"])' in patch.after
        assert "import ast" in patch.after
        ast.parse(patch.after)

    def test_exec_fires_without_a_patch(self):
        # exec exists to run statements; there is no data-only equivalent.
        result = _scan(_route('exec(request.args["e"])'))
        assert any(f.rule_id == self.RULE for f in result.findings)
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]

    def test_literal_eval_does_not_fire(self):
        source = (
            "import ast\nfrom flask import Flask, request\napp = Flask(__name__)\n"
            '@app.route("/x")\ndef h():\n'
            '    return ast.literal_eval(request.args["e"])\n'
        )
        assert self.RULE not in _rules_fired(source)

    def test_eval_with_a_globals_argument_declines_the_fix(self):
        # The caller wants an execution environment, which literal_eval has not got.
        result = _scan(_route('v = eval(request.args["e"], {}, {})'))
        assert any(f.rule_id == self.RULE for f in result.findings)
        assert not [p for p in result.valid_patches if p.rule_id == self.RULE]

    def test_eval_on_a_literal_is_never_low(self):
        source = 'v = eval("1 + 1")\n'
        assert _one(source, self.RULE).confidence.rank >= FindingConfidence.MEDIUM.rank
