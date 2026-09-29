"""A small, deliberately insecure Flask application.

Part of the F-07 fixture. See README.md. Do not deploy this.

Every line carrying a ``# SEEDED:`` marker is a planted defect with a known rule id;
every line carrying ``# SAFE:`` is the correct construction for the same rule and
must not be flagged. See ``sentinel.scan.seeded``.
"""

import os
import shlex
import sqlite3
import subprocess

from flask import Flask, render_template_string, request

from config import DATABASE_PATH, REPORT_DIR

app = Flask(__name__)


def _connect():
    return sqlite3.connect(DATABASE_PATH)


@app.route("/diagnostics/ping")
def diagnostics_ping():
    """Operator tool: check whether a host answers."""
    host = request.args.get("host", "localhost")
    output = subprocess.check_output("ping -c 1 " + host, shell=True)  # SEEDED: python.os-command-injection
    return output.decode("utf-8", errors="replace")


@app.route("/diagnostics/traceroute")
def diagnostics_traceroute():
    """The same tool, done correctly: an argv list and no shell."""
    host = request.args.get("host", "localhost")
    output = subprocess.check_output(["traceroute", "-m", "5", host])  # SAFE: python.os-command-injection
    return output.decode("utf-8", errors="replace")


@app.route("/diagnostics/dig")
def diagnostics_dig():
    """Correct too: the value is quoted before it reaches the shell."""
    host = request.args.get("host", "localhost")
    command = "dig +short " + shlex.quote(host)
    return subprocess.check_output(command, shell=True).decode()  # SAFE: python.os-command-injection


@app.route("/admin/rotate-logs")
def rotate_logs():
    """A fixed command. Dangerous construct, no attacker input."""
    os.system("logrotate -f /etc/logrotate.conf")  # INFO: python.os-system-injection
    return "rotated"


@app.route("/admin/archive")
def archive():
    name = request.args.get("name", "")
    os.system("tar czf /var/backups/%s.tgz /srv/app" % name)  # SEEDED: python.os-system-injection
    return "archived"


@app.route("/users/search")
def search_users():
    term = request.args.get("q", "")
    cursor = _connect().cursor()
    cursor.execute(f"SELECT id, email FROM users WHERE name LIKE '{term}%'")  # SEEDED: python.sql-injection
    return {"results": [dict(id=row[0], email=row[1]) for row in cursor.fetchall()]}


@app.route("/users/by-role")
def users_by_role():
    role = request.args.get("role", "member")
    cursor = _connect().cursor()
    cursor.execute(  # SEEDED: python.sql-injection
        "SELECT id, email FROM users WHERE role = '%s' ORDER BY id" % role
    )
    return {"results": cursor.fetchall()}


@app.route("/users/by-id")
def user_by_id():
    """Correct: the value is bound, not interpolated."""
    cursor = _connect().cursor()
    cursor.execute(  # SAFE: python.sql-injection
        "SELECT id, email FROM users WHERE id = ?", (int(request.args["id"]),)
    )
    return {"results": cursor.fetchall()}


@app.route("/reports/download")
def download_report():
    name = request.args.get("name", "")
    path = os.path.join(REPORT_DIR, name)
    with open(path, "rb") as handle:  # SEEDED: python.path-traversal
        return handle.read()


@app.route("/reports/attachment")
def download_attachment():
    """The same bug composed inline, which is where a mechanical fix can land."""
    with open(os.path.join(REPORT_DIR, request.args["file"]), "rb") as handle:  # SEEDED: python.path-traversal
        return handle.read()


@app.route("/reports/download-safe")
def download_report_safe():
    """Correct: the untrusted component is reduced to a bare filename."""
    name = os.path.basename(request.args.get("name", ""))
    with open(os.path.join(REPORT_DIR, name), "rb") as handle:  # SAFE: python.path-traversal
        return handle.read()


@app.route("/reports/render")
def render_report():
    title = request.args.get("title", "Report")
    return render_template_string("<h1>" + title + "</h1>")


@app.route("/metrics/formula")
def evaluate_formula():
    """A calculator endpoint. The classic way this is written, and the classic bug."""
    expression = request.args.get("expr", "0")
    return {"value": eval(expression)}  # SEEDED: python.code-injection-eval


@app.route("/metrics/threshold")
def parse_threshold():
    """Correct: literals only, so the input is data."""
    import ast as ast_module

    return {"value": ast_module.literal_eval(request.args.get("value", "0"))}  # SAFE: python.code-injection-eval
