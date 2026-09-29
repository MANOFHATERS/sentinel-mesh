"""Configuration for the F-07 fixture application. Do not deploy this."""

import os

DATABASE_PATH = "/var/lib/vulnerable-app/app.db"
REPORT_DIR = "/var/lib/vulnerable-app/reports"

# Credentials committed to the repository. The *values* are deliberately not shaped
# like any real provider's key format, and that is not laziness: a literal in a
# payment provider's live-key shape is indistinguishable from a leaked key to a
# secret scanner, and GitHub push protection rejected this file for exactly that
# reason.
# Nothing is lost by it — `python.hardcoded-credential` fires on the secret-shaped
# *name* beside a non-placeholder string literal, so the value's shape is irrelevant
# to what is being tested.
DB_PASSWORD = "Pr0d-Postgres-2024!"  # SEEDED: python.hardcoded-credential
BILLING_API_KEY = "fixture-only-billing-key-not-a-real-credential"  # SEEDED: python.hardcoded-credential

# Correct: the value comes from the environment.
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")  # SAFE: python.hardcoded-credential
# Correct: this names where a secret lives; it is not one.
TLS_PRIVATE_KEY_PATH = "/etc/ssl/private/app.pem"  # SAFE: python.hardcoded-credential
# Correct: an obvious placeholder in a checked-in example.
SMTP_PASSWORD = "changeme"  # SAFE: python.hardcoded-credential


def serve():
    from app import app

    app.run(debug=True, host="0.0.0.0", port=8080)  # SEEDED: python.flask-debug-enabled python.bind-all-interfaces


def serve_production():
    """Correct: no debugger, loopback only, proxied from outside."""
    from app import app

    app.run(debug=False, host="127.0.0.1", port=8080)  # SAFE: python.flask-debug-enabled python.bind-all-interfaces
