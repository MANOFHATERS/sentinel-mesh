"""Look after the demo identity provider's accounts.

    python scripts/dev_idp_accounts.py list
    python scripts/dev_idp_accounts.py code  maya.analyst@acme.example   # current 6-digit code
    python scripts/dev_idp_accounts.py uri   maya.analyst@acme.example   # otpauth:// for QR
    python scripts/dev_idp_accounts.py reset maya.analyst@acme.example   # new password

Passwords are stored only as hashes, so they cannot be shown again; ``reset`` is how a forgotten
one is replaced. ``code`` is a convenience for demos on a laptop without a phone to hand: it
computes what an authenticator app would show from the stored secret.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

from sentinel.dashboard.credentials import AccountStore, hash_password, otpauth_uri, totp_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["list", "code", "uri", "reset"])
    parser.add_argument("username", nargs="?")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(os.environ.get("SENTINEL_DEV_IDP_ACCOUNTS", "data/dev-idp-accounts.json")),
    )
    args = parser.parse_args()
    if not args.file.is_file():
        print(f"no account file at {args.file}; start the dashboard with --dev-idp first")
        return 1
    store = AccountStore.load(args.file)
    if args.command == "list":
        for account in store.all():
            mfa = "authenticator enrolled" if account.totp_secret else "no second factor"
            groups = ",".join(account.groups)
            print(f"{account.username:32s} {account.tenant:5s} {groups:14s} {mfa}")
        return 0
    account = store.get(args.username or "")
    if account is None:
        print("no such account")
        return 1
    if args.command == "reset":
        password = secrets.token_urlsafe(12)
        account.password_hash = hash_password(password)
        store.save()
        print(f"new password for {account.username}: {password}")
        return 0
    if not account.totp_secret:
        print("this account has no second factor enrolled")
        return 1
    if args.command == "code":
        print(totp_code(account.totp_secret))
    else:
        print(otpauth_uri(account.totp_secret, account.username))
    return 0


if __name__ == "__main__":
    sys.exit(main())
