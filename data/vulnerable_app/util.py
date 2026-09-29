"""Helpers for the F-07 fixture application. Do not deploy this."""

import hashlib
import os
import tempfile


def password_digest(password):
    """Storing a password under MD5. Collision-broken and far too fast."""
    return hashlib.md5(password.encode("utf-8")).hexdigest()  # SEEDED: python.weak-hash


def content_digest(payload):
    """Correct: SHA-256 for an integrity digest."""
    return hashlib.sha256(payload).hexdigest()  # SAFE: python.weak-hash


def cache_key(payload):
    """Correct: MD5 for a cache key, said out loud so nobody has to guess."""
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()  # SAFE: python.weak-hash


def staging_path(suffix=".csv"):
    """mktemp hands back a name and creates nothing, so the name can be claimed."""
    return tempfile.mktemp(suffix=suffix)  # SEEDED: python.insecure-temp-file


def staging_path_safe(suffix=".csv"):
    """Correct: the file is created atomically with restrictive permissions."""
    handle, path = tempfile.mkstemp(suffix=suffix)  # SAFE: python.insecure-temp-file
    os.close(handle)
    return path


def run_report(command):
    """A helper whose caller supplies the command. The source is not visible here,
    so the analyzer reports it at MEDIUM rather than HIGH — see scan/rules.py."""
    return os.system(command)  # SEEDED: python.os-system-injection
