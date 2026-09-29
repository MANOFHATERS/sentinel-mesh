"""Outbound integrations for the F-07 fixture application. Do not deploy this."""

import json
import pickle

import requests
import yaml
from flask import request
from jinja2 import Environment, FileSystemLoader

VENDOR_API = "https://vendor.example.com/api/v1"


def fetch_vendor_status(vendor_id):
    response = requests.get(  # SEEDED: python.tls-verification-disabled
        f"{VENDOR_API}/vendors/{vendor_id}/status", timeout=10, verify=False
    )
    return response.json()


def fetch_vendor_contract(vendor_id):
    """Correct: the certificate chain is verified."""
    response = requests.get(  # SAFE: python.tls-verification-disabled
        f"{VENDOR_API}/vendors/{vendor_id}/contract", timeout=10, verify=True
    )
    return response.json()


def load_tenant_profile():
    """The uploaded profile is YAML, and the default loader builds Python objects."""
    return yaml.load(request.files["profile"].read())  # SEEDED: python.yaml-unsafe-load


def load_bundled_defaults(text):
    """Correct: a loader that only ever produces plain data."""
    return yaml.safe_load(text)  # SAFE: python.yaml-unsafe-load


def restore_session(blob):
    """There is no safe pickle loader, which is why this rule ships without a fix."""
    return pickle.loads(blob)  # SEEDED: python.pickle-deserialization


def save_session(session):
    """Correct: serialising is not the dangerous direction. Only loading is."""
    return pickle.dumps(session)  # SAFE: python.pickle-deserialization


def load_session_json(blob):
    """Correct: a data-only format cannot name a class to instantiate."""
    return json.loads(blob)  # SAFE: python.pickle-deserialization


def template_environment():
    return Environment(loader=FileSystemLoader("templates"))  # SEEDED: python.template-autoescape-disabled


def safe_template_environment():
    """Correct: values rendered into a page are escaped."""
    return Environment(loader=FileSystemLoader("templates"), autoescape=True)  # SAFE: python.template-autoescape-disabled
