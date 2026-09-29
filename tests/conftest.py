"""Shared fixtures.

Design rule for this suite: fixtures build objects through the **real** code paths.
Synthetic alerts are produced by running the synthetic row generator through the
actual normalizer, not by calling ``Alert(...)`` with tidy values. A fixture that
hand-builds clean objects tests the tests; a fixture that goes through ingestion
means a normalizer regression breaks the suite, which is the point.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.audit.log import HashChainedAuditLog
from sentinel.core.clock import FrozenClock
from sentinel.core.schemas import (
    ActionType,
    AgentName,
    Alert,
    AlertSource,
    RiskTier,
    Severity,
    TriageDecision,
    TriageResult,
)
from sentinel.ingest.bus import InMemoryEventBus
from sentinel.ingest.normalizer import CICIDS2017Normalizer, UNSWNB15Normalizer
from sentinel.ml.datasets.synthetic import SyntheticCICGenerator, generate_alerts

FIXED_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(FIXED_NOW)


@pytest.fixture
def bus(clock: FrozenClock) -> InMemoryEventBus:
    return InMemoryEventBus(clock=clock, maxlen=10_000)


@pytest.fixture
def audit_log(tmp_path, clock: FrozenClock) -> HashChainedAuditLog:
    with HashChainedAuditLog(tmp_path / "audit.sqlite", clock=clock) as log:
        yield log


@pytest.fixture
def keyed_audit_log(tmp_path, clock: FrozenClock) -> HashChainedAuditLog:
    with HashChainedAuditLog(
        tmp_path / "keyed.sqlite", clock=clock, hmac_key=b"0123456789abcdef-test-key"
    ) as log:
        yield log


@pytest.fixture
def cic_normalizer() -> CICIDS2017Normalizer:
    return CICIDS2017Normalizer(tenant_id="acme", strict=True)


@pytest.fixture
def unsw_normalizer() -> UNSWNB15Normalizer:
    return UNSWNB15Normalizer(tenant_id="acme", strict=True)


@pytest.fixture
def cic_rows() -> list[dict]:
    """Twenty raw CIC-IDS2017-shaped rows, defects and all."""
    return list(SyntheticCICGenerator(seed=11).rows(20))


@pytest.fixture
def alert(cic_normalizer: CICIDS2017Normalizer, cic_rows: list[dict]) -> Alert:
    result = cic_normalizer.normalize(cic_rows[0], row_index=0)
    assert result is not None
    return result


@pytest.fixture
def small_alerts() -> list[Alert]:
    """400 enriched alerts — enough for schema and vectorizer tests, fast to build."""
    return generate_alerts(400, seed=101)


@pytest.fixture(scope="session")
def eval_alerts() -> list[Alert]:
    """A larger, session-scoped corpus for the model-quality tests.

    Session-scoped because generating and enriching 12,000 alerts costs a couple of
    seconds and several tests need the same corpus; the alerts are immutable, so
    sharing them cannot let one test affect another.
    """
    return generate_alerts(12_000, seed=20260928)


@pytest.fixture
def triage_result() -> TriageResult:
    return TriageResult(
        severity=Severity.HIGH,
        confidence=0.91,
        decision=TriageDecision.ESCALATE,
        technique_id="T1110",
        rationale="Sustained authentication attempts from one source across a 60s window.",
        supporting_fields=("src_flow_count_window", "dst_port"),
        model_version="triage-test-1",
        latency_ms=42.0,
        decided_at=FIXED_NOW,
    )


@pytest.fixture
def destructive_action_kwargs() -> dict:
    return {
        "alert_id": "alert-1",
        "tenant_id": "acme",
        "proposed_by": AgentName.CONTAINMENT,
        "action_type": ActionType.ISOLATE_HOST,
        "target": "host-42",
        "rationale": "Confirmed lateral movement from this host.",
        "risk_tier": RiskTier.RECOMMEND,
        "created_at": FIXED_NOW,
    }


@pytest.fixture
def minimal_alert_kwargs() -> dict:
    return {
        "alert_id": "alert-minimal",
        "tenant_id": "acme",
        "source": AlertSource.SIEM,
        "timestamp": FIXED_NOW,
        "ingested_at": FIXED_NOW,
        "asset_id": "host-1",
        "raw_payload": "connection from 10.0.0.5",
    }


@pytest.fixture(scope="session")
def mesh_models():
    """Part 5: the dashboard's trained models, built once per test session (per worker).

    Built through :meth:`MeshModels.build` — the same path ``python -m
    sentinel.dashboard`` takes — so the scenario flows the tests drive are the ones
    the live demo shows.
    """
    from sentinel.dashboard.workspace import MeshModels

    return MeshModels.build()
