"""The router: the last guardrail before the wire, and the writer of its audit record.

Each test drives the real router in front of real connectors talking to the live
sandbox. The negative tests assert on the emulators' request logs — a refusal that
still sent a request is not a refusal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentinel.agents.contain import verify_no_ungated_execution
from sentinel.audit.log import HashChainedAuditLog
from sentinel.connectors.base import Capability, ExecutionOutcome
from sentinel.connectors.config import ConfigurationError, router_from_env
from sentinel.connectors.journal import BlastRadiusLimiter, SqliteJournal
from sentinel.connectors.notify import LocalEnrichmentConnector
from sentinel.connectors.router import ConnectorRouter
from sentinel.connectors.sandbox import Sandbox
from sentinel.connectors.targets import TargetPolicy
from sentinel.core.clock import FrozenClock
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import (
    ActionRequest,
    ActionType,
    AgentName,
    AuditEventType,
    RiskTier,
)

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
HOST = "10.0.205.66"


def _action(action_type, target, *, tenant="acme", approve=True, tier=RiskTier.RECOMMEND):
    action = ActionRequest.propose(
        alert_id="alert-1", tenant_id=tenant, proposed_by=AgentName.CONTAINMENT,
        action_type=action_type, target=target, rationale="fixture", risk_tier=tier,
        created_at=NOW,
    )
    if approve and action.requires_human_approval:
        return action.approve(approver="soc@acme", at=NOW)
    return action


@pytest.fixture
def sandbox():
    with Sandbox(hosts=[HOST], clock=FrozenClock(NOW)) as box:
        yield box


@pytest.fixture
def log(tmp_path):
    with HashChainedAuditLog(tmp_path / "audit.sqlite", clock=FrozenClock(NOW)) as audit:
        yield audit


def _wire_requests(sandbox):
    return sum(len(s.requests) for s in (sandbox.github, sandbox.wazuh, sandbox.scim,
                                         sandbox.slack, sandbox.webhook))


class TestDispatch:
    @pytest.mark.parametrize(
        ("action_type", "target", "connector"),
        [
            (ActionType.ISOLATE_HOST, HOST, "wazuh"),
            (ActionType.BLOCK_IP, "203.0.113.7", "wazuh"),
            (ActionType.DISABLE_ACCOUNT, "alice@acme.example", "scim"),
            (ActionType.NOTIFY_ANALYST, HOST, "slack"),
            (ActionType.ENRICH_ONLY, HOST, "enrichment"),
            (ActionType.OPEN_PATCH_PR, "pkg:pypi/leftpad", "github"),
        ],
    )
    def test_every_action_type_reaches_the_connector_that_holds_it(
        self, sandbox, action_type, target, connector
    ):
        router = sandbox.router()
        assert router.connector_for(action_type) == connector
        outcome = router.execute(_action(action_type, target))
        assert outcome.succeeded
        assert router.executions[-1].connector == connector

    def test_the_effects_are_visible_in_the_remote_systems(self, sandbox):
        router = sandbox.router()
        router.execute(_action(ActionType.ISOLATE_HOST, HOST))
        router.execute(_action(ActionType.BLOCK_IP, "203.0.113.7"))
        router.execute(_action(ActionType.DISABLE_ACCOUNT, "alice@acme.example"))
        router.execute(_action(ActionType.NOTIFY_ANALYST, HOST))
        assert sandbox.wazuh.isolated_hosts() == {HOST}
        assert sandbox.wazuh.blocked_addresses() == {"203.0.113.7"}
        assert sandbox.scim.users["u-1"]["active"] is False
        assert len(sandbox.slack.messages) == 1

    def test_webhook_notifier_can_replace_slack(self, sandbox):
        router = sandbox.router(notify="webhook")
        router.execute(_action(ActionType.NOTIFY_ANALYST, HOST))
        assert len(sandbox.webhook.deliveries) == 1 and sandbox.slack.messages == []

    def test_executed_mirrors_the_simulated_connector_interface(self, sandbox):
        router = sandbox.router()
        action = _action(ActionType.BLOCK_IP, "203.0.113.7")
        router.execute(action)
        assert router.executed == [action]


class TestFailClosed:
    def test_an_action_with_no_connector_is_refused_not_dropped(self, log):
        router = ConnectorRouter(tenant_id="acme", connectors=[LocalEnrichmentConnector()],
                                 audit=log)
        with pytest.raises(GuardrailViolation, match=r"no connector holds ip.block"):
            router.execute(_action(ActionType.BLOCK_IP, "203.0.113.7"))
        rows = list(log.iter_records())
        assert [r.event_type for r in rows] == [AuditEventType.GUARDRAIL_BLOCKED]

    def test_a_draft_with_no_git_connector_is_refused(self):
        router = ConnectorRouter(tenant_id="acme")
        with pytest.raises(GuardrailViolation, match=r"pr.open_draft"):
            router.open_draft(_action(ActionType.OPEN_PATCH_PR, "sentinel/x"), object())

    def test_two_connectors_for_one_capability_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="held by both"):
            ConnectorRouter(tenant_id="acme",
                            connectors=[LocalEnrichmentConnector(), LocalEnrichmentConnector()])

    def test_a_draft_opener_without_the_capability_is_refused(self, sandbox):
        github = sandbox.github_connector()
        github._capabilities = frozenset({Capability.ISSUE_OPEN})
        with pytest.raises(ValueError, match=r"pr.open_draft"):
            ConnectorRouter(tenant_id="acme", draft_opener=github)


class TestTenantIsolation:
    def test_another_tenants_action_is_refused_before_the_wire(self, sandbox, log):
        router = sandbox.router(tenant_id="acme", audit=log)
        with pytest.raises(GuardrailViolation, match="tenant globex"):
            router.execute(_action(ActionType.ISOLATE_HOST, HOST, tenant="globex"))
        assert _wire_requests(sandbox) == 0
        assert log.count() == 1

    def test_a_router_must_name_its_tenant(self):
        with pytest.raises(ValueError):
            ConnectorRouter(tenant_id="  ")


class TestApprovalAtTheRouter:
    def test_unapproved_gated_action_is_refused_before_the_wire(self, sandbox):
        router = sandbox.router()
        with pytest.raises(GuardrailViolation, match="approval required"):
            router.execute(_action(ActionType.ISOLATE_HOST, HOST, approve=False))
        assert _wire_requests(sandbox) == 0

    def test_unattended_tier_executes_without_approval(self, sandbox):
        router = sandbox.router()
        action = _action(ActionType.BLOCK_IP, "203.0.113.7", tier=RiskTier.AUTONOMOUS)
        assert not action.requires_human_approval
        assert router.execute(action).succeeded


class TestTargets:
    def test_protected_infrastructure_is_refused(self, sandbox, log):
        router = sandbox.router(audit=log, targets=TargetPolicy(
            protected_networks=("10.0.0.0/16",)))
        with pytest.raises(GuardrailViolation, match="protected network"):
            router.execute(_action(ActionType.ISOLATE_HOST, HOST))
        assert _wire_requests(sandbox) == 0
        row = next(log.iter_records())
        assert row.event_type is AuditEventType.GUARDRAIL_BLOCKED
        assert row.payload["guardrail"] == "TargetRejected"


class TestBlastRadius:
    def test_the_ceiling_stops_a_runaway_and_is_audited(self, sandbox, log):
        limiter = BlastRadiusLimiter(max_actions=3, window=timedelta(hours=1),
                                     clock=FrozenClock(NOW))
        router = sandbox.router(audit=log, limiter=limiter)
        for i in range(3):
            router.execute(_action(ActionType.BLOCK_IP, f"203.0.113.{i + 1}"))
        with pytest.raises(GuardrailViolation, match="3 destructive"):
            router.execute(_action(ActionType.BLOCK_IP, "203.0.113.9"))
        assert sandbox.wazuh.blocked_addresses() == {"203.0.113.1", "203.0.113.2",
                                                     "203.0.113.3"}

    def test_notifications_do_not_count_against_the_ceiling(self, sandbox):
        limiter = BlastRadiusLimiter(max_actions=1, clock=FrozenClock(NOW))
        router = sandbox.router(limiter=limiter)
        for _ in range(3):
            router.execute(_action(ActionType.NOTIFY_ANALYST, HOST))
        router.execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))

    def test_a_refusal_before_the_wire_returns_the_slot(self, sandbox):
        limiter = BlastRadiusLimiter(max_actions=1, clock=FrozenClock(NOW))
        router = sandbox.router(limiter=limiter)
        with pytest.raises(GuardrailViolation):
            router.execute(_action(ActionType.ISOLATE_HOST, "10.9.9.9.9"))  # bad host
        router.execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))  # slot still free


class TestJournal:
    def test_a_repeated_execute_replays_without_a_second_call(self, sandbox):
        router = sandbox.router()
        action = _action(ActionType.ISOLATE_HOST, HOST)
        first = router.execute(action)
        calls = len(sandbox.wazuh.requests)
        second = router.execute(action)
        assert second.replayed and not first.replayed
        assert second.reference == first.reference
        assert len(sandbox.wazuh.requests) == calls
        assert len(sandbox.wazuh.executed) == 1

    def test_the_journal_survives_a_new_router_in_a_new_process_image(self, sandbox, tmp_path):
        path = tmp_path / "journal.sqlite"
        action = _action(ActionType.BLOCK_IP, "203.0.113.7")
        first = SqliteJournal(path)
        sandbox.router(journal=first).execute(action)
        first.close()
        second = SqliteJournal(path)
        outcome = sandbox.router(journal=second).execute(action)
        second.close()
        assert outcome.replayed and len(sandbox.wazuh.executed) == 1

    def test_a_failed_attempt_is_not_journalled_as_done(self, sandbox):
        router = sandbox.router()
        sandbox.wazuh.inject(503, times=3)
        action = _action(ActionType.BLOCK_IP, "203.0.113.7")
        with pytest.raises(Exception):  # noqa: B017 - any connector failure
            router.execute(action)
        assert router.execute(action).succeeded
        assert len(sandbox.wazuh.executed) == 1


class TestAuditRecord:
    def test_every_wire_attempt_is_a_connector_called_row(self, sandbox, log):
        router = sandbox.router(audit=log)
        action = _action(ActionType.ISOLATE_HOST, HOST)
        router.execute(action)
        rows = [r for r in log.iter_records() if r.event_type is AuditEventType.CONNECTOR_CALLED]
        # authenticate, resolve the agent, run the command
        assert [r.payload["route"] for r in rows] == ["auth", "agents.get",
                                                      "active_response.run"]
        assert all(r.subject_id == action.action_id for r in rows)
        assert all(r.actor == "connector:wazuh" for r in rows)
        assert all(r.payload["approval_status"] == "approved" for r in rows)
        assert log.verify().findings == ()

    def test_retries_are_individually_recorded(self, sandbox, log):
        router = sandbox.router(audit=log)
        sandbox.scim.inject(503)
        router.execute(_action(ActionType.DISABLE_ACCOUNT, "alice@acme.example"))
        search = [r.payload for r in log.iter_records()
                  if r.event_type is AuditEventType.CONNECTOR_CALLED
                  and r.payload["route"] == "users.search"]
        assert [(p["attempt"], p["status"]) for p in search] == [(1, 503), (2, 200)]

    def test_no_credential_ever_reaches_the_log(self, sandbox, log):
        router = sandbox.router(audit=log)
        for action_type, target in [
            (ActionType.ISOLATE_HOST, HOST),
            (ActionType.BLOCK_IP, "203.0.113.7"),
            (ActionType.DISABLE_ACCOUNT, "alice@acme.example"),
            (ActionType.NOTIFY_ANALYST, HOST),
            (ActionType.OPEN_PATCH_PR, "pkg:pypi/leftpad"),
        ]:
            router.execute(_action(action_type, target))
        rendered = " ".join(repr(r.payload) for r in log.iter_records())
        for secret in sandbox._secrets.values():
            assert secret not in rendered
        assert "Bearer" not in rendered and "Basic" not in rendered

    def test_the_f08_reader_still_needs_no_change(self, sandbox, log):
        # The router adds rows; it must not add a way for the F-08 check to be fooled.
        router = sandbox.router(audit=log)
        router.execute(_action(ActionType.BLOCK_IP, "203.0.113.7"))
        assert verify_no_ungated_execution(log) == ()


class TestConfigFromEnv:
    def test_tenant_is_required(self):
        with pytest.raises(ConfigurationError, match="SENTINEL_TENANT"):
            router_from_env({})

    def test_minimal_config_is_enrichment_only(self):
        router = router_from_env({"SENTINEL_TENANT": "acme"})
        assert router.capabilities == {Capability.ENRICH}

    def test_a_half_configured_connector_is_an_error(self):
        with pytest.raises(ConfigurationError, match="SENTINEL_WAZUH_PASSWORD"):
            router_from_env({"SENTINEL_TENANT": "acme",
                             "SENTINEL_WAZUH_URL": "https://wazuh.example:55000",
                             "SENTINEL_WAZUH_USER": "u",
                             "SENTINEL_WAZUH_SCOPES": "agent:read,active-response:command"})

    def test_full_config_builds_every_connector(self, tmp_path):
        env = {
            "SENTINEL_TENANT": "acme",
            "SENTINEL_GITHUB_REPO": "acme/billing",
            "SENTINEL_GITHUB_TOKEN": "t",
            "SENTINEL_GITHUB_SCOPES": "contents:write,pull_requests:write,issues:write",
            "SENTINEL_WAZUH_URL": "https://wazuh.example:55000",
            "SENTINEL_WAZUH_USER": "u",
            "SENTINEL_WAZUH_PASSWORD": "p",
            "SENTINEL_WAZUH_SCOPES": "agent:read,active-response:command",
            "SENTINEL_WAZUH_FIREWALL_AGENTS": "001",
            "SENTINEL_SCIM_URL": "https://dir.example/scim/v2",
            "SENTINEL_SCIM_TOKEN": "s",
            "SENTINEL_SCIM_SCOPES": "users:read,users:write",
            "SENTINEL_SLACK_WEBHOOK": "https://hooks.slack.com/services/T1/B1/abc",
            "SENTINEL_PROTECTED_NETWORKS": "10.255.0.0/16",
            "SENTINEL_JOURNAL": str(tmp_path / "j.sqlite"),
            "SENTINEL_MAX_ACTIONS_PER_HOUR": "10",
        }
        router = router_from_env(env)
        assert router.capabilities == set(Capability) - {Capability.PROCESS_KILL,
                                                        Capability.FILE_QUARANTINE}
        assert router.limiter.max_actions == 10
        assert router.targets.protected_networks == ("10.255.0.0/16",)

    def test_a_plaintext_url_in_production_config_is_refused(self):
        with pytest.raises(GuardrailViolation, match="https is required"):
            router_from_env({"SENTINEL_TENANT": "acme",
                             "SENTINEL_SCIM_URL": "http://dir.example/scim/v2",
                             "SENTINEL_SCIM_TOKEN": "s",
                             "SENTINEL_SCIM_SCOPES": "users:read,users:write"})

    def test_an_over_scoped_production_token_is_refused(self):
        with pytest.raises(GuardrailViolation, match="beyond"):
            router_from_env({"SENTINEL_TENANT": "acme",
                             "SENTINEL_GITHUB_REPO": "acme/billing",
                             "SENTINEL_GITHUB_TOKEN": "t",
                             "SENTINEL_GITHUB_SCOPES":
                                 "contents:write,pull_requests:write,administration:write"})


def test_execution_outcome_defaults_are_backward_compatible():
    outcome = ExecutionOutcome(succeeded=True, detail="x")
    assert outcome.reference is None and outcome.replayed is False
