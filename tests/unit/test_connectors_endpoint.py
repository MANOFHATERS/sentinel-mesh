"""EDR/firewall (Wazuh), identity (SCIM) and notification connectors, over live sockets.

Each connector is driven against a strict emulator of its API. The assertions are on
the emulator's *state* after the call — which host is isolated, which address is
blocked, which account is inactive, which message arrived — because that is what the
approval was for, and an outcome object saying "succeeded" is only the connector's
opinion.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.connectors.base import (
    Capability,
    ConnectorError,
    Credential,
    LeastPrivilegeError,
    Secret,
    TargetRejected,
)
from sentinel.connectors.http import HttpError
from sentinel.connectors.notify import (
    LocalEnrichmentConnector,
    SignedWebhookConnector,
    SlackWebhookConnector,
    escape_mrkdwn,
    sign_payload,
    verify_signature,
)
from sentinel.connectors.sandbox import (
    LiveServer,
    ScimEmulator,
    SlackEmulator,
    WazuhEmulator,
    WebhookReceiver,
)
from sentinel.connectors.scim import ScimIdentityConnector, scim_filter_value
from sentinel.connectors.wazuh import WazuhConnector
from sentinel.core.clock import FrozenClock
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import (
    ActionRequest,
    ActionType,
    AgentName,
    Evidence,
    EvidenceKind,
    RiskTier,
)

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)


def _action(action_type, target, *, approve=True, rationale="fixture", evidence=()):
    action = ActionRequest.propose(
        alert_id="alert-1", tenant_id="acme", proposed_by=AgentName.CONTAINMENT,
        action_type=action_type, target=target, rationale=rationale,
        risk_tier=RiskTier.RECOMMEND, created_at=NOW, evidence=evidence,
    )
    if approve and action.requires_human_approval:
        return action.approve(approver="soc@acme", at=NOW)
    return action


# --------------------------------------------------------------------------- #
# Wazuh
# --------------------------------------------------------------------------- #

AGENTS = [
    {"id": "000", "name": "wazuh-manager", "ip": "10.255.0.10", "status": "active"},
    {"id": "001", "name": "perimeter-fw", "ip": "10.255.0.1", "status": "active"},
    {"id": "002", "name": "ws-042", "ip": "10.0.205.66", "status": "active"},
    {"id": "003", "name": "ws-043", "ip": "10.0.91.48", "status": "disconnected"},
    {"id": "004", "name": "dup-a", "ip": "10.0.7.7", "status": "active"},
    {"id": "005", "name": "dup-b", "ip": "10.0.7.7", "status": "active"},
]


@pytest.fixture
def wazuh():
    emulator = WazuhEmulator(username="sentinel-ar", password="pw-sandbox", agents=AGENTS)
    with LiveServer(emulator) as server:
        yield emulator, server


def _wazuh(server, **kwargs):
    defaults = dict(
        base_url=server.url,
        credential=Credential(Secret("pw-sandbox"),
                              frozenset({"agent:read", "active-response:command"}),
                              username="sentinel-ar"),
        firewall_agents=("001",),
        allow_insecure_loopback=True,
        sleep=lambda _s: None,
    )
    defaults.update(kwargs)
    return WazuhConnector(**defaults)


class TestWazuhIsolation:
    def test_isolates_the_agent_that_owns_the_address(self, wazuh):
        emulator, server = wazuh
        outcome = _wazuh(server).execute(_action(ActionType.ISOLATE_HOST, "10.0.205.66"))
        assert outcome.succeeded
        assert emulator.isolated_hosts() == {"10.0.205.66"}
        assert emulator.executed[0]["agent"]["id"] == "002"

    def test_resolves_by_hostname(self, wazuh):
        emulator, server = wazuh
        _wazuh(server).execute(_action(ActionType.ISOLATE_HOST, "WS-042"))
        assert emulator.executed[0]["agent"]["id"] == "002"

    def test_the_manager_is_never_isolated(self, wazuh):
        emulator, server = wazuh
        with pytest.raises(GuardrailViolation, match="manager"):
            _wazuh(server).execute(_action(ActionType.ISOLATE_HOST, "10.255.0.10"))
        assert emulator.executed == []

    def test_a_disconnected_agent_is_a_failure_not_a_queued_success(self, wazuh):
        emulator, server = wazuh
        with pytest.raises(ConnectorError, match="disconnected"):
            _wazuh(server).execute(_action(ActionType.ISOLATE_HOST, "10.0.91.48"))
        assert emulator.executed == []

    def test_two_agents_with_one_address_is_refused(self, wazuh):
        emulator, server = wazuh
        with pytest.raises(GuardrailViolation, match="2 agents"):
            _wazuh(server).execute(_action(ActionType.ISOLATE_HOST, "10.0.7.7"))
        assert emulator.executed == []

    def test_an_unmanaged_host_is_a_failure(self, wazuh):
        _emu, server = wazuh
        with pytest.raises(ConnectorError, match="no agent"):
            _wazuh(server).execute(_action(ActionType.ISOLATE_HOST, "10.9.9.9"))

    def test_unapproved_isolation_touches_nothing(self, wazuh):
        emulator, server = wazuh
        with pytest.raises(GuardrailViolation, match="approval required"):
            _wazuh(server).execute(
                _action(ActionType.ISOLATE_HOST, "10.0.205.66", approve=False)
            )
        assert emulator.requests == []


class TestWazuhBlock:
    def test_blocks_on_the_firewall_agents_with_srcip(self, wazuh):
        emulator, server = wazuh
        outcome = _wazuh(server).execute(_action(ActionType.BLOCK_IP, "192.168.10.149"))
        assert outcome.succeeded
        assert emulator.blocked_addresses() == {"192.168.10.149"}
        run = emulator.executed[0]
        assert run["command"] == "!firewall-drop" and run["agent"]["id"] == "001"

    def test_the_address_is_canonicalised_before_it_is_sent(self, wazuh):
        emulator, server = wazuh
        _wazuh(server).execute(_action(ActionType.BLOCK_IP, "::ffff:203.0.113.5"))
        assert emulator.blocked_addresses() == {"203.0.113.5"}

    def test_a_range_is_refused_before_the_wire(self, wazuh):
        emulator, server = wazuh
        with pytest.raises(TargetRejected, match="range"):
            _wazuh(server).execute(_action(ActionType.BLOCK_IP, "10.0.0.0/8"))
        assert emulator.requests == []

    def test_partial_failure_is_reported_as_failure(self, wazuh):
        _emulator, server = wazuh
        outcome = _wazuh(server, firewall_agents=("001", "003")).execute(
            _action(ActionType.BLOCK_IP, "203.0.113.9")
        )
        assert not outcome.succeeded and "1 of 2" in outcome.detail

    def test_an_unknown_command_is_an_error(self, wazuh):
        _emu, server = wazuh
        connector = _wazuh(server, commands={Capability.IP_BLOCK: "not-installed"})
        with pytest.raises(HttpError, match="400"):
            connector.execute(_action(ActionType.BLOCK_IP, "203.0.113.9"))


class TestWazuhAuth:
    def test_one_login_serves_many_calls(self, wazuh):
        emulator, server = wazuh
        connector = _wazuh(server)
        for target in ("203.0.113.1", "203.0.113.2", "203.0.113.3"):
            connector.execute(_action(ActionType.BLOCK_IP, target))
        assert emulator.authentications == 1

    def test_a_revoked_token_reauthenticates_once(self, wazuh):
        emulator, server = wazuh
        connector = _wazuh(server)
        connector.execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))
        emulator.revoke_tokens()
        assert connector.execute(_action(ActionType.BLOCK_IP, "203.0.113.2")).succeeded
        assert emulator.authentications == 2

    def test_an_expired_token_is_refreshed_before_use(self, wazuh):
        emulator, server = wazuh
        now = [0.0]
        connector = _wazuh(server, token_lifetime_seconds=120, monotonic=lambda: now[0])
        connector.execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))
        now[0] = 61.0  # past lifetime minus the 60s slack
        connector.execute(_action(ActionType.BLOCK_IP, "203.0.113.2"))
        assert emulator.authentications == 2

    def test_a_transient_503_on_login_is_retried(self, wazuh):
        emulator, server = wazuh
        emulator.inject(503)
        assert _wazuh(server).execute(_action(ActionType.BLOCK_IP, "203.0.113.1")).succeeded
        assert emulator.authentications == 1

    def test_a_wrong_password_is_a_401(self, wazuh):
        _emu, server = wazuh
        connector = _wazuh(server, credential=Credential(
            Secret("wrong"), frozenset({"agent:read", "active-response:command"}),
            username="sentinel-ar"))
        with pytest.raises(HttpError, match="401"):
            connector.execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))

    def test_the_password_goes_only_to_the_auth_endpoint(self, wazuh):
        emulator, server = wazuh
        _wazuh(server).execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))
        basic = [r for r in emulator.requests
                 if r.headers.get("authorization", "").startswith("Basic ")]
        assert [r.path for r in basic] == ["/security/user/authenticate"]


class TestWazuhConfiguration:
    def test_ip_block_without_firewall_agents_is_refused(self, wazuh):
        _emu, server = wazuh
        with pytest.raises(ValueError, match="firewall_agents"):
            _wazuh(server, firewall_agents=())

    def test_isolation_only_needs_no_firewall(self, wazuh):
        _emu, server = wazuh
        _wazuh(server, capabilities=(Capability.HOST_ISOLATE,), firewall_agents=())

    def test_excess_scope_is_refused(self, wazuh):
        _emu, server = wazuh
        with pytest.raises(LeastPrivilegeError, match=r"\*:\*"):
            _wazuh(server, credential=Credential(
                Secret("pw"), frozenset({"agent:read", "active-response:command", "*:*"}),
                username="u"))

    def test_block_only_needs_no_agent_read(self, wazuh):
        _emu, server = wazuh
        connector = _wazuh(server, capabilities=(Capability.IP_BLOCK,),
                           credential=Credential(Secret("pw-sandbox"),
                                                 frozenset({"active-response:command"}),
                                                 username="sentinel-ar"))
        assert "agents.get" not in {r.name for r in connector.http.policy.routes}

    def test_other_action_types_are_refused(self, wazuh):
        _emu, server = wazuh
        with pytest.raises(GuardrailViolation):
            _wazuh(server).execute(_action(ActionType.DISABLE_ACCOUNT, "alice@acme.example"))


# --------------------------------------------------------------------------- #
# SCIM
# --------------------------------------------------------------------------- #

USERS = [
    {"id": "u-1", "userName": "alice@acme.example", "active": True},
    {"id": "u-2", "userName": "bob@acme.example", "active": False},
    {"id": "u-3", "userName": "twin@acme.example", "active": True},
    {"id": "u-4", "userName": "TWIN@acme.example", "active": True},
]


@pytest.fixture
def scim():
    emulator = ScimEmulator(token="scim-sandbox", users=USERS)
    with LiveServer(emulator) as server:
        yield emulator, server


def _scim(server, token="scim-sandbox"):
    return ScimIdentityConnector(
        base_url=server.url,
        credential=Credential(Secret(token), frozenset({"users:read", "users:write"})),
        allow_insecure_loopback=True, sleep=lambda _s: None,
    )


class TestScim:
    def test_disables_the_account(self, scim):
        emulator, server = scim
        outcome = _scim(server).execute(_action(ActionType.DISABLE_ACCOUNT, "alice@acme.example"))
        assert outcome.succeeded and emulator.users["u-1"]["active"] is False

    def test_patch_is_a_scim_patchop_on_active_only(self, scim):
        emulator, server = scim
        _scim(server).execute(_action(ActionType.DISABLE_ACCOUNT, "alice@acme.example"))
        patch = emulator.calls("PATCH")[0]
        assert patch.headers["content-type"] == "application/scim+json"
        body = patch.json()
        assert body["Operations"] == [{"op": "replace", "path": "active", "value": False}]

    def test_the_patch_carries_the_action_id_as_idempotency_key(self, scim):
        emulator, server = scim
        action = _action(ActionType.DISABLE_ACCOUNT, "alice@acme.example")
        _scim(server).execute(action)
        assert emulator.calls("PATCH")[0].headers["idempotency-key"] == action.action_id

    def test_a_503_on_the_patch_is_retried_because_it_is_keyed(self, scim):
        emulator, server = scim
        connector = _scim(server)
        action = _action(ActionType.DISABLE_ACCOUNT, "alice@acme.example")
        # The search succeeds; only the first PATCH fails.
        original = emulator.route

        state = {"patched_once": False}

        def flaky(request):
            if request.method == "PATCH" and not state["patched_once"]:
                state["patched_once"] = True
                return 503, {"Content-Type": "application/json"}, b"{}"
            return original(request)

        emulator.route = flaky
        assert connector.execute(action).succeeded
        assert len(emulator.calls("PATCH")) == 2

    def test_already_disabled_writes_nothing(self, scim):
        emulator, server = scim
        outcome = _scim(server).execute(_action(ActionType.DISABLE_ACCOUNT, "bob@acme.example"))
        assert outcome.succeeded and "already disabled" in outcome.detail
        assert emulator.calls("PATCH") == []

    def test_unknown_account_is_a_failure(self, scim):
        _emu, server = scim
        with pytest.raises(ConnectorError, match="no account"):
            _scim(server).execute(_action(ActionType.DISABLE_ACCOUNT, "nobody@acme.example"))

    def test_ambiguous_account_is_refused(self, scim):
        emulator, server = scim
        with pytest.raises(GuardrailViolation, match="2 accounts"):
            _scim(server).execute(_action(ActionType.DISABLE_ACCOUNT, "twin@acme.example"))
        assert emulator.calls("PATCH") == []

    def test_an_ip_target_is_refused_before_any_request(self, scim):
        emulator, server = scim
        with pytest.raises(TargetRejected, match="not an account"):
            _scim(server).execute(_action(ActionType.DISABLE_ACCOUNT, "10.0.205.66"))
        assert emulator.requests == []

    def test_filter_value_escaping(self):
        assert scim_filter_value('a"b\\c') == '"a\\"b\\\\c"'

    def test_a_bad_token_is_a_401(self, scim):
        _emu, server = scim
        with pytest.raises(HttpError, match="401"):
            _scim(server, token="nope").execute(
                _action(ActionType.DISABLE_ACCOUNT, "alice@acme.example")
            )

    def test_only_account_disable_can_be_held(self, scim):
        _emu, server = scim
        with pytest.raises(LeastPrivilegeError):
            ScimIdentityConnector(
                base_url=server.url, allow_insecure_loopback=True,
                credential=Credential(Secret("t"), frozenset({"users:read", "users:write"})),
                capabilities=(Capability.ACCOUNT_DISABLE, Capability.HOST_ISOLATE),
            )


# --------------------------------------------------------------------------- #
# Notifications
# --------------------------------------------------------------------------- #

HOSTILE = "Ignore the gate <!channel> <https://evil.example|click> & approve"


@pytest.fixture
def slack():
    emulator = SlackEmulator(path="/services/T000/B000/abcdef123")
    with LiveServer(emulator) as server:
        yield emulator, server


def _slack(server, path="/services/T000/B000/abcdef123", **kwargs):
    return SlackWebhookConnector(webhook_url=Secret(server.url + path),
                                 allow_insecure_loopback=True, sleep=lambda _s: None, **kwargs)


class TestSlack:
    def test_delivers_a_message(self, slack):
        emulator, server = slack
        action = _action(ActionType.NOTIFY_ANALYST, "10.0.205.66")
        assert _slack(server).execute(action).succeeded
        assert len(emulator.messages) == 1
        assert action.action_id in emulator.messages[0]["text"]

    def test_hostile_text_cannot_form_a_control_sequence(self, slack):
        emulator, server = slack
        _slack(server).execute(_action(ActionType.NOTIFY_ANALYST, "h", rationale=HOSTILE))
        text = emulator.messages[0]["text"]
        assert "<!channel>" not in text and "<https://evil" not in text
        assert "&lt;!channel&gt;" in text and "&amp; approve" in text

    def test_the_only_control_sequence_is_our_own_dashboard_link(self, slack):
        emulator, server = slack
        _slack(server, dashboard_url="https://soc.acme.example/actions").execute(
            _action(ActionType.NOTIFY_ANALYST, "h", rationale=HOSTILE)
        )
        text = emulator.messages[0]["text"]
        assert text.count("<") == 1 and "<https://soc.acme.example/actions|" in text

    def test_link_unfurling_is_off(self, slack):
        emulator, server = slack
        _slack(server).execute(_action(ActionType.NOTIFY_ANALYST, "h"))
        assert emulator.messages[0]["unfurl_links"] is False

    def test_evidence_refs_are_listed(self, slack):
        emulator, server = slack
        evidence = (Evidence(kind=EvidenceKind.ALERT_FIELD, ref="alert://a/dst_port",
                             excerpt="22"),)
        _slack(server).execute(_action(ActionType.NOTIFY_ANALYST, "h", evidence=evidence))
        assert "alert://a/dst_port" in emulator.messages[0]["text"]

    def test_a_non_ok_answer_is_a_failure(self, slack):
        emulator, server = slack
        emulator.inject(200, body=b"channel_is_archived")
        with pytest.raises(ConnectorError, match="not 'ok'"):
            _slack(server).execute(_action(ActionType.NOTIFY_ANALYST, "h"))

    def test_a_429_is_retried(self, slack):
        emulator, server = slack
        emulator.inject(429, headers={"Retry-After": "1"})
        assert _slack(server).execute(_action(ActionType.NOTIFY_ANALYST, "h")).succeeded
        assert len(emulator.messages) == 1

    def test_the_webhook_url_never_appears_in_an_error(self, slack):
        with pytest.raises(ValueError) as info:
            SlackWebhookConnector(webhook_url=Secret("https://hooks.slack.com/x/secretpart"))
        assert "secretpart" not in str(info.value)

    def test_the_webhook_url_is_not_in_the_connector_repr(self, slack):
        _emu, server = slack
        assert "abcdef123" not in repr(_slack(server).http)

    def test_only_notifications(self, slack):
        _emu, server = slack
        with pytest.raises(GuardrailViolation):
            _slack(server).execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))

    def test_escape_order(self):
        assert escape_mrkdwn("&lt;") == "&amp;lt;"
        assert escape_mrkdwn("<a>") == "&lt;a&gt;"


class TestSignedWebhook:
    def _setup(self, clock):
        secret = Secret("whsec-sandbox")
        receiver = WebhookReceiver(path="/hooks/sentinel", secret=secret, clock=clock)
        return receiver, secret

    def test_the_receiver_accepts_a_genuine_delivery(self):
        clock = FrozenClock(NOW)
        receiver, secret = self._setup(clock)
        with LiveServer(receiver) as server:
            connector = SignedWebhookConnector(url=server.url + "/hooks/sentinel",
                                               signing_secret=secret, clock=clock,
                                               allow_insecure_loopback=True)
            action = _action(ActionType.NOTIFY_ANALYST, "10.0.205.66")
            assert connector.execute(action).succeeded
        assert len(receiver.deliveries) == 1 and receiver.rejected == 0
        body = receiver.deliveries[0]["body"]
        assert body["action_id"] == action.action_id and body["event"] == "sentinel.action"

    def test_a_delivery_signed_with_the_wrong_secret_is_rejected(self):
        clock = FrozenClock(NOW)
        receiver, _secret = self._setup(clock)
        with LiveServer(receiver) as server:
            connector = SignedWebhookConnector(url=server.url + "/hooks/sentinel",
                                               signing_secret=Secret("wrong"), clock=clock,
                                               allow_insecure_loopback=True)
            with pytest.raises(HttpError, match="401"):
                connector.execute(_action(ActionType.NOTIFY_ANALYST, "h"))
        assert receiver.rejected == 1 and receiver.deliveries == []

    def test_a_redelivery_is_deduplicated_by_the_receiver(self):
        clock = FrozenClock(NOW)
        receiver, secret = self._setup(clock)
        with LiveServer(receiver) as server:
            connector = SignedWebhookConnector(url=server.url + "/hooks/sentinel",
                                               signing_secret=secret, clock=clock,
                                               allow_insecure_loopback=True)
            action = _action(ActionType.NOTIFY_ANALYST, "h")
            connector.execute(action)
            connector.execute(action)
        assert len(receiver.deliveries) == 1

    def test_verify_rejects_tampering(self):
        secret = Secret("s")
        body = b'{"a":1}'
        header = sign_payload(secret, body, int(NOW.timestamp()))
        assert verify_signature(secret, header, body, now=NOW)
        assert not verify_signature(secret, header, b'{"a":2}', now=NOW)

    def test_verify_rejects_replay_outside_the_window(self):
        secret = Secret("s")
        body = b"{}"
        header = sign_payload(secret, body, int(NOW.timestamp()) - 301)
        assert not verify_signature(secret, header, body, now=NOW)

    @pytest.mark.parametrize("header", ["", "t=abc,v1=00", "v1=00", "t=1", "garbage"])
    def test_verify_rejects_malformed_headers(self, header):
        assert not verify_signature(Secret("s"), header, b"{}", now=NOW)


class TestEnrichment:
    def test_records_and_touches_nothing(self):
        connector = LocalEnrichmentConnector()
        action = _action(ActionType.ENRICH_ONLY, "10.0.205.66")
        assert connector.execute(action).succeeded
        assert connector.recorded == [action]

    def test_only_enrichment(self):
        with pytest.raises(GuardrailViolation):
            LocalEnrichmentConnector().execute(_action(ActionType.BLOCK_IP, "203.0.113.1"))
