"""The connector layer's foundations: credentials, targets, egress, retries, journal.

Most of what makes a connector safe lives below the connector: the egress allowlist,
the refusal to follow redirects, the retry rules, the target validator and the
journal. Each is tested on its own here, and the redirect and size tests run against
a live socket, because "urllib does not forward the token on a redirect" is a claim
about urllib that only a second server can check.
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta

import pytest

from sentinel.connectors.base import (
    Credential,
    EgressDenied,
    ExecutionOutcome,
    LeastPrivilegeError,
    Secret,
    TargetRejected,
    check_scopes,
    require_executable,
)
from sentinel.connectors.http import (
    CallRecord,
    EgressPolicy,
    HttpError,
    HttpRequest,
    HttpResponse,
    RetryPolicy,
    Route,
    ScopedHttpClient,
    TransportError,
    UrllibTransport,
    _retry_after_seconds,
    quote_segment,
)
from sentinel.connectors.journal import (
    BlastRadiusExceeded,
    BlastRadiusLimiter,
    MemoryJournal,
    SqliteJournal,
)
from sentinel.connectors.sandbox import EmulatedService, LiveServer, RecordedRequest, _json
from sentinel.connectors.targets import (
    TargetPolicy,
    canonical_account,
    canonical_host,
    canonical_ip,
)
from sentinel.core.clock import FrozenClock
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import ActionRequest, ActionType, AgentName, RiskTier

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)


def _action(
    action_type: ActionType = ActionType.BLOCK_IP,
    *,
    tier: RiskTier = RiskTier.RECOMMEND,
    target: str = "203.0.113.7",
) -> ActionRequest:
    return ActionRequest.propose(
        alert_id="alert-1",
        tenant_id="acme",
        proposed_by=AgentName.CONTAINMENT,
        action_type=action_type,
        target=target,
        rationale="fixture",
        risk_tier=tier,
        created_at=NOW,
    )


# --------------------------------------------------------------------------- #
# Secrets and credentials
# --------------------------------------------------------------------------- #


class TestSecret:
    VALUE = "tok_live_looking_value_12345"

    def test_no_rendering_reveals_the_value(self):
        secret = Secret(self.VALUE)
        for rendered in (repr(secret), str(secret), f"{secret}", f"{secret!r}", f"{secret}"):
            assert self.VALUE not in rendered
            assert rendered == "Secret(***)"

    def test_reveal_is_the_only_way_out(self):
        assert Secret(self.VALUE).reveal() == self.VALUE

    def test_a_secret_inside_a_container_stays_redacted(self):
        assert self.VALUE not in repr({"token": Secret(self.VALUE)})
        assert self.VALUE not in repr([Secret(self.VALUE)])

    def test_cannot_be_pickled(self):
        with pytest.raises(TypeError):
            pickle.dumps(Secret(self.VALUE))

    def test_equality_is_by_value(self):
        assert Secret("a") == Secret("a")
        assert Secret("a") != Secret("b")

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_is_refused(self, bad):
        with pytest.raises(ValueError):
            Secret(bad)

    def test_credential_repr_hides_the_secret_but_shows_scopes(self):
        credential = Credential(Secret(self.VALUE), frozenset({"issues:write"}))
        assert self.VALUE not in repr(credential)
        assert "issues:write" in repr(credential)


class TestCheckScopes:
    def test_exact_match_passes(self):
        check_scopes(connector="x", declared={"a", "b"}, required={"a", "b"})

    def test_permitted_extra_passes(self):
        check_scopes(connector="x", declared={"a", "meta"}, required={"a"},
                     permitted_extra={"meta"})

    def test_missing_scope_is_refused(self):
        with pytest.raises(LeastPrivilegeError, match="missing"):
            check_scopes(connector="x", declared={"a"}, required={"a", "b"})

    def test_excess_scope_is_refused_and_named(self):
        with pytest.raises(LeastPrivilegeError, match="admin:org"):
            check_scopes(connector="x", declared={"a", "admin:org"}, required={"a"})

    def test_least_privilege_error_is_a_guardrail_violation(self):
        assert issubclass(LeastPrivilegeError, GuardrailViolation)
        assert issubclass(EgressDenied, GuardrailViolation)
        assert issubclass(TargetRejected, GuardrailViolation)


class TestRequireExecutable:
    def test_gated_pending_is_refused(self):
        with pytest.raises(GuardrailViolation, match="human approval required"):
            require_executable(_action(), connector="c")

    def test_gated_approved_passes(self):
        require_executable(_action().approve(approver="soc@acme", at=NOW), connector="c")

    def test_unattended_tier_pending_passes(self):
        action = _action(tier=RiskTier.AUTO_WITH_NOTIFY)
        assert not action.requires_human_approval
        require_executable(action, connector="c")

    def test_rejected_is_refused(self):
        with pytest.raises(GuardrailViolation, match="already rejected"):
            require_executable(_action().reject(approver="soc@acme", at=NOW), connector="c")

    def test_already_executed_is_refused(self):
        done = _action().approve(approver="soc@acme", at=NOW).mark_executed(at=NOW)
        with pytest.raises(GuardrailViolation, match="already executed"):
            require_executable(done, connector="c")

    def test_failed_is_refused(self):
        failed = _action().approve(approver="a", at=NOW).mark_failed(reason="x", at=NOW)
        with pytest.raises(GuardrailViolation, match="already failed"):
            require_executable(failed, connector="c")

    def test_non_destructive_at_recommend_passes(self):
        require_executable(_action(ActionType.NOTIFY_ANALYST, target="h"), connector="c")


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


class TestCanonicalIp:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("203.0.113.7", "203.0.113.7"),
            (" 192.168.10.149 ", "192.168.10.149"),
            ("::ffff:10.0.0.5", "10.0.0.5"),
            ("2001:DB8::1", "2001:db8::1"),
        ],
    )
    def test_valid_addresses_canonicalise(self, raw, expected):
        assert canonical_ip(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            ("10.0.0.0/8", "range"),
            ("0.0.0.0", "unspecified"),
            ("::", "unspecified"),
            ("127.0.0.1", "loopback"),
            ("::1", "loopback"),
            ("224.0.0.1", "multicast"),
            ("169.254.1.1", "link-local"),
            ("255.255.255.255", "broadcast"),
            ("010.000.000.005", "leading zeros"),
            ("10.0.05.1", "leading zeros"),
            ("host.example", "not an IP"),
            ("10.0.0.5; rm -rf /", "not an IP"),
            ("", "not an IP"),
        ],
    )
    def test_unsafe_or_malformed_addresses_are_refused(self, raw, reason):
        with pytest.raises(TargetRejected, match=reason):
            canonical_ip(raw)


class TestCanonicalHost:
    def test_ip_passes_through_canonically(self):
        assert canonical_host("::ffff:10.0.0.9") == "10.0.0.9"

    def test_hostname_is_lowercased(self):
        assert canonical_host("WS-042.Corp.Example") == "ws-042.corp.example"

    @pytest.mark.parametrize(
        "raw", ["-bad.example", "a..b", "x" * 64 + ".example", "host/../admin", "10.0.0.999"]
    )
    def test_invalid_hostnames_are_refused(self, raw):
        with pytest.raises(TargetRejected):
            canonical_host(raw)

    def test_loopback_is_still_refused_as_a_host(self):
        with pytest.raises(TargetRejected, match="loopback"):
            canonical_host("127.0.0.1")


class TestCanonicalAccount:
    @pytest.mark.parametrize("raw", ["alice@acme.example", "svc-backup", "j.doe+ops@x.io"])
    def test_accounts_pass(self, raw):
        assert canonical_account(raw) == raw

    def test_an_ip_address_is_not_an_account(self):
        # _TARGET_FIELD sends disable_account the asset id, which on the network
        # corpus is an IP. The connector must refuse it rather than search for it.
        with pytest.raises(TargetRejected, match="not an account"):
            canonical_account("10.0.205.66")

    @pytest.mark.parametrize("raw", ['alice" or userName pr "', "a b", "(x)", "", "a\\b"])
    def test_filter_syntax_is_refused(self, raw):
        with pytest.raises(TargetRejected):
            canonical_account(raw)


class TestTargetPolicy:
    def test_protected_network_is_refused_regardless_of_approval(self):
        policy = TargetPolicy(protected_networks=("10.255.0.0/16",))
        with pytest.raises(TargetRejected, match="protected network"):
            policy.validate(ActionType.BLOCK_IP, "10.255.3.4")

    def test_protected_host_is_refused_case_insensitively(self):
        policy = TargetPolicy(protected_hosts=frozenset({"DC01.corp.example"}))
        with pytest.raises(TargetRejected, match="protected list"):
            policy.validate(ActionType.ISOLATE_HOST, "dc01.CORP.example")

    def test_protected_account(self):
        policy = TargetPolicy(protected_hosts=frozenset({"breakglass@acme.example"}))
        with pytest.raises(TargetRejected, match="protected"):
            policy.validate(ActionType.DISABLE_ACCOUNT, "breakglass@acme.example")

    def test_unprotected_target_is_returned_canonically(self):
        policy = TargetPolicy(protected_networks=("10.255.0.0/16",))
        assert policy.validate(ActionType.BLOCK_IP, "::ffff:203.0.113.9") == "203.0.113.9"

    def test_non_destructive_targets_are_not_parsed(self):
        assert TargetPolicy().validate(ActionType.NOTIFY_ANALYST, " anything ") == "anything"

    def test_a_typo_in_the_protected_list_fails_at_construction(self):
        with pytest.raises(ValueError):
            TargetPolicy(protected_networks=("10.0.0.0/33",))


# --------------------------------------------------------------------------- #
# Egress policy
# --------------------------------------------------------------------------- #

ROUTES = (
    Route("thing.get", "GET", r"/things/[^/]+", query={"view"}),
    Route("thing.create", "POST", r"/things"),
)


class TestEgressPolicy:
    def test_https_origin_with_matching_route(self):
        policy = EgressPolicy("https://api.example", ROUTES)
        route, url = policy.authorize("GET", "/things/abc", {"view": "full"})
        assert route.name == "thing.get"
        assert url == "https://api.example/things/abc?view=full"

    def test_base_path_prefix_is_kept(self):
        policy = EgressPolicy("https://dir.example/scim/v2/", ROUTES)
        assert policy.authorize("POST", "/things")[1] == "https://dir.example/scim/v2/things"

    def test_unlisted_method_is_denied(self):
        policy = EgressPolicy("https://api.example", ROUTES)
        with pytest.raises(EgressDenied, match="not an allowed route"):
            policy.authorize("DELETE", "/things/abc")

    def test_unlisted_path_is_denied(self):
        policy = EgressPolicy("https://api.example", ROUTES)
        with pytest.raises(EgressDenied):
            policy.authorize("GET", "/things/abc/secrets")

    def test_unlisted_query_key_is_denied(self):
        policy = EgressPolicy("https://api.example", ROUTES)
        with pytest.raises(EgressDenied, match="query key"):
            policy.authorize("GET", "/things/abc", {"admin": "1"})

    @pytest.mark.parametrize("path", ["/things/..", "/things/.", "/things/%2E%2E"])
    def test_dot_segments_are_denied(self, path):
        policy = EgressPolicy("https://api.example", ROUTES)
        with pytest.raises(EgressDenied, match="dot segment"):
            policy.authorize("GET", path)

    def test_plaintext_origin_is_refused(self):
        with pytest.raises(EgressDenied, match="https is required"):
            EgressPolicy("http://api.example", ROUTES)

    def test_plaintext_loopback_needs_the_explicit_flag(self):
        with pytest.raises(EgressDenied):
            EgressPolicy("http://127.0.0.1:9000", ROUTES)
        EgressPolicy("http://127.0.0.1:9000", ROUTES, allow_insecure_loopback=True)

    def test_the_flag_does_not_allow_plaintext_to_a_real_host(self):
        with pytest.raises(EgressDenied):
            EgressPolicy("http://api.example", ROUTES, allow_insecure_loopback=True)

    def test_credentials_in_the_url_are_refused(self):
        with pytest.raises(EgressDenied, match="credentials"):
            EgressPolicy("https://user:pw@api.example", ROUTES)

    def test_no_routes_is_refused(self):
        with pytest.raises(EgressDenied):
            EgressPolicy("https://api.example", ())

    def test_route_repr_hides_the_pattern(self):
        # A webhook's route pattern is its URL path, which is the credential.
        assert "services" not in repr(Route("hook", "POST", "/services/T/B/secret"))


class TestQuoteSegment:
    def test_slash_is_encoded_so_it_cannot_add_a_segment(self):
        assert quote_segment("a/b") == "a%2Fb"

    @pytest.mark.parametrize("bad", ["", ".", ".."])
    def test_dot_segments_are_refused(self, bad):
        with pytest.raises(EgressDenied):
            quote_segment(bad)


# --------------------------------------------------------------------------- #
# The scoped client, against a scripted transport
# --------------------------------------------------------------------------- #


class ScriptedTransport:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent: list[HttpRequest] = []

    def send(self, request, *, timeout):
        self.sent.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _ok(status=200, body=b"{}", headers=None):
    return HttpResponse(status=status, headers=headers or {}, body=body)


def _client(transport, **kwargs):
    sleeps: list[float] = []
    records: list[CallRecord] = []
    client = ScopedHttpClient(
        connector="test",
        policy=EgressPolicy("https://api.example", ROUTES),
        transport=transport,
        auth=lambda: {"Authorization": "Bearer t"},
        sleep=sleeps.append,
        on_call=records.append,
        **kwargs,
    )
    return client, sleeps, records


class TestScopedClient:
    def test_denied_route_never_reaches_the_transport(self):
        transport = ScriptedTransport()
        client, _s, records = _client(transport)
        with pytest.raises(EgressDenied):
            client.request("PUT", "/things/x")
        assert transport.sent == [] and records == []

    def test_get_is_retried_on_503_then_succeeds(self):
        transport = ScriptedTransport(_ok(503), _ok(200))
        client, sleeps, records = _client(transport)
        assert client.request("GET", "/things/x").status == 200
        assert [r.status for r in records] == [503, 200]
        assert [r.attempt for r in records] == [1, 2]
        assert sleeps == [RetryPolicy().backoff(1)]

    def test_post_without_idempotency_key_is_not_retried(self):
        transport = ScriptedTransport(_ok(503))
        client, sleeps, _r = _client(transport)
        with pytest.raises(HttpError) as info:
            client.request("POST", "/things", json_body={"a": 1})
        assert info.value.status == 503 and len(transport.sent) == 1 and sleeps == []

    def test_post_with_idempotency_key_is_retried_and_carries_it(self):
        transport = ScriptedTransport(_ok(502), _ok(201))
        client, _s, _r = _client(transport)
        client.request("POST", "/things", json_body={}, idempotency_key="act-1")
        assert len(transport.sent) == 2
        assert all(dict(r.headers)["Idempotency-Key"] == "act-1" for r in transport.sent)

    def test_retryable_override_retries_an_unkeyed_post(self):
        transport = ScriptedTransport(_ok(503), _ok(200))
        client, _s, _r = _client(transport)
        client.request("POST", "/things", json_body={}, retryable=True)
        assert len(transport.sent) == 2
        assert "Idempotency-Key" not in dict(transport.sent[0].headers)

    def test_retryable_false_disables_retries_on_get(self):
        transport = ScriptedTransport(_ok(503))
        client, _s, _r = _client(transport)
        with pytest.raises(HttpError):
            client.request("GET", "/things/x", retryable=False)
        assert len(transport.sent) == 1

    def test_retry_after_seconds_is_honoured(self):
        transport = ScriptedTransport(_ok(429, headers={"retry-after": "3"}), _ok())
        client, sleeps, _r = _client(transport)
        client.request("GET", "/things/x")
        assert sleeps == [3.0]

    def test_retry_after_beyond_the_ceiling_fails_now(self):
        transport = ScriptedTransport(_ok(429, headers={"retry-after": "3600"}))
        client, sleeps, _r = _client(transport)
        with pytest.raises(HttpError, match="ceiling"):
            client.request("GET", "/things/x")
        assert sleeps == []

    def test_4xx_is_not_retried(self):
        transport = ScriptedTransport(_ok(404, body=b'{"message":"nope"}'))
        client, _s, _r = _client(transport)
        with pytest.raises(HttpError, match="404"):
            client.request("GET", "/things/x")
        assert len(transport.sent) == 1

    def test_expected_non_2xx_is_an_answer(self):
        transport = ScriptedTransport(_ok(404))
        client, _s, _r = _client(transport)
        assert client.request("GET", "/things/x", expect=(200, 404)).status == 404

    def test_redirect_is_a_failure_and_is_not_followed(self):
        transport = ScriptedTransport(_ok(302, headers={"location": "https://evil.example"}))
        client, _s, _r = _client(transport)
        with pytest.raises(HttpError, match="redirect"):
            client.request("GET", "/things/x")
        assert len(transport.sent) == 1

    def test_transport_errors_are_retried_for_idempotent_calls(self):
        transport = ScriptedTransport(TransportError("reset"), _ok())
        client, _s, records = _client(transport)
        client.request("GET", "/things/x")
        assert records[0].status is None and records[0].error == "reset"

    def test_transport_errors_exhaust_and_raise(self):
        transport = ScriptedTransport(*(TransportError("down") for _ in range(3)))
        client, _s, _r = _client(transport)
        with pytest.raises(TransportError):
            client.request("GET", "/things/x")
        assert len(transport.sent) == 3

    def test_call_record_carries_a_body_digest_and_nothing_sensitive(self):
        transport = ScriptedTransport(_ok(201))
        client, _s, records = _client(transport)
        client.request("POST", "/things", json_body={"password": "hunter2"})
        payload = records[0].as_payload()
        assert payload["request_sha256"] and len(payload["request_sha256"]) == 64
        assert "hunter2" not in repr(payload) and "Bearer" not in repr(payload)

    def test_auth_is_computed_per_attempt(self):
        tokens = iter(["one", "two"])
        transport = ScriptedTransport(_ok(503), _ok())
        client = ScopedHttpClient(
            connector="t",
            policy=EgressPolicy("https://api.example", ROUTES),
            transport=transport,
            auth=lambda: {"Authorization": next(tokens)},
            sleep=lambda _s: None,
        )
        client.request("GET", "/things/x")
        assert [dict(r.headers)["Authorization"] for r in transport.sent] == ["one", "two"]


class TestRetryAfterParsing:
    def test_seconds(self):
        assert _retry_after_seconds("12", NOW) == 12.0

    def test_http_date(self):
        assert _retry_after_seconds("Tue, 29 Sep 2026 12:00:30 GMT", NOW) == 30.0

    def test_past_date_is_zero(self):
        assert _retry_after_seconds("Tue, 29 Sep 2026 11:00:00 GMT", NOW) == 0.0

    def test_garbage_is_none(self):
        assert _retry_after_seconds("soon", NOW) is None


# --------------------------------------------------------------------------- #
# The real transport, against live sockets
# --------------------------------------------------------------------------- #


class _Redirector(EmulatedService):
    def __init__(self, location):
        super().__init__()
        self.location = location

    def route(self, request: RecordedRequest):
        return 302, {"Location": self.location}, b""


class _Echo(EmulatedService):
    def __init__(self, body=b'{"ok":true}'):
        super().__init__()
        self.body = body

    def route(self, request: RecordedRequest):
        return 200, {"Content-Type": "application/json"}, self.body


class TestUrllibTransportLive:
    def test_a_redirect_is_not_followed_and_the_token_never_reaches_the_target(self):
        with LiveServer(_Echo()) as attacker:
            redirector = _Redirector(attacker.url + "/steal")
            with LiveServer(redirector) as origin:
                client = ScopedHttpClient(
                    connector="t",
                    policy=EgressPolicy(origin.url, ROUTES, allow_insecure_loopback=True),
                    auth=lambda: {"Authorization": "Bearer do-not-leak"},
                    sleep=lambda _s: None,
                )
                with pytest.raises(HttpError, match="redirect"):
                    client.request("GET", "/things/x")
            assert redirector.requests[0].headers["authorization"] == "Bearer do-not-leak"
            assert attacker.service.requests == []  # never contacted at all

    def test_oversized_responses_are_refused(self):
        with LiveServer(_Echo(body=b"x" * 4096)) as server:
            client = ScopedHttpClient(
                connector="t",
                policy=EgressPolicy(server.url, ROUTES, allow_insecure_loopback=True),
                transport=UrllibTransport(max_response_bytes=1024),
            )
            with pytest.raises(Exception, match="exceeded 1024 bytes"):
                client.request("GET", "/things/x")

    def test_connection_refused_is_a_transport_error(self):
        server = LiveServer(_Echo()).start()
        url = server.url
        server.stop()
        client = ScopedHttpClient(
            connector="t",
            policy=EgressPolicy(url, ROUTES, allow_insecure_loopback=True),
            retry=RetryPolicy(max_attempts=2),
            sleep=lambda _s: None,
            timeout=2.0,
        )
        with pytest.raises(TransportError):
            client.request("GET", "/things/x")

    def test_http_error_statuses_come_back_as_responses(self):
        class NotFound(EmulatedService):
            def route(self, request):
                return _json(404, {"message": "Not Found"})

        with LiveServer(NotFound()) as server:
            response = UrllibTransport().send(
                HttpRequest("GET", server.url + "/things/x"), timeout=5
            )
        assert response.status == 404 and response.json() == {"message": "Not Found"}

    def test_injected_faults_drive_real_retries(self):
        service = _Echo()
        service.inject(503, times=2)
        with LiveServer(service) as server:
            client = ScopedHttpClient(
                connector="t",
                policy=EgressPolicy(server.url, ROUTES, allow_insecure_loopback=True),
                sleep=lambda _s: None,
            )
            assert client.request("GET", "/things/x").json() == {"ok": True}
        assert len(service.requests) == 3


# --------------------------------------------------------------------------- #
# Journal and blast radius
# --------------------------------------------------------------------------- #


@pytest.fixture(params=["memory", "sqlite"])
def journal(request, tmp_path):
    if request.param == "memory":
        yield MemoryJournal()
    else:
        j = SqliteJournal(tmp_path / "journal.sqlite")
        yield j
        j.close()


class TestJournal:
    def test_unknown_action_has_no_entry(self, journal):
        assert journal.lookup("a") is None

    def test_started_is_not_completed(self, journal):
        journal.start("a", connector="wazuh")
        entry = journal.lookup("a")
        assert entry is not None and not entry.completed and entry.outcome is None

    def test_completed_records_the_outcome(self, journal):
        journal.start("a", connector="wazuh")
        journal.complete("a", ExecutionOutcome(True, "done", reference="ref-1"))
        entry = journal.lookup("a")
        assert entry.completed and entry.outcome == ExecutionOutcome(True, "done", "ref-1")

    def test_restarting_a_completed_action_does_not_erase_it(self, journal):
        journal.start("a", connector="wazuh")
        journal.complete("a", ExecutionOutcome(True, "done"))
        journal.start("a", connector="wazuh")
        assert journal.lookup("a").completed


class TestSqliteJournalSurvivesTheProcess:
    def test_reopened_journal_remembers(self, tmp_path):
        path = tmp_path / "journal.sqlite"
        first = SqliteJournal(path)
        first.start("a", connector="github")
        first.complete("a", ExecutionOutcome(True, "opened #1", reference="url"))
        first.close()
        second = SqliteJournal(path)
        entry = second.lookup("a")
        second.close()
        assert entry.completed and entry.outcome.reference == "url"


class TestBlastRadius:
    def test_refuses_past_the_ceiling(self):
        limiter = BlastRadiusLimiter(max_actions=2, clock=FrozenClock(NOW))
        limiter.acquire("acme")
        limiter.acquire("acme")
        with pytest.raises(BlastRadiusExceeded, match="2 destructive"):
            limiter.acquire("acme")

    def test_tenants_are_independent(self):
        limiter = BlastRadiusLimiter(max_actions=1, clock=FrozenClock(NOW))
        limiter.acquire("acme")
        limiter.acquire("globex")

    def test_the_window_slides(self):
        clock = FrozenClock(NOW)
        limiter = BlastRadiusLimiter(max_actions=1, window=timedelta(minutes=10), clock=clock)
        limiter.acquire("acme")
        clock.advance(599)
        with pytest.raises(BlastRadiusExceeded):
            limiter.acquire("acme")
        clock.advance(1)
        limiter.acquire("acme")

    def test_release_returns_a_slot(self):
        limiter = BlastRadiusLimiter(max_actions=1, clock=FrozenClock(NOW))
        limiter.acquire("acme")
        limiter.release("acme")
        assert limiter.remaining("acme") == 1
        limiter.acquire("acme")

    def test_zero_is_not_a_ceiling(self):
        with pytest.raises(ValueError):
            BlastRadiusLimiter(max_actions=0)
