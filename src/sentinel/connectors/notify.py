"""Telling a human: Slack and signed webhooks (PRD Figure 2, "Slack/webhook stub").

``NOTIFY_ANALYST`` is the one action every trust tier may take without approval, so
it is the one most likely to fire in volume and the one whose content is least
reviewed. Two hazards follow, and both are handled here rather than trusted to the
caller:

1.  **Chat injection.** Slack's ``mrkdwn`` treats ``<...>`` as a control sequence:
    ``<!channel>`` pings everyone, ``<https://x|click here>`` is a disguised link. An
    action's rationale quotes triage output, which quotes alert fields, which an
    attacker can shape. Slack's documented escaping (``&`` ``<`` ``>`` to entities) is
    applied to every interpolated string, so no sequence in the text can become a
    control sequence in the message.
2.  **Forgery and replay** for generic webhooks. The receiver of a "contain this
    host" notification must be able to tell it came from here and is not a replay of
    last week's. :class:`SignedWebhookConnector` signs ``timestamp.body`` with
    HMAC-SHA256 and sends the action id as a delivery id; :func:`verify_signature` is
    the receiver's half, with a tolerance window and a constant-time compare.

Neither connector sends the alert's raw payload. The message carries the action,
its target, its rationale and the refs of its evidence — what an analyst needs to
decide whether to open the dashboard, and nothing an attacker wrote verbatim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import urllib.parse
from collections.abc import Callable
from datetime import datetime
from typing import Final

from sentinel.connectors.base import (
    Capability,
    ConnectorError,
    ExecutionOutcome,
    Secret,
    require_executable,
)
from sentinel.connectors.http import (
    CallRecord,
    EgressPolicy,
    RetryPolicy,
    Route,
    ScopedHttpClient,
    Transport,
    UrllibTransport,
)
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import GuardrailViolation
from sentinel.core.schemas import ActionRequest, ActionType

__all__ = [
    "SIGNATURE_HEADER",
    "LocalEnrichmentConnector",
    "SignedWebhookConnector",
    "SlackWebhookConnector",
    "escape_mrkdwn",
    "notification_fields",
    "sign_payload",
    "verify_signature",
]

SIGNATURE_HEADER: Final[str] = "X-Sentinel-Signature"
DELIVERY_HEADER: Final[str] = "X-Sentinel-Delivery"
_SLACK_PATH: Final[re.Pattern[str]] = re.compile(
    r"^/services/[A-Za-z0-9]{1,32}/[A-Za-z0-9]{1,32}/[A-Za-z0-9]{1,64}$"
)
_SLACK_TEXT_LIMIT: Final[int] = 2900


def escape_mrkdwn(text: str) -> str:
    """Slack's documented escaping. Order matters: ``&`` first."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def notification_fields(action: ActionRequest) -> dict[str, object]:
    """What a notification may say about an action. The single source for both kinds."""
    return {
        "action_id": action.action_id,
        "alert_id": action.alert_id,
        "tenant_id": action.tenant_id,
        "action_type": action.action_type.value,
        "target": action.target,
        "proposed_by": action.proposed_by.value,
        "risk_tier": action.risk_tier.value,
        "approval_status": action.approval_status.value,
        "rationale": action.rationale[:1000],
        "evidence": [item.ref for item in action.evidence[:16]],
    }


class SlackWebhookConnector:
    """Posts to one Slack incoming webhook. The webhook URL is itself the secret."""

    name = "slack"

    def __init__(
        self,
        *,
        webhook_url: Secret,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        allow_insecure_loopback: bool = False,
        dashboard_url: str | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(webhook_url.reveal())
        if not _SLACK_PATH.match(parsed.path):
            # Deliberately does not echo the URL: it is a bearer credential.
            raise ValueError("slack: webhook URL does not have the /services/T/B/X shape")
        self._path = parsed.path
        self._capabilities = frozenset({Capability.NOTIFY})
        self.dashboard_url = dashboard_url
        self.http = ScopedHttpClient(
            connector=self.name,
            policy=EgressPolicy(
                base_url=f"{parsed.scheme}://{parsed.netloc}",
                routes=(Route("webhook.post", "POST", re.escape(parsed.path)),),
                allow_insecure_loopback=allow_insecure_loopback,
            ),
            transport=transport or UrllibTransport(),
            retry=retry or RetryPolicy(),
            **({"sleep": sleep} if sleep is not None else {}),
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def observe(self, callback: Callable[[CallRecord], None] | None) -> None:
        self.http.on_call = callback

    def render(self, action: ActionRequest) -> dict[str, object]:
        """The Slack payload. Public so the escaping can be tested without a network."""
        fields = notification_fields(action)
        header = (
            f"*Sentinel Mesh* · `{escape_mrkdwn(str(fields['action_type']))}` on "
            f"`{escape_mrkdwn(str(fields['target']))}` "
            f"({escape_mrkdwn(str(fields['risk_tier']))}, "
            f"{escape_mrkdwn(str(fields['approval_status']))})"
        )
        body = escape_mrkdwn(str(fields["rationale"]))
        refs = ", ".join(f"`{escape_mrkdwn(str(ref))}`" for ref in fields["evidence"])  # type: ignore[union-attr]
        footer = f"action `{fields['action_id']}` · alert `{fields['alert_id']}`"
        if self.dashboard_url:
            footer += f" · <{self.dashboard_url}|open in dashboard>"
        parts = (header, body, refs and f"Evidence: {refs}", footer)
        text = "\n".join(part for part in parts if part)
        if len(text) > _SLACK_TEXT_LIMIT:
            text = text[: _SLACK_TEXT_LIMIT - 20] + "\n_(truncated)_"
        return {
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
        }

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        require_executable(action, connector=self.name)
        if action.action_type is not ActionType.NOTIFY_ANALYST:
            raise GuardrailViolation(
                f"slack: cannot execute {action.action_type.value}; it sends notifications"
            )
        # Slack webhooks have no idempotency key; a retry after a timeout can
        # duplicate a message. For a notification that is the right trade — a
        # duplicate ping is noise, a lost one is a missed incident — so the POST is
        # retried on 429/5xx by passing the action id as the key.
        response = self.http.request(
            "POST",
            self._path,
            json_body=self.render(action),
            idempotency_key=action.action_id,
            expect=(200,),
        )
        if response.text().strip() != "ok":
            raise ConnectorError(f"slack: webhook answered {response.text()[:60]!r}, not 'ok'")
        return ExecutionOutcome(
            succeeded=True,
            detail=f"notified Slack about {action.action_type.value} on {action.target}",
            reference=f"slack:{action.action_id}",
        )


def sign_payload(secret: Secret, body: bytes, timestamp: int) -> str:
    """``t=<unix>,v1=<hex HMAC-SHA256(secret, "<unix>.<body>")>``."""
    mac = hmac.new(
        secret.reveal().encode("utf-8"),
        str(timestamp).encode("ascii") + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    return f"t={timestamp},v1={mac}"


def verify_signature(
    secret: Secret,
    header: str,
    body: bytes,
    *,
    now: datetime,
    tolerance_seconds: int = 300,
) -> bool:
    """The receiver's check. False on any malformation, skew, or mismatch."""
    parts: dict[str, str] = {}
    for item in header.split(","):
        key, sep, value = item.strip().partition("=")
        if not sep:
            return False
        parts.setdefault(key, value)
    timestamp_text = parts.get("t", "")
    signature = parts.get("v1", "")
    if not timestamp_text.isdigit() or not signature:
        return False
    timestamp = int(timestamp_text)
    if abs(now.timestamp() - timestamp) > tolerance_seconds:
        return False
    expected = sign_payload(secret, body, timestamp).split("v1=", 1)[1]
    return hmac.compare_digest(expected, signature)


class SignedWebhookConnector:
    """POSTs a signed JSON notification to one configured endpoint."""

    name = "webhook"

    def __init__(
        self,
        *,
        url: str,
        signing_secret: Secret,
        clock: Clock | None = None,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        allow_insecure_loopback: bool = False,
    ) -> None:
        parsed = urllib.parse.urlsplit(url)
        if not parsed.path or parsed.path == "/":
            raise ValueError("webhook: the URL needs a path")
        self._secret = signing_secret
        self._clock = clock or SystemClock()
        self._path = parsed.path
        self._capabilities = frozenset({Capability.NOTIFY})
        self.http = ScopedHttpClient(
            connector=self.name,
            policy=EgressPolicy(
                base_url=f"{parsed.scheme}://{parsed.netloc}",
                routes=(Route("webhook.post", "POST", re.escape(parsed.path)),),
                allow_insecure_loopback=allow_insecure_loopback,
            ),
            transport=transport or UrllibTransport(),
            retry=retry or RetryPolicy(),
            **({"sleep": sleep} if sleep is not None else {}),
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def observe(self, callback: Callable[[CallRecord], None] | None) -> None:
        self.http.on_call = callback

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        require_executable(action, connector=self.name)
        if action.action_type is not ActionType.NOTIFY_ANALYST:
            raise GuardrailViolation(
                f"webhook: cannot execute {action.action_type.value}; it sends notifications"
            )
        payload = {"event": "sentinel.action", **notification_fields(action)}
        # Signed over the exact bytes sent, so the body is serialised here and the
        # client is told not to re-serialise it differently.
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        timestamp = int(self._clock.now().timestamp())
        self.http.request(
            "POST",
            self._path,
            json_body=payload,
            headers={
                SIGNATURE_HEADER: sign_payload(self._secret, body, timestamp),
                DELIVERY_HEADER: action.action_id,
            },
            idempotency_key=action.action_id,
            expect=(200, 202, 204),
        )
        return ExecutionOutcome(
            succeeded=True,
            detail=f"delivered signed notification for {action.action_type.value}",
            reference=f"webhook:{action.action_id}",
        )


class LocalEnrichmentConnector:
    """``ENRICH_ONLY`` has no external side effect, and this connector proves it.

    It exists so the router can refuse *every* action it has no connector for — fail
    closed — without making "monitor this" an error. It records the request and
    touches nothing outside the process.
    """

    name = "enrichment"

    def __init__(self) -> None:
        self._capabilities = frozenset({Capability.ENRICH})
        self.recorded: list[ActionRequest] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def execute(self, action: ActionRequest) -> ExecutionOutcome:
        require_executable(action, connector=self.name)
        if action.action_type is not ActionType.ENRICH_ONLY:
            raise GuardrailViolation(
                f"enrichment: cannot execute {action.action_type.value}; it has no side effects"
            )
        self.recorded.append(action)
        return ExecutionOutcome(
            succeeded=True,
            detail=f"enrichment requested for {action.target}; no external side effect",
        )
