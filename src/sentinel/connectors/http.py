"""The only way a connector talks to the network: a scoped, allowlisted HTTP client.

Every real connector in this package is a thin translation from an
:class:`~sentinel.core.schemas.ActionRequest` to a handful of HTTP calls. What makes
them safe is not the translation — it is that every call goes through
:class:`ScopedHttpClient`, which enforces, in this order:

1.  **The egress allowlist** (:class:`EgressPolicy`). A connector is bound to one
    origin and a fixed set of ``(method, path-pattern)`` routes. Anything else raises
    :class:`~sentinel.connectors.base.EgressDenied` before a socket is opened. This
    is where "the Git connector has no merge" stops being a missing method and
    becomes a property of the process: ``PUT /repos/o/r/pulls/7/merge`` matches no
    route, whatever code asks for it.
2.  **No redirects.** Python's ``urllib`` follows redirects by default *and forwards
    the ``Authorization`` header to the new location*, including a different host.
    A compromised or misconfigured endpoint answering ``302 Location:
    https://attacker/`` would receive the token. :class:`UrllibTransport` refuses to
    follow; a 3xx is returned as a response and treated as a failure. Tested against
    a second live server that asserts it never sees the header.
3.  **Bounded retries, only where retrying is safe.** ``GET``/``PUT``/``DELETE`` are
    idempotent; ``POST``/``PATCH`` are retried only when the caller supplies an
    idempotency key. A ``429`` or ``503`` honours ``Retry-After`` up to a ceiling;
    past the ceiling the call fails now rather than parking the graph thread for an
    hour. ``4xx`` other than ``429`` is never retried — it will not get better.
4.  **Bounded responses.** A body larger than ``max_response_bytes`` is refused. A
    connector reads a few kilobytes of JSON; a hostile endpoint streaming gigabytes
    should cost the process nothing.
5.  **A record of every attempt** (:class:`CallRecord`), handed to ``on_call``. The
    router writes these to the audit chain as ``connector_called`` rows: method, route
    *name*, status, attempt, duration and a digest of the request body — never the
    body, never a header, never the URL's query string.
"""

from __future__ import annotations

import email.utils
import hashlib
import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from sentinel.connectors.base import ConnectorError, EgressDenied

__all__ = [
    "CallRecord",
    "EgressPolicy",
    "HttpError",
    "HttpRequest",
    "HttpResponse",
    "RetryPolicy",
    "Route",
    "ScopedHttpClient",
    "Transport",
    "TransportError",
    "UrllibTransport",
    "quote_segment",
]

DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_MAX_RESPONSE_BYTES: Final[int] = 1 << 20  # 1 MiB
_IDEMPOTENT_METHODS: Final[frozenset[str]] = frozenset({"GET", "HEAD", "PUT", "DELETE"})


class TransportError(ConnectorError):
    """The request did not produce an HTTP response (DNS, refused, timeout, reset)."""


class HttpError(ConnectorError):
    """The remote system answered with a status the connector cannot proceed past."""

    def __init__(self, message: str, *, status: int, route: str) -> None:
        super().__init__(message)
        self.status = status
        self.route = route


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    url: str
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes | None = None


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    #: Lower-cased header names.
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConnectorError(f"response body is not JSON: {exc}") from exc

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class Transport(Protocol):
    def send(self, request: HttpRequest, *, timeout: float) -> HttpResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Return 3xx responses to the caller instead of following them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(slots=True)
class UrllibTransport:
    """A stdlib HTTP transport that never follows redirects and caps response size.

    Stdlib rather than ``requests`` or ``httpx`` for the reason the rest of the repo
    uses NumPy rather than torch: the one-command install stays dependency-free, and
    the few behaviours that matter for security (redirects, size, timeouts) are
    explicit here rather than defaults in a library this code does not own.
    """

    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    _opener: urllib.request.OpenerDirector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._opener = urllib.request.build_opener(_NoRedirect())

    def send(self, request: HttpRequest, *, timeout: float) -> HttpResponse:
        raw = urllib.request.Request(
            request.url,
            data=request.body,
            method=request.method,
            headers=dict(request.headers),
        )
        try:
            response = self._opener.open(raw, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # An HTTP status is a response, not a transport failure. Includes 3xx,
            # which _NoRedirect declined to follow.
            with exc:
                body = self._read(exc)
                return HttpResponse(
                    status=exc.code,
                    headers={k.lower(): v for k, v in exc.headers.items()},
                    body=body,
                )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise TransportError(f"{request.method} failed: {reason}") from exc
        with response:
            body = self._read(response)
            return HttpResponse(
                status=response.status,
                headers={k.lower(): v for k, v in response.headers.items()},
                body=body,
            )

    def _read(self, stream) -> bytes:
        body = stream.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise ConnectorError(
                f"response exceeded {self.max_response_bytes} bytes; refusing to buffer it"
            )
        return body


def quote_segment(value: str) -> str:
    """Percent-encode one path segment. Refuses ``.`` and ``..`` outright.

    ``urllib.parse.quote`` leaves dots alone, so a target of ``..`` would survive
    encoding and a server normalising the path would resolve it to the parent
    resource. An action target is attacker-influenced text that a human approved as a
    name, and it must stay a name.
    """
    text = str(value)
    if text in {"", ".", ".."}:
        raise EgressDenied(f"refusing path segment {text!r}")
    return urllib.parse.quote(text, safe="")


@dataclass(frozen=True, slots=True)
class Route:
    """One permitted call: a method, a path pattern, and the query keys it may carry."""

    name: str
    method: str
    #: Not in ``repr``: a webhook's route pattern is its URL path, which is the secret.
    pattern: str = field(repr=False)
    query: frozenset[str] = field(default_factory=frozenset)
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", self.method.upper())
        object.__setattr__(self, "query", frozenset(self.query))
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def matches(self, method: str, path: str) -> bool:
        return method == self.method and self._compiled.fullmatch(path) is not None


#: One path segment in a route pattern: anything but a slash. Encoded, so a ``/``
#: inside a value arrives as ``%2F`` and cannot add a segment.
SEGMENT: Final[str] = r"[^/]+"


@dataclass(frozen=True, slots=True)
class EgressPolicy:
    """An origin and the routes a connector may call on it.

    ``https`` is required. The single exception is a loopback origin, which is what
    the local sandbox and a developer's port-forward look like; it is still an
    explicit flag rather than a silent default, and a non-loopback ``http`` origin
    raises even with the flag set.
    """

    base_url: str
    routes: tuple[Route, ...]
    allow_insecure_loopback: bool = False
    _scheme: str = field(init=False, repr=False, compare=False)
    _netloc: str = field(init=False, repr=False, compare=False)
    _prefix: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise EgressDenied(f"unusable connector origin {self.base_url!r}")
        if parsed.username or parsed.password:
            raise EgressDenied("credentials in a connector URL end up in logs; use a Credential")
        if parsed.query or parsed.fragment:
            raise EgressDenied("a connector origin carries no query string or fragment")
        if parsed.scheme == "http" and not (
            self.allow_insecure_loopback and _is_loopback(parsed.hostname)
        ):
            raise EgressDenied(
                f"plaintext origin {self.base_url!r} refused; https is required "
                "except for an explicitly allowed loopback sandbox"
            )
        if not self.routes:
            raise EgressDenied("an egress policy with no routes permits nothing; say so")
        object.__setattr__(self, "_scheme", parsed.scheme)
        object.__setattr__(self, "_netloc", parsed.netloc.lower())
        object.__setattr__(self, "_prefix", parsed.path.rstrip("/"))

    @property
    def origin(self) -> str:
        return f"{self._scheme}://{self._netloc}"

    def authorize(
        self, method: str, path: str, query: Mapping[str, str] | None = None
    ) -> tuple[Route, str]:
        """Return the matching route and the full URL, or raise ``EgressDenied``."""
        method = method.upper()
        if not path.startswith("/"):
            raise EgressDenied(f"path {path!r} must be absolute")
        decoded_segments = [urllib.parse.unquote(s) for s in path.split("/")[1:]]
        if any(segment in {".", ".."} for segment in decoded_segments):
            raise EgressDenied(f"path {path!r} contains a dot segment")
        for route in self.routes:
            if route.matches(method, path):
                keys = frozenset((query or {}).keys())
                if not keys <= route.query:
                    raise EgressDenied(
                        f"route {route.name} does not permit query key(s) "
                        f"{sorted(keys - route.query)}"
                    )
                url = f"{self.origin}{self._prefix}{path}"
                if query:
                    url += "?" + urllib.parse.urlencode(sorted(query.items()))
                return route, url
        raise EgressDenied(
            f"{method} {path} is not an allowed route for {self.origin}; this connector "
            f"may call only: {', '.join(r.name for r in self.routes)}"
        )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 4.0
    #: A Retry-After longer than this fails the call rather than waiting.
    max_retry_after: float = 30.0
    retry_statuses: frozenset[int] = frozenset({429, 502, 503, 504})

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def backoff(self, attempt: int) -> float:
        return min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One attempt, as the audit chain records it. Contains nothing sensitive."""

    connector: str
    route: str
    method: str
    status: int | None
    attempt: int
    duration_ms: float
    request_sha256: str | None
    error: str | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "connector": self.connector,
            "route": self.route,
            "method": self.method,
            "status": self.status,
            "attempt": self.attempt,
            "duration_ms": round(self.duration_ms, 3),
            "request_sha256": self.request_sha256,
            "error": self.error,
        }


def _retry_after_seconds(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - now).total_seconds())


@dataclass(slots=True)
class ScopedHttpClient:
    """An HTTP client that can only reach what its :class:`EgressPolicy` permits."""

    connector: str
    policy: EgressPolicy
    transport: Transport = field(default_factory=UrllibTransport)
    #: Called per attempt to produce auth headers, so a token can be refreshed.
    auth: Callable[[], Mapping[str, str]] | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    sleep: Callable[[float], None] = time.sleep
    on_call: Callable[[CallRecord], None] | None = None
    user_agent: str = "sentinel-mesh-connector/0.1"
    base_headers: Mapping[str, str] = field(default_factory=dict)

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        content_type: str = "application/json",
        use_auth: bool = True,
        expect: tuple[int, ...] = (200, 201, 204),
        retryable: bool | None = None,
    ) -> HttpResponse:
        """Make one logical call (possibly several attempts). Raises on failure.

        ``expect`` lists the statuses that count as success. A status outside it
        raises :class:`HttpError` after retries are exhausted — except that callers
        may include e.g. ``404`` or ``422`` in ``expect`` when those are answers
        rather than failures (a lookup that found nothing, a ref that exists).

        ``retryable`` overrides the method-based default for a call the caller knows
        is safe to repeat without a key — a login, say, which is a ``POST`` only
        because it carries credentials in a body.
        """
        method = method.upper()
        route, url = self.policy.authorize(method, path, query)
        body: bytes | None = None
        digest: str | None = None
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
            digest = hashlib.sha256(body).hexdigest()

        if retryable is None:
            retryable = method in _IDEMPOTENT_METHODS or idempotency_key is not None
        attempts = self.retry.max_attempts if retryable else 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            merged: dict[str, str] = {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                **self.base_headers,
            }
            if body is not None:
                merged["Content-Type"] = content_type
            if idempotency_key is not None:
                merged["Idempotency-Key"] = idempotency_key
            if use_auth and self.auth is not None:
                merged.update(self.auth())
            if headers:
                merged.update(headers)
            request = HttpRequest(
                method=method, url=url, headers=tuple(merged.items()), body=body
            )
            started = time.perf_counter()
            try:
                response = self.transport.send(request, timeout=self.timeout)
            except TransportError as exc:
                self._record(route, method, None, attempt, started, digest, str(exc))
                last_error = exc
                if attempt < attempts:
                    self.sleep(self.retry.backoff(attempt))
                    continue
                raise
            self._record(route, method, response.status, attempt, started, digest, None)

            if response.status in expect:
                return response
            if 300 <= response.status < 400:
                raise HttpError(
                    f"{route.name}: refusing to follow a {response.status} redirect "
                    "(it would carry the credential to another location)",
                    status=response.status,
                    route=route.name,
                )
            if response.status in self.retry.retry_statuses and attempt < attempts:
                wait = _retry_after_seconds(
                    response.headers.get("retry-after"), datetime.now(UTC)
                )
                if wait is not None and wait > self.retry.max_retry_after:
                    raise HttpError(
                        f"{route.name}: remote asked to retry after {wait:.0f}s, beyond "
                        f"the {self.retry.max_retry_after:.0f}s ceiling",
                        status=response.status,
                        route=route.name,
                    )
                self.sleep(wait if wait is not None else self.retry.backoff(attempt))
                continue
            raise HttpError(
                f"{route.name}: {method} returned {response.status}: "
                f"{_short(response.text())}",
                status=response.status,
                route=route.name,
            )
        raise ConnectorError(  # pragma: no cover - the loop always returns or raises
            f"{route.name}: exhausted retries ({last_error})"
        )

    def _record(
        self,
        route: Route,
        method: str,
        status: int | None,
        attempt: int,
        started: float,
        digest: str | None,
        error: str | None,
    ) -> None:
        if self.on_call is None:
            return
        self.on_call(
            CallRecord(
                connector=self.connector,
                route=route.name,
                method=method,
                status=status,
                attempt=attempt,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                request_sha256=digest,
                error=None if error is None else error[:200],
            )
        )


def _short(text: str, limit: int = 200) -> str:
    """A truncated, single-line rendering of a remote error body for a message."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."
