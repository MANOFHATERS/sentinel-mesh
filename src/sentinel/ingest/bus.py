"""Layer 2 — the event bus.

The PRD specifies Redis Streams as a Kafka stand-in. This module defines the
narrow :class:`EventBus` protocol the rest of the system codes against, plus two
implementations:

:class:`InMemoryEventBus`
    A faithful reimplementation of the Redis Streams consumer-group semantics the
    platform actually depends on — monotonic ``ms-seq`` ids, per-group cursors, a
    pending-entries list, acknowledgement, claiming of stale entries, and
    ``MAXLEN`` trimming. It exists so the entire pipeline, including Part 3's
    agents, is testable and demoable with no server running. Tests assert the two
    implementations agree.

:class:`RedisStreamsEventBus`
    The production path. ``redis`` is imported lazily, so the foundation neither
    requires nor installs it.

Delivery semantics are **at-least-once**, deliberately. Exactly-once across a
network is a marketing claim; at-least-once plus idempotent alert ids (see
:meth:`~sentinel.core.schemas.Alert.derive_id`) is a design that actually holds,
and it is why replaying a dataset twice produces the same alert ids rather than
duplicates the pipeline cannot recognise.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import BusError

__all__ = [
    "EventBus",
    "InMemoryEventBus",
    "Message",
    "RedisStreamsEventBus",
    "StreamId",
]


@dataclass(frozen=True, order=True, slots=True)
class StreamId:
    """A Redis-Streams-shaped id: milliseconds plus an intra-millisecond counter.

    Ordering is lexicographic-by-component, not string order, because ``"10-0"``
    sorts before ``"9-0"`` as a string and that bug is silent and awful.
    """

    ms: int
    seq: int

    def __str__(self) -> str:
        return f"{self.ms}-{self.seq}"

    @classmethod
    def parse(cls, text: str) -> StreamId:
        try:
            ms_text, seq_text = text.split("-", 1)
            return cls(int(ms_text), int(seq_text))
        except (ValueError, AttributeError) as exc:
            raise BusError(f"malformed stream id {text!r}; expected '<ms>-<seq>'") from exc

    def next_seq(self) -> StreamId:
        return StreamId(self.ms, self.seq + 1)


#: The id every consumer group's cursor starts at, and the sentinel "before
#: everything" value. Declared outside the dataclass body so it stays a class
#: attribute rather than becoming a third field.
StreamId.ZERO = StreamId(0, 0)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class Message:
    """One delivered event."""

    id: StreamId
    topic: str
    payload: Mapping[str, Any]
    delivery_count: int = 1

    @property
    def is_redelivery(self) -> bool:
        return self.delivery_count > 1


@runtime_checkable
class EventBus(Protocol):
    """The only bus surface the platform depends on."""

    def publish(self, topic: str, payload: Mapping[str, Any]) -> StreamId:
        """Append one event and return its id."""
        ...

    def consume(
        self,
        topic: str,
        *,
        group: str,
        consumer: str,
        count: int = 10,
        create_group: bool = True,
    ) -> list[Message]:
        """Read up to ``count`` undelivered events for ``group``."""
        ...

    def ack(self, topic: str, *, group: str, ids: list[StreamId]) -> int:
        """Acknowledge processing; returns how many ids were actually pending."""
        ...

    def pending(self, topic: str, *, group: str) -> list[StreamId]:
        """Ids delivered to ``group`` but not yet acknowledged."""
        ...

    def length(self, topic: str) -> int:
        """Number of events currently retained in ``topic``."""
        ...


# --------------------------------------------------------------------------- #
# In-memory implementation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _PendingEntry:
    id: StreamId
    consumer: str
    delivered_at: float
    delivery_count: int


@dataclass(slots=True)
class _Group:
    cursor: StreamId = StreamId.ZERO
    pending: OrderedDict[StreamId, _PendingEntry] = field(default_factory=OrderedDict)


@dataclass(slots=True)
class _Stream:
    entries: OrderedDict[StreamId, Mapping[str, Any]] = field(default_factory=OrderedDict)
    groups: dict[str, _Group] = field(default_factory=dict)
    last_id: StreamId = StreamId.ZERO


class InMemoryEventBus:
    """Process-local bus with Redis-Streams consumer-group semantics.

    ``maxlen`` caps retention per topic the way ``XADD ... MAXLEN`` does, so a
    long replay cannot exhaust memory. Trimming an entry that is still pending is
    reported through :meth:`dropped_unacked`, because silently losing an
    unacknowledged security event is precisely the failure a SOC must not have
    happen invisibly.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        maxlen: int | None = 100_000,
    ) -> None:
        if maxlen is not None and maxlen < 1:
            raise ValueError("maxlen must be positive or None for unbounded")
        self._clock = clock or SystemClock()
        self._maxlen = maxlen
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.RLock()
        self._dropped_unacked = 0
        self._ms_counter = itertools.count()

    # --- internals ----------------------------------------------------------

    def _stream(self, topic: str) -> _Stream:
        if not topic or "\n" in topic:
            raise ValueError(f"invalid topic {topic!r}")
        stream = self._streams.get(topic)
        if stream is None:
            stream = _Stream()
            self._streams[topic] = stream
        return stream

    def _next_id(self, stream: _Stream) -> StreamId:
        # Derive milliseconds from the injected clock so a FrozenClock yields a
        # deterministic id sequence, and fall back to the intra-ms counter when
        # several events share a millisecond — exactly what Redis does.
        now_ms = int(self._clock.monotonic() * 1000)
        if now_ms > stream.last_id.ms:
            return StreamId(now_ms, 0)
        return stream.last_id.next_seq()

    # --- EventBus -----------------------------------------------------------

    def publish(self, topic: str, payload: Mapping[str, Any]) -> StreamId:
        if not isinstance(payload, Mapping):
            raise BusError(f"payload must be a mapping, got {type(payload).__name__}")
        with self._lock:
            stream = self._stream(topic)
            message_id = self._next_id(stream)
            stream.entries[message_id] = dict(payload)
            stream.last_id = message_id
            self._trim(stream)
            return message_id

    def _trim(self, stream: _Stream) -> None:
        if self._maxlen is None:
            return
        while len(stream.entries) > self._maxlen:
            oldest_id, _ = stream.entries.popitem(last=False)
            for group in stream.groups.values():
                if group.pending.pop(oldest_id, None) is not None:
                    self._dropped_unacked += 1

    def consume(
        self,
        topic: str,
        *,
        group: str,
        consumer: str,
        count: int = 10,
        create_group: bool = True,
    ) -> list[Message]:
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            stream = self._stream(topic)
            grp = stream.groups.get(group)
            if grp is None:
                if not create_group:
                    raise BusError(f"consumer group {group!r} does not exist on {topic!r}")
                grp = _Group()
                stream.groups[group] = grp

            out: list[Message] = []
            for message_id, payload in stream.entries.items():
                if message_id <= grp.cursor:
                    continue
                grp.cursor = message_id
                grp.pending[message_id] = _PendingEntry(
                    id=message_id,
                    consumer=consumer,
                    delivered_at=self._clock.monotonic(),
                    delivery_count=1,
                )
                out.append(Message(id=message_id, topic=topic, payload=payload))
                if len(out) >= count:
                    break
            return out

    def ack(self, topic: str, *, group: str, ids: list[StreamId]) -> int:
        with self._lock:
            stream = self._stream(topic)
            grp = stream.groups.get(group)
            if grp is None:
                return 0
            return sum(1 for message_id in ids if grp.pending.pop(message_id, None) is not None)

    def pending(self, topic: str, *, group: str) -> list[StreamId]:
        with self._lock:
            stream = self._stream(topic)
            grp = stream.groups.get(group)
            return [] if grp is None else list(grp.pending)

    def length(self, topic: str) -> int:
        with self._lock:
            return len(self._stream(topic).entries)

    # --- recovery -----------------------------------------------------------

    def claim_stale(
        self,
        topic: str,
        *,
        group: str,
        consumer: str,
        min_idle_seconds: float,
        count: int = 10,
    ) -> list[Message]:
        """Reassign entries a dead consumer never acknowledged (``XAUTOCLAIM``).

        Without this, one crashed worker parks alerts in its pending list forever
        — a silent detection gap, which is the worst kind.
        """
        if min_idle_seconds < 0:
            raise ValueError("min_idle_seconds must be >= 0")
        with self._lock:
            stream = self._stream(topic)
            grp = stream.groups.get(group)
            if grp is None:
                return []
            now = self._clock.monotonic()
            claimed: list[Message] = []
            for entry in list(grp.pending.values()):
                if now - entry.delivered_at < min_idle_seconds:
                    continue
                payload = stream.entries.get(entry.id)
                if payload is None:  # trimmed away
                    grp.pending.pop(entry.id, None)
                    continue
                entry.consumer = consumer
                entry.delivered_at = now
                entry.delivery_count += 1
                claimed.append(
                    Message(
                        id=entry.id,
                        topic=topic,
                        payload=payload,
                        delivery_count=entry.delivery_count,
                    )
                )
                if len(claimed) >= count:
                    break
            return claimed

    def dropped_unacked(self) -> int:
        """How many unacknowledged events retention has discarded. Should stay 0."""
        with self._lock:
            return self._dropped_unacked

    def topics(self) -> list[str]:
        with self._lock:
            return sorted(self._streams)

    def reset(self) -> None:
        """Drop all state. Tests only."""
        with self._lock:
            self._streams.clear()
            self._dropped_unacked = 0


# --------------------------------------------------------------------------- #
# Redis implementation
# --------------------------------------------------------------------------- #


class RedisStreamsEventBus:
    """Redis Streams bus. Requires the ``bus`` extra (``pip install -e .[bus]``).

    Payload values are JSON-encoded per field because Redis stream fields are
    flat bytes; decoding is symmetric so a round-trip through this bus is
    value-preserving for the JSON-native types the canonical layer already
    guarantees.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        maxlen: int | None = 1_000_000,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._redis = client
        else:
            try:
                import redis
            except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
                raise BusError(
                    "RedisStreamsEventBus needs the 'redis' package: pip install -e '.[bus]'. "
                    "Use InMemoryEventBus for local development and tests."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._maxlen = maxlen

    @staticmethod
    def _encode(payload: Mapping[str, Any]) -> dict[str, str]:
        import json

        return {key: json.dumps(value, separators=(",", ":")) for key, value in payload.items()}

    @staticmethod
    def _decode(fields: Mapping[str, Any]) -> dict[str, Any]:
        import json

        out: dict[str, Any] = {}
        for key, value in fields.items():
            text = value.decode() if isinstance(value, bytes) else str(value)
            try:
                out[key] = json.loads(text)
            except json.JSONDecodeError:
                out[key] = text  # tolerate hand-written entries from redis-cli
        return out

    def publish(self, topic: str, payload: Mapping[str, Any]) -> StreamId:
        raw = self._redis.xadd(topic, self._encode(payload), maxlen=self._maxlen, approximate=True)
        return StreamId.parse(raw if isinstance(raw, str) else raw.decode())

    def consume(
        self,
        topic: str,
        *,
        group: str,
        consumer: str,
        count: int = 10,
        create_group: bool = True,
    ) -> list[Message]:
        if create_group:
            self._ensure_group(topic, group)
        response = self._redis.xreadgroup(group, consumer, {topic: ">"}, count=count, block=None)
        messages: list[Message] = []
        for _stream_name, entries in response or []:
            for raw_id, fields in entries:
                message_id = StreamId.parse(raw_id if isinstance(raw_id, str) else raw_id.decode())
                messages.append(
                    Message(id=message_id, topic=topic, payload=self._decode(fields))
                )
        return messages

    def _ensure_group(self, topic: str, group: str) -> None:
        try:
            self._redis.xgroup_create(topic, group, id="0", mkstream=True)
        except Exception as exc:  # redis raises ResponseError for BUSYGROUP
            if "BUSYGROUP" not in str(exc):
                raise BusError(f"could not create consumer group {group!r}: {exc}") from exc

    def ack(self, topic: str, *, group: str, ids: list[StreamId]) -> int:
        if not ids:
            return 0
        return int(self._redis.xack(topic, group, *[str(i) for i in ids]))

    def pending(self, topic: str, *, group: str) -> list[StreamId]:
        try:
            entries = self._redis.xpending_range(topic, group, min="-", max="+", count=1000)
        except Exception:
            return []
        out: list[StreamId] = []
        for entry in entries:
            raw = entry["message_id"]
            out.append(StreamId.parse(raw if isinstance(raw, str) else raw.decode()))
        return out

    def length(self, topic: str) -> int:
        return int(self._redis.xlen(topic))
