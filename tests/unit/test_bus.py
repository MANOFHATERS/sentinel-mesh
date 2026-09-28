"""Event bus: consumer-group semantics, at-least-once delivery, bounded retention."""

from __future__ import annotations

import pytest

from sentinel.core.clock import FrozenClock
from sentinel.core.errors import BusError
from sentinel.ingest.bus import InMemoryEventBus, StreamId

TOPIC = "alerts.raw"


class TestStreamId:
    def test_parses_and_renders(self):
        assert str(StreamId.parse("1234-5")) == "1234-5"

    def test_ordering_is_numeric_not_lexicographic(self):
        # "10-0" < "9-0" as strings. Getting this wrong silently reorders a stream.
        assert StreamId(9, 0) < StreamId(10, 0)
        assert sorted([StreamId(10, 0), StreamId(9, 0)]) == [StreamId(9, 0), StreamId(10, 0)]

    def test_intra_millisecond_ordering(self):
        assert StreamId(5, 1) < StreamId(5, 2)

    def test_next_seq_increments_within_the_millisecond(self):
        assert StreamId(5, 1).next_seq() == StreamId(5, 2)

    @pytest.mark.parametrize("bad", ["", "abc", "1", "x-y", None])
    def test_malformed_ids_rejected(self, bad):
        with pytest.raises(BusError, match="malformed stream id"):
            StreamId.parse(bad)  # type: ignore[arg-type]

    def test_zero_sentinel(self):
        assert StreamId(0, 0) == StreamId.ZERO


class TestPublishConsume:
    def test_published_event_is_consumable(self, bus):
        bus.publish(TOPIC, {"kind": "alert", "id": "a1"})
        messages = bus.consume(TOPIC, group="triage", consumer="worker-1")
        assert len(messages) == 1
        assert messages[0].payload["id"] == "a1"

    def test_ids_increase_monotonically(self, bus, clock):
        ids = []
        for index in range(5):
            clock.advance(0.01)
            ids.append(bus.publish(TOPIC, {"i": index}))
        assert ids == sorted(ids)
        assert len(set(ids)) == 5

    def test_same_millisecond_publishes_get_distinct_ids(self, bus):
        # The clock does not advance, so every id shares a millisecond.
        ids = [bus.publish(TOPIC, {"i": i}) for i in range(4)]
        assert len(set(ids)) == 4
        assert {i.ms for i in ids} == {ids[0].ms}

    def test_order_is_preserved(self, bus):
        for index in range(10):
            bus.publish(TOPIC, {"i": index})
        messages = bus.consume(TOPIC, group="g", consumer="c", count=10)
        assert [m.payload["i"] for m in messages] == list(range(10))

    def test_count_limits_the_batch(self, bus):
        for index in range(10):
            bus.publish(TOPIC, {"i": index})
        assert len(bus.consume(TOPIC, group="g", consumer="c", count=3)) == 3

    def test_cursor_advances_between_reads(self, bus):
        for index in range(6):
            bus.publish(TOPIC, {"i": index})
        first = bus.consume(TOPIC, group="g", consumer="c", count=3)
        second = bus.consume(TOPIC, group="g", consumer="c", count=3)
        assert [m.payload["i"] for m in first] == [0, 1, 2]
        assert [m.payload["i"] for m in second] == [3, 4, 5]

    def test_empty_topic_yields_nothing(self, bus):
        assert bus.consume(TOPIC, group="g", consumer="c") == []

    def test_length_reports_retained_events(self, bus):
        for index in range(4):
            bus.publish(TOPIC, {"i": index})
        assert bus.length(TOPIC) == 4

    def test_non_mapping_payload_rejected(self, bus):
        with pytest.raises(BusError, match="must be a mapping"):
            bus.publish(TOPIC, ["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_invalid_count_rejected(self, bus):
        with pytest.raises(ValueError, match="count must be"):
            bus.consume(TOPIC, group="g", consumer="c", count=0)

    def test_invalid_topic_rejected(self, bus):
        with pytest.raises(ValueError, match="invalid topic"):
            bus.publish("", {"a": 1})

    def test_payload_is_copied_not_aliased(self, bus):
        payload = {"mutable": 1}
        bus.publish(TOPIC, payload)
        payload["mutable"] = 999
        assert bus.consume(TOPIC, group="g", consumer="c")[0].payload["mutable"] == 1


class TestConsumerGroups:
    def test_groups_have_independent_cursors(self, bus):
        for index in range(3):
            bus.publish(TOPIC, {"i": index})
        triage = bus.consume(TOPIC, group="triage", consumer="c", count=10)
        anomaly = bus.consume(TOPIC, group="anomaly", consumer="c", count=10)
        # Fan-out: every agent group sees every alert. This is why Triage and the
        # anomaly scorer can both consume the same feed without coordinating.
        assert len(triage) == 3
        assert len(anomaly) == 3

    def test_consumers_in_one_group_share_the_work(self, bus):
        for index in range(6):
            bus.publish(TOPIC, {"i": index})
        first = bus.consume(TOPIC, group="triage", consumer="w1", count=3)
        second = bus.consume(TOPIC, group="triage", consumer="w2", count=3)
        seen = [m.payload["i"] for m in first] + [m.payload["i"] for m in second]
        assert sorted(seen) == list(range(6))

    def test_create_group_false_refuses_an_unknown_group(self, bus):
        bus.publish(TOPIC, {"i": 0})
        with pytest.raises(BusError, match="does not exist"):
            bus.consume(TOPIC, group="ghost", consumer="c", create_group=False)


class TestAcknowledgement:
    def test_delivered_events_are_pending_until_acked(self, bus):
        bus.publish(TOPIC, {"i": 0})
        messages = bus.consume(TOPIC, group="g", consumer="c")
        assert bus.pending(TOPIC, group="g") == [messages[0].id]
        assert bus.ack(TOPIC, group="g", ids=[messages[0].id]) == 1
        assert bus.pending(TOPIC, group="g") == []

    def test_acking_an_unknown_id_counts_zero(self, bus):
        bus.publish(TOPIC, {"i": 0})
        bus.consume(TOPIC, group="g", consumer="c")
        assert bus.ack(TOPIC, group="g", ids=[StreamId(999, 999)]) == 0

    def test_acking_twice_counts_once(self, bus):
        bus.publish(TOPIC, {"i": 0})
        message = bus.consume(TOPIC, group="g", consumer="c")[0]
        assert bus.ack(TOPIC, group="g", ids=[message.id]) == 1
        assert bus.ack(TOPIC, group="g", ids=[message.id]) == 0

    def test_ack_on_unknown_group_is_a_no_op(self, bus):
        assert bus.ack(TOPIC, group="ghost", ids=[StreamId(1, 0)]) == 0

    def test_pending_on_unknown_group_is_empty(self, bus):
        assert bus.pending(TOPIC, group="ghost") == []


class TestRecovery:
    def test_stale_pending_entries_can_be_claimed(self, bus, clock):
        """A crashed worker must not park alerts forever — that is a detection gap."""
        bus.publish(TOPIC, {"i": 0})
        bus.consume(TOPIC, group="g", consumer="dead-worker")
        clock.advance(120)
        claimed = bus.claim_stale(
            TOPIC, group="g", consumer="live-worker", min_idle_seconds=60
        )
        assert len(claimed) == 1
        assert claimed[0].delivery_count == 2
        assert claimed[0].is_redelivery

    def test_fresh_entries_are_not_claimed(self, bus, clock):
        bus.publish(TOPIC, {"i": 0})
        bus.consume(TOPIC, group="g", consumer="worker")
        clock.advance(5)
        assert bus.claim_stale(TOPIC, group="g", consumer="other", min_idle_seconds=60) == []

    def test_acked_entries_are_never_claimed(self, bus, clock):
        bus.publish(TOPIC, {"i": 0})
        message = bus.consume(TOPIC, group="g", consumer="w")[0]
        bus.ack(TOPIC, group="g", ids=[message.id])
        clock.advance(600)
        assert bus.claim_stale(TOPIC, group="g", consumer="o", min_idle_seconds=1) == []

    def test_claim_on_unknown_group_is_empty(self, bus):
        assert bus.claim_stale(TOPIC, group="ghost", consumer="c", min_idle_seconds=0) == []

    def test_negative_idle_rejected(self, bus):
        with pytest.raises(ValueError, match="min_idle_seconds"):
            bus.claim_stale(TOPIC, group="g", consumer="c", min_idle_seconds=-1)


class TestRetention:
    def test_maxlen_trims_the_oldest(self, clock):
        bus = InMemoryEventBus(clock=clock, maxlen=5)
        for index in range(10):
            bus.publish(TOPIC, {"i": index})
        assert bus.length(TOPIC) == 5
        remaining = [m.payload["i"] for m in bus.consume(TOPIC, group="g", consumer="c", count=10)]
        assert remaining == [5, 6, 7, 8, 9]

    def test_trimming_an_unacked_event_is_counted_not_silent(self, clock):
        """Losing an unacknowledged security event invisibly is the worst failure."""
        bus = InMemoryEventBus(clock=clock, maxlen=3)
        for index in range(3):
            bus.publish(TOPIC, {"i": index})
        bus.consume(TOPIC, group="g", consumer="c", count=3)  # all pending
        assert bus.dropped_unacked() == 0
        for index in range(3, 6):
            bus.publish(TOPIC, {"i": index})
        assert bus.dropped_unacked() == 3

    def test_unbounded_retention_is_available(self, clock):
        bus = InMemoryEventBus(clock=clock, maxlen=None)
        for index in range(100):
            bus.publish(TOPIC, {"i": index})
        assert bus.length(TOPIC) == 100

    def test_invalid_maxlen_rejected(self):
        with pytest.raises(ValueError, match="maxlen"):
            InMemoryEventBus(maxlen=0)

    def test_claiming_a_trimmed_entry_drops_it_from_pending(self, clock):
        bus = InMemoryEventBus(clock=clock, maxlen=2)
        bus.publish(TOPIC, {"i": 0})
        bus.consume(TOPIC, group="g", consumer="c")
        for index in range(1, 5):
            bus.publish(TOPIC, {"i": index})
        clock.advance(600)
        assert bus.claim_stale(TOPIC, group="g", consumer="c2", min_idle_seconds=1) == []


class TestHousekeeping:
    def test_topics_are_listed_sorted(self, bus):
        bus.publish("z.topic", {"a": 1})
        bus.publish("a.topic", {"a": 1})
        assert bus.topics() == ["a.topic", "z.topic"]

    def test_reset_clears_everything(self, bus):
        bus.publish(TOPIC, {"a": 1})
        bus.consume(TOPIC, group="g", consumer="c")
        bus.reset()
        assert bus.topics() == []
        assert bus.length(TOPIC) == 0

    def test_satisfies_the_eventbus_protocol(self, bus):
        from sentinel.ingest.bus import EventBus

        assert isinstance(bus, EventBus)

    def test_redis_bus_also_satisfies_the_protocol(self):
        # Checked structurally without a server, so the two implementations cannot
        # drift apart in signature.
        from sentinel.ingest.bus import EventBus, RedisStreamsEventBus

        for method in ("publish", "consume", "ack", "pending", "length"):
            assert hasattr(RedisStreamsEventBus, method)
        assert set(EventBus.__protocol_attrs__) <= set(dir(RedisStreamsEventBus))

    def test_redis_bus_reports_a_useful_error_without_the_extra(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "redis":
                raise ModuleNotFoundError("No module named 'redis'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from sentinel.ingest.bus import RedisStreamsEventBus

        with pytest.raises(BusError, match=r"\[bus\]"):
            RedisStreamsEventBus()

    def test_redis_payload_encoding_round_trips(self):
        from sentinel.ingest.bus import RedisStreamsEventBus

        payload = {"kind": "alert", "n": 3, "f": 1.5, "b": True, "nested": {"a": [1, 2]}}
        encoded = RedisStreamsEventBus._encode(payload)
        assert RedisStreamsEventBus._decode(encoded) == payload

    def test_redis_decode_tolerates_hand_written_entries(self):
        from sentinel.ingest.bus import RedisStreamsEventBus

        # Someone doing XADD from redis-cli writes bare strings, not JSON.
        assert RedisStreamsEventBus._decode({"note": "hello"}) == {"note": "hello"}


class TestDeterminism:
    def test_frozen_clock_gives_a_reproducible_id_sequence(self):
        def run() -> list[str]:
            bus = InMemoryEventBus(clock=FrozenClock())
            ids = []
            for index in range(5):
                ids.append(str(bus.publish(TOPIC, {"i": index})))
            return ids

        assert run() == run()
