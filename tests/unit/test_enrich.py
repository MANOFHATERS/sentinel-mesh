"""Session-context enrichment: causality, bounded memory, and attack signatures.

:meth:`TestCausality.test_prefix_enrichment_matches_full_stream_prefix` is the test
this module exists for. A window feature built by looking at the whole dataset
scores wonderfully offline and is unimplementable in production, because at serving
time the future has not happened yet. That failure is invisible in aggregate
metrics, so it has to be pinned structurally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentinel.core.schemas import Alert, AlertSource
from sentinel.ingest.enrich import CONTEXT_FEATURES, SessionContextEnricher

BASE = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def make_alert(
    *,
    index: int,
    seconds: float,
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.9",
    dst_port: int = 443,
    total_bytes: float = 1000.0,
) -> Alert:
    when = BASE + timedelta(seconds=seconds)
    return Alert(
        alert_id=f"a{index}",
        tenant_id="acme",
        source=AlertSource.NETWORK_IDS,
        timestamp=when,
        ingested_at=when,
        asset_id=dst_ip,
        raw_payload=f"flow {index}",
        features={"total_bytes": total_bytes, "dst_port": dst_port},
        src_ip=src_ip,
        dst_ip=dst_ip,
        dst_port=dst_port,
    )


class TestContract:
    def test_every_context_feature_is_added(self):
        enriched = SessionContextEnricher().enrich(make_alert(index=0, seconds=0))
        for name in CONTEXT_FEATURES:
            assert name in enriched.features

    def test_original_features_are_preserved(self):
        enriched = SessionContextEnricher().enrich(make_alert(index=0, seconds=0))
        assert enriched.features["total_bytes"] == 1000.0

    def test_original_alert_is_not_mutated(self):
        alert = make_alert(index=0, seconds=0)
        SessionContextEnricher().enrich(alert)
        assert "src_flow_count_window" not in alert.features

    def test_identity_fields_are_unchanged(self):
        alert = make_alert(index=0, seconds=0)
        enriched = SessionContextEnricher().enrich(alert)
        assert enriched.alert_id == alert.alert_id
        assert enriched.timestamp == alert.timestamp

    def test_all_context_features_are_finite(self):
        enricher = SessionContextEnricher()
        for index in range(50):
            enriched = enricher.enrich(make_alert(index=index, seconds=index * 0.1))
            for name in CONTEXT_FEATURES:
                assert isinstance(enriched.features[name], float)


class TestCausality:
    def test_first_alert_sees_an_empty_window(self):
        enriched = SessionContextEnricher().enrich(make_alert(index=0, seconds=0))
        assert enriched.features["src_flow_count_window"] == 0.0
        assert enriched.features["dst_flow_count_window"] == 0.0

    def test_an_alert_never_counts_itself(self):
        enricher = SessionContextEnricher()
        first = enricher.enrich(make_alert(index=0, seconds=0))
        second = enricher.enrich(make_alert(index=1, seconds=1))
        assert first.features["src_flow_count_window"] == 0.0
        assert second.features["src_flow_count_window"] == 1.0

    def test_prefix_enrichment_matches_full_stream_prefix(self):
        """The anti-leakage invariant: features depend only on the past.

        If enriching the first 10 alerts gave different answers than enriching all
        100 and keeping the first 10, the enricher would be reading the future.
        """
        stream = [make_alert(index=i, seconds=i * 0.3) for i in range(100)]
        full = SessionContextEnricher().enrich_all(stream)
        prefix = SessionContextEnricher().enrich_all(stream[:10])
        for whole, part in zip(full[:10], prefix, strict=True):
            assert {k: whole.features[k] for k in CONTEXT_FEATURES} == {
                k: part.features[k] for k in CONTEXT_FEATURES
            }

    def test_events_outside_the_window_are_evicted(self):
        enricher = SessionContextEnricher(window_seconds=60.0)
        for index in range(5):
            enricher.enrich(make_alert(index=index, seconds=index))
        # 10 minutes later, nothing from the earlier burst should remain in scope.
        late = enricher.enrich(make_alert(index=99, seconds=600))
        assert late.features["src_flow_count_window"] == 0.0

    def test_window_boundary_is_inclusive_of_exactly_window_seconds_ago(self):
        enricher = SessionContextEnricher(window_seconds=60.0)
        enricher.enrich(make_alert(index=0, seconds=0))
        boundary = enricher.enrich(make_alert(index=1, seconds=60.0))
        assert boundary.features["src_flow_count_window"] == 1.0

    def test_just_outside_the_window_is_excluded(self):
        enricher = SessionContextEnricher(window_seconds=60.0)
        enricher.enrich(make_alert(index=0, seconds=0))
        outside = enricher.enrich(make_alert(index=1, seconds=60.001))
        assert outside.features["src_flow_count_window"] == 0.0

    def test_out_of_order_arrivals_are_counted_not_hidden(self):
        enricher = SessionContextEnricher()
        enricher.enrich(make_alert(index=0, seconds=10))
        enricher.enrich(make_alert(index=1, seconds=5))
        assert enricher.stats()["out_of_order_events"] == 1

    def test_deterministic_for_the_same_stream(self):
        stream = [make_alert(index=i, seconds=i * 0.2) for i in range(60)]
        first = SessionContextEnricher().enrich_all(stream)
        second = SessionContextEnricher().enrich_all(stream)
        assert [a.canonical_hash() for a in first] == [a.canonical_hash() for a in second]


class TestAttackSignatures:
    def test_brute_force_shows_high_count_on_one_port(self):
        enricher = SessionContextEnricher()
        last = None
        for index in range(300):
            last = enricher.enrich(
                make_alert(index=index, seconds=index * 0.1, dst_port=22, total_bytes=1500)
            )
        assert last.features["src_flow_count_window"] > 200
        assert last.features["src_distinct_dst_ports_window"] == 1.0

    def test_port_scan_shows_high_distinct_ports(self):
        enricher = SessionContextEnricher()
        last = None
        for index in range(400):
            last = enricher.enrich(
                make_alert(index=index, seconds=index * 0.01, dst_port=(index % 1000) + 1)
            )
        assert last.features["src_distinct_dst_ports_window"] > 300

    def test_ddos_shows_fan_in_but_no_per_source_signal(self):
        """The distributed case: every per-source count stays at ~1 by construction."""
        enricher = SessionContextEnricher()
        last = None
        for index in range(500):
            last = enricher.enrich(
                make_alert(
                    index=index,
                    seconds=index * 0.01,
                    src_ip=f"172.16.{index // 250}.{index % 250}",
                    dst_ip="10.0.0.9",
                    dst_port=80,
                )
            )
        assert last.features["src_flow_count_window"] <= 2.0
        assert last.features["dst_flow_count_window"] > 400
        assert last.features["dst_distinct_src_ips_window"] > 400

    def test_beacon_has_near_zero_interarrival_variation(self):
        enricher = SessionContextEnricher()
        last = None
        for index in range(20):
            # Perfectly regular 3-second beacon.
            last = enricher.enrich(make_alert(index=index, seconds=index * 3.0, dst_port=4444))
        assert last.features["src_interarrival_mean_window"] == pytest.approx(3.0)
        assert last.features["src_interarrival_cv_window"] < 0.01

    def test_human_traffic_has_high_interarrival_variation(self):
        enricher = SessionContextEnricher()
        gaps = [0.2, 3.0, 0.1, 12.0, 0.4, 7.5, 0.3, 20.0, 1.1, 5.0]
        elapsed = 0.0
        last = None
        for index, gap in enumerate(gaps):
            elapsed += gap
            last = enricher.enrich(make_alert(index=index, seconds=elapsed))
        assert last.features["src_interarrival_cv_window"] > 0.5

    def test_target_sweep_shows_distinct_destinations(self):
        enricher = SessionContextEnricher()
        last = None
        for index in range(100):
            last = enricher.enrich(
                make_alert(index=index, seconds=index * 0.1, dst_ip=f"10.0.1.{index % 200}")
            )
        assert last.features["src_distinct_dst_ips_window"] > 90

    def test_byte_volume_accumulates(self):
        enricher = SessionContextEnricher()
        last = None
        for index in range(10):
            last = enricher.enrich(
                make_alert(index=index, seconds=index * 0.5, total_bytes=1000.0)
            )
        assert last.features["src_bytes_sum_window"] == pytest.approx(9000.0)

    def test_sent_from_and_sent_to_histories_do_not_bleed_into_each_other(self):
        """Fan-out and fan-in are separate indexes, deliberately.

        Merging them would blur the one signal that catches a distributed attack:
        a DDoS victim's fan-in count would be diluted by its own outbound traffic,
        and a scanner's fan-out count by the replies it receives.
        """
        enricher = SessionContextEnricher()
        enricher.enrich(make_alert(index=0, seconds=0, src_ip="1.1.1.1", dst_ip="2.2.2.2"))
        reverse = enricher.enrich(
            make_alert(index=1, seconds=1, src_ip="2.2.2.2", dst_ip="1.1.1.1")
        )
        # 2.2.2.2 has been a destination but never a source; 1.1.1.1 the reverse.
        # So on the return flow, both histories are legitimately empty.
        assert reverse.features["src_flow_count_window"] == 0.0
        assert reverse.features["dst_flow_count_window"] == 0.0

        # Repeating the same direction does accumulate.
        again = enricher.enrich(
            make_alert(index=2, seconds=2, src_ip="1.1.1.1", dst_ip="2.2.2.2")
        )
        assert again.features["src_flow_count_window"] == 1.0
        assert again.features["dst_flow_count_window"] == 1.0


class TestBoundedMemory:
    def test_key_cardinality_is_capped_with_lru_eviction(self):
        """An attacker choosing a million source addresses must not exhaust memory."""
        enricher = SessionContextEnricher(max_tracked_keys=100)
        for index in range(1000):
            enricher.enrich(
                make_alert(
                    index=index,
                    seconds=index * 0.01,
                    src_ip=f"172.16.{index // 250}.{index % 250}",
                )
            )
        stats = enricher.stats()
        assert stats["tracked_sources"] <= 100
        assert stats["evictions"] > 0

    def test_per_key_events_are_capped(self):
        enricher = SessionContextEnricher(per_key_cap=50)
        last = None
        for index in range(500):
            last = enricher.enrich(make_alert(index=index, seconds=index * 0.01))
        # Saturating is correct: beyond a few thousand flows/minute from one host
        # the exact number carries no additional information.
        assert last.features["src_flow_count_window"] <= 50

    def test_stats_are_reported(self):
        enricher = SessionContextEnricher()
        for index in range(5):
            enricher.enrich(make_alert(index=index, seconds=index))
        stats = enricher.stats()
        assert stats["processed"] == 5
        assert stats["tracked_sources"] == 1

    def test_reset_clears_state(self):
        enricher = SessionContextEnricher()
        enricher.enrich(make_alert(index=0, seconds=0))
        enricher.reset()
        assert enricher.stats()["processed"] == 0
        after = enricher.enrich(make_alert(index=1, seconds=1))
        assert after.features["src_flow_count_window"] == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"window_seconds": 0}, "window_seconds"),
            ({"window_seconds": -1}, "window_seconds"),
            ({"max_tracked_keys": 0}, "max_tracked_keys"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SessionContextEnricher(**kwargs)


class TestMissingFields:
    def test_alert_without_ips_falls_back_to_the_asset_id(self):
        when = BASE
        alert = Alert(
            alert_id="a0",
            tenant_id="acme",
            source=AlertSource.EDR,
            timestamp=when,
            ingested_at=when,
            asset_id="laptop-17",
            raw_payload="process started",
            features={},
        )
        enricher = SessionContextEnricher()
        enricher.enrich(alert)
        second = enricher.enrich(alert.updated(alert_id="a1"))
        # Grouping by asset keeps host-level telemetry (EDR, cloud) usable even with
        # no network addresses present.
        assert second.features["src_flow_count_window"] == 1.0

    def test_missing_total_bytes_contributes_zero(self):
        when = BASE
        alert = Alert(
            alert_id="a0",
            tenant_id="acme",
            source=AlertSource.EDR,
            timestamp=when,
            ingested_at=when,
            asset_id="laptop-17",
            raw_payload="x",
            features={},
        )
        enricher = SessionContextEnricher()
        enricher.enrich(alert)
        second = enricher.enrich(alert.updated(alert_id="a1"))
        assert second.features["src_bytes_sum_window"] == 0.0
