"""Normalizers: F-01 — real dataset quirks, unit harmonization, canonical output.

The test that earns its keep here is
:meth:`TestUnitHarmonization.test_duration_units_agree_across_datasets`. CIC-IDS2017
reports flow duration in microseconds and UNSW-NB15 in seconds. A model trained on
one and validated on the other is off by 10^6, which crashes nothing and silently
invalidates every cross-dataset number in the evaluation report.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from sentinel.core.errors import NormalizationError
from sentinel.core.schemas import AlertSource, Severity
from sentinel.ingest.normalizer import (
    UNIFIED_FEATURES,
    CICIDS2017Normalizer,
    UNSWNB15Normalizer,
    canonicalize_column,
    canonicalize_row,
    get_normalizer,
    severity_for_family,
)

CIC_ROW = {
    "Flow ID": "flow-1",
    " Source IP": "192.168.10.5",
    " Source Port": "52134",
    " Destination IP": "10.0.0.9",
    " Destination Port": "443",
    " Protocol": "6",
    " Timestamp": "5/7/2017 3:30:15",
    " Flow Duration": "2000000",
    " Total Fwd Packets": "10",
    " Total Backward Packets": "8",
    "Total Length of Fwd Packets": "1200",
    " Total Length of Bwd Packets": "800",
    "Flow Bytes/s": "1000.0",
    " Flow Packets/s": "9.0",
    " Label": "BENIGN",
}

UNSW_ROW = {
    "id": "1",
    "srcip": "175.45.176.3",
    "sport": "52134",
    "dstip": "149.171.126.9",
    "dsport": "443",
    "proto": "tcp",
    "state": "FIN",
    "dur": "2.0",
    "sbytes": "1200",
    "dbytes": "800",
    "spkts": "10",
    "dpkts": "8",
    "Stime": "1500000000",
    "attack_cat": "",
    "Label": "0",
}


class TestColumnCanonicalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (" Destination Port", "destination_port"),
            ("Flow Bytes/s", "flow_bytes_s"),
            ("Total Length of Fwd Packets", "total_length_of_fwd_packets"),
            (" Label", "label"),
            ("attack_cat", "attack_cat"),
            ("Flow ID", "flow_id"),
            ("  MULTIPLE   SPACES  ", "multiple_spaces"),
            ("ct_dst_src_ltm", "ct_dst_src_ltm"),
            ("Fwd IAT Min", "fwd_iat_min"),
        ],
    )
    def test_real_column_names(self, raw, expected):
        assert canonicalize_column(raw) == expected

    def test_is_idempotent(self):
        once = canonicalize_column(" Destination Port")
        assert canonicalize_column(once) == once

    def test_row_keys_are_canonicalized(self):
        assert canonicalize_row({" Label": "BENIGN"}) == {"label": "BENIGN"}

    def test_ambiguous_header_is_rejected(self):
        # Two differently-spelled columns collapsing to one name with different
        # values means the file's header is broken; picking one silently loses data.
        with pytest.raises(NormalizationError, match="ambiguous"):
            canonicalize_row({" Label": "BENIGN", "Label": "DDoS"})

    def test_identical_values_under_a_collision_are_tolerated(self):
        assert canonicalize_row({" Label": "BENIGN", "Label": "BENIGN"}) == {"label": "BENIGN"}


class TestUnitHarmonization:
    def test_duration_units_agree_across_datasets(self, cic_normalizer, unsw_normalizer):
        """2,000,000 us (CIC) and 2.0 s (UNSW) describe the same flow."""
        cic = cic_normalizer.normalize(CIC_ROW, row_index=0)
        unsw = unsw_normalizer.normalize(UNSW_ROW, row_index=0)
        assert cic.features["duration_seconds"] == pytest.approx(2.0)
        assert unsw.features["duration_seconds"] == pytest.approx(2.0)

    def test_identical_flows_produce_identical_unified_features(
        self, cic_normalizer, unsw_normalizer
    ):
        # The whole point of the unified space: the same physical flow expressed in
        # two schemas must land on the same vector, or cross-dataset validation is
        # measuring a schema difference rather than a distribution shift.
        cic = cic_normalizer.normalize(CIC_ROW, row_index=0)
        unsw = unsw_normalizer.normalize(UNSW_ROW, row_index=0)
        for name in UNIFIED_FEATURES:
            assert cic.features[name] == pytest.approx(unsw.features[name]), name

    def test_scale_constants_are_declared_and_differ(self):
        assert CICIDS2017Normalizer.DURATION_SCALE_TO_SECONDS == 1e-6
        assert UNSWNB15Normalizer.DURATION_SCALE_TO_SECONDS == 1.0


class TestCicDefects:
    def test_infinity_in_rate_columns_is_sanitized(self, cic_normalizer):
        row = {**CIC_ROW, "Flow Bytes/s": "Infinity", " Flow Packets/s": "Infinity"}
        alert = cic_normalizer.normalize(row, row_index=0)
        # Rates are recomputed from the counters, so the dataset's Infinity never
        # reaches a feature at all.
        assert math.isfinite(alert.features["bytes_per_second"])
        assert alert.features["bytes_per_second"] == pytest.approx(1000.0)

    def test_nan_in_rate_columns_is_sanitized(self, cic_normalizer):
        row = {**CIC_ROW, "Flow Bytes/s": "NaN"}
        alert = cic_normalizer.normalize(row, row_index=0)
        assert all(
            not isinstance(v, float) or math.isfinite(v) for v in alert.features.values()
        )

    def test_zero_duration_flow_yields_a_large_finite_rate(self, cic_normalizer):
        row = {**CIC_ROW, " Flow Duration": "0", "Flow Bytes/s": "Infinity"}
        alert = cic_normalizer.normalize(row, row_index=0)
        rate = alert.features["bytes_per_second"]
        assert math.isfinite(rate)
        # 2000 bytes over the 1-microsecond measurement floor.
        assert rate == pytest.approx(2000 / 1e-6)

    @pytest.mark.parametrize(
        "label",
        [
            "Web Attack – Brute Force",  # proper en-dash
            "Web Attack \x96 Brute Force",  # cp1252 mojibake in mirrored copies
            "Web Attack - Brute Force",  # plain hyphen
            "Web Attack — Brute Force",  # em-dash
            "  web attack – brute force  ",  # padded and lowercase
        ],
    )
    def test_every_web_attack_label_spelling_maps_to_one_family(self, cic_normalizer, label):
        alert = cic_normalizer.normalize({**CIC_ROW, " Label": label}, row_index=0)
        assert alert.ground_truth_label == "web_attack"

    @pytest.mark.parametrize(
        ("label", "family"),
        [
            ("BENIGN", "benign"),
            ("DDoS", "ddos"),
            ("DoS Hulk", "dos"),
            ("DoS GoldenEye", "dos"),
            ("DoS slowloris", "dos"),
            ("DoS Slowhttptest", "dos"),
            ("PortScan", "recon"),
            ("FTP-Patator", "brute_force"),
            ("SSH-Patator", "brute_force"),
            ("Bot", "botnet"),
            ("Infiltration", "infiltration"),
            ("Heartbleed", "exploit"),
        ],
    )
    def test_all_real_cic_labels_are_mapped(self, cic_normalizer, label, family):
        alert = cic_normalizer.normalize({**CIC_ROW, " Label": label}, row_index=0)
        assert alert.ground_truth_label == family

    def test_unknown_label_is_reported_not_silently_benign(self):
        # Defaulting an unrecognised label to benign would delete attacks from the
        # evaluation set and inflate every metric.
        normalizer = CICIDS2017Normalizer(tenant_id="acme")
        alert = normalizer.normalize({**CIC_ROW, " Label": "NewAttack2027"}, row_index=0)
        assert alert.ground_truth_label == "unknown"
        assert normalizer.report.unknown_labels == {"newattack2027": 1}

    def test_timestamp_is_parsed_day_first(self, cic_normalizer):
        # The capture ran 3-7 July 2017, so 5/7/2017 is 5 July, not 7 May.
        alert = cic_normalizer.normalize(CIC_ROW, row_index=0)
        assert alert.timestamp == datetime(2017, 7, 5, 3, 30, 15, tzinfo=UTC)

    def test_unparseable_timestamp_falls_back_rather_than_guessing(self, cic_normalizer):
        alert = cic_normalizer.normalize({**CIC_ROW, " Timestamp": "garbage"}, row_index=0)
        assert alert.timestamp.year == 2017

    def test_explicit_observed_at_overrides_the_row_timestamp(self, cic_normalizer):
        when = datetime(2026, 1, 1, tzinfo=UTC)
        alert = cic_normalizer.normalize(CIC_ROW, row_index=0, observed_at=when)
        assert alert.timestamp == when

    def test_a_non_cic_row_is_rejected_rather_than_coerced(self, cic_normalizer):
        with pytest.raises(NormalizationError, match="does not look like CIC-IDS2017"):
            cic_normalizer.normalize({"totally": "unrelated"}, row_index=0)

    def test_non_numeric_value_is_rejected_in_strict_mode(self, cic_normalizer):
        with pytest.raises(NormalizationError, match="non-numeric"):
            cic_normalizer.normalize({**CIC_ROW, " Flow Duration": "not-a-number"}, row_index=0)


class TestUnswDefects:
    @pytest.mark.parametrize(
        ("attack_cat", "family"),
        [
            ("", "benign"),
            ("Normal", "benign"),
            ("DoS", "dos"),
            ("Reconnaissance", "recon"),
            (" Reconnaissance ", "recon"),
            ("Exploits", "exploit"),
            ("Fuzzers", "fuzzers"),
            (" Fuzzers ", "fuzzers"),
            ("Generic", "generic"),
            ("Backdoor", "backdoor"),
            ("Backdoors", "backdoor"),  # the real CSVs use both spellings
            ("Analysis", "analysis"),
            ("Shellcode", "shellcode"),
            ("Worms", "worm"),
        ],
    )
    def test_all_real_unsw_labels_are_mapped(self, unsw_normalizer, attack_cat, family):
        row = {**UNSW_ROW, "attack_cat": attack_cat, "Label": "0" if family == "benign" else "1"}
        alert = unsw_normalizer.normalize(row, row_index=0)
        assert alert.ground_truth_label == family

    def test_blank_attack_cat_with_positive_label_is_not_benign(self, unsw_normalizer):
        # The train/test CSVs leave attack_cat blank but set label=1. Reading that
        # as benign would relabel real attacks as normal traffic.
        row = {**UNSW_ROW, "attack_cat": "", "Label": "1"}
        alert = unsw_normalizer.normalize(row, row_index=0)
        assert alert.ground_truth_label == "unknown"

    def test_blank_attack_cat_with_zero_label_is_benign(self, unsw_normalizer):
        alert = unsw_normalizer.normalize({**UNSW_ROW, "Label": "0"}, row_index=0)
        assert alert.ground_truth_label == "benign"

    def test_protocol_name_is_mapped_to_a_number(self, unsw_normalizer):
        alert = unsw_normalizer.normalize(UNSW_ROW, row_index=0)
        assert alert.features["protocol_number"] == 6
        assert alert.protocol == "tcp"

    @pytest.mark.parametrize(("proto", "number"), [("tcp", 6), ("udp", 17), ("icmp", 1)])
    def test_common_protocols(self, unsw_normalizer, proto, number):
        alert = unsw_normalizer.normalize({**UNSW_ROW, "proto": proto}, row_index=0)
        assert alert.features["protocol_number"] == number

    def test_unknown_protocol_becomes_none_not_zero(self, unsw_normalizer):
        # Mapping an unknown protocol to 0 would make it indistinguishable from
        # HOPOPT, and the one-hot 'other' bucket would never fire.
        alert = unsw_normalizer.normalize({**UNSW_ROW, "proto": "quic-ish"}, row_index=0)
        assert alert.features["protocol_number"] is None

    def test_epoch_timestamp_is_parsed(self, unsw_normalizer):
        alert = unsw_normalizer.normalize(UNSW_ROW, row_index=0)
        assert alert.timestamp == datetime.fromtimestamp(1_500_000_000, tz=UTC)

    def test_implausible_epoch_is_ignored(self, unsw_normalizer):
        # The train/test CSVs have no Stime; a stray small integer must not be read
        # as a 1970 timestamp.
        alert = unsw_normalizer.normalize({**UNSW_ROW, "Stime": "42"}, row_index=0)
        assert alert.timestamp.year >= 2017

    def test_a_non_unsw_row_is_rejected(self, unsw_normalizer):
        with pytest.raises(NormalizationError, match="does not look like UNSW-NB15"):
            unsw_normalizer.normalize({"totally": "unrelated"}, row_index=0)


class TestCanonicalOutput:
    def test_emits_exactly_the_unified_feature_contract(self, cic_normalizer):
        alert = cic_normalizer.normalize(CIC_ROW, row_index=0)
        assert set(UNIFIED_FEATURES) <= set(alert.features)

    def test_derived_features_are_correct(self, cic_normalizer):
        alert = cic_normalizer.normalize(CIC_ROW, row_index=0)
        f = alert.features
        assert f["total_bytes"] == 2000.0
        assert f["total_packets"] == 18.0
        assert f["src_mean_packet_bytes"] == pytest.approx(120.0)
        assert f["dst_mean_packet_bytes"] == pytest.approx(100.0)
        assert f["bytes_ratio_src_to_total"] == pytest.approx(0.6)

    def test_zero_packet_flow_does_not_divide_by_zero(self, cic_normalizer):
        row = {**CIC_ROW, " Total Backward Packets": "0", " Total Length of Bwd Packets": "0"}
        alert = cic_normalizer.normalize(row, row_index=0)
        assert alert.features["dst_mean_packet_bytes"] == 0.0
        assert alert.features["bytes_ratio_src_to_total"] == 1.0

    def test_zero_byte_flow_does_not_divide_by_zero(self, cic_normalizer):
        row = {
            **CIC_ROW,
            "Total Length of Fwd Packets": "0",
            " Total Length of Bwd Packets": "0",
        }
        alert = cic_normalizer.normalize(row, row_index=0)
        assert alert.features["bytes_ratio_src_to_total"] == 0.0

    def test_raw_payload_preserves_the_original_row(self, cic_normalizer):
        alert = cic_normalizer.normalize(CIC_ROW, row_index=0)
        import json

        restored = json.loads(alert.raw_payload.raw)
        assert restored[" Destination Port"] == "443"
        assert restored[" Label"] == "BENIGN"

    def test_raw_payload_survives_non_finite_source_values(self, cic_normalizer):
        # The payload is the forensic record, so Infinity must be preserved there as
        # a string rather than dropped or turned into a JSON-invalid literal.
        alert = cic_normalizer.normalize({**CIC_ROW, "Flow Bytes/s": "Infinity"}, row_index=0)
        import json

        assert json.loads(alert.raw_payload.raw)["Flow Bytes/s"] == "Infinity"

    def test_tenant_is_carried(self, cic_normalizer):
        assert cic_normalizer.normalize(CIC_ROW, row_index=0).tenant_id == "acme"

    def test_source_is_recorded(self, cic_normalizer):
        assert cic_normalizer.normalize(CIC_ROW, row_index=0).source is AlertSource.NETWORK_IDS

    def test_ids_are_deterministic_across_runs(self):
        first = CICIDS2017Normalizer(tenant_id="acme").normalize(CIC_ROW, row_index=5)
        second = CICIDS2017Normalizer(tenant_id="acme").normalize(CIC_ROW, row_index=5)
        assert first.alert_id == second.alert_id

    def test_ids_differ_between_tenants(self):
        a = CICIDS2017Normalizer(tenant_id="acme").normalize(CIC_ROW, row_index=5)
        b = CICIDS2017Normalizer(tenant_id="globex").normalize(CIC_ROW, row_index=5)
        assert a.alert_id != b.alert_id

    def test_ports_are_parsed(self, cic_normalizer):
        alert = cic_normalizer.normalize(CIC_ROW, row_index=0)
        assert alert.dst_port == 443
        assert alert.src_port == 52134

    def test_out_of_range_port_becomes_none(self, cic_normalizer):
        alert = cic_normalizer.normalize({**CIC_ROW, " Destination Port": "99999"}, row_index=0)
        assert alert.dst_port is None


class TestLenientMode:
    def test_lenient_mode_returns_none_and_records_the_rejection(self):
        normalizer = CICIDS2017Normalizer(tenant_id="acme", strict=False)
        assert normalizer.normalize({"nope": 1}, row_index=0) is None
        assert normalizer.report.rows_rejected == 1
        assert normalizer.report.failure_rate == 1.0

    def test_report_tracks_throughput(self, cic_normalizer):
        for index in range(5):
            cic_normalizer.normalize(CIC_ROW, row_index=index)
        assert cic_normalizer.report.rows_seen == 5
        assert cic_normalizer.report.alerts_emitted == 5
        assert cic_normalizer.report.failure_rate == 0.0

    def test_report_summary_is_informative(self, cic_normalizer):
        cic_normalizer.normalize(CIC_ROW, row_index=0)
        summary = cic_normalizer.report.summary()
        assert "1/1 rows normalized" in summary
        assert "0 rejected" in summary


class TestRegistryAndTaxonomy:
    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("cic-ids2017", CICIDS2017Normalizer),
            ("cicids2017", CICIDS2017Normalizer),
            ("CIC-IDS2017", CICIDS2017Normalizer),
            ("unsw-nb15", UNSWNB15Normalizer),
            ("unsw_nb15", UNSWNB15Normalizer),
        ],
    )
    def test_lookup(self, name, cls):
        assert isinstance(get_normalizer(name, tenant_id="t"), cls)

    def test_unknown_dataset_is_refused_with_the_known_list(self):
        with pytest.raises(NormalizationError, match="known:"):
            get_normalizer("not-a-dataset")

    def test_tenant_id_is_mandatory(self):
        with pytest.raises(ValueError, match="multi-tenant"):
            CICIDS2017Normalizer(tenant_id="")

    @pytest.mark.parametrize(
        ("family", "at_least"),
        [
            ("benign", Severity.INFO),
            ("recon", Severity.MEDIUM),
            ("brute_force", Severity.HIGH),
            ("botnet", Severity.CRITICAL),
            ("infiltration", Severity.CRITICAL),
        ],
    )
    def test_severity_priors_are_ordered_sensibly(self, family, at_least):
        assert severity_for_family(family) == at_least

    def test_unknown_family_is_not_silently_low(self):
        # An unmapped family defaulting to low severity would hide novel attacks.
        assert severity_for_family("something-new") >= Severity.MEDIUM
