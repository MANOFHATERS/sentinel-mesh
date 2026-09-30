"""The real-IP flow loader: window choice, enrichment, and the workspace it feeds.

The Parquet file is a generated fixture in the full UNSW-NB15 column layout (with source and
destination IPs and capture times); it exists so these tests need no download and is never
presented as real data.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentinel.dashboard.workspace import MeshModels
from sentinel.real.network import (
    RealDataError,
    _window_rows,
    analyse_flows,
    detect_format,
    find_dataset,
    has_addresses,
    train_real,
)
from tests.unit.test_real_data import write_unsw

EPOCH = 1_424_223_000  # 18 Feb 2015, the day the real capture was taken
FAMILIES = ["Exploits", "DoS", " Reconnaissance ", "Generic", " Fuzzers ", "Backdoor"]


def build_rows(n: int, seed: int = 4):
    """A time-ordered capture: benign client->server flows, with attack bursts from one host."""
    rng = random.Random(seed)
    rows = []
    burst_until = 0
    family = ""
    for i in range(n):
        if i >= burst_until and rng.random() < 0.02:
            burst_until = i + rng.randint(30, 80)
            family = rng.choice(FAMILIES)
        attack = i < burst_until
        cat = family if attack else None
        rows.append(
            {
                "srcip": f"175.45.176.{rng.randint(0, 3)}"
                if attack
                else f"59.166.0.{rng.randint(0, 9)}",
                "sport": str(rng.randint(1024, 65000)),
                "dstip": f"149.171.126.{rng.randint(0, 3) if attack else rng.randint(0, 19)}",
                "dsport": str(
                    rng.choice([21, 22, 25, 53, 80, 443, 445, 3389])
                    if attack
                    else rng.choice([80, 443, 53])
                ),
                "proto": "udp" if attack and rng.random() < 0.5 else "tcp",
                "state": "INT" if attack else "FIN",
                "dur": rng.expovariate(60) if attack else rng.expovariate(2),
                "sbytes": rng.randint(40, 400) if attack else rng.randint(300, 9000),
                "dbytes": rng.randint(0, 200) if attack else rng.randint(200, 20000),
                "spkts": rng.randint(1, 6) if attack else rng.randint(4, 40),
                "dpkts": rng.randint(0, 3) if attack else rng.randint(3, 40),
                "Stime": EPOCH + i // 3,
                "Ltime": EPOCH + i // 3,
                "attack_cat": cat,
                "label": 1 if attack else 0,
            }
        )
    return rows


def write_parquet(path, n=3600, row_group=250, seed=4):
    pq.write_table(pa.Table.from_pylist(build_rows(n, seed)), path, row_group_size=row_group)
    return path


def test_the_loader_prefers_a_file_with_real_addresses_over_one_without(tmp_path):
    folder = tmp_path / "data" / "raw" / "unsw-nb15"
    folder.mkdir(parents=True)
    write_unsw(folder / "UNSW_NB15_training-set.csv", rows=30)
    assert find_dataset(tmp_path).suffix == ".csv"
    write_parquet(folder / "full.parquet", n=300)
    found = find_dataset(tmp_path)
    assert found.suffix == ".parquet" and has_addresses(found)
    assert not has_addresses(folder / "UNSW_NB15_training-set.csv")


def test_a_parquet_file_without_flow_columns_is_not_a_flow_file(tmp_path):
    pq.write_table(pa.table({"a": [1, 2], "b": [3, 4]}), tmp_path / "x.parquet")
    assert detect_format(tmp_path / "x.parquet") is None and not has_addresses(
        tmp_path / "x.parquet"
    )
    assert detect_format(tmp_path / "missing.parquet") is None


def test_the_window_is_a_contiguous_run_chosen_for_families_and_attack_share(tmp_path):
    path = write_parquet(tmp_path / "full.parquet", n=6000)
    rows, total, detail = _window_rows(path, 1500)
    assert total == 6000 and len(rows) == 1500 and detail["window_rows"] == 1500
    indexes = [i for i, _ in rows]
    assert indexes == list(range(indexes[0], indexes[0] + 1500)), "a run, not a sample"
    assert detail["window_start_row"] == indexes[0]
    attacks = sum(r["label"] for _i, r in rows) / 1500
    families = {str(r["attack_cat"]).strip().lower() for _i, r in rows if r["attack_cat"]}
    assert len(families) >= 3 and 0.1 < attacks < 0.7
    assert all(v is not None for _i, r in rows for v in r.values()), "nulls become empty strings"
    assert _window_rows(path, 1500)[2] == detail, "deterministic"


def test_a_window_may_start_in_the_middle_of_a_row_group_and_ask_for_more_than_exists(tmp_path):
    path = write_parquet(tmp_path / "full.parquet", n=1000, row_group=300)
    rows, total, _ = _window_rows(path, 333)  # 333 is not a multiple of the 300-row groups
    assert len(rows) == 333
    every, _, detail = _window_rows(path, 5000)
    assert len(every) == 1000 == detail["window_rows"] and total == 1000


def test_real_flows_are_ordered_in_time_and_enriched_with_what_each_host_was_doing(tmp_path):
    path = write_parquet(tmp_path / "full.parquet", n=4000)
    _model, test, report = train_real(path, limit=2500, seed=3)
    assert report["has_addresses"] is True
    assert (
        report["window"]["window_rows"] == 2500
        and report["time_span"]["start"] < report["time_span"]["end"]
    )
    assert report["metrics"]["roc_auc"] > 0.9, (
        "bursts from one host are exactly what the context sees"
    )
    alert = test[0]
    assert alert.src_ip and alert.dst_ip and alert.asset_id == alert.dst_ip
    assert alert.timestamp.year == 2015
    assert (
        "src_flow_count_window" in alert.features
        and "dst_distinct_src_ips_window" in alert.features
    )
    assert any("real source and destination addresses" in note.lower() for note in report["notes"])


def test_context_features_actually_separate_the_attack_bursts(tmp_path):
    path = write_parquet(tmp_path / "full.parquet", n=4000)
    _model, test, _report = train_real(path, limit=2500, seed=3)
    attack = [a.features["src_flow_count_window"] for a in test if a.ground_truth_label != "benign"]
    benign = [a.features["src_flow_count_window"] for a in test if a.ground_truth_label == "benign"]
    assert attack and benign
    assert sum(attack) / len(attack) > 1.5 * sum(benign) / len(benign)


def test_a_file_too_small_to_evaluate_is_a_clear_error(tmp_path):
    path = write_parquet(tmp_path / "tiny.parquet", n=100)
    with pytest.raises(RealDataError, match=r"usable rows|attack"):
        analyse_flows(path, limit=1000)


def test_the_real_feed_keeps_real_time_order_when_the_file_has_capture_times(mesh_models, tmp_path):
    path = write_parquet(tmp_path / "full.parquet", n=4000)
    attack = tmp_path / "attack.json"
    attack.write_text('{"type": "bundle", "objects": []}')
    from tests.unit.test_real_workspace import stix_for_workspace

    attack.write_text(__import__("json").dumps(stix_for_workspace()))
    models = MeshModels.build_real(
        mesh_models, dataset_path=path, attack_path=attack, seed=3, limit=2500
    )
    stamps = [a.timestamp for a in models.feed]
    assert stamps == sorted(stamps), "a feed with real times streams in the order things happened"
    assert models.real_report["has_addresses"] is True
    assert {a.src_ip for a in models.feed if a.ground_truth_label != "benign"} <= {
        f"175.45.176.{i}" for i in range(4)
    }
