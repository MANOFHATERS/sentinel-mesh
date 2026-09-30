"""Grounding: D3FEND for what the actions are, CISA KEV and EPSS for how urgent a flaw is.

The ontology, catalogue and score file here are small hand-made fixtures in the real shapes.
"""

from __future__ import annotations

import gzip
import json

import pytest

from sentinel.core.schemas import ActionType
from sentinel.real.grounding import (
    ACTION_D3FEND,
    GroundingError,
    countermeasures,
    load_exploitation,
    load_ontology,
)
from sentinel.real.service import RealData
from sentinel.real.supplychain import annotate_exploitation, graph_from_snapshot, real_advisories
from tests.unit.test_real_supplychain import snapshot


def technique(node_id, d3id, label, *, enables=None, parent=None, definition="Defined."):
    node = {
        "@id": node_id,
        "d3f:d3fend-id": d3id,
        "rdfs:label": label,
        "d3f:definition": definition,
    }
    if enables:
        node["d3f:enables"] = {"@id": enables}
    if parent:
        node["rdfs:subClassOf"] = [{"@id": "_:b1"}, {"@id": parent}]
    return node


def ontology_file(tmp_path, *, drop=None):
    graph = [
        {"@id": "d3f:Isolate", "@type": ["d3f:DefensiveTactic"]},
        {"@id": "d3f:Evict", "@type": ["d3f:DefensiveTactic"]},
        {"@id": "d3f:Harden", "@type": ["d3f:DefensiveTactic"]},
        technique("d3f:NetworkIsolation", "D3-NI", "Network Isolation", enables="d3f:Isolate"),
        technique(
            "d3f:NetworkTrafficFiltering",
            "D3-NTF",
            "Network Traffic Filtering",
            parent="d3f:NetworkIsolation",
        ),
        # the tactic is inherited from an ancestor, as it is for most techniques in the real file
        technique(
            "d3f:InboundTrafficFiltering",
            "D3-ITF",
            "Inbound Traffic Filtering",
            parent="d3f:NetworkTrafficFiltering",
        ),
        technique("d3f:AccountLocking", "D3-AL", "Account Locking", enables="d3f:Evict"),
        technique("d3f:ProcessTermination", "D3-PT", "Process Termination", enables="d3f:Evict"),
        technique("d3f:FileEviction", "D3-FEV", "File Eviction", enables="d3f:Evict"),
        technique("d3f:SoftwareUpdate", "D3-SU", "Software Update", enables="d3f:Harden"),
    ]
    graph = [n for n in graph if n.get("d3f:d3fend-id") != drop]
    path = tmp_path / "d3fend.json"
    path.write_text(json.dumps({"@context": {}, "@graph": graph}))
    return path


def test_every_response_action_has_a_mapping_so_a_new_action_cannot_be_forgotten():
    assert set(ACTION_D3FEND) == {a.value for a in ActionType}


def test_actions_map_to_their_d3fend_technique_with_the_tactic_from_the_ontology(tmp_path):
    rows = {r["action"]: r for r in countermeasures(load_ontology(ontology_file(tmp_path)))}
    assert (
        rows["isolate_host"]["label"] == "Network Isolation"
        and rows["isolate_host"]["tactic"] == "Isolate"
    )
    assert (
        rows["block_ip"]["d3fend_id"] == "D3-ITF" and rows["block_ip"]["tactic"] == "Isolate"
    )  # inherited
    assert (
        rows["disable_account"]["tactic"] == "Evict" and rows["open_patch_pr"]["tactic"] == "Harden"
    )
    assert (
        rows["block_ip"]["url"] == "https://d3fend.mitre.org/technique/d3f:InboundTrafficFiltering/"
    )
    assert rows["notify_analyst"]["d3fend_id"] is None and rows["enrich_only"]["label"] is None


def test_a_mapping_to_an_id_the_ontology_lacks_is_refused_rather_than_shipped(tmp_path):
    with pytest.raises(GroundingError, match="D3-AL"):
        countermeasures(load_ontology(ontology_file(tmp_path, drop="D3-AL")))


def test_a_missing_or_wrong_ontology_says_how_to_get_it(tmp_path):
    with pytest.raises(GroundingError, match="fetch_real_data"):
        load_ontology(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    with pytest.raises(GroundingError, match="not a D3FEND"):
        load_ontology(bad)


def write_exploitation(tmp_path, *, kev=True, epss=True):
    if kev:
        (tmp_path / "known_exploited_vulnerabilities.json").write_text(
            json.dumps(
                {
                    "vulnerabilities": [
                        {
                            "cveID": "CVE-2020-0004",
                            "dateAdded": "2021-11-03",
                            "product": "oldlib",
                            "knownRansomwareCampaignUse": "Known",
                        },
                        {
                            "cveID": "CVE-1999-0001",
                            "dateAdded": "2022-01-01",
                            "product": "x",
                            "knownRansomwareCampaignUse": "Unknown",
                        },
                    ]
                }
            )
        )
    if epss:
        with gzip.open(tmp_path / "epss_scores-current.csv.gz", "wt", encoding="utf-8") as handle:
            handle.write("#model_version:v2025.03.14,score_date:2026-09-29T12:00:00Z\n")
            handle.write("cve,epss,percentile\nCVE-2020-0001,0.25,0.9\nCVE-2020-0004,0.01,0.5\n")
    return tmp_path


def test_kev_membership_and_epss_scores_are_looked_up_by_cve(tmp_path):
    exploitation = load_exploitation(write_exploitation(tmp_path))
    assert exploitation.epss_date == "2026-09-29T12:00:00Z" and len(exploitation.kev) == 2
    hit = exploitation.of("CVE-2020-0004")
    assert hit["kev"] == {"added": "2021-11-03", "ransomware": True, "product": "oldlib"}
    assert hit["epss"] == 0.01 and hit["epss_percentile"] == 0.5
    assert exploitation.of("CVE-1999-0001")["kev"]["ransomware"] is False
    assert exploitation.of("CVE-2020-0001") == {"epss": 0.25, "epss_percentile": 0.9}
    assert exploitation.of("CVE-0000-0000") == {}


def test_either_file_alone_is_enough_and_neither_is_none(tmp_path):
    assert load_exploitation(tmp_path) is None
    only_kev = load_exploitation(write_exploitation(tmp_path, epss=False))
    assert only_kev.epss == {} and only_kev.epss_date is None and "CVE-2020-0004" in only_kev.kev


def test_advisories_are_ranked_by_real_exploitation_before_severity(tmp_path):
    graph, facts = graph_from_snapshot(snapshot())
    plain = real_advisories(graph, facts)
    assert list(plain) == ["CVE-2020-0001"]  # worst by severity alone (CRITICAL)
    listed = annotate_exploitation(facts, load_exploitation(write_exploitation(tmp_path)))
    assert listed == 1
    ranked = real_advisories(graph, facts)
    assert list(ranked) == ["CVE-2020-0004"], "the issue on the KEV list outranks a more severe one"
    summary = ranked["CVE-2020-0004"].summary
    assert "exploited in the wild" in summary and "2021-11-03" in summary and "1% chance" in summary


def test_the_grounding_endpoint_reports_what_is_and_is_not_on_disk(tmp_path):
    empty = RealData(tmp_path).grounding()
    assert empty["available"] is False and empty["exploitation"] is None
    base = tmp_path / "data" / "real"
    base.mkdir(parents=True)
    ontology_file(base).rename(base / "d3fend.json")
    write_exploitation(base)
    found = RealData(tmp_path).grounding()
    assert found["available"] and len(found["countermeasures"]) == len(ACTION_D3FEND)
    assert found["exploitation"] == {
        "kev_listed": 2,
        "epss_scored": 2,
        "epss_date": "2026-09-29T12:00:00Z",
    }
