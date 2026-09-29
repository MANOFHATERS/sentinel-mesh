"""The scripted demo scenarios (Part 5, PRD Section 9.2).

A scenario may choose *addressing* and *which advisory lands where*; it may not
choose what the mesh concludes. These tests hold both halves of that line.
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from sentinel.core.schemas import TriageDecision
from sentinel.core.untrusted import InjectionVerdict, UntrustedText
from sentinel.dashboard.scenarios import (
    ASSET_INVENTORY,
    INJECTION_NOTE,
    PROTECTED_NETWORKS,
    RISK_SOURCE_CVES,
    SCENARIOS,
    Advisory,
    AdvisoryKind,
    ScenarioError,
    ScenarioName,
    ScriptedFlow,
    cve_advisory,
    exposure_scope,
    malicious_advisory,
    pick_story_flow,
    risk_sources,
    script_alert,
    select_cve_package,
    select_malicious_package,
)
from sentinel.graph.explain import explain_node
from sentinel.graph.schema import NodeKind
from sentinel.graph.synthetic import SyntheticGraphGenerator
from sentinel.ml.datasets.synthetic import generate_alerts

LAUNCH = datetime(2026, 9, 29, 9, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def flows():
    return [a for a in generate_alerts(4000, seed=7) if a.ground_truth_label != "benign"]


@pytest.fixture(scope="module")
def graph():
    built, _truth = SyntheticGraphGenerator(seed=20260928).generate()
    return built


FLOW = ScriptedFlow(
    caption="probe",
    family="recon",
    src_ip="10.20.4.17",
    dst_ip="10.20.1.10",
    asset_id="10.20.1.10",
    seconds_before_launch=120.0,
)


class TestScriptAlert:
    def test_features_are_carried_over_untouched(self, flows):
        base = flows[0]
        scripted = script_alert(base, FLOW, tenant_id="acme", scenario="s", index=0,
                                launched_at=LAUNCH)
        assert scripted.features == base.features
        assert scripted.signature == base.signature
        assert (scripted.src_port, scripted.dst_port, scripted.protocol) == (
            base.src_port, base.dst_port, base.protocol,
        )

    def test_envelope_is_readdressed_consistently(self, flows):
        scripted = script_alert(flows[0], FLOW, tenant_id="acme", scenario="s", index=0,
                                launched_at=LAUNCH)
        assert (scripted.src_ip, scripted.dst_ip, scripted.asset_id) == (
            "10.20.4.17", "10.20.1.10", "10.20.1.10",
        )
        row = json.loads(scripted.raw_payload.raw)
        # The raw log line the Investigation Agent cites must agree with the fields.
        assert row[" Source IP"] == "10.20.4.17"
        assert row[" Destination IP"] == "10.20.1.10"
        assert scripted.tenant_id == "acme"

    def test_time_is_the_story_s_not_the_capture_s(self, flows):
        scripted = script_alert(flows[0], FLOW, tenant_id="acme", scenario="s", index=0,
                                launched_at=LAUNCH)
        assert scripted.ingested_at == LAUNCH
        assert (LAUNCH - scripted.timestamp).total_seconds() == 120.0
        assert scripted.dwell_seconds() == 120.0

    def test_ids_are_deterministic_and_distinct_per_step(self, flows):
        one = script_alert(flows[0], FLOW, tenant_id="acme", scenario="s", index=0,
                           launched_at=LAUNCH)
        again = script_alert(flows[0], FLOW, tenant_id="acme", scenario="s", index=0,
                             launched_at=LAUNCH)
        other = script_alert(flows[0], FLOW, tenant_id="acme", scenario="s", index=1,
                             launched_at=LAUNCH)
        tenant = script_alert(flows[0], FLOW, tenant_id="globex", scenario="s", index=0,
                              launched_at=LAUNCH)
        assert one.alert_id == again.alert_id
        assert len({one.alert_id, other.alert_id, tenant.alert_id}) == 3

    def test_payload_note_is_attacker_text_and_is_scanned(self, flows):
        flow = replace(FLOW, payload_note=INJECTION_NOTE)
        scripted = script_alert(flows[0], flow, tenant_id="acme", scenario="s", index=0,
                                launched_at=LAUNCH)
        assert json.loads(scripted.raw_payload.raw)["Payload Note"] == INJECTION_NOTE
        assert scripted.injection_scan.verdict is InjectionVerdict.LIKELY_INJECTION

    def test_non_json_payload_is_refused(self, flows):
        broken = flows[0].updated(raw_payload="not json")
        with pytest.raises(ScenarioError, match="JSON"):
            script_alert(broken, FLOW, tenant_id="acme", scenario="s", index=0,
                         launched_at=LAUNCH)


class _FakeAgent:
    """Escalates exactly the alerts whose scripted payload carries a marker."""

    def __init__(self, escalate_ids):
        self.escalate_ids = escalate_ids
        self.seen = []

    def triage(self, alert):
        self.seen.append(alert)

        class _Verdict:
            decision = (
                TriageDecision.ESCALATE
                if alert.features.get("_probe") in self.escalate_ids
                else TriageDecision.MONITOR
            )

        return _Verdict()


class TestPickStoryFlow:
    def test_picks_the_first_flow_triage_escalates_as_scripted(self, flows):
        recon = [a for a in flows if a.ground_truth_label == "recon"][:4]
        pool = [a.updated(features={**a.features, "_probe": i}) for i, a in enumerate(recon)]
        agent = _FakeAgent({2, 3})
        chosen = pick_story_flow(pool, FLOW, agent=agent)
        assert chosen.alert_id == pool[2].alert_id
        # It judged the scripted envelope, not the raw corpus row.
        assert all(seen.src_ip == FLOW.src_ip for seen in agent.seen)

    def test_used_flows_are_skipped(self, flows):
        recon = [a for a in flows if a.ground_truth_label == "recon"][:4]
        pool = [a.updated(features={**a.features, "_probe": i}) for i, a in enumerate(recon)]
        chosen = pick_story_flow(pool, FLOW, agent=_FakeAgent({2, 3}),
                                 used={pool[2].alert_id})
        assert chosen.alert_id == pool[3].alert_id

    def test_never_forces_a_verdict(self, flows):
        recon = [a for a in flows if a.ground_truth_label == "recon"][:4]
        with pytest.raises(ScenarioError, match="escalated"):
            pick_story_flow(recon, FLOW, agent=_FakeAgent(set()))


class TestAdvisories:
    def test_apply_returns_a_copy_and_changes_only_the_package(self, graph):
        package = select_cve_package(graph)
        before = graph.node(package)
        advisory = cve_advisory(package)
        after = advisory.apply(graph)
        assert graph.node(package) == before, "the base graph must never be mutated"
        changed = after.node(package)
        assert changed.cve_exposure_count == before.cve_exposure_count + 4
        assert changed.is_unmaintained
        assert [n for n in after.nodes if n.node_id != package] == [
            n for n in graph.nodes if n.node_id != package
        ]
        assert after.edges == graph.edges

    def test_the_advisory_makes_the_package_a_risk_source(self, graph):
        for make in (cve_advisory, malicious_advisory):
            package = select_cve_package(graph)
            after = make(package).apply(graph)
            assert package not in risk_sources(graph)
            assert package in risk_sources(after)
            assert explain_node(after, package).is_intrinsically_risky

    def test_malicious_advisory_records_the_breach(self, graph):
        package = select_malicious_package(graph)
        after = malicious_advisory(package).apply(graph)
        assert after.node(package).breach_history == graph.node(package).breach_history + 1
        assert after.node(package).cve_exposure_count >= RISK_SOURCE_CVES

    def test_a_cve_advisory_adds_one_unit_per_cve(self):
        with pytest.raises(ScenarioError, match="per CVE"):
            Advisory("CVE-X", AdvisoryKind.CVE, "pkg-0001", "t", "s",
                     cve_ids=("CVE-2026-1",), exposure_added=3)

    def test_an_advisory_cannot_target_a_vendor(self, graph):
        vendor = graph.nodes_of_kind(NodeKind.VENDOR)[0].node_id
        with pytest.raises(ScenarioError, match="not a package"):
            cve_advisory(vendor).apply(graph)


class TestPackageSelection:
    def test_cve_package_is_a_fourth_order_dependency(self, graph):
        package = select_cve_package(graph)
        reach = graph.downstream(package, max_hops=4)
        orgs = {n: hops for n, hops in reach.items()
                if graph.node(n).kind is NodeKind.ORGANIZATION}
        assert len(orgs) >= 3
        assert set(orgs.values()) == {4}, "every organisation is exactly four hops away"
        assert not risk_sources(graph) & set(reach), "no competing risk source in scope"

    def test_malicious_package_is_the_widest_eligible(self, graph):
        cve = select_cve_package(graph)
        cve_reach = set(graph.downstream(cve, max_hops=4))
        package = select_malicious_package(graph, exclude={cve, *cve_reach},
                                           must_not_reach={cve})
        reach = graph.downstream(package, max_hops=4)
        assert package != cve and package not in cve_reach
        assert cve not in reach
        assert any(graph.node(n).kind is NodeKind.ORGANIZATION for n in reach)

    def test_selection_is_deterministic(self, graph):
        assert select_cve_package(graph) == select_cve_package(graph)
        assert select_malicious_package(graph) == select_malicious_package(graph)

    def test_exposure_scope_keeps_every_path(self, graph):
        package = select_cve_package(graph)
        after = cve_advisory(package).apply(graph)
        scope = exposure_scope(after, package)
        assert scope[0] != "" and package in scope
        sub = after.subgraph(scope)
        # Every path the explainer finds in the full graph survives the scoping.
        for node_id in scope:
            if node_id == package:
                continue
            full = {p.nodes for p in explain_node(after, node_id, max_paths=50).paths
                    if p.source == package}
            scoped = {p.nodes for p in explain_node(sub, node_id, max_paths=50).paths}
            assert full <= scoped, node_id


class TestScenarioSpecs:
    def test_three_scenarios_matching_the_prd(self):
        assert set(SCENARIOS) == set(ScenarioName)
        titles = " ".join(spec.title.lower() for spec in SCENARIOS.values())
        for phrase in ("phishing", "lateral", "vendor", "cve", "malicious", "package"):
            assert phrase in titles

    def test_all_five_agents_are_exercised(self):
        agents = {agent for spec in SCENARIOS.values() for agent in spec.agents}
        assert agents == {"triage", "investigation", "containment", "supply-chain",
                          "code-scan"}

    def test_protected_assets_are_inside_the_protected_networks(self):
        networks = [ipaddress.ip_network(n) for n in PROTECTED_NETWORKS]
        for asset in ASSET_INVENTORY.values():
            inside = any(ipaddress.ip_address(asset.address) in n for n in networks)
            assert inside == asset.protected, asset

    def test_no_real_host_is_named(self):
        for asset in ASSET_INVENTORY.values():
            address = ipaddress.ip_address(asset.address)
            documentation = any(
                address in ipaddress.ip_network(block)
                for block in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
            )
            assert address.is_private or documentation, asset

    def test_every_flow_addresses_a_known_asset(self):
        for spec in SCENARIOS.values():
            for flow in spec.flows:
                for address in (flow.src_ip, flow.dst_ip, flow.asset_id):
                    assert address in ASSET_INVENTORY, (spec.name, address)
                assert flow.seconds_before_launch > 0

    def test_the_lateral_step_comes_from_protected_infrastructure(self):
        # Scenario 1 is designed to exercise the router's protected-target refusal.
        lateral = SCENARIOS[ScenarioName.PHISHING_LATERAL].flows[-1]
        assert ASSET_INVENTORY[lateral.src_ip].protected

    def test_the_injection_note_is_detected(self):
        scan = UntrustedText(INJECTION_NOTE, origin="test").scan()
        assert scan.verdict is InjectionVerdict.LIKELY_INJECTION
