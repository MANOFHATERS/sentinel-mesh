"""Seeded synthetic flow generator.

Purpose: the repository ships no datasets — CIC-IDS2017 is ~2.8M flows across
several hundred megabytes, and UNSW-NB15 is a separate multi-gigabyte download.
Every test, the CI run and a first-time demo still need data, so this module
generates flows that are *shaped like* the real thing.

Two properties make it useful rather than decorative:

**It emits raw source rows, not clean feature vectors.** The CIC generator
produces dictionaries with the genuine column names — leading spaces,
``"Flow Bytes/s"``, ``" Label"`` — and deliberately injects the dataset's real
defects: ``Infinity`` in rate columns for zero-duration flows, the en-dash in
``"Web Attack – Brute Force"``, and its cp1252 mojibake twin. So the tests
exercise the normalizer's actual job instead of bypassing it.

**The classes overlap on purpose.** Attack profiles are drawn from log-normal
distributions that partially overlap benign traffic, controlled by
``separability``. A generator that made attacks trivially separable would let a
detector report AUC 0.999 and prove nothing; PRD F-03 asks for AUC >= 0.90, and at
the default separability this generator produces a problem where a well-built
ensemble lands in the low-to-mid 0.90s and a careless one does not. Class
imbalance is set to roughly the real ratio (~80% benign), which is what makes the
minority-class augmentation in Phase 2 worth building.

Nothing here is presented as real data. PRD Section 10 lists "synthetic data is
unrealistic and undermines credibility" as a risk, with the mitigation being
explicitness — so every alert generated carries ``dataset="cic-ids2017-synthetic"``
and no output of this module is ever labelled as a real capture.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np

from sentinel.ingest.normalizer import (
    BENIGN,
    BOTNET,
    BRUTE_FORCE,
    DDOS,
    DOS,
    INFILTRATION,
    RECON,
    WEB_ATTACK,
)

__all__ = [
    "ATTACK_MIX",
    "PROFILES",
    "FlowProfile",
    "SyntheticCICGenerator",
    "SyntheticUNSWGenerator",
    "generate_alerts",
]


@dataclass(frozen=True, slots=True)
class FlowProfile:
    """Log-normal parameters for one traffic class.

    Means are ``log``-space, so ``mu_src_bytes=8.0`` is roughly 3 kB. Using
    log-space parameters is not cosmetic: real flow sizes are multiplicative
    processes spanning six orders of magnitude, and a Gaussian in linear space
    would generate negative byte counts.
    """

    family: str
    cic_label: str
    unsw_label: str
    mu_duration_log_us: float
    sigma_duration: float
    mu_src_bytes: float
    sigma_src_bytes: float
    mu_dst_bytes: float
    sigma_dst_bytes: float
    mu_src_packets: float
    sigma_src_packets: float
    dst_ports: tuple[int, ...]
    protocols: tuple[int, ...]
    #: Probability the responder sends nothing at all.
    p_no_response: float = 0.02
    #: Probability the flow is a single packet (scans, beacon probes).
    p_single_packet: float = 0.01
    #: Probability of a zero-duration flow, which is what makes the real dataset
    #: emit ``Infinity`` in its rate columns.
    p_zero_duration: float = 0.005


# Parameters chosen to reflect the qualitative signature of each family rather
# than to maximise separability: a port scan really is one tiny packet with no
# reply, slowloris really is a long-lived flow carrying almost nothing, and
# exfiltration really is a large upload with a small acknowledgement stream.
PROFILES: Final[dict[str, FlowProfile]] = {
    BENIGN: FlowProfile(
        family=BENIGN,
        cic_label="BENIGN",
        unsw_label="Normal",
        mu_duration_log_us=13.0,
        sigma_duration=2.2,
        mu_src_bytes=6.2,
        sigma_src_bytes=1.9,
        mu_dst_bytes=7.6,
        sigma_dst_bytes=2.3,
        mu_src_packets=2.3,
        sigma_src_packets=1.1,
        dst_ports=(80, 443, 53, 22, 25, 110, 143, 993, 3306, 8080),
        protocols=(6, 6, 6, 17, 17),
        p_no_response=0.03,
        p_single_packet=0.02,
        p_zero_duration=0.01,
    ),
    DDOS: FlowProfile(
        family=DDOS,
        cic_label="DDoS",
        unsw_label="DoS",
        # Very short flows, very high packet rate, small payloads.
        mu_duration_log_us=8.5,
        sigma_duration=1.4,
        mu_src_bytes=4.2,
        sigma_src_bytes=1.0,
        mu_dst_bytes=1.5,
        sigma_dst_bytes=1.3,
        mu_src_packets=3.4,
        sigma_src_packets=0.9,
        dst_ports=(80, 443),
        protocols=(6,),
        p_no_response=0.45,
        p_zero_duration=0.08,
    ),
    DOS: FlowProfile(
        family=DOS,
        cic_label="DoS slowloris",
        unsw_label="DoS",
        # Slowloris: hold the socket open, send almost nothing.
        mu_duration_log_us=16.2,
        sigma_duration=1.1,
        mu_src_bytes=4.8,
        sigma_src_bytes=0.8,
        mu_dst_bytes=3.0,
        sigma_dst_bytes=1.4,
        mu_src_packets=1.6,
        sigma_src_packets=0.7,
        dst_ports=(80, 443, 8080),
        protocols=(6,),
        p_no_response=0.20,
    ),
    RECON: FlowProfile(
        family=RECON,
        cic_label="PortScan",
        unsw_label="Reconnaissance",
        mu_duration_log_us=5.5,
        sigma_duration=1.6,
        mu_src_bytes=3.8,
        sigma_src_bytes=0.6,
        mu_dst_bytes=0.5,
        sigma_dst_bytes=1.0,
        mu_src_packets=0.3,
        sigma_src_packets=0.5,
        dst_ports=tuple(range(1, 1024, 37)),
        protocols=(6, 6, 1),
        p_no_response=0.70,
        p_single_packet=0.55,
        p_zero_duration=0.15,
    ),
    BRUTE_FORCE: FlowProfile(
        family=BRUTE_FORCE,
        cic_label="SSH-Patator",
        unsw_label="Exploits",
        mu_duration_log_us=12.4,
        sigma_duration=1.3,
        mu_src_bytes=7.1,
        sigma_src_bytes=0.8,
        mu_dst_bytes=7.3,
        sigma_dst_bytes=0.9,
        mu_src_packets=2.9,
        sigma_src_packets=0.6,
        dst_ports=(22, 21, 3389, 445),
        protocols=(6,),
        p_no_response=0.05,
    ),
    WEB_ATTACK: FlowProfile(
        family=WEB_ATTACK,
        # The en-dash spelling, verbatim from the real CSVs.
        cic_label="Web Attack – Brute Force",
        unsw_label="Exploits",
        mu_duration_log_us=13.6,
        sigma_duration=1.5,
        mu_src_bytes=8.6,
        sigma_src_bytes=1.1,
        mu_dst_bytes=8.2,
        sigma_dst_bytes=1.4,
        mu_src_packets=2.7,
        sigma_src_packets=0.8,
        dst_ports=(80, 443, 8080, 8443),
        protocols=(6,),
    ),
    BOTNET: FlowProfile(
        family=BOTNET,
        cic_label="Bot",
        unsw_label="Backdoor",
        # Periodic small beacons to a high port: low volume, regular, long-lived.
        mu_duration_log_us=14.8,
        sigma_duration=0.9,
        mu_src_bytes=5.4,
        sigma_src_bytes=0.5,
        mu_dst_bytes=5.1,
        sigma_dst_bytes=0.6,
        mu_src_packets=1.9,
        sigma_src_packets=0.4,
        dst_ports=(4444, 8443, 50050, 1337, 6667),
        protocols=(6,),
    ),
    INFILTRATION: FlowProfile(
        family=INFILTRATION,
        cic_label="Infiltration",
        unsw_label="Backdoor",
        # Exfiltration: large upload, tiny download, long flow.
        mu_duration_log_us=15.6,
        sigma_duration=1.2,
        mu_src_bytes=13.2,
        sigma_src_bytes=1.3,
        mu_dst_bytes=5.0,
        sigma_dst_bytes=1.1,
        mu_src_packets=5.8,
        sigma_src_packets=0.9,
        dst_ports=(443, 22, 8443, 9001),
        protocols=(6,),
    ),
}

#: Attack mix, normalized over the non-benign families. Roughly proportional to
#: CIC-IDS2017's own composition, where DoS/DDoS and PortScan dominate and
#: Infiltration is genuinely rare (36 flows in 2.8M) — which is exactly the
#: imbalance the diffusion augmentation in Phase 2 exists to address.
ATTACK_MIX: Final[dict[str, float]] = {
    DDOS: 0.34,
    DOS: 0.26,
    RECON: 0.21,
    BRUTE_FORCE: 0.10,
    WEB_ATTACK: 0.05,
    BOTNET: 0.032,
    INFILTRATION: 0.008,
}


class _BaseGenerator:
    """Shared sampling logic."""

    def __init__(
        self,
        *,
        seed: int = 20260928,
        benign_fraction: float = 0.80,
        separability: float = 1.0,
        attack_mix: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 < benign_fraction < 1.0:
            raise ValueError("benign_fraction must be in (0, 1)")
        if separability <= 0:
            raise ValueError("separability must be positive")
        self.rng = np.random.default_rng(seed)
        self.benign_fraction = benign_fraction
        # separability < 1 shrinks the distance between attack means and benign
        # means, making the problem harder; > 1 pulls them apart.
        self.separability = separability
        mix = attack_mix or ATTACK_MIX
        total = sum(mix.values())
        if total <= 0:
            raise ValueError("attack_mix weights must sum to a positive number")
        self._families = list(mix)
        self._weights = np.array([mix[f] / total for f in self._families], dtype=float)
        #: Flows per campaign, filled in by the planner (see _CampaignMixin.plan).
        self._planned_burst: dict[str, int] = {}

    def _pick_family(self) -> str:
        if self.rng.random() < self.benign_fraction:
            return BENIGN
        index = int(self.rng.choice(len(self._families), p=self._weights))
        return self._families[index]

    def _sample_flow(self, profile: FlowProfile) -> dict[str, float | int]:
        """Sample raw physical quantities for one flow."""
        benign = PROFILES[BENIGN]

        def blend(attack_mu: float, benign_mu: float) -> float:
            """Move an attack mean toward benign as separability drops."""
            if profile.family == BENIGN:
                return attack_mu
            return benign_mu + (attack_mu - benign_mu) * self.separability

        duration_us = float(
            self.rng.lognormal(
                blend(profile.mu_duration_log_us, benign.mu_duration_log_us),
                profile.sigma_duration,
            )
        )
        if self.rng.random() < profile.p_zero_duration:
            duration_us = 0.0

        src_bytes = float(
            self.rng.lognormal(
                blend(profile.mu_src_bytes, benign.mu_src_bytes), profile.sigma_src_bytes
            )
        )
        dst_bytes = float(
            self.rng.lognormal(
                blend(profile.mu_dst_bytes, benign.mu_dst_bytes), profile.sigma_dst_bytes
            )
        )
        if self.rng.random() < profile.p_no_response:
            dst_bytes = 0.0

        src_packets = max(
            1.0,
            float(
                self.rng.lognormal(
                    blend(profile.mu_src_packets, benign.mu_src_packets),
                    profile.sigma_src_packets,
                )
            ),
        )
        if self.rng.random() < profile.p_single_packet:
            src_packets = 1.0
            src_bytes = min(src_bytes, 120.0)

        # Packet counts must be consistent with byte counts: a 1-byte packet is
        # impossible, and a detector that learns from physically impossible flows
        # learns an artefact of the generator.
        src_packets = min(src_packets, max(1.0, src_bytes / 40.0))
        dst_packets = 0.0 if dst_bytes == 0 else max(1.0, min(src_packets * 1.2, dst_bytes / 40.0))

        return {
            "duration_us": duration_us,
            "src_bytes": round(src_bytes),
            "dst_bytes": round(dst_bytes),
            "src_packets": round(src_packets),
            "dst_packets": round(dst_packets),
            "dst_port": int(self.rng.choice(profile.dst_ports)),
            "src_port": int(self.rng.integers(49152, 65535)),
            "protocol": int(self.rng.choice(profile.protocols)),
        }


@dataclass(frozen=True, slots=True)
class CampaignShape:
    """How an attack family distributes itself over hosts and time.

    This is the structural half of the generator, and it matters more than the
    per-flow distributions. An attack scattered uniformly across random hosts is
    not detectable by anything a real SOC uses, and a generator that produced such
    data would let a per-flow detector look adequate while teaching nothing.
    Real captures look like campaigns: one attacker, a burst of flows, a coherent
    target set, inside a bounded window.

    ``burst_min``/``burst_max``
        Flows per campaign.
    ``inter_arrival_seconds``
        Mean gap between flows within the campaign. Sub-second for automated
        tooling; tens of seconds for a C2 beacon.
    ``arrival_jitter``
        Coefficient of variation of that gap. Near zero is machine-regular — the
        property that gives away a beacon and that the enricher measures directly.
    ``distributed_sources``
        True for DDoS: many sources converging on one destination. This is the
        fan-in signature, and it is invisible to any per-source feature.
    ``port_sweep``
        True for a port scan: one source, one target, every port.
    ``target_sweep``
        True when the campaign walks across destination hosts rather than ports.
    """

    burst_min: int
    burst_max: int
    inter_arrival_seconds: float
    arrival_jitter: float = 0.8
    distributed_sources: bool = False
    port_sweep: bool = False
    target_sweep: bool = False


CAMPAIGNS: Final[dict[str, CampaignShape]] = {
    # Patator-style credential stuffing: one source, one service, hundreds of tries.
    BRUTE_FORCE: CampaignShape(
        burst_min=60, burst_max=400, inter_arrival_seconds=0.25, arrival_jitter=0.5
    ),
    # nmap-style sweep: one source, thousands of ports, sub-millisecond spacing.
    RECON: CampaignShape(
        burst_min=150, burst_max=1200,
        inter_arrival_seconds=0.02, arrival_jitter=0.6, port_sweep=True,
    ),
    # Distributed: the fan-in is the whole signal.
    DDOS: CampaignShape(
        burst_min=200, burst_max=900,
        inter_arrival_seconds=0.01, arrival_jitter=0.9, distributed_sources=True,
    ),
    # Slowloris: fewer flows, each long-lived, all to one victim port.
    DOS: CampaignShape(
        burst_min=40, burst_max=250, inter_arrival_seconds=0.6, arrival_jitter=0.7
    ),
    # Manual-ish web probing: modest volume, human-scale spacing.
    WEB_ATTACK: CampaignShape(
        burst_min=15, burst_max=90, inter_arrival_seconds=1.5, arrival_jitter=0.9
    ),
    # C2 beacon: low volume, long-lived, and conspicuously regular.
    BOTNET: CampaignShape(
        burst_min=8, burst_max=40, inter_arrival_seconds=30.0, arrival_jitter=0.06
    ),
    # Exfiltration: a handful of large transfers.
    INFILTRATION: CampaignShape(
        burst_min=1, burst_max=6, inter_arrival_seconds=45.0, arrival_jitter=1.0,
        target_sweep=True,
    ),
}


@dataclass(slots=True)
class _Campaign:
    """Mutable state for one in-flight campaign."""

    family: str
    remaining: int
    shape: CampaignShape
    source_ip: str
    target_ip: str
    target_port: int


class _CampaignMixin(_BaseGenerator):
    """Campaign scheduling and monotonic stream time for both dataset generators.

    Why the schedule is *planned* rather than sampled
    ------------------------------------------------
    The obvious design is to pick a family per episode from a probability vector.
    Two measured failures killed it:

    1.  Sampling per episode and then emitting a 200-1,200 flow burst made a nominal
        80% benign rate produce a stream that was **99.5% attack flows**. Solving the
        episode probabilities from the requested flow mix fixed the aggregate.

    2.  It could not fix the variance. Because a family's episode probability must be
        proportional to ``w_f / E_f``, the large campaigns become vanishingly rare
        *as episodes*: a 20,000-flow stream contained only 7-15 attack campaigns in
        total, spread across seven families. Whole families vanished at random --
        ``recon`` and ``brute_force`` were absent from several seeds, one seed came
        out 99.3% benign, and an evaluation built on such a corpus is either
        undefined (single-class ROC-AUC) or silently measuring a different problem
        than the one it reports.

    So the episode sequence is **planned** instead: derive how many flows each family
    should contribute, split that into campaigns of realistic size, and interleave
    them with benign flows in a seeded shuffle. This is stratified sampling rather
    than rejection sampling, and it is the standard way to build a balanced
    evaluation corpus. Campaign structure -- one attacker, one target set, a
    contiguous burst, a bounded window -- is fully preserved, which is what the
    detector actually needs; what is removed is sampling noise in the *class mix*,
    which is a property of the harness rather than of the phenomenon.

    Real captures do contain only a handful of campaigns per family, and a
    true-to-life single capture is the wrong instrument for measuring a detector.
    This generator's job is a corpus of known, reproducible composition; the real
    downloads (``docs/DATA.md``) are the instrument for the other question.
    """

    #: Stream start, epoch seconds. 3 July 2017 08:00 UTC -- the Monday CIC-IDS20171s
    #: capture week begins, so generated timestamps sit in a plausible range.
    STREAM_START_EPOCH: Final[float] = 1_499_068_800.0

    #: Mean gap between benign flows, seconds. Sets the background rate that
    #: campaign bursts stand out against.
    BENIGN_INTER_ARRIVAL: Final[float] = 0.35

    #: Largest share of a stream any single campaign may occupy. Keeps campaign
    #: structure at every scale: without it, a 500-flow request could be a single
    #: truncated 1,200-flow burst, i.e. a single-class corpus.
    MAX_CAMPAIGN_SHARE: Final[float] = 0.05

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._now = self.STREAM_START_EPOCH
        self._campaign: _Campaign | None = None
        self._schedule: list[str | None] = []
        self._schedule_position = 0

    # --- planning -----------------------------------------------------------

    def plan(self, count: int) -> dict[str, int]:
        """Build the episode schedule for a ``count``-flow stream.

        Returns the planned flow count per family, which is what
        ``test_flow_level_class_balance_matches_the_request`` asserts against.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        self._schedule = []
        self._schedule_position = 0
        self._campaign = None
        self._now = self.STREAM_START_EPOCH
        if count == 0:
            return {}

        burst_cap = max(2, int(count * self.MAX_CAMPAIGN_SHARE))
        attack_flow_budget = round(count * (1.0 - self.benign_fraction))
        planned: dict[str, int] = {}
        episodes: list[str | None] = []

        for family, weight in zip(self._families, self._weights, strict=True):
            shape = CAMPAIGNS[family]
            low = min(shape.burst_min, burst_cap)
            high = max(1, min(shape.burst_max, burst_cap))
            family_flows = attack_flow_budget * float(weight)

            # Every family gets at least one campaign, so no evaluation silently
            # omits a class. A family whose share rounds below one flow still
            # appears once -- which is also how a genuinely rare family behaves
            # (Infiltration is 36 flows in CIC-IDS2017's 2.8M).
            n_campaigns = max(1, math.ceil(family_flows / high))
            per_campaign = max(1, round(family_flows / n_campaigns))
            per_campaign = max(min(per_campaign, high), 1)
            if per_campaign < low <= family_flows:
                per_campaign = low

            planned[family] = n_campaigns * per_campaign
            episodes.extend([family] * n_campaigns)
            self._planned_burst[family] = per_campaign

        benign_flows = max(0, count - sum(planned.values()))
        planned[BENIGN] = benign_flows
        episodes.extend([None] * benign_flows)

        # Seeded shuffle: campaigns land at unpredictable but reproducible points in
        # the stream, so the enricher sees realistic interleaving rather than a tidy
        # block structure it could learn instead of the attack.
        order = self.rng.permutation(len(episodes))
        self._schedule = [episodes[index] for index in order]
        return planned

    # --- address helpers ----------------------------------------------------

    def _random_internal_ip(self) -> str:
        return f"192.168.10.{int(self.rng.integers(1, 251))}"

    def _random_external_ip(self) -> str:
        return f"172.16.{int(self.rng.integers(0, 256))}.{int(self.rng.integers(1, 251))}"

    def _start_campaign(self, family: str) -> _Campaign:
        shape = CAMPAIGNS[family]
        profile = PROFILES[family]
        return _Campaign(
            family=family,
            remaining=self._planned_burst.get(family, shape.burst_min),
            shape=shape,
            source_ip=self._random_external_ip(),
            target_ip=f"10.0.{int(self.rng.integers(0, 256))}.{int(self.rng.integers(1, 251))}",
            target_port=int(self.rng.choice(profile.dst_ports)),
        )

    def _advance(self, mean_gap: float, jitter: float) -> None:
        """Move the stream clock forward by a jittered, strictly positive gap."""
        if jitter <= 0.0:
            self._now += max(mean_gap, 1e-4)
            return
        # Gamma with shape 1/cv^2 has coefficient of variation exactly cv and is
        # strictly positive -- so a low-jitter beacon stays regular and time never
        # runs backwards, which the causal enricher depends on.
        shape_k = max(1.0 / (jitter**2), 0.05)
        gap = float(self.rng.gamma(shape_k, mean_gap / shape_k))
        self._now += max(gap, 1e-4)

    def _next_event(self) -> tuple[str, dict[str, float | int], str, str, int]:
        """Produce the next (family, flow, src_ip, dst_ip, dst_port) in stream order."""
        if self._campaign is not None and self._campaign.remaining <= 0:
            self._campaign = None

        if self._campaign is None:
            family = self._next_scheduled()
            if family is None:
                self._advance(self.BENIGN_INTER_ARRIVAL, 1.0)
                profile = PROFILES[BENIGN]
                flow = self._sample_flow(profile)
                return (
                    BENIGN,
                    flow,
                    self._random_internal_ip(),
                    f"10.0.{int(self.rng.integers(0, 256))}.{int(self.rng.integers(1, 251))}",
                    int(flow["dst_port"]),
                )
            self._campaign = self._start_campaign(family)

        campaign = self._campaign
        campaign.remaining -= 1
        shape = campaign.shape
        profile = PROFILES[campaign.family]
        self._advance(shape.inter_arrival_seconds, shape.arrival_jitter)
        flow = self._sample_flow(profile)

        source_ip = (
            # A distributed attack draws a fresh source every flow, so no per-source
            # feature can see it; only the destination fan-in can.
            self._random_external_ip()
            if shape.distributed_sources
            else campaign.source_ip
        )
        target_ip = campaign.target_ip
        if shape.target_sweep:
            target_ip = f"10.0.{int(self.rng.integers(0, 256))}.{int(self.rng.integers(1, 251))}"
        port = int(flow["dst_port"]) if shape.port_sweep else campaign.target_port
        return campaign.family, flow, source_ip, target_ip, port

    def _next_scheduled(self) -> str | None:
        """Pop the next planned episode, cycling if the stream outruns the plan."""
        if not self._schedule:
            return None
        if self._schedule_position >= len(self._schedule):
            self._schedule_position = 0
        family = self._schedule[self._schedule_position]
        self._schedule_position += 1
        return family


class SyntheticCICGenerator(_CampaignMixin):
    """Emits raw rows with genuine CIC-IDS2017 column spellings and defects."""

    dataset = "cic-ids2017-synthetic"

    #: Fraction of web-attack labels written with the cp1252 mojibake dash instead
    #: of the proper en-dash, matching what mirrored copies of the CSVs contain.
    MOJIBAKE_RATE: Final[float] = 0.35

    def rows(self, count: int) -> Iterator[dict[str, Any]]:
        if count < 0:
            raise ValueError("count must be non-negative")
        self.plan(count)
        for index in range(count):
            family, flow, source_ip, target_ip, port = self._next_event()
            profile = PROFILES[family]

            duration_us = float(flow["duration_us"])
            total_bytes = float(flow["src_bytes"]) + float(flow["dst_bytes"])
            total_packets = float(flow["src_packets"]) + float(flow["dst_packets"])

            # The real dataset divides by a zero duration and writes the result out.
            if duration_us == 0.0:
                bytes_rate: Any = "Infinity" if total_bytes > 0 else "NaN"
                packets_rate: Any = "Infinity" if total_packets > 0 else "NaN"
            else:
                bytes_rate = total_bytes / (duration_us / 1e6)
                packets_rate = total_packets / (duration_us / 1e6)

            label = profile.cic_label
            if family == WEB_ATTACK and self.rng.random() < self.MOJIBAKE_RATE:
                label = label.replace("–", "\x96")

            moment = datetime.fromtimestamp(self._now, tz=UTC)
            yield {
                "Flow ID": f"synthetic-{index}",
                " Source IP": source_ip,
                " Source Port": flow["src_port"],
                " Destination IP": target_ip,
                " Destination Port": port,
                " Protocol": flow["protocol"],
                # CIC-IDS2017's own day-first, unpadded, timezone-free spelling.
                " Timestamp": (
                    f"{moment.day}/{moment.month}/{moment.year} "
                    f"{moment.hour}:{moment.minute:02d}:{moment.second:02d}"
                ),
                " Flow Duration": int(duration_us),
                " Total Fwd Packets": int(flow["src_packets"]),
                " Total Backward Packets": int(flow["dst_packets"]),
                "Total Length of Fwd Packets": int(flow["src_bytes"]),
                " Total Length of Bwd Packets": int(flow["dst_bytes"]),
                "Flow Bytes/s": bytes_rate,
                " Flow Packets/s": packets_rate,
                " Label": label,
            }


class SyntheticUNSWGenerator(_CampaignMixin):
    """Emits raw rows with UNSW-NB15 column spellings, including the padding quirks.

    Uses the same campaign engine as the CIC generator but the other dataset's
    schema, spelling, units and label vocabulary — which is what makes the
    cross-dataset validation test meaningful. If both generators shared a schema,
    "train on CIC, validate on UNSW" would only be testing the random seed.
    """

    dataset = "unsw-nb15-synthetic"

    #: Fraction of attack_cat values padded with stray spaces, as in the real CSVs.
    PADDING_RATE: Final[float] = 0.30

    #: UNSW-NB15's own address ranges: the ACCS testbed used these two /24s for
    #: attacker and victim traffic respectively.
    ATTACKER_PREFIX: Final[str] = "175.45.176"
    VICTIM_PREFIX: Final[str] = "149.171.126"

    def _random_external_ip(self) -> str:
        return f"{self.ATTACKER_PREFIX}.{int(self.rng.integers(1, 251))}"

    def _random_internal_ip(self) -> str:
        return f"{self.VICTIM_PREFIX}.{int(self.rng.integers(1, 251))}"

    def rows(self, count: int) -> Iterator[dict[str, Any]]:
        if count < 0:
            raise ValueError("count must be non-negative")
        self.plan(count)
        protocol_names = {6: "tcp", 17: "udp", 1: "icmp"}
        for index in range(count):
            family, flow, source_ip, target_ip, port = self._next_event()
            profile = PROFILES[family]
            duration_seconds = float(flow["duration_us"]) / 1e6

            attack_cat = "" if family == BENIGN else profile.unsw_label
            if attack_cat and self.rng.random() < self.PADDING_RATE:
                attack_cat = f" {attack_cat} "

            yield {
                "id": index,
                "srcip": source_ip,
                "sport": flow["src_port"],
                # UNSW's victim subnet, so the two datasets do not share addresses.
                "dstip": target_ip.replace("10.0", self.VICTIM_PREFIX[:6], 1)
                if target_ip.startswith("10.0")
                else target_ip,
                "dsport": port,
                "proto": protocol_names.get(int(flow["protocol"]), "tcp"),
                "state": "FIN",
                # Seconds, not microseconds. The asymmetry with CIC is the point.
                "dur": duration_seconds,
                "sbytes": int(flow["src_bytes"]),
                "dbytes": int(flow["dst_bytes"]),
                "spkts": int(flow["src_packets"]),
                "dpkts": int(flow["dst_packets"]),
                "Stime": int(self._now),
                "Ltime": int(self._now) + max(1, int(duration_seconds)),
                "attack_cat": attack_cat,
                "Label": 0 if family == BENIGN else 1,
            }


def generate_alerts(
    count: int,
    *,
    seed: int = 20260928,
    dataset: str = "cic",
    benign_fraction: float = 0.80,
    separability: float = 1.0,
    tenant_id: str = "demo",
    enrich: bool = True,
) -> list[Any]:
    """Convenience: generate ``count`` canonical alerts through the real normalizer.

    Going through the normalizer rather than constructing alerts directly is the
    point — it means every test that uses synthetic data also exercises the
    ingestion path, so a normalizer regression cannot hide behind clean fixtures.
    """
    from sentinel.ingest.normalizer import CICIDS2017Normalizer, UNSWNB15Normalizer

    if dataset.startswith("cic"):
        generator: _BaseGenerator = SyntheticCICGenerator(
            seed=seed, benign_fraction=benign_fraction, separability=separability
        )
        normalizer = CICIDS2017Normalizer(tenant_id=tenant_id, strict=True)
    elif dataset.startswith("unsw"):
        generator = SyntheticUNSWGenerator(
            seed=seed, benign_fraction=benign_fraction, separability=separability
        )
        normalizer = UNSWNB15Normalizer(tenant_id=tenant_id, strict=True)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; use 'cic' or 'unsw'")

    alerts = []
    for row_index, row in enumerate(generator.rows(count)):  # type: ignore[attr-defined]
        alert = normalizer.normalize(row, row_index=row_index)
        if alert is not None:
            alerts.append(alert)

    if not enrich:
        return alerts

    # Enrichment runs here, in stream order, through the same object the serving
    # path uses. Callers get alerts that already carry session context, so no test
    # or training script can accidentally build a model on the per-flow features
    # alone and then be served enriched ones (or vice versa).
    from sentinel.ingest.enrich import SessionContextEnricher

    return SessionContextEnricher().enrich_all(alerts)
