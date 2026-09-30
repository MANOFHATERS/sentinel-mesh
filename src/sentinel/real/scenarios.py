"""Launchable cases for the real-data workspace, built from what the real data actually contains.

The synthetic demo's three scenarios are scripted stories (which host, which advisory). A real
workspace has no script, so its "scenarios" are *selections from real data* that the analyst can
launch, in the same interface:

``campaign-<family>``
    The next few real flows of one attack family from the held-out capture, through triage,
    investigation and containment. Repeatable: each launch takes the next flows.
``advisory-<id>``
    A real OSV advisory against a real pinned package, reviewed by the Supply-Chain Agent over the
    real dependency graph (exposure paths, scored and explained).
``code-scan``
    The Code-Scan Agent on the real primary project's own source, with drafted patches gated behind
    an approval that opens a draft pull request on the local GitHub emulator.

Nothing here is invented: every flow, package, advisory and file is real, and the cards say which.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["CAMPAIGN_FLOWS", "RealScenario", "advisory_slug", "real_scenarios"]

CAMPAIGN_FLOWS: Final[int] = 5
_MAX_CAMPAIGNS: Final[int] = 5
_MIN_FAMILY_FLOWS: Final[int] = 8

_FAMILY_TITLES: Final[dict[str, str]] = {
    "exploit": "Exploit traffic",
    "generic": "Generic (block-cipher) attack traffic",
    "fuzzers": "Fuzzing traffic",
    "dos": "Denial-of-service traffic",
    "recon": "Reconnaissance (scanning) traffic",
    "analysis": "Analysis (probing) traffic",
    "backdoor": "Backdoor traffic",
    "shellcode": "Shellcode traffic",
    "worm": "Worm traffic",
}


@dataclass(frozen=True, slots=True)
class RealScenario:
    id: str
    kind: str  # campaign | advisory | code
    title: str
    summary: str
    agents: tuple[str, ...]
    walkthrough: tuple[str, ...]
    #: campaign: the attack family; advisory: the advisory id; code: the repository.
    subject: str


def advisory_slug(advisory_id: str) -> str:
    return "advisory-" + re.sub(r"[^A-Za-z0-9]+", "-", advisory_id).strip("-").lower()


def real_scenarios(models: Any) -> list[RealScenario]:
    """The launchable real cases for a real :class:`MeshModels`."""
    dataset = (models.real_report or {}).get("dataset", "the real capture")
    out: list[RealScenario] = []

    families = Counter(
        a.ground_truth_label
        for a in models.feed
        if a.ground_truth_label not in (None, "benign", "unlabelled")
    )
    for family, count in families.most_common():
        if count < _MIN_FAMILY_FLOWS or len(out) >= _MAX_CAMPAIGNS:
            continue
        name = _FAMILY_TITLES.get(family, f"{family.title()} traffic")
        out.append(
            RealScenario(
                id=f"campaign-{family}",
                kind="campaign",
                title=f"Real {name.lower()}",
                summary=(
                    f"The next {CAMPAIGN_FLOWS} real {family} flows from the held-out {dataset} "
                    f"capture ({count:,} in the feed), labelled by the researchers who captured "
                    "them. Triage has never seen these flows. Repeat to take the next batch."
                ),
                agents=("triage", "investigation", "containment"),
                walkthrough=(
                    "Triage scores each real flow and dismisses, monitors or escalates it.",
                    "Investigation cites MITRE's real ATT&CK entry where the family maps to one.",
                    "Containment proposes a response; a destructive one waits for your approval.",
                    "This capture has no host addresses, so the asset is the flow record and "
                    "the action runs against the local emulators.",
                ),
                subject=family,
            )
        )

    for advisory_id, advisory in models.real_advisories.items():
        out.append(
            RealScenario(
                id=advisory_slug(advisory_id),
                kind="advisory",
                title=f"Real advisory {advisory_id}",
                summary=advisory.summary,
                agents=("supply-chain",),
                walkthrough=(
                    f"{advisory.package_id.removeprefix('pypi:')} is pinned by real projects "
                    "in the dependency graph.",
                    "The Supply-Chain Agent scores every dependent and explains each score as "
                    "a concrete path through the real graph.",
                    "A package with a known vulnerability and a stale pin is proposed for a "
                    "reviewed dependency change; nothing is edited automatically.",
                ),
                subject=advisory_id,
            )
        )

    if models.repository != "acme/billing":
        out.append(
            RealScenario(
                id="code-scan",
                kind="code",
                title=f"Real code scan of {models.repository}",
                summary=(
                    f"The source of {models.repository}, a real public project, scanned by the "
                    "same analyzer the demo uses. Findings link to the code; validated patches "
                    "are drafted behind an approval and open a draft pull request on the local "
                    "GitHub emulator, never on the real repository."
                ),
                agents=("code-scan",),
                walkthrough=(
                    "The analyzer parses the real Python files (nothing is run).",
                    "Findings are reported; validated patches are drafted.",
                    "You approve or reject the draft pull request; it is opened on the emulator.",
                ),
                subject=models.repository,
            )
        )
    return out
