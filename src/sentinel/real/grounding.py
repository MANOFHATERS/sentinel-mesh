"""What the response layer can be grounded in, and what it cannot.

The Containment Agent's policy learns which response to pick from a *simulator* whose reward table
was written for this project. No public dataset records real analysts' decisions and their
outcomes, so that cannot be replaced with real experience for free. What real, open sources do
give is grounding for two things around it:

**What the actions are.** MITRE D3FEND is the catalogue of defensive techniques. Each of the
project's response actions is mapped to the D3FEND technique it implements (network isolation,
inbound traffic filtering, account locking ...), and the ontology supplies the name, definition and
tactic (Isolate / Evict / Harden ...). The mapping is written by hand and checked against the loaded
ontology, so an id that does not exist cannot ship. An action with no defensive counterpart
(notifying an analyst, enrichment) is reported as having none, not forced into the catalogue.

**How urgent a vulnerability is.** CISA's Known Exploited Vulnerabilities catalogue lists CVEs that
are being exploited in the wild, and FIRST's EPSS gives the estimated probability that a CVE is
exploited in the next 30 days. Both are attached to the real supply chain's vulnerabilities, and
the advisories the workspace offers for review are ranked by them.

What stays ours: the reward magnitudes the policy is trained on.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from sentinel.core.errors import SentinelError

__all__ = [
    "ACTION_D3FEND",
    "Exploitation",
    "GroundingError",
    "countermeasures",
    "load_exploitation",
    "load_ontology",
]

D3FEND_PATH: Final[Path] = Path("data/real/d3fend.json")

#: Response action -> D3FEND technique id (``None``: there is no defensive counterpart).
ACTION_D3FEND: Final[dict[str, str | None]] = {
    "isolate_host": "D3-NI",  # Network Isolation
    "block_ip": "D3-ITF",  # Inbound Traffic Filtering
    "disable_account": "D3-AL",  # Account Locking
    "kill_process": "D3-PT",  # Process Termination
    "quarantine_file": "D3-FEV",  # File Eviction
    "open_patch_pr": "D3-SU",  # Software Update
    "notify_analyst": None,  # a message to a person, not a countermeasure
    "enrich_only": None,  # collects context; changes nothing
}

_TACTICS: Final[frozenset[str]] = frozenset(
    {"Model", "Harden", "Detect", "Isolate", "Deceive", "Evict", "Restore"}
)


class GroundingError(SentinelError):
    """A grounding file is missing or malformed."""


def load_ontology(path: Path | str = D3FEND_PATH) -> dict[str, dict[str, Any]]:
    """The D3FEND ontology as ``{node id: node}``."""
    file = Path(path)
    if not file.is_file():
        raise GroundingError(f"{file} not found; run scripts/fetch_real_data.py")
    try:
        graph = json.loads(file.read_text(encoding="utf-8"))["@graph"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GroundingError(f"{file} is not a D3FEND ontology: {exc}") from exc
    return {node["@id"]: node for node in graph if "@id" in node}


def _label(node: dict[str, Any]) -> str:
    label = node.get("rdfs:label", "")
    return str(label.get("@value", "") if isinstance(label, dict) else label)


def _parents(node: dict[str, Any]) -> list[str]:
    raw = node.get("rdfs:subClassOf", [])
    raw = raw if isinstance(raw, list) else [raw]
    return [
        p["@id"] for p in raw if isinstance(p, dict) and str(p.get("@id", "")).startswith("d3f:")
    ]


def _tactic(ontology: dict[str, dict[str, Any]], node: dict[str, Any]) -> str | None:
    """The D3FEND tactic (Isolate, Evict, Harden ...) a technique enables, on it or an ancestor."""
    seen: set[str] = set()
    frontier = [node["@id"]]
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        entry = ontology.get(current)
        if entry is None:
            continue
        enables = entry.get("d3f:enables")
        for item in enables if isinstance(enables, list) else [enables] if enables else []:
            name = str(item.get("@id", "")).removeprefix("d3f:")
            if name in _TACTICS:
                return name
        frontier.extend(_parents(entry))
    return None


def countermeasures(ontology: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Each response action with the D3FEND technique it implements, or none."""
    by_id = {n["d3f:d3fend-id"]: n for n in ontology.values() if "d3f:d3fend-id" in n}
    rows: list[dict[str, Any]] = []
    for action, d3id in ACTION_D3FEND.items():
        if d3id is None:
            rows.append(
                {
                    "action": action,
                    "d3fend_id": None,
                    "label": None,
                    "definition": None,
                    "tactic": None,
                    "url": None,
                }
            )
            continue
        node = by_id.get(d3id)
        if node is None:
            raise GroundingError(f"{action} maps to {d3id}, which is not in this D3FEND ontology")
        rows.append(
            {
                "action": action,
                "d3fend_id": d3id,
                "label": _label(node),
                "definition": str(node.get("d3f:definition", ""))[:300],
                "tactic": _tactic(ontology, node),
                "url": f"https://d3fend.mitre.org/technique/{node['@id']}/",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Exploitation likelihood
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Exploitation:
    """CISA KEV membership and EPSS scores by CVE id."""

    kev: dict[str, dict[str, Any]]
    epss: dict[str, tuple[float, float]]
    epss_date: str | None

    def of(self, cve: str) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if cve in self.kev:
            entry = self.kev[cve]
            info["kev"] = {
                "added": entry.get("dateAdded"),
                "ransomware": entry.get("knownRansomwareCampaignUse") == "Known",
                "product": entry.get("product"),
            }
        if cve in self.epss:
            info["epss"] = self.epss[cve][0]
            info["epss_percentile"] = self.epss[cve][1]
        return info


def load_exploitation(directory: Path | str = "data/real") -> Exploitation | None:
    """KEV and EPSS from ``directory``. ``None`` if neither file is there."""
    base = Path(directory)
    kev: dict[str, dict[str, Any]] = {}
    epss: dict[str, tuple[float, float]] = {}
    date = None
    kev_file = base / "known_exploited_vulnerabilities.json"
    if kev_file.is_file():
        for entry in json.loads(kev_file.read_text(encoding="utf-8")).get("vulnerabilities", []):
            kev[entry["cveID"]] = entry
    epss_file = base / "epss_scores-current.csv.gz"
    if epss_file.is_file():
        with gzip.open(epss_file, "rt", encoding="utf-8") as handle:
            first = handle.readline()
            if first.startswith("#"):
                date = next(
                    (
                        p.split(":", 1)[1]
                        for p in first.strip("#\n").split(",")
                        if p.startswith("score_date")
                    ),
                    None,
                )
                rows = csv.DictReader(handle)
            else:
                rows = csv.DictReader([first, *handle])
            for row in rows:
                epss[row["cve"]] = (float(row["epss"]), float(row["percentile"]))
    if not kev and not epss:
        return None
    return Exploitation(kev, epss, date)
