"""Download the real public data the REAL DATA workspace uses.

Nothing is downloaded unless ``--yes`` is given; without it the script only prints what it
would fetch, from where, and how big it is. Each file is checked after it arrives (size and
shape) and is written only if it passes.

    python scripts/fetch_real_data.py             # lists the files, downloads nothing
    python scripts/fetch_real_data.py --yes       # downloads both
    python scripts/fetch_real_data.py --yes --only attack

Files
-----
network  UNSW_NB15_training-set.csv  ~32 MB   the UNSW-NB15 dataset (Australian Centre for
         Cyber Security, UNSW Canberra), from a public Hugging Face mirror of the official
         training partition       -> data/raw/unsw-nb15/
attack   enterprise-attack.json      ~54 MB   MITRE ATT&CK for Enterprise as STIX 2.1, from
         github.com/mitre-attack/attack-stix-data   -> data/real/
kev      known_exploited_vulnerabilities.json  ~3 MB   CISA's catalogue of vulnerabilities known
         to be exploited in the wild (cisa.gov)      -> data/real/
epss     epss_scores-current.csv.gz  ~6 MB   FIRST's daily probability that each CVE is exploited
         (epss.empiricalsecurity.com)                -> data/real/
d3fend   d3fend.json                 ~5 MB   MITRE D3FEND, the catalogue of defensive techniques
         and the ATT&CK techniques they counter      -> data/real/
flows    UNSW_NB15_full_shard2.parquet  ~106 MB   the full UNSW-NB15, one of two shards, with real
         source and destination IPs and timestamps (a public Hugging Face mirror of the official
         data)                                       -> data/raw/unsw-nb15/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Asset:
    key: str
    filename: str
    url: str
    destination: Path
    expected_bytes: int
    description: str
    min_bytes: int | None = None


ASSETS = (
    Asset(
        "network",
        "UNSW_NB15_training-set.csv",
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
        ROOT / "data" / "raw" / "unsw-nb15",
        32_293_018,
        "UNSW-NB15 training partition (labelled network flows)",
    ),
    Asset(
        "kev",
        "known_exploited_vulnerabilities.json",
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        ROOT / "data" / "real",
        2_675_299,
        "CISA Known Exploited Vulnerabilities catalogue",
    ),
    Asset(
        "epss",
        "epss_scores-current.csv.gz",
        "https://epss.empiricalsecurity.com/epss_scores-current.csv.gz",
        ROOT / "data" / "real",
        6_000_000,
        "FIRST EPSS: probability that each CVE is exploited in the next 30 days",
        min_bytes=1_000_000,
    ),
    Asset(
        "d3fend",
        "d3fend.json",
        "https://d3fend.mitre.org/ontologies/d3fend.json",
        ROOT / "data" / "real",
        4_833_645,
        "MITRE D3FEND defensive-technique ontology",
    ),
    Asset(
        "flows",
        "UNSW_NB15_full_shard2.parquet",
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/data/"
        "train-00001-of-00002.parquet",
        ROOT / "data" / "raw" / "unsw-nb15",
        105_596_940,
        "UNSW-NB15, full flows with real IPs and timestamps (shard 2 of 2)",
    ),
    Asset(
        "attack",
        "enterprise-attack.json",
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
        "enterprise-attack/enterprise-attack.json",
        ROOT / "data" / "real",
        53_835_637,
        "MITRE ATT&CK for Enterprise (STIX 2.1)",
    ),
)


def validate(asset: Asset, path: Path) -> str | None:
    """A problem description, or ``None`` if the file looks right."""
    size = path.stat().st_size
    floor = asset.min_bytes if asset.min_bytes is not None else asset.expected_bytes * 0.5
    if size < floor:
        return f"only {size:,} bytes; expected about {asset.expected_bytes:,}"
    if asset.key == "kev":
        try:
            catalogue = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            return f"not valid JSON: {exc}"
        if not catalogue.get("vulnerabilities"):
            return "no vulnerabilities in the catalogue"
        return None
    if asset.key == "d3fend":
        try:
            ontology = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            return f"not valid JSON: {exc}"
        return None if "@graph" in ontology else "not a JSON-LD ontology"
    if asset.key == "epss":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            head = [handle.readline() for _ in range(3)]
        return None if any(line.startswith("cve,epss") for line in head) else "not an EPSS file"
    if asset.key == "flows":
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(path).schema_arrow.names)
        need = {"srcip", "dstip", "Stime", "attack_cat", "label"}
        return None if need <= names else f"missing columns {sorted(need - names)}"
    if asset.key == "attack":
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            return f"not valid JSON: {exc}"
        if not isinstance(bundle, dict) or bundle.get("type") != "bundle":
            return "not a STIX bundle"
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if not {"sbytes", "dbytes", "dur", "label"} <= set(header):
            return "not the UNSW-NB15 column layout"
    return None


def download(asset: Asset) -> bool:
    asset.destination.mkdir(parents=True, exist_ok=True)
    target = asset.destination / asset.filename
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {asset.filename} ...", flush=True)
    try:
        with httpx.stream("GET", asset.url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            done = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    print(f"\r  {done / 1e6:6.1f} MB", end="", flush=True)
        print()
    except httpx.HTTPError as exc:
        print(f"  failed: {type(exc).__name__}: {exc}")
        partial.unlink(missing_ok=True)
        return False
    problem = validate(asset, partial)
    if problem:
        print(f"  refused: {problem}")
        partial.unlink(missing_ok=True)
        return False
    partial.replace(target)
    print(f"  saved {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--yes", action="store_true", help="actually download (default: list only)")
    parser.add_argument("--only", choices=[a.key for a in ASSETS], help="fetch just one file")
    args = parser.parse_args()
    chosen = [a for a in ASSETS if args.only in (None, a.key)]

    print("Real public data used by the dashboard's 'Real data' page:\n")
    for asset in chosen:
        exists = (asset.destination / asset.filename).is_file()
        note = "(already present)" if exists else ""
        print(f"  {asset.filename}  ~{asset.expected_bytes / 1e6:.0f} MB  {note}")
        print(f"    {asset.description}")
        print(f"    from {asset.url}\n    to   {asset.destination.relative_to(ROOT)}/\n")
    if not args.yes:
        print("Nothing downloaded. Re-run with --yes to fetch these files.")
        return 0
    ok = all([download(a) for a in chosen])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
