# Datasets

This repository ships **no data**. CIC-IDS2017 is ~2.8M flows across several hundred
megabytes; UNSW-NB15 is a separate multi-gigabyte download. Both are freely available
for research, and neither is redistributable here.

Everything in the project runs and every test passes without them, because
[`ml/datasets/synthetic.py`](../src/sentinel/ml/datasets/synthetic.py) generates flows
shaped like the real files — including their real defects. What the synthetic corpus
**cannot** give you is genuine distribution shift; see the caveat at the bottom.

## CIC-IDS2017

Canadian Institute for Cybersecurity, University of New Brunswick.
<https://www.unb.ca/cic/datasets/ids-2017.html>

Download the **MachineLearningCSV** archive and unpack the CSVs into `data/raw/`:

```
data/raw/cic-ids2017/
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  Wednesday-workingHours.pcap_ISCX.csv
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  Friday-WorkingHours-Morning.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Then:

```python
from sentinel.ingest.normalizer import CICIDS2017Normalizer
from sentinel.ingest.replay import CsvReplaySource, ReplayService

source = CsvReplaySource(path="data/raw/cic-ids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
service = ReplayService(normalizer=CICIDS2017Normalizer(tenant_id="acme", strict=True), bus=bus)
stats = service.run(source)
print(stats.summary())
```

### Quirks the normalizer already handles

Every one of these is exercised by a test, because each of them has silently broken
somebody's pipeline before:

| Quirk | Why it matters |
|---|---|
| Column names carry **leading spaces** (`" Destination Port"`) | A naive `df["Destination Port"]` raises `KeyError` on some files and not others |
| `Flow Bytes/s` and `Flow Packets/s` contain literal **`Infinity`** and **`NaN`** | Zero-duration flows. Propagates as silent NaN through training; also makes an alert unhashable |
| `Flow Duration` is in **microseconds** | UNSW-NB15's `dur` is in seconds. Mixing them is a factor-of-10⁶ error that crashes nothing |
| Web-attack labels use a Unicode **en-dash** (`"Web Attack – Brute Force"`) | Widely-mirrored copies mojibake it to the cp1252 byte `\x96`, so a string match on one spelling misses the other |
| `Timestamp` is `D/M/YYYY H:MM(:SS)`, unpadded, no timezone | Day-first and genuinely ambiguous for days 1–12. The capture ran 3–7 July 2017, so day-first is correct |
| File encodings are inconsistent across mirrors | `CsvReplaySource` tries UTF-8 then falls back to latin-1 |

### Class imbalance

Roughly 80% benign. `Infiltration` is **36 flows out of 2.8M** — which is the reason
the diffusion augmentation in PRD §5.5.5 exists, and the reason
[`metrics.py`](../src/sentinel/ml/metrics.py) reports per-family recall next to the
aggregate. An aggregate AUC of 0.99 is compatible with never detecting Infiltration
at all.

## UNSW-NB15

Australian Centre for Cyber Security, UNSW Canberra.
<https://research.unsw.edu.au/projects/unsw-nb15-dataset>

Both variants work — the four-part full CSVs (`UNSW-NB15_1.csv` … `_4.csv`) and the
pre-split `UNSW_NB15_training-set.csv` / `UNSW_NB15_testing-set.csv`. Put them in
`data/raw/unsw-nb15/` and use `UNSWNB15Normalizer`.

### Quirks the normalizer already handles

| Quirk | Why it matters |
|---|---|
| `attack_cat` values are **space-padded** (`" Reconnaissance "`) | A dict lookup on the raw value misses them |
| `Backdoor` **and** `Backdoors` both appear | Two spellings, one family |
| Blank `attack_cat` with `label = 1` | The train/test CSVs do this. Reading blank as benign **relabels real attacks as normal traffic** |
| `dur` is in **seconds** | See the CIC duration note |
| The full CSVs have `Stime`/`Ltime`; the train/test CSVs do not | A stray small integer must not be read as a 1970 timestamp |

## The unified feature space

The two datasets do not share a feature set — ~78 columns versus ~47, defined
differently. Cross-dataset validation is only meaningful in the intersection, so
[`normalizer.py`](../src/sentinel/ingest/normalizer.py) defines
`UNIFIED_FEATURES`: eleven bidirectional flow statistics plus port and protocol, with
**rates recomputed from the counters** rather than copied from each source's own
derived columns (which is both where the `Infinity` values come from and where the two
datasets quietly disagree).

Session-context features (`sentinel.ingest.enrich`) are layered on top of that space
and computed identically for both datasets, so a model trained on one scores the other
without translation.

The verbatim source row is preserved in `Alert.raw_payload` (untrusted-wrapped), so a
richer per-dataset feature set remains available later without re-ingesting.

## Synthetic data — what it is and is not for

`generate_alerts(n)` produces canonical alerts by running generated
dataset-shaped rows through the **real** normalizer and enricher, so every test that
uses synthetic data also exercises the ingestion path.

The generator deliberately:

- emits the raw column spellings and the real defects above, including `Infinity` and
  both dash spellings;
- structures attacks as **campaigns** (one attacker, one target set, a contiguous
  burst) rather than scattered flows, because scattered attacks are undetectable by
  anything a real SOC uses;
- overlaps attack and benign distributions (`separability`), so a detector cannot
  score 0.999 and prove nothing;
- plans its class mix rather than sampling it, so the composition is exact and
  reproducible at every corpus size.

**What it cannot do:** prove generalisation. Both synthetic corpora come from one
campaign engine, so the cross-dataset test validates the *unit and schema
harmonisation* — a real and frequently-fatal bug class — rather than a genuine
distribution shift. For that, download the real files. PRD §10 lists "synthetic data
is unrealistic and undermines credibility" as a risk whose mitigation is being
explicit about what is real, so: nothing generated here is ever labelled as a real
capture, and every synthetic alert carries `dataset="cic-ids2017-synthetic"`.

## Real data in the dashboard (Part 5.6)

The **REAL DATA** login uses two downloadable files and one runtime download. All of it is
optional; the synthetic demo needs none of it.

| What | File / source | Size | Where it goes |
|---|---|---|---|
| Real network flows | `UNSW_NB15_training-set.csv` (UNSW Canberra; a public Hugging Face mirror of the official training partition) | ~32 MB | `data/raw/unsw-nb15/` |
| MITRE ATT&CK for Enterprise | `enterprise-attack.json`, STIX 2.1, from github.com/mitre-attack/attack-stix-data | ~54 MB | `data/real/` |
| A real repository | fetched at scan time from `codeload.github.com` for the URL you paste | 25 MB cap | memory only |

`python scripts/fetch_real_data.py` lists these and downloads nothing without `--yes`; each file is
checked (size and shape) before it is kept. Beyond the two above it can also fetch: the **full
UNSW-NB15** (one 106 MB Parquet shard with real IPs and capture times, from a public Hugging Face
mirror), **CISA KEV** (1.8 MB), **FIRST EPSS** (2.7 MB) and **MITRE D3FEND** (4.8 MB). Two more things are
built from free APIs, no key needed, and cached: `python scripts/build_real_supply_chain.py` (deps.dev and
OSV.dev, about five minutes the first time) and `python scripts/build_scanner_eval.py` (GitHub's API; its
unauthenticated limit of 60 requests an hour is enough, and `GITHUB_TOKEN` lifts it). CIC-IDS2017 works too (see above) but its official
download is behind a form, so it is not fetched automatically; put its CSVs in
`data/raw/cic-ids2017/`.

What a real run cannot tell you: the UNSW train/test CSVs carry no capture timestamps, so the
session-context features the synthetic corpus has do not exist for it (per-flow detection only);
and a real repository has no answer key, so scanner recall and false-positive rate are unknown.
Results are reported as measured, whatever they are.

## Other sources (Part 2)

| Source | Use | Status |
|---|---|---|
| MITRE ATT&CK STIX corpus | RAG knowledge base for investigation reports | Part 2.4 |
| NVD / CVE JSON feeds | Grounds the Investigation and Code-Scan agents | Part 2.4 |
| CycloneDX / SPDX SBOMs | Real supply-chain graph input | PRD Phase 2 |
| OSV.dev | Live package vulnerability feed | PRD Phase 2 |
