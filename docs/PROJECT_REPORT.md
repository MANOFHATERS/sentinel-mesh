# Sentinel Mesh — Project Report

*What was built, how it works, what was measured, and what is honestly not done.*
*Repository: `MANOFHATERS/sentinel-mesh` · implements `Sentinel_Mesh_PRD.docx` · state as of commit `660b6a0` plus the documentation update that adds this file.*

---

## 1. What this project is

Sentinel Mesh is an **autonomous, agentic security-operations platform** for mid-market companies and the managed-security providers that protect them. It reads a stream of security alerts, decides which are real, investigates them, proposes a response, and — only after a human approves — carries the response out through real connectors (firewall/EDR, identity, GitHub, Slack). It also watches the software supply chain and scans code for vulnerabilities.

The design idea that everything else follows from: **an AI system that can act must be bounded by arithmetic and structure, not by good behaviour.** So the model can raise an alarm but never lower one; it can narrate a patch but never write one; a destructive action cannot be constructed without an approval flag; and every step lands in a tamper-evident audit chain.

| At a glance | |
|---|---|
| Source code | ~37,000 lines of Python in `src/` |
| Tests | ~29,500 lines · **3,312 Python tests + 54 front-end tests**, ruff clean |
| Documents | `README.md`, `docs/BUILD_PLAN.md` (ledger, ~1,450 lines), `docs/ARCHITECTURE.md`, `docs/DATA.md`, this report |
| Commits on `main` | Parts 1 → 5.3, one part per session, each pushed after the full suite and the evaluation gate passed |
| PRD status | **Every Must/Should criterion F-01 … F-12 met and measured by one command** (`scripts/evaluate.py` exits non-zero if any gate fails). F-13 (PPO) and F-14 (live production integration) are post-sprint by the PRD's own scoping |
| Runs with | No datasets, no API keys, **no LLM call**, no Redis, **no PyTorch**, no outbound network |

---

## 2. Architecture (PRD Figure 2, six layers)

```
Layer 1  Sources        replay a dataset as a live feed
Layer 2  Ingestion      event bus · normalizers · streaming session enrichment
Layer 3  Intelligence   feature store · anomaly ensemble · NN engine · autoencoder
                        supply-chain graph + GraphSAGE · RAG knowledge base
                        contextual-bandit response policy · diffusion augmentation
                        static analysis with mechanical fixes
Layer 4  Orchestration  checkpointed state machine · five agents on three graphs
Layer 5  Action         least-privilege router · Wazuh · SCIM · GitHub · Slack/webhooks
Layer 6  Oversight      hash-chained audit log · Analyst Copilot dashboard · SSO
```

**The five agents** (all run on three checkpointed graphs that share one state machine, one Human Approval Gate and one audit chain):

| Agent | Job | PRD |
|---|---|---|
| Triage | Score an alert (detector + classifier + novelty gate), dismiss, monitor or escalate | F-02 |
| Investigation | Write a cited, MITRE-mapped narrative over the knowledge base | F-05 |
| Containment | Propose a response (informed by the learned policy) and stop at the approval gate | F-08 |
| Code-Scan / Patch | Find vulnerabilities, generate validated patches, draft a pull request | F-07 |
| Supply-Chain | Score dependency risk through the graph, explain it as a concrete path | F-06 |

---

## 3. What was built, part by part

### Part 1 — Foundation
Layers 1–3 (detection) and 6 (audit) plus the contracts everything depends on.
- **Contracts** (`Alert`, `TriageResult`, `InvestigationReport`, `ActionRequest`, `Evidence`) as frozen pydantic models. The guardrails are validators, not prompt text: an `ActionRequest` for a destructive action at or below the "recommend" tier with `requires_human_approval=False` **cannot be constructed**.
- **Untrusted-input discipline:** attacker text renders as a redacted fingerprint; reaching the content needs `.raw` (greppable in review); prompt fences use a single-use random nonce; a prompt-injection scanner.
- **Hash-chained, append-only audit log** (SQLite triggers enforce append-only). The limits are tested too: a test performs the forgery an unkeyed chain allows, and another shows HMAC keying stops it.
- **Ingestion:** event bus (Redis Streams semantics + in-memory), CIC-IDS2017 and UNSW-NB15 normalizers, replay service, and a strictly causal, bounded-memory **session enricher** — the fix for a real finding (single-flow features cannot see a brute-force campaign; ROC-AUC went 0.839 → 0.997).
- **Feature store** with one feature definition (no `fit(dataframe)` overload), bit-identical batch/single-row paths, spec fingerprints.
- **Anomaly ensemble** (Isolation Forest + PCA) with calibration to an empirical CDF so scores are comparable.
- **One evaluation pipeline** (`scripts/evaluate.py`, F-12) and a campaign-structured synthetic dataset generator shaped like CIC-IDS2017, defects included.

### Part 2 — Intelligence core
- **Hand-written NumPy neural-network engine** with a finite-difference `gradient_check` that is itself proven able to fail (three injected backprop bugs). Chosen over PyTorch on purpose: 5,546 parameters, one-command reproducibility, no compiled extensions on Windows.
- **Denoising autoencoder** behind the `AnomalyDetector` protocol; a weight floor stops an AUC-optimal tuner from deleting the Isolation Forest (cost measured: 0.0017 AUC).
- **Supply-chain graph + GraphSAGE** (500 nodes, 1,080 edges). Risk is generated as *intrinsic* + *inherited* so the graph must beat a features-only baseline on inherited nodes (0.50 vs 0.00). The PRD's "2-layer GraphSAGE reaches fourth-order dependencies" is arithmetically impossible; `n_layers` is configurable and the explainer walks arbitrarily deep.
- **RAG knowledge base** over ATT&CK/CVE (BM25/TF-IDF/LSA), with lookup-before-ranking for identifiers.
- **Contextual-bandit response policy** (Thompson sampling, trust-tier action mask, shaped reward) — F-09.
- **Diffusion augmentation** and an adversarial **calibration probe** (measured finding: softmax confidence *rises* out of distribution → became the novelty gate).

### Part 3 — Orchestration and the agents
- **State machine** with frozen `IncidentState`, hash-verified chain-linked checkpoints (memory + SQLite), `interrupt_before`/resume. F-04 is asserted by resuming at **every** node and requiring a hash-identical result.
- **`monotone_caution`:** an LLM verdict may only raise severity, move toward escalation, lower confidence (min), and cite fields that exist. A fully compromised model cannot suppress a real alert. `HostileEngine` is the adversarial test asset.
- **Triage, Investigation, Containment + approval gate, orchestrator**, `SimulationClock` (real node durations plus scripted human time, so MTTD/MTTC are honest).
- **Code-Scan Agent:** hermetic static analysis (import resolution, per-rule taint tracking, 14 rules) with **mechanical fixes** — the model is a *narrator*; no engine field is read when building a patch. A patch is accepted only if it parses, its diff replays to the patched text through an independent applier, the finding is gone, and no other rule gained an instance.
- **Supply-Chain Agent:** scheduled, continuous, on its own graph reusing the same state machine and gate; package route is gated, vendor route (notify only) deliberately is not.

### Part 4 — Connector layer
Real HTTP clients for **Wazuh** (host isolation, IP blocks), **SCIM 2.0** (account disablement), **GitHub** (draft PRs and issues — the client has no `PUT/PATCH/DELETE`, so merging is impossible at the network layer), **Slack/HMAC-signed webhooks**, behind a **router** that checks tenant, approval, journal, capability, target, blast radius and audit. Every connector has an egress allowlist of `(method, path)`, no redirects, bounded retries. Live local emulators over real sockets stand in for the services. A crash between "call made" and "checkpoint written" executes once with the journal, twice without.

### Part 5 — Analyst Copilot dashboard (F-10)
A FastAPI app with a zero-build front end (no npm; a d3-force-semantics layout in ~200 lines; `textContent`-only DOM helper, strict CSP with no inline script).
- **Pages:** Overview, Scenarios, Approval queue, Incidents (+ detail), Supply chain (interactive risk map), Code scan, Wire & guardrails, Audit log, Models, Evaluation.
- **Three demo scenarios** completable from the UI alone: *Phishing → lateral movement*, *Vendor-dependency CVE*, *Malicious open-source package* (whose payload addresses "AI security scanners" and is flagged and rendered inert).
- **Security model:** bearer tokens (digest-only registry), tenant scoping from the identity (never the request), and **the approver is the token, never the body**; another tenant's id returns the same 404 as a non-existent one.
- Measured over a real socket: 3/3 scenarios, 7 decisions, 0 ungated executions, 22/22 boundary probes refused.

### Part 5.1 — The policy in the live flow, and the Models page
The learned response policy now decides every live incident, behind a **triage floor** (a learned component may add caution, never remove it). Added the F-12 alert-reduction chart and the §9.1 cumulative-regret curve, and a Models page showing every model's training record. It reported, rather than hid, that diffusion augmentation does not help recall on this corpus.

### Part 5.2 — Edge-case audit and theme
A **130-case adversarial audit** across all five parts: 72 correct, 48 cleanly refused, 6 plain `ValueError`, **3 real bugs, all fixed** — (1) one crafted flow (1e308 bytes) could keep an alert away from every human, (2) an empty batch leaked a library error, (3) a comment addressed to "the AI reviewer" went unflagged. Kept as a permanent 99-test suite. The dashboard also got a single bright theme.

### Part 5.3 — Single sign-on and role-aware interface *(built in this working session)*
Prompted by three real observations while demoing (empty pages, dead Launch buttons for a viewer, and "how does a new user sign in?").

| Standard | What was built |
|---|---|
| **SSO (OIDC)** | Authorization-code + PKCE + `state` + `nonce`. ID token validated for signature (JWKS, RS256/ES256 only — `alg:none` and HMAC-confusion refused), issuer, audience, expiry, nonce, verified email |
| **Roles from groups** | `SOC-Analyst` → analyst, `Auditor` → viewer; no mapped group → **refused**, not defaulted; both groups → the more privileged |
| **MFA** | `amr`/`acr` must show a second factor (on by default) |
| **SCIM 2.0** | `/scim/v2/Users` create/list/filter/patch/delete under its own token; deactivation revokes live sessions at once; JIT provisioning can be turned off |
| **Short-lived sessions** | 15-minute access token, rotating refresh token, 12-hour cap; a refresh token used twice revokes the whole session; only digests stored |
| **Audit of sign-ins** | Every success, refusal, refresh, logout and SCIM change in a hash-chained log, shown on the Audit page |
| **Least-privilege UI** | Launch, Replay, Approve/Reject and Recover are *not rendered* for viewers; a read-only banner explains why; the server still returns 403 |
| **Demo identity provider** | `--dev-idp` mounts a stand-in OIDC provider (Maya = analyst, Omar = viewer, Nina = no MFA → refused, Carl = no group → refused) |

The browser still uses a bearer header (no cookie, so no CSRF surface); the token reaches the page as a one-time handoff code in the URL *fragment*.

### Part 5.4 — Live feed and live runs *(built in this working session)*
- **Live feed:** an analyst-only toggle streams held-out alerts through the incident graph (4 every 2 s), so the Overview counters and the approval queue move on their own.
- **Live runs:** the Models and Evaluation pages open on saved results; an analyst can re-train the models under any seed (~12 s), run a quick evaluation (~18 s) or the full evaluation (~2.5 min) and see the result beside the saved one. One run at a time, cancellable, and evaluations run as a subprocess of the one pipeline so the saved report is never overwritten.
- **Two findings, both fixed:** a crash-recovery experiment that only worked for some seeds, and agent-layer checks that drove the *first* N flows of the test split (under seed 7 almost all benign, so three gates failed). They now sample evenly across the split; the full evaluation passes every gate under seed 7 and under the published seed.

### Part 5.5 — Motion *(built in this working session)*
- **Supply-chain map:** computed first, then grown node by node from the highest-risk node along the graph's own links (each new node is linked to one already shown); every node pops in and every new link flashes. Zoom (+, −, wheel, keys), pan, Fit, Rebuild and Skip; hovering a node lights up its connections.
- **Charts:** bars grow from the baseline and lines draw left to right the first time they scroll into view, on every visit to the Models and Evaluation pages.
- **Pages:** every page animates in when opened (headings, tiles, cards, table rows staggered); a background refresh never replays it. All motion is CSS, off under `prefers-reduced-motion`.

---

## 4. Measured results

All from one command: `python scripts/evaluate.py --n 20000 --cross-dataset --graph --kb --policy --agents --codescan --supplychain --connectors --dashboard` (seed 20260928).

| ID | Criterion | Result |
|---|---|---|
| F-01 | ≥1,000 alerts/min, zero schema failures | >1,000,000/min, 0 failures |
| F-02 | Triage agreement ≥ 85% | **0.9921** |
| F-03 | Detector ROC-AUC ≥ 0.90 | **0.9971** (cross-dataset UNSW 0.9969) |
| F-04 | Pause/resume at any node | Asserted at **every** node, hash-identical to an uninterrupted run |
| F-05 | Every claim cited | 0 uncited, 0 unresolvable over 50 reports |
| F-06 | Supply-chain top-10 precision ≥ 0.80 | **0.80** (mean of 10 seeds; met, not comfortably); 10/10 flagged nodes explained by a path |
| F-07 | ≥3 seeded vulns + valid patch PRs | **18/18** found, **0** false positives on 18 controls, **15** validated patches |
| F-08 | Zero ungated executions | **0** on all three graphs; **0** requests on the wire before approval |
| F-09 | Regret decreases | 60.6 vs 301.4 for a non-learning policy |
| F-10 | 3 scenarios from the UI alone | **3/3** over HTTP, 22/22 boundary probes refused |
| F-11 | Tamper detection < 1 s | 50,000 rows sub-second, every tamper class detected |
| F-12 | Report from the demo pipeline | One `evaluate.py`, no separate path |
| §9.1 | ≥60% fewer alerts to humans | **88.9%** at a 1% attack rate; 76.7% measured at triage |
| §9.1 | MTTD < 30 s, MTTC < 3 min | **0.03 s** / **12.12 s** worst case |
| §5.7 | Connectors least-privilege | 11/11 out-of-scope probes refused before the wire |

---

## 5. Defects found and fixed during the build

The ledger records every finding that changed the design; the notable ones:

- **Dataset label broke the injection scanner** — CIC-IDS2017's own `" Label":"BENIGN"` column looked like verdict manipulation, flagging 77% of alerts. Fixed at the scanner *and* by redacting labels at the prompt boundary.
- **The approval gate was unreachable** at the trust tier every customer starts on — F-08 passed vacuously (200 incidents, 133 executions, 0 approvals). Fixed by separating "act without a human" from "recommend for approval".
- **MTTD was nine years** because the generator left the capture timestamp as ingest time; both definitions are now reported.
- **Two "correct-looking" patches were silently wrong** (`'?'` and `'?%'` placeholders) — the fixer now declines rather than emit them.
- **A merge of competing edits produced `host="1127.0.0.1"`** — patches now compose over tokens, not characters.
- **"Which action may execute now" was written three times, the third wrong** — now one function.
- **Part 5.2's three bugs** (overflow to infinity, empty batch, unflagged reviewer-directed comment).
- **This session:** (1) a viewer saw greyed-out buttons with no explanation → controls hidden, banner added; (2) sign-out kept the last page, so the next user landed on Evaluation → route reset; (3) the SSO button was an anchor that the front end's own `safeHref` filter silently stripped → replaced with a button and a regression test.

---

## 6. How to run it

```bash
pip install -e ".[dev,api]"
python -m pytest -q                                   # 3,312 tests (+54 front-end)
python scripts/evaluate.py --n 20000 --cross-dataset --graph --kb --policy \
    --agents --codescan --supplychain --connectors --dashboard

python -m sentinel.dashboard             # API tokens printed once; http://127.0.0.1:8765/
python -m sentinel.dashboard --dev-idp   # single sign-on via the demo identity provider
```

**Demo order:** sign in as Maya (analyst) → launch all three scenarios → approve/reject in the Approval queue → look at Incidents, Code scan, Wire & guardrails, Audit log (including *Sign-in activity*) → sign out → sign in as Omar (viewer) and show the read-only interface → try Nina and Carl to show the MFA and no-group refusals.

---

## 7. Honest limits

- **No LLM has been called.** `AnthropicEngine` is written and tested against a fake transport, but every number was produced with `NullEngine`. The quality a model adds to narratives is unmeasured.
- **No real external service has been contacted.** Connectors are real HTTP clients tested against strict local emulators; F-14 is post-sprint by the PRD.
- **SSO has only been tested against the built-in demo provider.** A real Okta / Azure AD tenant is three settings away but untried. Sessions, the SCIM directory and the sign-in log are in memory (a restart signs everyone out); a group change applies at the next sign-in.
- **F-06 sits on its bar** (0.80 vs 0.80; per-seed 0.60–0.90). Rank correlation (0.65) is the steadier figure.
- **F-07 is a measurement on a 268-line fixture**, not a false-positive rate on a production codebase.
- **The supply-chain graph is synthetic** (PRD §5.5.3), so the Supply-Chain Agent drafts no manifest edit; that needs CycloneDX ingestion.
- **`mypy --strict` has not been run** (not installed in the build environment).
- **One timing test** (F-11's 50k-row verification) can exceed its budget under heavy parallel load.
- **Uncommitted work:** `src/sentinel/agents/runtime.py` has a local, unpushed change (a LangGraph-backed rewrite of the state-machine module). It is not part of the pushed state or of any number above.

---

## 8. What comes next (PRD 90-day plan)

1. Point the Wazuh connector at a real free-tier Wazuh instance — the first real external service.
2. Make the first real LLM calls through `AnthropicEngine` and measure against the `NullEngine` baseline.
3. Replace the synthetic supply-chain graph with CycloneDX SBOM ingestion (which also lets the agent draft a real manifest change).
4. Connect a real identity provider to the SSO layer and move sessions/directory/audit to a database.
