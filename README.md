# Sentinel Mesh

An autonomous, agentic security operations platform for the mid-market and the MSSPs
that protect it. Implementation of [`Sentinel_Mesh_PRD.docx`](Sentinel_Mesh_PRD.docx).

**Status: Parts 1–5.4 complete; every PRD acceptance criterion F-01 to F-12 is met** — the foundation (ingestion, contracts,
tamper-evident audit, anomaly detection), the intelligence core (deep detector,
supply-chain GNN, RAG knowledge base, bandit response policy, diffusion augmentation),
the full agent layer (**all five PRD agents** across three checkpointed graphs sharing
one state machine, one Human Approval Gate and one audit chain), and the **connector
layer**: real HTTP connectors for Wazuh (EDR/firewall), SCIM (identity), GitHub (draft
PRs, never merges) and Slack/signed webhooks, behind a least-privilege router, exercised
end to end against live local API emulators, and the **Analyst Copilot dashboard**
(Part 5), from which all three PRD demo scenarios complete end to end. Since Part 5.3 people
sign in through **single sign-on** (OIDC with MFA, roles from identity-provider groups,
SCIM provisioning, short-lived sessions, an audited sign-in trail), and the interface shows
each role only the controls it may use. See [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) for exactly
what is done, what is measured, every finding that changed the design, and where the next
session picks up, and [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for a complete report of
what was built.

```
3,289 Python tests + 38 front-end tests · ruff clean
```

## What works today

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/evaluate.py --n 20000 --cross-dataset --graph --kb --policy \
    --agents --codescan --supplychain --connectors --dashboard
python -m sentinel.dashboard             # Analyst Copilot: prints API tokens, serves http://127.0.0.1:8765/
python -m sentinel.dashboard --dev-idp   # the same with single sign-on, through a demo identity provider
```

No datasets, no API keys, **no LLM call**, no Redis, **no torch**, and no outbound
network — the connector gate talks HTTP to emulators it starts on `127.0.0.1` — the evaluation
generates flows shaped like CIC-IDS2017 (defects included) and runs them through the
real pipeline, including all three real orchestration graphs, the real static
analyzer over a real seeded repository, and the real audit log.

```
[deep (IF + denoising autoencoder)]
[test] n=7,199 (33.3% attack)
  ROC-AUC          0.9971   PASS (F-03 needs >= 0.90)
  PR-AUC           0.9950
  @threshold=0.8904: P=0.8298 R=0.9975 F1=0.9059
  alerts to human  40.1% of raw feed (59.9% reduction)
  FPR on benign    0.1023   reduction @1% base rate 88.9% (PRD 9.1 needs >= 60%)
  per-detector AUC denoising_autoencoder=0.9988, isolation_forest=0.9698
  recall by family:
    web_attack     0.9832
    dos            0.9953
    brute_force    0.9959
    botnet         1.0000
    ddos           1.0000
    infiltration   1.0000
    recon          1.0000
[cross-dataset unsw] ROC-AUC 0.9969 (in-domain 0.9971, drop +0.0002)

[supply-chain graph] 500 nodes, 1080 edges
  54 high-risk nodes: 25 intrinsic, 29 inherited
  top-10 precision   0.9000   PASS (F-06 needs >= 0.80)
  rank corr vs truth 0.6682
  features-only      0.5000   <- the graph's contribution is the gap
  inherited-risk only: GNN 0.5000 vs features-only 0.0000
  top flagged nodes:
    pkg-0129     risk=0.967 driver=own_features
      pkg-0155 (11 CVEs, 1961 days stale) reaches pkg-0129 in 1 hop(s); contribution 0.642

[triage (F-02)] n=6,211 (22.7% attack)
  label agreement  0.9921   PASS (F-02 needs >= 0.85)
  recall           0.9965   precision 0.9697   alert reduction 76.7%
  technique match  0.9929 on 1,411 attacks
  max latency      0.23 ms (budget 5,000)

[orchestration (F-04)] 200 incidents, sampled evenly across the test split
  dismissed at triage 150 · stopped at the gate 35 · completed 50 · failed 0

[investigation (F-05)] 50 reports — uncited claims 0, unresolvable refs 0
[approval gate (F-08)] ungated executions 0, audit chain findings 0
[Section 9.1] MTTD 0.03s worst (< 30s) · MTTC 12.12s worst (< 180s)

[code-scan (F-07)] 14 rules, 4 files, 268 lines
  seeded detected        18/18   recall 1.000   PASS (F-07 needs >= 3)
  false positives        0 of 18 SAFE controls   PASS (needs 0)
  validated patches      15      PASS (each parses, replays through its own diff,
                                 removes its finding, and adds none)
  rejected patches       0
  gate                   stopped for approval; 0 PRs opened before it
  report                 39 cited claims, 0 uncited, 24 CVE citations
  findings by rule:
    python.sql-injection            2 found, 1 patched
    python.os-command-injection     1 found, 1 patched
    python.os-system-injection      3 found, 2 patched
    python.hardcoded-credential     2 found, 2 patched
    python.path-traversal           2 found, 1 patched
    python.code-injection-eval      1 found, 1 patched
    python.pickle-deserialization   1 found, 0 patched  <- no safe loader exists
    ... 7 more rules, all patched

[supply-chain agent (F-06 guardrail)] 500 nodes, 1080 edges
  flagged                10  (7 packages, 3 organizations)
  explainable            10/10 = 1.000   PASS (needs >= 0.80)
  graph-path citations   27 · cited claims 23 · uncited 0
  package route          gated (open_patch_pr is destructive)
  vendor route           not gated (notify_analyst has no side effect)
  attribution conflicts  1, reported rather than hidden
  #4 org-19 (organization) risk=0.9522 driver=supply_chain (own 5% / chain 95%)
     pkg-0129 (17 CVEs, 1541 days stale) reaches org-19 in 2 hop(s)
     via pkg-0129 -> vendor-014 -> org-19; contribution 0.490

[connectors (Part 4)] 300 incidents through Wazuh/SCIM/GitHub/Slack over HTTP
  gated 29 · approved 22 · rejected 7 · failed actions 0 · router refusals 0
  state agreement        PASS  (what Wazuh blocked == what was approved)
  draft PRs 1 · merges 0 · methods on the wire GET, POST · pushed files re-scan clean
  wire before approval   0     (no byte left the process before a human said yes)
  out-of-scope probes    11/11 refused, 0 reached the remote
  crash between the call and the checkpoint: 1 execution with the journal, 2 without
```

## PRD acceptance criteria met

| ID | Criterion | Result |
|---|---|---|
| **F-01** | 1,000+ alerts/min sustained, zero schema failures | >1,000,000/min, 0 failures |
| **F-02** | Triage ≥ 85% agreement with ground truth on held-out split | **0.9921** |
| **F-03** | Anomaly detector ROC-AUC ≥ 0.90 on held-out split | **0.9971** |
| **F-04** | Any node can pause for human input and resume with full context | Resume asserted at **every** node, across a real SQLite file in a fresh process image, hash-identical to an uninterrupted run |
| **F-05** | Every factual claim traces to a KB chunk or raw log line | 0 uncited claims, 0 unresolvable refs over 50 reports |
| **F-06** | Supply-chain top-10 precision ≥ 0.80 | **0.8000** (mean over 10 fixed seeds); guardrail: **10/10** flagged nodes explained by a concrete graph path |
| **F-07** | ≥ 3 seeded vulnerabilities found, a syntactically valid patch PR for each | **18/18** seeded found (recall **1.000**), **0** false positives on 18 SAFE controls, **15** validated patches, **0** rejected |
| **F-08** | Zero actions executed without a logged approval | **0** ungated executions on **all three** graphs, read back from the audit chain; with real connectors, **0** requests on the wire for a gated action before its approval |
| **F-10** | All 3 demo scenarios completable end to end from the UI alone | **3/3** completed over HTTP alone by `evaluate.py --dashboard` (7 decisions, 0 ungated, 22/22 boundary probes refused), and walked through by hand in a browser |
| **F-09** | Simulated regret decreases over a 200-episode replay | regret **60.6** vs **301.4** for a non-learning policy |
| **F-11** | Chain verification detects any tampering in < 1s | 50,000 rows verified sub-second; every tampering class detected |
| **F-12** | Report generated by the same pipeline as the demo | One `scripts/evaluate.py`, no separate reporting path |
| §5.7 | Every connector least-privilege, never an admin credential | Capabilities, declared scopes and an egress allowlist per connector; **11/11** out-of-scope probes refused before the wire |
| §9.1 | ≥ 60% fewer alerts reaching a human | **88.9%** at a realistic 1% attack base rate; **76.7%** measured directly at triage |
| §9.1 | MTTD < 30s, MTTC < 3 min | **0.03s** / **12.12s** worst case, real node durations |

Every Must and Should criterion in PRD Section 6 is met. F-13 (PPO) and F-14 (a live
production integration) are scoped by the PRD itself as post-sprint.

Part 1's honest weak spot — `botnet` recall 0.608 — is closed: **1.000**, with no
family regressing. Five caveats stated plainly rather than buried:

- **F-06 sits right on its bar.** 0.8000 against a 0.80 target, per-seed 0.60–0.90.
  Met, not comfortably met; rank correlation (0.65) is the stabler figure.
- **§9.1's split-level number went *down*** (60.8% → 59.9%) precisely *because* recall
  went up. On a 33%-attack evaluation split every extra attack correctly caught is one
  more alert reaching a human, so that metric penalises a better detector. Rather than
  detune to recover a percentage point, both numbers are reported and the measured FPR
  and recall are projected onto a realistic feed.
- **No LLM has been called.** `AnthropicEngine` is written, injectable and tested
  against a fake transport, but every number above was produced with `NullEngine`. That
  is the honest configuration to measure in — see *monotone caution* below — but it
  means the *quality* a model would add to an investigation narrative is unmeasured.
- **F-07's numbers are a measurement of these rules against these shapes**, on a
  268-line fixture. Recall 1.000 with zero false positives on 18 controls is real and
  falsifiable — the controls are the *correct* construction for each rule, sitting beside
  the defective one — but it is not a false-positive rate on a production codebase, and
  that is the number that decides adoption.
- **No real external service has been contacted.** The Part 4 connectors are real HTTP
  clients for the documented Wazuh, GitHub, SCIM and Slack APIs, tested against strict
  emulators over live sockets — not against production instances. F-14 (a live
  production integration) remains, as the PRD scopes it, post-sprint.
- **The Supply-Chain Agent drafts no dependency-manifest edit.** PRD §5.5.3 scopes the
  sprint graph as synthetic, so `package-0129` is not a real project and a version pin
  would be invented. The Code-Scan Agent's patches are real because they rewrite real
  syntax; a manifest patch becomes real with Phase 2's CycloneDX ingestion. The action's
  own rationale says so, not just this README.

## Architecture

Six layers, per PRD Figure 2. Layers 1–5 and 6's audit trail are built, and all five
PRD agents exist.

```
Layer 1  Sources        sentinel.ingest.replay        dataset replay as a live feed   ✅
Layer 2  Ingestion      sentinel.ingest.bus           event bus (Redis Streams)       ✅
                        sentinel.ingest.normalizer    heterogeneous -> canonical      ✅
                        sentinel.ingest.enrich        streaming session context       ✅
Layer 3  Intelligence   sentinel.ml.featurestore      one feature definition          ✅
                        sentinel.ml.anomaly           detection ensemble              ✅
                        sentinel.ml.nn                gradient-checked NN engine      ✅
                        sentinel.ml.deep              denoising autoencoder           ✅
                        sentinel.graph                supply-chain graph + GraphSAGE  ✅
                        sentinel.ml.diffusion         tabular DDPM augmentation       ✅
                        sentinel.ml.robustness        calibration probe, novelty gate ✅
                        sentinel.kb                   RAG over ATT&CK/CVE             ✅
                        sentinel.rl                   contextual bandit policy        ✅
                        sentinel.scan                 hermetic static analysis + fixes ✅
Layer 4  Orchestration  sentinel.agents.runtime       checkpointed state machine      ✅
                        sentinel.agents.triage        Triage Agent (F-02)             ✅
                        sentinel.agents.investigate   Investigation Agent (F-05)      ✅
                        sentinel.agents.contain       Containment + approval gate     ✅
                        sentinel.agents.codescan      Code-Scan / Patch (F-07)        ✅
                        sentinel.agents.supplychain   Supply-Chain Agent (F-06)       ✅
                        sentinel.agents.engine        LLM seam + monotone caution     ✅
Layer 5  Action         sentinel.connectors.router    least-privilege dispatch        ✅
                        sentinel.connectors.wazuh     EDR isolation, firewall blocks  ✅
                        sentinel.connectors.scim      account disablement (SCIM 2.0)  ✅
                        sentinel.connectors.github    draft PRs + issues, no merge    ✅
                        sentinel.connectors.notify    Slack, signed webhooks          ✅
                        sentinel.connectors.sandbox   live local API emulators        ✅
Layer 6  Oversight      sentinel.audit                hash-chained audit log          ✅
                        sentinel.dashboard            Analyst Copilot (F-10)          ✅
                        sentinel.dashboard.sso        OIDC SSO, SCIM, sessions, audit ✅
```

## The design decisions worth knowing about

Each of these is a place where the obvious implementation is quietly wrong, and each is
pinned by a test.

**Monotone caution: the LLM can raise an alarm and cannot lower one.** Every agent
computes a deterministic verdict first — detectors, classifier, injection pre-scan,
knowledge base. The model is then asked for an opinion, and `monotone_caution` merges it
under one rule: severity may only be *raised*, a decision may only move *toward*
escalation, confidence is taken as the **minimum**, and a technique mapping is accepted
only if every alert field it cites actually exists. So a model fully compromised by
prompt injection can raise false alarms and **cannot suppress a real alert** — a bound
from arithmetic rather than from the model's cooperation. `HostileEngine` (which
dismisses everything at confidence 1.0, fabricates citations, and writes an instruction
aimed at whoever reads next) is a test asset that drives this end to end. The
confidence-minimum rule also cascades: a *mutually agreed* dismissal whose confidence
the engine lowered below Appendix A's 0.6 floor becomes an escalation.

**The model cannot write code, and that is arithmetic rather than policy.** Monotone
caution bounds what a compromised engine can do to a *verdict*. For the Code-Scan Agent
the output is a *diff*, so the bound is stricter: **no field of an engine response is read
when building a patch.** Every hunk is an AST-positioned splice generated from the rule
that fired. An engine fully controlled by an attacker — through a poisoned advisory, a
hostile code comment, a crafted commit — can raise a severity and add a cited sentence,
and cannot place one character into a pull request. `HostileEngine` drives this through
the live graph and the pushed diff comes out byte-identical to `NullEngine`'s. That
inverts how LLM-drafted patches usually work, and it is what makes shipping the result as
a draft PR defensible: the language model is a *narrator*.

**"A syntactically valid patch" is four checks, and the second one is the one that
matters.** `ast.parse` on the result clears F-07's literal wording, and clearing it
literally produces exactly the tool developers already ignore. So a patch is accepted only
once the patched source parses, **replaying the diff against the original reproduces the
patched text**, one instance of the rule is gone, and no rule gained an instance. Check
two is not redundant: the diff is the artifact that reaches the pull request, and a diff
generated from one string while a different string was validated is a patch that passes
review and breaks the build. It runs through an independent unified-diff applier that
verifies every context and removal line — independent because if it shared an
implementation with the edit machinery, agreement between them would prove nothing.

**Two correct-looking patches that were silently wrong, both caught by the fourth gate's
absence rather than its presence.** A concatenated query `"... a = '" + n + "'"` became
`"... a = '?'"` — the opening quote lives in the left fragment and the closing quote in
the right, so the quote-stripping ran in each and matched in neither. The query runs,
matches the literal string `?`, returns nothing, and looks fixed. Then `LIKE '{term}%'`
became `LIKE '?%'`, a search for the two-character string `?%`; that one *cannot* be fixed
by substitution, because the real remedy moves the wildcard into the bound value, so the
fixer now **declines** and reports the finding without a patch. Both passed all four
validation gates — syntactically perfect, finding removed — which is the failure mode a
patch-generating scanner has, and the reason the per-rule tests assert the resulting
*text* rather than the existence of a diff.

**Sanitizers have to be per-rule, and one boolean cannot express it.**
`os.path.basename(user_input)` genuinely fixes a path traversal and does nothing at all
for a command injection — `basename` returns `"a; rm -rf /"` unchanged. A single
"sanitised" flag must therefore be wrong in one direction or the other: treat it as clean
and the command injection is missed; treat it as tainted and every correctly fixed path
handler keeps reporting, which is how a scanner gets uninstalled. So a taint fact records
*which rules* a value has been cleared for. The same distinction separates "sanitised"
from "unknown": watching `shlex.quote` run on a value is positive evidence the developer
handled it and suppresses the finding, while a helper's parameter whose source is in a
caller the analysis does not follow is reported at MEDIUM.

**The F-07 fixture's ground truth lives in the fixture, and the SAFE half is the half
that matters.** `data/vulnerable_app` marks each planted defect with a comment on its own
line rather than in a manifest whose line numbers go stale the moment anyone edits the
file — a drift no test can catch, because both halves stay correct in isolation. Recall
alone is satisfiable by a scanner that flags every line, so each seeded defect sits beside
the **correct construction for the same rule**, and flagging one of those fails the build
as hard as missing a defect. There is a third state, `INFO`, because two cannot express
`os.system("logrotate -f /etc/logrotate.conf")`: the construct really is `os.system`, so
silence hides something, and the argument is a literal, so calling it a vulnerability is
crying wolf. And nothing in the analyzer reads a marker — a test strips every one and
asserts the finding set is identical, because otherwise a rule could come to key on the
comment and the fixture would be grading the scanner on reading its own answer key.

**Supply-chain risk is continuous, so it schedules itself rather than waiting for an
alert.** Nothing happened; a dependency unmaintained for four years was equally
unmaintained yesterday. Bolting a sixth node onto the incident graph would re-score 500
nodes per network flow and produce output unrelated to the triggering alert; a standalone
job with its own storage would need its own approval gate and audit log, so F-08 would
hold in two implementations and the second would be the untested one. Instead a scheduled
monitor mints its own `VENDOR_FEED` alert — deterministic per `(tenant, assessment)`, so
re-running a nightly tick *resumes* rather than forking — and a second graph reuses the
state machine, the checkpoint chain, the gate and the log unchanged. Three graphs, one
implementation of every guarantee.

**What the Supply-Chain Agent proposes depends on what it found, and the routing
follows.** A package can be upgraded, pinned or dropped, so the action is
`OPEN_PATCH_PR` — destructive, therefore gated. A vendor cannot be patched at any phase,
so the action is `NOTIFY_ANALYST`, which has no side effect on a monitored system and is
therefore **not** gated: asking a human to approve telling a human is a gate that teaches
people to click through. Both paths run in the same graph and both are measured.

**"Which action may execute now" was written three times and the third copy was wrong.**
At a trust tier permitting unattended execution an action never becomes `APPROVED`,
because nobody approves it — it stays `PENDING` with `requires_human_approval` false. The
code-scan graph's node looked only for `APPROVED`, so an autonomous-tier scan failed its
run while the routing that sent it there was correct. It now lives once, beside
`verify_no_ungated_execution`, because it is the positive form of the same security
predicate. Three copies of such a predicate means the rule holds in three implementations
and only one of them had a test.

**The novelty gate, promoted from diagnostic to safety mechanism.** Part 2.2 measured
that a classifier's softmax *rises* on far-out-of-distribution input — most confident
exactly where it knows least, and label smoothing made it worse. In Part 3 that
measurement became operational: triage confidence is discounted by distance from the
training manifold *before* the dismissal floor is applied. The result is the behaviour a
security team actually wants — **the stranger an alert is, the harder it is to
auto-dismiss** — which is precisely the case where a supervised triage model is
confidently wrong.

**The dataset's own label field broke two things at once.** CIC-IDS2017 ships its
ground-truth column inside every row, spelled `" Label":"BENIGN"`. That is verb,
separator, target, in exactly the order the injection scanner's verdict-manipulation
rule looked for — so **77% of a 12,000-alert corpus was flagged as prompt injection**
and every alert escalated as an attack on the agents. It is also the answer: fencing
that payload into a triage prompt hands the model the label F-02 scores it on. Both are
fixed, differently on purpose — the scanner rule now excludes structural punctuation
between verb and object, and label redaction happens at the *prompt boundary*, not in
the normalizer, because `raw_payload` is the audit record of what the source sent.

**The approval gate was unreachable at the tier every customer starts on.** §5.5.4 gives
the policy `auto_contain` and the action mask correctly requires the `auto_with_notify`
tier for it; §5.7 says every action type starts at `recommend`. Together: the mask
removed `auto_contain`, escalation mapped only to `notify_analyst`, no destructive action
was ever proposed, and **200 incidents produced 133 executions and 0 approval requests**
— F-08 passing vacuously on a system that had never gated anything. "Contain" names two
things. The policy's `auto_contain` means *act without a human*; *recommending* a
containment action for approval is what the `recommend` tier is *for*.

**Lookup beats ranking when you already have an identifier.** Triage names a technique,
and a name is an identifier, not a query. Searching "brute force authentication" returns
`T1187 Forced Authentication` second, and a narrative built on it is fluent, cited and
about the wrong technique. Worse, the relevance score does not separate them — the
correct hit scored **0.025** against a wrong neighbour's **0.066** — so no threshold
would have caught it. Evidence now comes from the identifier's own chunks plus the
knowledge base's relation graph; free-text search runs only when there is no identifier
to look up.

**MTTD was nine years, and a frozen clock would have hidden it.** The offline generator
leaves `ingested_at` at the capture timestamp, so §9.1's literal definition measures the
age of CIC-IDS2017. Both definitions are now reported with the feed lag between them,
and a test drives the *real* replay service to show they coincide on the streamed path
the demo uses. Separately, a `FrozenClock` reports every node as taking 0.00s — a
30-second budget measured on one passes regardless of how slow the pipeline is — while a
`SystemClock` cannot inject §9.1's scripted human click. `SimulationClock` does both.

**Guardrails are schema invariants, not prompt instructions.** PRD F-08 requires that no
destructive action executes without a logged approval. That is enforced by making the
ungated object *unconstructible*: `ActionRequest` rejects a destructive action at or
below the `recommend` trust tier with `requires_human_approval=False`, and
`ActionRequest.propose()` derives the flag so a caller never gets to set it. Likewise,
Appendix A's "if confidence < 0.6 you MUST escalate" is a validator on `TriageResult`,
not a sentence in a prompt an attacker can argue with. This covers code paths that have
not been written yet — which is the point of having built it before the agents.

**Untrusted text cannot be rendered by accident.** `UntrustedText.__str__`, `__repr__`
and `__format__` all return a redacted fingerprint (`<untrusted:siem sha256=1a2b3c4d
len=812>`). Reaching the content requires `.raw`, which is greppable in review.
`for_prompt()` fences it with a single-use random nonce, so an attacker who writes
`</untrusted_data>` into a log line cannot close a fence they cannot predict.

**One feature definition, structurally.** `AlertVectorizer` accepts only canonical
`Alert` objects — there is no `fit(dataframe)` overload, so a training job cannot read a
column the serving path never sees. Batch and single-row paths share one extraction and
a test asserts they are *bit*-identical, because the classic skew bug agrees to six
decimal places and is wrong forever. Fitted artifacts carry a spec fingerprint and
refuse to score under a changed feature space.

**Calibration before ensembling.** Isolation Forest returns a negated path length around
[-0.5, 0.5]; PCA returns squared error in [0, ∞). Averaging them raw lets whichever has
more numeric spread dominate, and "tuned weights" silently absorb a unit conversion.
Both are mapped through an empirical CDF fitted on training scores, so 0.99 means "more
anomalous than 99% of benign traffic" for both, and the threshold becomes an
interpretable false-positive budget — which is how a SOC with two analysts actually
reasons about it.

**The audit log's limits are tested, not just documented.** A plain SHA-256 chain is
tamper-*evident*, not tamper-proof: an attacker with file write access who knows the
scheme can rewrite it.
`test_unkeyed_chain_can_be_fully_rewritten` performs exactly that forgery and asserts it
succeeds; `test_keyed_chain_resists_the_same_rewrite` shows what HMAC keying buys. The
append-only guarantee is enforced by SQLite triggers, not convention.

**The detection gap that per-flow features cannot close.** The first honest evaluation
came in at ROC-AUC 0.839, with brute-force recall of 0.02. A single SSH brute-force flow
is statistically identical to a legitimate login, because that is what it is; what
distinguishes it is four hundred of them from one host in ninety seconds. The fix was a
strictly-causal, bounded-memory streaming enricher — and a regression test that blinds
those columns and asserts detection degrades, so the finding cannot rot.

**NumPy instead of PyTorch, bought with a gradient check.** The PRD names PyTorch and
PyTorch Geometric. Both were replaced by a hand-written engine
([`ml/nn.py`](src/sentinel/ml/nn.py)), because the models are small (5,546 parameters;
500 graph nodes), `torch-geometric` on Windows needs compiled extensions pinned to a
specific torch build, and Part 1's "clone it and reproduce the number in one command"
property is worth protecting. What makes that defensible rather than lazy is
`gradient_check`: every analytic gradient is compared against a central finite
difference to ~1e-9, and the check is proven able to *fail* by injecting the three bugs
hand-rolled backprop actually ships with. `AnomalyDetector` is still a Protocol, so a
torch backend remains a one-line swap.

**An AUC-optimal tuner will delete your ensemble.** Given the autoencoder, weight
tuning assigned the Isolation Forest **0.0** — optimal for the metric, and it
reconstructs exactly the single point of failure PRD §5.5.2 wants an ensemble to
remove. The fix is a weight floor whose cost was measured rather than assumed: 0.10
costs 0.0017 AUC and keeps every family's recall; 0.20 starts eating the botnet recall
Part 2 exists to fix.

**The textbook autoencoder warning did not survive measurement.** "A bottleneck as wide
as the input learns the identity and stops detecting anything" is the standard caution.
Sweeping the bottleneck 2→31 on the real feature space, ROC-AUC stayed flat at
0.9978–0.9989 — and a *linear full-rank* control separated correlation violations 1141x
against a rank-2 bottleneck's 5x, i.e. the unconstrained model was **better**. Gradient
descent from small init finds the projector onto the benign manifold, not the identity,
because nothing off-manifold is in the training objective. The guard stayed; its stated
reason was corrected to what was measured.

**The Git connector cannot merge, and that is a property of the network layer.** Part 3's
stand-in had no `merge` method. A real GitHub token with `contents:write` *can* merge, so
the guarantee moved down a layer: every connector's HTTP client is bound to an egress
allowlist of `(method, path)` routes, and the Git connector's list contains no `PUT`,
`PATCH` or `DELETE` at all. Merging, closing, force-pushing or editing a PR out of draft
is an `EgressDenied` before a socket opens. The emulator implements the merge endpoint
precisely so a test can assert it was called **zero** times.

**Several patches to one file compose over tokens, not characters.** Each fix is
validated against the original file, so five fixes in `app.py` need composing into one
commit. A first character-level merge of two *competing* rewrites of `"0.0.0.0"` produced
`host="1127.0.0.1"` — parseable, plausible, wrong — and passed until a test asserted that
competing edits must conflict. Literals and identifiers are now atomic.

**Edge direction is the whole ballgame in the graph.** Every edge points the way risk
flows, so a dependency edge runs `dependency -> dependent` — the reverse of SBOM
notation. Transposing it yields a model that still trains, still converges, and
confidently answers a meaningless question. It is stated three times in the source,
enforced by node-kind pairing rules, and pinned by an **asymmetric** test fixture,
because a symmetric one would pass either way.

**The graph has to earn its place, so a baseline is reported beside it.** F-06's 0.80
bar is clearable by a model that ignores every edge, if the ground truth is a function
of node features. So risk is generated in two separable parts — *intrinsic* (visible in
a node's own features) and *inherited* (visible **only** through the graph) — and the
GNN is held to beating a features-only logistic regression on the inherited nodes
specifically: **0.50 vs 0.00**. If that margin ever closes, the graph has become
decoration and the test says so.

**PRD §5.5.3 contradicts itself and the contradiction is arithmetic.** A *k*-layer
message-passing network sees exactly *k* hops, so "2-layer GraphSAGE" cannot deliver
"exposure through a fourth-order dependency". Measured by perturbing a node three hops
away: a 2-layer model's score moves by *exactly zero*. Rather than pick whichever
clause read better, `n_layers` is configurable, the default is the PRD's 2, the
depth/overfitting trade is measured across 1–4 layers, and the structural explainer
walks the graph to arbitrary depth so the fourth-order *explanation* exists regardless
of the model's receptive field.

## Layout

```
src/sentinel/
  core/         contracts, canonical bytes, clocks, untrusted text
  audit/        hash-chained append-only log
  ingest/       replay, bus, normalizers, session enrichment
  ml/           feature store, anomaly ensemble, NN engine, autoencoder,
                diffusion, family classifier, calibration probe, metrics
  graph/        supply-chain graph, synthetic generator, GraphSAGE, path explainer
  kb/           corpus, chunking, BM25/TF-IDF/LSA retrieval, citation resolution
  rl/           response actions, trust-tier mask, shaped reward, Thompson bandit
  scan/         import resolution, taint analysis, 14 rules + mechanical fixes,
                unified diffs and a verifying applier, fixture ground truth
  agents/       state, checkpoints, runtime, prompts, engine, triage,
                investigate, contain, codescan, supplychain, orchestrator
  connectors/   egress allowlist, targets, journal, router, and the Wazuh,
                SCIM, GitHub and Slack/webhook connectors; live API emulators
  dashboard/    the three demo scenarios, the live workspace, auth (API tokens),
                sso (OIDC, SCIM, sessions, sign-in audit), a demo identity provider,
                the FastAPI app and its zero-build front end (static/)
data/
  vulnerable_app/  the F-07 fixture: 18 seeded defects, 18 SAFE controls,
                   with its ground truth as marker comments in the source
tests/
  unit/         the module-level suites
  integration/  PRD acceptance criteria end to end
  js/           front-end unit tests (node --test), run from pytest
scripts/
  evaluate.py   the single evaluation pipeline (F-12) — every gate, one command
docs/
  PROJECT_REPORT.md a complete report of what was built, by part
  BUILD_PLAN.md  what is built, what is measured, every finding, what is next
  ARCHITECTURE.md the decisions, and why the obvious alternative is wrong
  DATA.md        dataset downloads and every quirk the normalizers handle
```

## Optional extras

Parts 1 and 2 install with **four** dependencies (pydantic, numpy, pandas,
scikit-learn) and need no network access. Neither `[deep]` nor `[graph]` is required —
see the NumPy note above. Later parts add theirs:

```bash
pip install -e ".[bus]"     # redis — live event bus instead of the in-memory one
pip install -e ".[agents]"  # langgraph, anthropic — Part 3
pip install -e ".[api]"     # fastapi, uvicorn, httpx, pyjwt — the dashboard and SSO
pip install -e ".[deep]"    # torch — optional alternative backend, not required
```

## Signing in and roles

The dashboard has two ways in, for two kinds of caller:

| Caller | How | Notes |
|---|---|---|
| A person | **Single sign-on** (OIDC authorization code + PKCE) through the organisation's identity provider | ID token verified (signature, issuer, audience, expiry, nonce); a second factor is required; `SOC-Analyst` → analyst, `Auditor` → viewer, no mapped group → refused |
| A machine | A static API key (`SENTINEL_DASHBOARD_TOKENS`) | With SSO on, none exist unless configured explicitly |

An **analyst** can launch scenarios and approve or reject actions; a **viewer** reads
everything for their tenant and sees none of those controls (the API answers 403 as well —
hiding a button is the interface, the check on the server is the security). SCIM at
`/scim/v2/Users` provisions and deactivates users, and deactivating one ends their live
sessions at once. Sessions are a 15-minute access token with a rotating refresh token; every
sign-in, refusal and refresh is written to a hash-chained log shown on the Audit page.

**Live runs.** The Models and Evaluation pages open on saved results. An analyst can also
re-train the models under a chosen seed (~12 s), run a quick evaluation (~18 s) or the full
evaluation (~3 min, published seed) and see the result beside the saved one; the saved report
is never overwritten. The Overview has a live-feed toggle that streams alerts through the
pipeline on a timer.

`--dev-idp` starts a stand-in identity provider so this can be demonstrated on a laptop.
**It is the only provider this has been tested against**; a real Okta or Azure AD tenant is
three settings away (`SENTINEL_OIDC_ISSUER`, `_CLIENT_ID`, `_CLIENT_SECRET`) but has not
been tried. Sessions, the SCIM directory and the sign-in log live in memory.
