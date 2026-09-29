# Architecture Notes

PRD Section 5 specifies the six-layer architecture. This document records the
implementation decisions inside those layers: what was chosen, what the obvious
alternative was, and why it was rejected. It is deliberately about the *non-obvious*
calls — the ones where a reviewer would otherwise have to guess whether something was
considered.

```
┌─ Layer 1 · Sources ──────────────────────────────────────────────────┐
│  ingest.replay        dataset -> live alert stream                    │
│    ├─ CsvReplaySource      streaming, encoding-tolerant               │
│    └─ TimeModel            WALL_CLOCK (demo) | VIRTUAL (eval, tests)  │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ Layer 2 · Ingestion ────────▼───────────────────────────────────────┐
│  ingest.normalizer   CIC-IDS2017 | UNSW-NB15  ->  canonical Alert     │
│  ingest.enrich       strictly-causal sliding-window session context   │
│  ingest.bus          Redis Streams semantics (in-memory + real impl)  │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ Layer 3 · Intelligence ─────▼───────────────────────────────────────┐
│  ml.featurestore     one FeatureSpec, fingerprinted, train==serve     │
│  ml.anomaly/deep     IsolationForest + denoising AE, calibrated       │
│  ml.classify         supervised family classifier (F-02)              │
│  ml.robustness       calibration probe + novelty gate                 │
│  graph               supply-chain GraphSAGE + path explainer          │
│  kb                  ATT&CK/CVE corpus, BM25+TF-IDF+LSA, citations    │
│  rl                  trust-tier mask, shaped reward, Thompson bandit  │
│  ml.metrics          leakage-refusing splits, per-family reporting    │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ Layer 4 · Orchestration ────▼───────────────────────────────────────┐
│  agents.runtime      checkpointed state machine, interrupt + resume   │
│  agents.engine       ReasoningEngine seam + monotone_caution          │
│  agents.triage       F-02 · investigate F-05 · contain F-08           │
│  agents.orchestrator PRD Figure 3, with the gate as its own node      │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ Layer 6 · Oversight ────────▼───────────────────────────────────────┐
│  audit.log           SHA-256/HMAC chain, DB-enforced append-only      │
└──────────────────────────────────────────────────────────────────────┘
        core.schemas · core.canonical · core.untrusted · core.clock
        (cross-cutting: contracts, hashable bytes, injection defense, time)
```

---

## Layer 0 — cross-cutting

### Canonical bytes come before everything

`core.canonical` defines one byte representation per logical value, and every hash,
chain link and signature in the platform derives from it. It is strict by design:
`NaN`, `Infinity`, sets, naive datetimes and unknown types are **rejected** rather than
coerced.

*Alternative rejected:* `json.dumps(obj, default=str)`. It never fails, which sounds
convenient until a `NaN` in a feature dict silently serializes as `NaN` (invalid JSON
that some parsers accept and others don't), or a set serializes in a different order on
a different run and the audit chain fails to verify for no discoverable reason.

One subtle call: strings are **not** NFC-normalized. Folding `é` (U+00E9) and `é`
(`e` + U+0301) onto one hash would hand an attacker a way to alter stored text without
breaking the chain. Lone surrogates are rejected instead, since they cannot be encoded
at all.

### Contracts are immutable and revalidating

Every contract in `core.schemas` is a frozen pydantic model with `extra="forbid"`.
State transitions go through `Contract.updated()`, which round-trips through
`model_validate`.

*This was a bug, found by a test.* `model_copy(update=...)` is the idiomatic way to
write those transitions and pydantic deliberately **skips validation** on it — so
`ActionRequest.mark_executed()` was able to produce an object in `EXECUTED` state with
no approver, which is precisely the state PRD F-08 forbids. Revalidating costs a few
microseconds per transition and makes the invariant unconditional.

### Guardrails as type invariants, not prompt text

The pattern used throughout: where the PRD states a rule an agent must follow, the rule
is encoded so that violating objects **cannot be constructed**.

| PRD rule | Where it is enforced |
|---|---|
| "If confidence < 0.6 you MUST escalate rather than dismiss" (App. A) | `TriageResult` model validator |
| "Never claim a technique mapping without citing the alert field(s)" (App. A) | `TriageResult` model validator |
| "Every destructive action requires the Human Approval Gate" (§5.1, F-08) | `ActionRequest` model validator + `propose()` deriving the flag |
| "Every factual claim must cite a retrieved source or raw log line" (F-05) | `InvestigationReport` model validator |
| Prompt-injection content must escalate (§5.7) | `TriageResult` model validator |

The reason this matters more than it looks: a prompt rule is advice to a model an
attacker is actively trying to talk out of it. A validator that rejects the resulting
object holds regardless of what the model returns — including in code paths that have
not been written yet, which is why Part 1 was built before the agents.

### Untrusted text is a type, not a convention

`UntrustedText` returns a redacted fingerprint from `__str__`, `__repr__` **and**
`__format__`, so accidental interpolation into a prompt is visibly safe. `for_prompt()`
fences content with a single-use random nonce.

*Alternative rejected:* a fixed delimiter such as `<untrusted>...</untrusted>`. An
attacker who knows the delimiter writes it into a log line and escapes the data region.
A nonce they cannot predict cannot be closed — and an attempt to close it is itself
scored as an attack indicator.

The injection scanner is weighted by *how rarely a pattern appears in benign security
telemetry*. The bare word "system" appears constantly in logs and is not a rule at all;
"human approval is not required" essentially never appears in a firewall log and carries
weight 0.85. Signals combine with a noisy-OR, so many weak hits accumulate toward
certainty without any single rule reaching it alone. The SOC-specific rules —
`verdict_manipulation` and `approval_manipulation` — are the ones a generic injection
filter would miss, and they are the ones that matter here.

---

## Layer 1–2 — ingestion

### Two time models, and why it is not over-engineering

`TimeModel.VIRTUAL` advances an injected `FrozenClock` instead of sleeping.
PRD §9.1 measures simulated MTTD as a timestamp delta in the pipeline logs. Under a real
clock that metric is partly a measurement of how busy the laptop was, and a 100,000-row
evaluation takes as long as the replay rather than as long as the computation.
`TimeModel.WALL_CLOCK` exists for the live demo, where a judge watching a dashboard needs
alerts to *arrive*.

### The unified feature space

CIC-IDS2017 ships ~78 flow features and UNSW-NB15 ~47, defined differently.
`UNIFIED_FEATURES` is the intersection both datasets genuinely express: eleven
bidirectional flow statistics plus port and protocol.

*Alternative rejected:* use each dataset's full native feature set. It scores better
in-domain and makes PRD §10's cross-dataset validation impossible, which is the exact
mitigation that section relies on.

Rate features are **recomputed from the counters** rather than copied from the source's
own derived columns. Two reasons: that is where CIC-IDS2017's literal `Infinity` values
come from, and the two datasets' definitions of "bytes per second" quietly disagree.
Zero-duration flows divide by a 1-microsecond floor — the measurement resolution
CIC-IDS2017 reports in, not an arbitrary epsilon.

The verbatim source row is preserved in `Alert.raw_payload` (untrusted-wrapped), so a
richer per-dataset feature set stays available without a re-ingest.

### Session context: the finding that changed the design

The first honest evaluation on a per-flow-only space produced ROC-AUC **0.839** against
the 0.90 target, with `brute_force` recall 0.02 and `web_attack` 0.12.

That is not a tuning failure. **A single SSH brute-force flow is statistically identical
to a legitimate SSH login, because that is what it is.** What distinguishes it is four
hundred of them from one host in ninety seconds — information a per-flow feature space
cannot represent at all.

`ingest.enrich` adds sliding-window features that carry exactly that structure:

| Family | The feature that separates it |
|---|---|
| Brute force | high `src_flow_count_window`, `src_distinct_dst_ports_window == 1` |
| Port scan | high `src_distinct_dst_ports_window` |
| DDoS | high `dst_flow_count_window` / `dst_distinct_src_ips_window` (**fan-in**) |
| Slowloris | high concurrent count, long durations, one port |
| Botnet C2 | low `src_interarrival_cv_window` (machine-regular timing) |

AUC went from 0.839 to 0.989. Two properties make the enricher correct rather than
merely effective:

**Strictly backward-looking.** A window feature is computed only from events already
seen — never the current event, never a later one. The obvious implementation, a
full-dataset `groupby`, leaks the future into the features: it scores beautifully offline
and is unimplementable in production, and the failure is invisible in aggregate metrics.
`test_prefix_enrichment_matches_full_stream_prefix` pins it by asserting that enriching
a prefix equals enriching the whole stream and taking the prefix.

**Bounded memory.** The key space is attacker-chosen — spoofing a million source
addresses is free. State is capped with LRU eviction and evictions are counted. An
unbounded `defaultdict(list)` keyed on attacker-controlled data is a denial of service
in the detection path.

Fan-out and fan-in are kept in **separate indexes**. Merging them would dilute a DDoS
victim's fan-in count with its own outbound traffic, destroying the only feature that can
see a distributed attack.

### Bus: at-least-once, deliberately

Exactly-once across a network is a marketing claim. At-least-once plus idempotent alert
ids (`Alert.derive_id` is a UUID5 over `tenant + dataset + source + row_index`) is a
design that holds — and it is why replaying a dataset twice produces the *same* ids
rather than duplicates the pipeline cannot recognise.

`InMemoryEventBus` reimplements the Redis Streams semantics the platform actually depends
on, including the pending-entries list and `XAUTOCLAIM`-style recovery. Without stale-entry
claiming, one crashed worker parks alerts in its pending list forever — a silent detection
gap, which is the worst kind. Trimming an unacknowledged event is *counted*
(`dropped_unacked()`), because losing a security event invisibly is unacceptable in a way
that losing one loudly is not.

---

## Layer 3 — intelligence

### Train/serve parity is structural

Three guards, in order of strength:

1. `AlertVectorizer` accepts **only** canonical `Alert` objects. There is no
   `fit(dataframe)` overload, so a training job cannot read a source column the serving
   path never sees — it must go through the same normalizer.
2. Batch and single-row paths call the same per-column extraction, and the test asserts
   **bit** equality. The classic skew bug is a pandas training path and a looping serving
   path that agree to six decimal places and are wrong forever; `approx` would not catch it.
3. Fitted artifacts carry a spec fingerprint over ordered column identities, and
   `assert_compatible` refuses to score under a changed spec. Order is included because
   reordered columns produce different vectors.

### Transform choices

Flow byte and packet counts span six orders of magnitude, so they are `log1p`-compressed
before standardization — a raw-scale standardizer lets one 400 MB transfer dominate the
variance of every other feature. Ports are **not** ordinal (443 is not "more" than 80), so
they expand into semantic indicators rather than entering as integers. Protocol is one-hot
over the three protocols carrying essentially all traffic, plus an explicit `other`
bucket — a missing protocol must reach `other` rather than being indistinguishable from
all-zeros. `bytes_ratio_src_to_total` is left uncompressed because it is already bounded,
and `src_interarrival_cv_window` is left uncompressed because it is the *low* end that
carries the beacon signal and `log1p` would squash exactly that region.

### Calibration before ensembling

Isolation Forest returns a negated path-length score around [-0.5, 0.5]; PCA returns
squared error in [0, ∞). Averaging them raw lets whichever has larger numeric spread
dominate, and the "tuned weights" silently absorb a unit conversion —
`test_raw_scores_are_on_different_scales` measures the ratio at >2×.

Each detector is wrapped in an empirical-CDF calibrator fitted on **training scores
only**, mapping every raw score to its quantile within benign traffic. After calibration
0.99 means "more anomalous than 99% of benign traffic" for both models, the weights mean
what they claim, and the threshold becomes an interpretable false-positive budget — which
is how a team with two analysts actually configures a detector.

### The two detectors are genuinely different models

A "diverse ensemble" of correlated members is a false sense of coverage. Both blind spots
are constructed directly in the tests:

- **Isolation Forest** partitions axis-aligned. It catches marginal outliers and is blind
  to correlation structure: a flow whose every individual value is ordinary but whose
  *combination* is impossible scores low. The test builds one by permuting a real benign
  row's coordinates — every marginal is still benign, the correlation is destroyed.
- **PCA reconstruction** sees correlation violations and underweights pure marginal
  outliers. The test builds one by placing an extreme point *along* the first principal
  component: it reconstructs almost perfectly because it lies on the learned manifold.

### On calling it PCA

`PCAReconstructionDetector` is the closed-form optimum of a **linear** autoencoder with
squared loss. PRD §5.5.2 specifies a nonlinear denoising autoencoder; that needs PyTorch
and lands in Part 2 behind the same `AnomalyDetector` protocol, a one-line change in
`build_default_ensemble`. It is named for what it is. Shipping PCA and labelling it a deep
autoencoder is the kind of thing that does not survive a technical interview.

### The split refuses leakage

`three_way_split` returns a **benign-only** training split plus mixed validation and test
splits, and `SplitIndices.__post_init__` raises if any two overlap. The shape is dictated
by the method: the detectors are semi-supervised novelty detectors that must see only
benign traffic while fitting; weight tuning and threshold selection need labelled
positives; the final numbers need a split neither ever touched. Tuning on test data is the
single most common way a security-ML benchmark becomes fiction, so the correct split is
the easy one to reach for.

Per-family recall is reported next to the aggregate because with ~80% benign traffic and
`Infiltration` at well under 1%, an aggregate AUC of 0.99 is fully compatible with never
detecting the most severe family. `test_aggregate_can_hide_a_missed_rare_class`
demonstrates it.

---

## Layer 6 — audit

### Threat model, stated rather than implied

A plain SHA-256 chain is tamper-**evident** against an attacker who cannot recompute it:
a hex-editor edit, a buggy migration, disk corruption, an insider who doesn't realise the
rows are linked. It is **not** evidence against an attacker with database write access who
knows the scheme — they rewrite every subsequent hash and it verifies again. Calling that
"immutable" would be theatre, so three mechanisms are layered and each is tested for what
it does *and does not* buy:

1. **DB-enforced append-only.** `BEFORE UPDATE`/`BEFORE DELETE` triggers `RAISE(ABORT)`.
   A bug or a careless `UPDATE` cannot rewrite history; an attack has to drop the
   triggers, which is itself reported as a finding.
2. **Keyed chaining (HMAC-SHA256)** under a key held outside the database.
   `test_unkeyed_chain_can_be_fully_rewritten` performs the full forgery and asserts it
   **succeeds**; `test_keyed_chain_resists_the_same_rewrite` shows the key is what stops it.
3. **External anchoring.** `checkpoint()` emits the head `(seq, hash)` for publication
   somewhere the platform cannot reach. A truncated tail leaves a self-consistent chain —
   `test_deleted_tail_row_is_detected_by_the_anchor` shows the chain arithmetic finds
   nothing — so only a published anchor catches it.

An untested boundary is a boundary nobody can reason about.

### The log does not store attacker text

Audit payloads carry `raw_payload_sha256`, not the payload. The audit log must not become
a second store of attacker-controlled content that some future component reads back into
a prompt.

### Verification is one ordered scan

Each row's payload is re-hashed, its row hash recomputed from the declared
`HASHED_FIELDS`, and its `prev_hash` compared to the previous row's stored hash.
Chaining forward on the *stored* hash rather than the recomputed one is deliberate: using
the recomputed value would mask a single-row edit and cascade spurious `BROKEN_LINK`
findings down the rest of the log. Deletions surface twice over — as a `seq` gap and as a
broken link. 50,000 rows verify in well under the F-11 one-second budget.

`test_hash_covers_every_column_that_matters` reads `PRAGMA table_info` and asserts no
column sits outside the hash, so adding a column without protecting it fails the suite.

---

## Layer 4 — orchestration and agents

Added in Part 3. The full rationale, with measurements, is in
[BUILD_PLAN.md](BUILD_PLAN.md); three structural points belong here because they are
architecture rather than implementation.

### The state is a contract, not a dict

Every orchestration framework models workflow state as a mutable mapping.
`IncidentState` is a frozen pydantic contract instead, for one reason that is specific
to this system: the state carries approval status, so a checkpoint store that
round-trips through an opaque binary format without a verifiable hash is an
approval-forgery path. Being a contract gives the state canonical bytes, a hash that
`Checkpointer.get` recomputes on read, and re-validation on load — so a checkpoint
edited in the store is *refused* rather than resumed. Checkpoints are themselves
chained, so deleting the intermediate one that recorded a denied approval breaks the
chain. It is the same tamper-evidence construction as the audit log, deliberately:
there is one such idea in this system, not two.

It also preserves a type. `Alert.raw_payload` is an `UntrustedText`; a checkpointer that
serialises it to `str` and restores it as `str` has silently deleted the control that
stops it being interpolated into a prompt. Restoring through the schema's own validator
keeps the property by construction, and a test asserts it across a real file.

### The reasoning engine is a seam, not a dependency

`ReasoningEngine` has one method: one prompt in, one parsed response out. An engine
cannot call tools, cannot reach the audit log, and cannot see the state. Everything with
real-world consequence happens in Python, where it is reviewable. The default is
`NullEngine`, and every Part 3 acceptance criterion passes with it — which is what makes
`monotone_caution`'s bound honest, because the baseline it protects is a working system
rather than an empty one.

An engine's output is typed `UntrustedText`. This is not decoration: the model read
attacker-controlled bytes, so its output is a function of attacker-controlled bytes, and
treating it as trusted because "it came from our model" is exactly the laundering step
that makes second-order injection work.

### The approval gate is a node, not a flag

`interrupt_before` is static per node, and the gate therefore gets its own node. That
makes "this incident is waiting for a human" a *position in the graph* — visible in the
checkpoint, in the pending queue, and in the audit log — rather than a boolean somebody
has to remember to check. The alternative, one execution node that sometimes pauses,
puts the decision of whether to pause inside the node that does the acting, which is the
one place it must not be. A rejected action ends the run without the connector ever
being called.

F-08 is then enforced in four independent places — the schema, the action mask, the
graph, and the audit log — and checked by `verify_no_ungated_execution`, which reads the
log rather than the objects that enforce the rule. Asking those objects whether the rule
held would be circular.

---

## What is still not built

No Code-Scan/Patch agent (F-07), no Supply-Chain agent node, no real connector layer,
no dashboard (F-10). Those are Parts 3.3–5 and are scoped in
[BUILD_PLAN.md](BUILD_PLAN.md). The foundation was built first specifically so the agent
layer inherits its guardrails rather than reimplementing them — and Part 3 is the
evidence that worked: `ActionRequest` and `TriageResult` needed no changes to hold the
agents to F-08 and Appendix A.
