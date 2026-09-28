# Sentinel Mesh — Build Ledger

The PRD specifies a 5-agent autonomous SOC with a 6-layer architecture. This file
tracks what is actually built and verified, and what the next session picks up.

**Rule for this ledger:** nothing is marked done unless a test asserts it. A feature
with code but no test is listed as *partial*.

---

## Part 1 — Foundation ✅ COMPLETE

Layers 1, 2, 3 (detection) and 6 (audit) of PRD Figure 2, plus the Appendix B
contracts every later part depends on.

### Delivered

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| Appendix B | Canonical contracts: `Alert`, `TriageResult`, `InvestigationReport`, `ActionRequest`, `Evidence` | [schemas.py](../src/sentinel/core/schemas.py) | `tests/unit/test_schemas.py` |
| §5.7 | Canonical byte serialization (basis of all hashing) | [canonical.py](../src/sentinel/core/canonical.py) | `tests/unit/test_canonical.py` |
| §5.7, App. A | Untrusted-input discipline + prompt-injection scanner | [untrusted.py](../src/sentinel/core/untrusted.py) | `tests/unit/test_untrusted.py` |
| **F-11** | Hash-chained append-only audit log | [audit/log.py](../src/sentinel/audit/log.py) | `tests/unit/test_audit_log.py` |
| **F-01** | Event bus (Redis Streams semantics + in-memory impl) | [ingest/bus.py](../src/sentinel/ingest/bus.py) | `tests/unit/test_bus.py` |
| **F-01** | CIC-IDS2017 + UNSW-NB15 normalizers | [ingest/normalizer.py](../src/sentinel/ingest/normalizer.py) | `tests/unit/test_normalizer.py` |
| **F-01** | Replay service (wall-clock + virtual time) | [ingest/replay.py](../src/sentinel/ingest/replay.py) | `tests/integration/test_pipeline.py` |
| §5.5.1 | Streaming session-context enrichment | [ingest/enrich.py](../src/sentinel/ingest/enrich.py) | `tests/unit/test_enrich.py` |
| §7.3 | Feature store with train/serve parity guarantees | [ml/featurestore.py](../src/sentinel/ml/featurestore.py) | `tests/unit/test_featurestore.py` |
| **F-03** | Anomaly ensemble (Isolation Forest + PCA reconstruction) | [ml/anomaly.py](../src/sentinel/ml/anomaly.py) | `tests/unit/test_anomaly.py` |
| §9.1, §9.3 | Metrics + leakage-refusing three-way split | [ml/metrics.py](../src/sentinel/ml/metrics.py) | `tests/unit/test_metrics.py` |
| **F-12** | Single offline evaluation pipeline | [scripts/evaluate.py](../scripts/evaluate.py) | `tests/integration/test_pipeline.py` |
| §7.1 | Campaign-structured synthetic dataset generator | [ml/datasets/synthetic.py](../src/sentinel/ml/datasets/synthetic.py) | `tests/integration/test_pipeline.py` |

### Measured results

From `python scripts/evaluate.py --n 20000 --cross-dataset` (n=20,000 synthetic
CIC-shaped flows, 20% attack, seed 20260928):

```
ROC-AUC (held-out test)      0.9894    F-03 target >= 0.90   PASS
PR-AUC                       0.9823
Precision / Recall @ 10% FPR 0.833 / 0.978
Alert volume reduction       60.8%     PRD 9.1 target >= 60% PASS
Cross-dataset AUC (UNSW)     0.9900    (drop -0.0006)
Ingestion throughput         >1,000,000 alerts/min   F-01 target >= 1,000  PASS
Schema failures              0                       F-01 target 0         PASS
Audit verification, 50k rows < 1s                     F-11 target < 1s      PASS
```

Per-family recall at the 10%-FPR threshold — reported because the aggregate hides
the hard classes:

| Family | Recall | Note |
|---|---|---|
| recon | 1.000 | port sweep is unmistakable in the fan-out features |
| infiltration | 1.000 | large upload, visible per-flow |
| brute_force | 0.996 | only detectable via the window features |
| dos | 0.991 | |
| ddos | 0.985 | caught by destination fan-in |
| web_attack | 0.966 | |
| **botnet** | **0.608** | **honest weak spot** — low-rate C2 beacons are the hardest class; the regularity indicator helps but does not solve it. Candidate for the Phase 2 work. |

### Notable findings during the build

These were found by measurement, not anticipated, and each changed the design:

1. **A per-flow-only feature space cannot reach the F-03 target.** The first honest
   evaluation came in at ROC-AUC **0.839** with `brute_force` recall 0.02 and
   `web_attack` 0.12. Root cause: a single SSH brute-force flow is statistically
   identical to a legitimate SSH login, because that is what it is. Fix: the
   streaming session-context enricher ([enrich.py](../src/sentinel/ingest/enrich.py)),
   which lifted AUC to 0.989. A regression test
   (`test_context_features_are_what_make_brute_force_detectable`) blinds the context
   columns and asserts detection degrades, so the finding cannot silently rot.

2. **Pydantic's `model_copy(update=...)` skips validators.** Every state-transition
   helper on `ActionRequest` was bypassing the F-08 approval invariants — an action
   could reach `EXECUTED` with no approver. Fixed with `Contract.updated()`, which
   round-trips through `model_validate`.

3. **Campaign structure silently broke the class balance.** Sampling a family per
   *episode* and then emitting a 200–1,200 flow burst turned a nominal 80% benign
   rate into a stream that was 99.5% attack flows. Solving the episode probabilities
   fixed the aggregate but not the variance: a 20,000-flow stream held only 7–15
   campaigns total, and whole attack families vanished at random. Replaced with a
   planned, stratified schedule; the mix is now exact and every family is present at
   every scale.

4. **Chain verification breached its own F-11 budget under load.** A 50,000-row
   verify took 1.77s on a contended machine against a 1s requirement; profiling put
   83% of it in the generic recursive canonicalizer. `row_material` emits the
   fixed row schema's canonical JSON directly (2.9s to 0.41s, a 7x speedup), with
   `TestFastPathEquivalence` asserting byte equality against the generic path over
   3,000 random and every single-field placement of 18 adversarial strings — a fast
   path that is only *usually* equivalent would be a silent forgery generator.

5. **The microsecond/second duration mismatch.** CIC-IDS2017 reports flow duration in
   microseconds, UNSW-NB15 in seconds. Nothing crashes — the cross-dataset validation
   just silently becomes meaningless. Pinned by
   `test_duration_units_agree_across_datasets`.

### Honest caveats

- **The cross-dataset number does not yet prove generalisation.** Both corpora come
  from one campaign engine, so it validates the *unit and schema harmonisation*
  (which is the bug class it exists to catch) rather than a genuine distribution
  shift. Real generalisation needs the actual downloads — see [DATA.md](DATA.md).
- **The "autoencoder" in the ensemble is a linear one (PCA reconstruction).** That is
  exactly what it is called in the code and docs. The nonlinear denoising
  autoencoder from PRD §5.5.2 needs PyTorch and lands in Part 2, behind the same
  `AnomalyDetector` protocol — a one-line swap in `build_default_ensemble`.
- **The unkeyed audit chain is tamper-*evident*, not tamper-proof.** An attacker with
  file write access who knows the scheme can rewrite it. `HashChainedAuditLog`
  therefore supports HMAC keying and external anchoring, and
  `test_unkeyed_chain_can_be_fully_rewritten` proves the limit rather than papering
  over it.
- **No LLM, no agents, no dashboard yet.** That is Parts 3–5.

---

## Part 2 — Intelligence core: 2.1 and 2.3 ✅ COMPLETE

`943 tests passing` (Part 1 ended at 639). Ruff clean. `mypy` is configured in
`pyproject.toml` but **is not installed in this environment, so it has not been
run** — that is a gap, not a pass.

### 2.1 Deep anomaly detection (PRD §5.5.2) ✅

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| §5.5.2 | Gradient-checked NumPy NN engine (Dense/ReLU/Tanh/Sigmoid, AdamW, early stopping) | [ml/nn.py](../src/sentinel/ml/nn.py) | `tests/unit/test_nn.py` (79) |
| §5.5.2 | Nonlinear denoising autoencoder behind the `AnomalyDetector` protocol | [ml/deep.py](../src/sentinel/ml/deep.py) | `tests/unit/test_deep.py` (46) |
| §9.1 | Base-rate-corrected alert-reduction reporting | [ml/metrics.py](../src/sentinel/ml/metrics.py) | `tests/unit/test_metrics.py` |

**Deviation from the PRD, deliberately.** The PRD names PyTorch. This is a
hand-written NumPy engine instead, because the models are small (`32-64-10-64-32`,
5,546 parameters; GraphSAGE over 500 nodes), Part 1's whole-repo reproducibility in
one command is worth protecting, and `torch-geometric` on Windows needs compiled
extensions pinned to a specific torch build. The trade that makes it defensible is
`gradient_check`: every analytic gradient is compared against a central finite
difference to ~1e-9, so the chain rule itself is under test rather than delegated.
`AnomalyDetector` is a Protocol, so a torch backend remains a drop-in.

**Measured** (`python scripts/evaluate.py --n 20000 --cross-dataset`):

| Metric | Part 1 (IF + PCA) | Part 2.1 (IF + autoencoder) |
|---|---|---|
| ROC-AUC (held-out test) | 0.9894 | **0.9971** |
| PR-AUC | 0.9823 | **0.9950** |
| Recall @ 10% FPR | 0.9783 | **0.9975** |
| **botnet recall** | **0.6081** | **1.0000** |
| web_attack recall | 0.9664 | 0.9832 |
| Cross-dataset (UNSW) AUC | 0.9900 | 0.9969 |
| Autoencoder alone, AUC | — | 0.9988 |

Every family is now ≥ 0.983 and **no family regressed**. The documented Part 1 weak
spot is closed: `botnet` 0.608 → 1.000.

### 2.3 Supply-chain risk graph + GNN (PRD F-06, §5.5.3) ✅

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| §5.5.3 | Nodes, directed edges, adjacency oriented by risk flow | [graph/schema.py](../src/sentinel/graph/schema.py) | `tests/unit/test_graph_schema.py` (42) |
| §7.1 | 500-node graph + graded ground truth | [graph/synthetic.py](../src/sentinel/graph/synthetic.py) | `tests/unit/test_graph_synthetic.py` (40) |
| **F-06** | 2-layer GraphSAGE risk propagation | [graph/gnn.py](../src/sentinel/graph/gnn.py) | `tests/unit/test_graph_gnn.py` (50) |
| **F-06** | Path attribution + model ablation | [graph/explain.py](../src/sentinel/graph/explain.py) | `tests/unit/test_graph_explain.py` (31) |

**Measured** (`python scripts/evaluate.py --graph`, seed 20260928):

```
[supply-chain graph] 500 nodes, 1080 edges
  54 high-risk nodes: 25 intrinsic, 29 inherited
  top-10 precision   0.9000   PASS (F-06 needs >= 0.80)
  rank corr vs truth 0.6682
  features-only      0.5000   <- the graph's contribution is the gap
  inherited-risk only: GNN 0.5000 vs features-only 0.0000
```

F-06 acceptance is asserted as the **mean over 10 fixed seeds: 0.8000**, not on one
seed. Top-10 precision on a 150-node test split moves 0.1 per node and the per-seed
spread is 0.60–0.90, so a single-seed gate would be a coin flip. Hyperparameters were
chosen on seeds 100–119 and evaluated once on the disjoint set — a 27-config grid
search tuned on 12 seeds scored 0.825 there and **0.740** on held-out seeds, which is
why that separation is enforced rather than assumed.

### Findings during Part 2 — each changed the design

1. **The textbook autoencoder "identity-map collapse" could not be reproduced.**
   The standard warning is that `bottleneck >= n_features` learns the identity and
   reconstruction error vanishes for anomalies too. Swept the bottleneck 2→31 on the
   real 32-feature space: ROC-AUC stayed flat at **0.9978–0.9989**, and a near-full-rank
   bottleneck of 31 scored as well as anything. A control with a *linear* full-rank
   autoencoder separated correlation-violating inputs by **1141x** versus a rank-2
   bottleneck's **5x** — the unconstrained model was *better*. Gradient descent from
   small initialisation converges to the projector onto the benign manifold, not the
   identity, because nothing off-manifold appears in the training objective. The real
   failure is **under**-capacity. The guard stayed (a bottleneck that does not
   bottleneck is an incoherent architecture) but its docstring now says what was
   measured instead of repeating folklore.

2. **An AUC-optimal tuner destroys the ensemble.** Against the autoencoder,
   `tune_weights` assigned the Isolation Forest weight **0.0** — correct for the
   metric, and it re-creates the single point of failure §5.5.2 asks for an ensemble
   to remove. Added `min_weight`; measured cost of a 0.10 floor is 0.0017 AUC
   (0.9988 → 0.9971) with botnet recall still 1.000, while 0.20 starts eating it.

3. **PRD §9.1's reduction target fights recall.** Part 1 scored 60.8% reduction at
   recall 0.978; Part 2 scores **59.9% at recall 0.9975**. The better detector reports
   the worse number, because on a 33%-attack split every extra attack caught is one
   more alert reaching a human. Rather than detune, `reduction_at_base_rate` projects
   the measured FPR and recall onto a realistic feed: **88.9% at a 1% attack rate**,
   where benign suppression dominates. Both numbers are now reported.

4. **`last < first` is not a convergence test.** With `batch_size >= n_rows` each
   epoch sums the same values in a different order, and float64 addition is not
   associative, so the loss jitters ~1e-16 *in a random direction*. A model at
   `lr=1e-20` — weights provably frozen — reported a "decreasing" loss and passed.
   Fixed with `MIN_RELATIVE_IMPROVEMENT`.

5. **Adam is very hard to diverge.** `lr=1e6` on data scaled by 1e8 trained to a
   terrible model without a single non-finite value, because Adam normalises its step
   to ≈`lr` regardless of gradient magnitude. Divergence here signals *loss* overflow,
   not a large gradient — both behaviours are now pinned.

6. **A transposed aggregation is undetectable in a one-layer GNN.** The bug corrupts
   the gradient a layer *returns to its input*; for the first layer that value is
   discarded, so `gradient_check` (parameter gradients only) measured **7.3e-10** —
   indistinguishable from correct. In a two-layer stack the same bug measures ~1e-1.
   The blind spot is documented in `gradient_check` and both behaviours are tested.

7. **Three ground-truth designs were wrong before one worked.** (a) Binary
   reachability with 4 organisations put **32/32 vendors and 4/4 orgs** in the
   high-risk set — nothing to rank. (b) Fixing the depth distribution left **41
   isolated packages**, which do not exist in an SBOM. (c) Defining high-risk as the
   *union* of "intrinsically risky" and "exposure ≥ threshold" was incoherent:
   intrinsic packages were labelled positive while scoring **0.0** exposure, so a
   regression ranked the genuinely compromised packages **last** and still reported a
   healthy 0.61 rank correlation while top-10 precision fell to 0.60. Now one
   continuous quantity with one threshold, and `intrinsic ⊂ high_risk` by construction.

8. **Regression beats classification, but only once the target is coherent.** Before
   fix 7(c): classification 0.800 vs regression 0.600. After: regression **0.825** vs
   classification 0.800, with rank correlation 0.64 vs 0.39. The label is a threshold
   on a continuous quantity, so training on the boolean throws away the ordering that
   F-06 then measures — and §3.4 sells a *"continuously scored"* graph, not a flag.

9. **`sum` aggregation lost, contradicting a good argument.** Exposure genuinely
   accumulates and a mean divides the count away, so `sum` should match the additive
   ground truth. Measured: `sum` wins for classification (0.800 vs 0.788) and **loses**
   for regression (0.750 vs 0.825), which is the objective that ships. Degree varies
   1–12 in a preferential-attachment graph, so an unnormalised sum feeds the next layer
   activations scaled by degree rather than risk. `mean` is the default on evidence;
   `sum` is kept and tested because the argument is not wrong in principle.

10. **PRD §5.5.3 contradicts itself, and the contradiction is arithmetic.** A
    *k*-layer message-passing network sees exactly *k* hops, so "2-layer GraphSAGE"
    cannot deliver "exposure through a **fourth-order** dependency" — the information
    never enters the final layer's input. Measured by perturbing a node 3 hops away:
    a 2-layer model's score moves by **exactly 0.0**; a 4-layer model's does move.
    Measured across depths, top-10 precision *falls* with depth (0.83 / 0.78 / 0.70 /
    0.63 at 1/2/3/4 layers) while rank correlation *rises* (0.573 / 0.641 / 0.618 /
    0.685) — deeper receptive fields capture more propagation and simultaneously
    overfit a 275-node training split. The PRD's 2 layers is a defensible middle of
    that trade, `n_layers` is configurable, and the structural explainer walks the
    graph to arbitrary depth so the fourth-order *explanation* exists regardless.

11. **Organisations saturate, and no single threshold fixes it.** They aggregate 4–12
    vendors each carrying 9–13 dependencies, so they sit 2–4 hops from most of the
    graph. Raising the threshold to spread them out pushes a fourth-order contribution
    (`0.7**4 = 0.24`) below relevance and deletes the PRD's headline scenario from the
    benchmark; lowering it flags every organisation. This is §2.4's thesis as a
    measurement — essentially every mid-market company *has* third-party exposure — so
    the binary label discriminates on vendors and packages while organisations are
    evaluated by ranking, and F-06 precision is reported **per node kind** so it
    cannot hide in an aggregate.

### Honest caveats for Part 2

- **`mypy --strict` has not been run** (not installed here). Annotations were written
  to satisfy it but that is unverified.
- **F-06 sits right on its bar.** 0.8000 mean over 10 seeds against a 0.80 target,
  per-seed 0.60–0.90. It is met, and it is not comfortably met. Rank correlation
  (0.65) is the stabler number and the one worth quoting in a demo.
- **The deep-exposure-only case cannot be measured at 500 nodes.** Nodes whose *only*
  route to risk is ≥3 hops number 1–4 per graph, so no seed produced enough for a
  stable statistic. Testing the PRD's fourth-order claim end to end needs a larger
  graph or a deliberately planted scenario.
- **The graph is synthetic and its ground truth is a model I wrote.** F-06 measures
  whether the GNN recovers a propagation rule from structure — which is a real and
  falsifiable claim, and is why the features-only baseline is reported beside it — but
  it is not evidence about real SBOMs. Phase 2's CycloneDX/OSV ingestion is what would
  make it so.
- **Transductive, not inductive.** The whole graph's features and edges are visible
  during training; only test labels are hidden. Correct for the deployment (an MSSP's
  graph is fully known) but not comparable to an inductive benchmark.

---

## Part 2 — REMAINING for the next session

Suggested order: **2.4 → 2.5 → 2.2**. The KB unblocks Part 3's Investigation Agent,
the bandit needs the graph and detector outputs as context, and augmentation is the
most optional of the three.

### 2.4 RAG knowledge base (PRD §5.5.6)
- `pip install -e ".[agents]"`
- `kb/index.py`: FAISS index over MITRE ATT&CK technique descriptions + an NVD/CVE
  snapshot; chunk, embed, persist.
- `kb/retrieve.py`: returns `Evidence` objects (already defined in `schemas.py`), so
  the Investigation Agent's citations are typed from day one.
- Tests: known-technique retrieval (a lateral-movement query must return T1021), and
  a test that every returned `Evidence.ref` resolves to real indexed content — the
  `InvestigationReport` validator already rejects citations that point at nothing.

### 2.5 Contextual bandit response policy (PRD F-09, §5.5.4)
- `rl/bandit.py`: Thompson sampling over `{auto_contain, escalate, monitor, dismiss}`,
  context = alert/investigation embedding + asset criticality.
- `rl/reward.py`: the shaped reward from §5.5.4 — a reversed auto-contain is penalised
  more heavily than an over-cautious escalation.
- Acceptance test: cumulative regret trends down over 200 simulated episodes against
  an oracle policy, and the policy **never** proposes an action outside the current
  trust tier (`RiskTier` is already defined and ordered).

### Notes for whoever picks up 2.4 / 2.5 / 2.2

Things Part 2 established that these three should reuse rather than rediscover:

- **`ml/nn.py` is the shared engine.** 2.2's diffusion denoiser is another MLP; build
  it from `Dense`/`ReLU`/`Adam`/`train` and gradient-check it. Do **not** add torch.
- **`gradient_check` is parameter-only.** Check the composition at the depth you ship
  (finding 6).
- **Hermetic tests, no downloads.** 2.4 needs an embedder; `sentence-transformers`
  pulls a model over the network, which would make the suite non-reproducible and
  CI-hostile. Prefer a deterministic in-repo embedder (hashed TF-IDF over ATT&CK
  technique text) with sentence-transformers as an optional backend behind the same
  interface, exactly as `AnomalyDetector` allows a torch swap. Same for FAISS: exact
  cosine search over a few thousand chunks is a matmul.
- **`Evidence` and `EvidenceKind.KB_CHUNK` already exist** and
  `InvestigationReport._every_claim_is_grounded` already rejects citations that point
  at nothing — so `kb/retrieve.py` should return typed `Evidence`, as
  `graph/explain.py:NodeExplanation.as_evidence` now does.
- **Tune on seeds you do not report on** (finding: a 27-config grid scored 0.825 on
  its tuning seeds and 0.740 held out). 2.5's regret curve will be noisier than it
  looks; fix the seed set in advance.
- **For 2.5, `RiskTier` is ordered and `ActionRequest.propose` derives
  `requires_human_approval`** — the bandit cannot construct an ungated destructive
  action even if its policy asks for one. Test that the *action mask* is correct, not
  that the schema holds; the schema is already tested.
- **For 2.2, augment train only.** The acceptance test must measure rare-family recall
  on a held-out split containing **no** synthetic rows, and must assert the majority
  distribution is unchanged. Note the detector's per-family recall is now ≥ 0.983
  everywhere, so there is much less headroom than when 2.2 was first planned —
  consider whether it still earns its place, or whether its real value is now the
  adversarial-robustness check from §5.5.5 (boundary-adjacent perturbed samples
  probing the Triage Agent's confidence calibration).

---

## Part 3 — LangGraph orchestrator + 5 agents (F-02, F-04, F-05, F-07, F-08)

`agents/` — one node per agent on a checkpointed LangGraph state machine, with the
Human Approval Gate as an interrupt node. The guardrails these agents must respect
are **already enforced by the schemas**, which is the point of having built Part 1
first: the Triage Agent cannot emit a low-confidence dismissal, and the Containment
Agent cannot construct an ungated destructive action, regardless of what the model
returns. Agent tests should therefore focus on *tool-use correctness* and
*prompt-injection resistance* (feed the hostile payloads from `test_untrusted.py`
through a live Triage call and assert escalation), not on re-testing the invariants.

## Part 4 — Connector layer (Layer 5)
Mocked EDR / firewall / Git PR / Slack connectors behind a least-privilege interface;
Semgrep integration for F-07.

## Part 5 — Analyst Copilot dashboard (F-10, Layer 6)
Next.js + d3-force. Consumes the bus and the audit log; the three demo scenarios must
be completable end-to-end from the UI alone.

---

## Running what exists

```bash
pip install -e ".[dev]"
python -m pytest -q                              # full suite, 943 tests
python -m pytest -q -m "not slow"                # skip the multi-seed benchmarks
python -m ruff check src tests scripts

# Detection (F-03) + supply chain (F-06). Exits non-zero if either gate fails.
python scripts/evaluate.py --n 20000 --cross-dataset --graph

python scripts/evaluate.py --n 20000 --shallow   # the Part 1 linear baseline, for comparison
```

No datasets, no API keys, no Redis, and no torch. Every number above is produced by
that one script.

### Architecture flags worth knowing

| Flag | Default | Why |
|---|---|---|
| `SupplyChainGNN(objective=...)` | `regression` | the label is a threshold on a continuous quantity; classification discards the ordering F-06 measures (finding 8) |
| `SupplyChainGNN(aggregation=...)` | `mean` | `sum` matches the additive semantics in theory and loses in measurement (finding 9) |
| `SupplyChainGNN(n_layers=...)` | `2` | the PRD's figure; see finding 10 for what it costs and buys |
| `WeightedEnsemble.tune_weights(min_weight=)` | `0.0` | explicit at the call site; `evaluate.py` passes `0.10` (finding 2) |
| `DenoisingAutoencoderDetector(bottleneck=)` | `n_features // 3` | measured optimum, shallow (finding 1) |
