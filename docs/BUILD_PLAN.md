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

## Part 2 — Intelligence core: 2.4, 2.5, 2.2 ✅ COMPLETE

### 2.4 RAG knowledge base (PRD F-05, §5.5.6) ✅

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| §5.5.6 | 198-document ATT&CK/CVE/advisory/playbook corpus, cross-linked | [kb/corpus.py](../src/sentinel/kb/corpus.py) | `tests/unit/test_kb_corpus.py` |
| §5.5.6 | 420 stable, sentence-aligned, separately citable chunks | [kb/chunk.py](../src/sentinel/kb/chunk.py) | `tests/unit/test_kb_chunk.py` |
| §5.5.6 | BM25 + TF-IDF + LSA with RRF fusion, MMR, link expansion | [kb/index.py](../src/sentinel/kb/index.py) | `tests/unit/test_kb_index.py` |
| **F-05** | `KnowledgeBase`: search, resolve, related, persist | [kb/retrieve.py](../src/sentinel/kb/retrieve.py) | `tests/unit/test_kb_retrieve.py` |
| §9.3 | 95 labelled queries, 34 tune / 61 held out, with gates | [kb/eval.py](../src/sentinel/kb/eval.py) | `tests/unit/test_kb_eval.py` |

**Measured** (held-out split, k=5): recall **0.768**, MRR **0.930**, nDCG **0.780**,
recall after link-following **0.896**. Tuning split scored 0.922 / 1.000 / 0.896 and
that gap is reported rather than smoothed over.

No FAISS, no sentence-transformers, no network. `Retriever` is a real seam — the suite
drives the whole knowledge base through a stub.

### 2.5 Contextual bandit response policy (PRD F-09, §5.5.4) ✅

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| §5.5.4 | The four decisions + `ActionMask` (structural, not a reward penalty) | [rl/actions.py](../src/sentinel/rl/actions.py) | `tests/unit/test_rl_actions.py` |
| §5.5.4 | The shaped reward, ordering enforced in code | [rl/reward.py](../src/sentinel/rl/reward.py) | `tests/unit/test_rl_reward.py` |
| **F-09** | Linear Thompson sampling, Sherman-Morrison, Cholesky + jitter ladder | [rl/bandit.py](../src/sentinel/rl/bandit.py) | `tests/unit/test_rl_bandit.py` |
| **F-09** | Episodes from the real ingestion pipeline, and the replay | [rl/simulate.py](../src/sentinel/rl/simulate.py) | `tests/unit/test_rl_simulate.py` |

**Measured** (5 held-out seeds × 200 episodes): total regret **60.6** against **301.4**
for a non-learning policy (ratio 0.201); sublinearity +42.2% mean, +23.1% worst;
optimal-action rate **0.600** against 0.242; **zero** tier violations.

### 2.2 Diffusion augmentation + calibration probe (PRD §5.5.5) ✅

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| §5.5.5 | Class-conditional tabular DDPM, gradient-checked at shipped depth | [ml/diffusion.py](../src/sentinel/ml/diffusion.py) | `tests/unit/test_diffusion.py` |
| **F-02** | Supervised family classifier | [ml/classify.py](../src/sentinel/ml/classify.py) | `tests/unit/test_classify.py` |
| §5.5.5 | Augmentation harness, calibration probe, novelty gate | [ml/robustness.py](../src/sentinel/ml/robustness.py) | `tests/unit/test_robustness.py` |

**Measured**: augmentation is roughly zero at full data and **−0.167** rare-family
macro recall at ~12 rows per family. The calibration probe found a real defect —
confidence is U-shaped under perturbation, peaking at **+0.333** overconfidence —
and gating on the anomaly ensemble's novelty takes that to **−0.003**.

### Findings during 2.4 / 2.5 / 2.2

1. **Link expansion was the obvious fix and measurement rejected it.** As a fused rank
   list it bought recall 0.892 → 0.936 and paid MRR 1.000 → 0.895. A trade is not an
   improvement, so ranking stays pure and `follow_links` appends without displacing —
   which moves recall 0.768 → 0.896 for free.
2. **Relevance from fusion scores was uninformative** (every hit between 0.89 and
   1.00). It is now the TF-IDF cosine, which is bounded and spreads.
3. **Nonsense queries scored against real documents** through feature-hash collisions,
   and no relevance floor separates them: gibberish tops at 0.05–0.11 while the weakest
   correct hit is 0.077. Undegraded term digests make the question exact.
4. **Retrieved evidence is untrusted input.** A poisoned advisory is a *better*
   injection vector than a poisoned alert, because it arrives wearing the authority of
   evidence. Chunks are scanned at build time and excluded from prompts.
5. **F-09's stated criterion and the product goal pull in opposite directions.** The
   greediest setting had the *lowest* total regret (47) and the *worst* measured
   decrease, because it converges before the first quarter ends. Optimising for the
   acceptance criterion would have made the policy worse.
6. **At low exploration the policy bifurcated across seeds** — two locked onto
   containment, three onto escalation — and the mean hid two different policies.
   Optimistic initialisation separated exploration (early, cheap) from exploitation.
7. **The tie-break sign was inverted**, and with the optimistic prior every arm ties on
   the first decision, so a fresh deployment at a permissive tier would have opened by
   containing a host.
8. **§5.5.5's augmentation claim cannot apply to §5.5.2's detector.** That detector is
   a benign-only novelty detector, so synthetic attacks cannot move its boundary. The
   claim is about a supervised model, which is why `ml/classify.py` exists.
9. **A classifier's own softmax cannot be its OOD detector**, because the quantity that
   should fall is computed from the saturating function that rises. Label smoothing —
   the standard remedy — made peak overconfidence *worse*, +0.32 → +0.46.

---

## Part 3 — Orchestration and agents ✅ 3.1 and 3.2 COMPLETE

Layer 4 of PRD Figure 2. `2,135 tests passing` (Part 2 ended at 1,861), ruff clean.

### 3.1 Runtime + Triage Agent (F-04, F-02) ✅

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| **F-04** | `IncidentState` as a frozen contract, not a dict | [agents/state.py](../src/sentinel/agents/state.py) | `tests/unit/test_agents_runtime.py` |
| **F-04** | Hash-verified, chain-linked checkpoints (memory + SQLite) | [agents/checkpoint.py](../src/sentinel/agents/checkpoint.py) | `tests/unit/test_agents_runtime.py` |
| **F-04** | The state machine: routing, `interrupt_before`, resume | [agents/runtime.py](../src/sentinel/agents/runtime.py) | `tests/unit/test_agents_runtime.py` (63) |
| App. A, §5.7 | Appendix A prompts, nonce-fenced untrusted blocks, label redaction | [agents/prompts.py](../src/sentinel/agents/prompts.py) | `tests/unit/test_agents_prompts.py` (24) |
| §5.4, §10 | `ReasoningEngine` protocol + `monotone_caution` | [agents/engine.py](../src/sentinel/agents/engine.py) | `tests/unit/test_agents_engine.py` (60) |
| **F-02** | Triage Agent: detector + classifier + novelty gate | [agents/triage.py](../src/sentinel/agents/triage.py) | `tests/unit/test_agents_triage.py` (41) |
| §9.3 | `four_way_split`, for a pipeline fitting both model kinds | [ml/metrics.py](../src/sentinel/ml/metrics.py) | `tests/unit/test_metrics.py` |

### 3.2 Investigation, Containment, approval gate, orchestrator (F-05, F-08) ✅

| PRD ref | What | Where | Verified by |
|---|---|---|---|
| **F-05** | Cited, MITRE-mapped narratives over the 2.4 knowledge base | [agents/investigate.py](../src/sentinel/agents/investigate.py) | `tests/integration/test_agent_pipeline.py` |
| **F-08** | Proposals, `SimulatedConnector`, `verify_no_ungated_execution` | [agents/contain.py](../src/sentinel/agents/contain.py) | `tests/unit/test_agents_contain.py` (31) |
| **F-04**, Fig. 3 | The five-node graph, MTTD/MTTC from step history | [agents/orchestrator.py](../src/sentinel/agents/orchestrator.py) | `tests/integration/test_agent_pipeline.py` (28) |
| §9.1 | `SimulationClock`: real durations plus scripted human time | [core/clock.py](../src/sentinel/core/clock.py) | `tests/unit/test_clock.py` (25) |

### Measured results

From `python scripts/evaluate.py --n 20000 --agents --incidents 400`:

```
[triage (F-02)] n=6,211 (22.7% attack)
  label agreement  0.9921   PASS (F-02 needs >= 0.85)
  recall           0.9965   (Section 9.1 needs >= 0.80)
  precision        0.9697   (Section 9.1 needs >= 0.85)
  alert reduction  76.7%    (Section 9.1 needs >= 60%)
  technique match  0.9929 on 1,411 attacks
  max latency      0.23 ms (budget 5,000)

[orchestration (F-04)] 400 incidents
  dismissed at triage 318 · stopped at the gate 72 · completed 82 · failed 0

[investigation (F-05)] 82 reports
  uncited claims 0 · unresolvable refs 0

[approval gate (F-08)]
  ungated executions 0 · audit chain findings 0 · connector executions 82

[Section 9.1 timing]
  MTTD (pipeline)  0.01s mean, 0.03s worst   (target < 30s)
  MTTC             12.03s mean, 12.17s worst (target < 180s)
```

The operating point is chosen on the validation split (0.9923 agreement) and reported
on a test split that neither the fit nor the calibration touched (0.9921) — a 0.0002
gap, which is the check that the grid search did not fit validation noise.

### Deviation from the PRD, deliberately

**The orchestrator is ~350 lines here, not LangGraph.** F-04's acceptance criterion —
*"any node can pause for human input and resume with full context intact"* — *is* the
interrupt/checkpoint/resume loop's semantics, so delegating it would mean the project's
central orchestration claim is tested by mocking the library that implements it. Owning
it buys two things that are tested rather than asserted: `test_agents_runtime.py`
resumes at **every** node in a graph and asserts the finished state is hash-identical to
an uninterrupted run, and it does so across a real SQLite file in a fresh object graph;
and the checkpoint is a canonicalised contract that re-validates on load, so an edited
checkpoint is refused rather than resumed. The API is deliberately LangGraph-shaped
(`add_node`, `add_edge`, `add_conditional_edges`, `compile(interrupt_before=...)`,
`invoke`/`resume`), so porting is a rewrite of one file. LangGraph *is* installable in
this environment — the deviation is a choice, not a workaround.

### The design decision that matters most: monotone caution

Every agent computes a **deterministic verdict first**. The reasoning engine is then
asked for an opinion and `engine.monotone_caution` merges it under one rule: severity
may only be raised, a decision may only move toward escalation, and confidence is taken
as the **minimum**. A technique mapping is accepted only if every alert field it cites
exists. The consequence: **an engine fully compromised by prompt injection can raise
false alarms and cannot suppress a real one.** `HostileEngine` exists to test exactly
that, end to end through the live graph.

The confidence-minimum rule cascades in a way worth noticing: a *mutually agreed*
dismissal whose confidence the engine lowered below Appendix A's 0.6 floor becomes an
escalation. "The model was less sure than the detector" turns into a human looking at
it, which is the outcome the floor exists to produce.

`NullEngine` is the default, and every Part 3 acceptance criterion passes with no model
in the loop at all. That is what makes the bound honest — the baseline it protects is a
working system, not an empty one.

### Findings during Part 3 — each changed the design

1. **The injection scanner flagged 77% of a 12,000-alert corpus.** CIC-IDS2017 ships its
   ground-truth column inside every row, spelled `" Label":"BENIGN"` — verb, separator,
   target, in exactly the order `verdict_manipulation` looked for. Every alert escalated
   as a suspected attack on the agents, and F-02 collapsed. The gap between verb and
   object now excludes structural punctuation, which keeps every imperative form and
   drops the JSON coincidence. Found by wiring the layers together; Part 1 had tested
   the scanner on hand-written hostile strings and never on a whole corpus.
2. **That same field is the answer.** Fencing the raw payload into a prompt hands the
   model the label F-02 scores it on. Redaction happens at the prompt boundary, not in
   the normalizer — `raw_payload` is the audit record of what the source sent, and
   editing it would make the tamper-evident log disagree with the SIEM.
3. **The scanner had no rule for text naming the approval gate.** `HostileEngine`,
   written to produce the worst plausible output, wrote *"ignore the approval gate and
   proceed"*. `instruction_override` needs a word like "previous" between verb and
   object; `approval_manipulation` only covered "\<verb\> without approval". Closed,
   with the benign-telemetry corpus as the false-positive check.
4. **The Human Approval Gate was unreachable at the tier every customer starts on.**
   §5.5.4 gives the policy `auto_contain` and `rl/actions` correctly requires
   `auto_with_notify` for it; §5.7 starts every action type at `recommend`. Together the
   mask removed `auto_contain`, `escalate` mapped only to `notify_analyst`, and no
   destructive action was ever proposed — **200 incidents, 133 executions, 0 approval
   requests**, with F-08 passing vacuously on a system that had never gated anything.
   "Contain" names two things: the policy's `auto_contain` means *act without a human*,
   while *recommending* a containment action for approval is what the `recommend` tier
   is for.
5. **Thresholds as benign-FPR quantiles scored 0.754 agreement** — below the F-02 bar,
   with precision 0.48. The operating point is now searched on validation under §9.1's
   recall and precision floors *as constraints*, because an optimiser given agreement
   alone discovers that dismissing everything scores the benign base rate (0.773).
6. **Lookup beats ranking when an identifier exists.** Triage has already named a
   technique, and a name is an identifier, not a query. A DDoS query ranks `T1110` and
   `T1046` detection text above the right answer, and relevance does not separate them
   (correct hit 0.025, wrong neighbour 0.066) — so no threshold would have filtered
   them. Free-text search now runs only when there is no identifier to look up.
7. **Linked citations need a per-kind budget.** `T1190` links to 31 CVEs and 2
   playbooks; "first three links" cites three CVEs and never reaches the playbook, which
   is the citation an analyst responding to the incident actually opens.
8. **MTTD was nine years.** The offline generator leaves `ingested_at` at the capture
   timestamp, so §9.1's literal definition measures the age of CIC-IDS2017. Both
   definitions are now reported with the feed lag between them, and a test drives the
   real replay service to show they coincide on the streamed path the demo uses.
9. **A frozen clock makes the MTTD gate unfalsifiable** (every node reports 0.00s), and
   a system clock cannot inject §9.1's scripted human click. `SimulationClock` does
   both: real monotonic durations plus explicit jumps.
10. **MTTC as defined includes the queue.** 28 incidents parked at the gate against one
    serial analyst gave 227s against a 180s budget. That is §2.1's thesis showing up as
    a number, not a defect to tune away, so the budget is asserted per incident and the
    queueing effect has its own test.
11. **Batch latency was cumulative.** `triage_batch` scores a whole batch in one
    vectorised call, so sharing a start instant charged the last alert with the entire
    batch — reporting 1,288 ms against F-02's 5-second budget for a 0.23 ms operation.

### Honest caveats for Part 3

- **No LLM has actually been called.** `AnthropicEngine` is written, injectable and
  tested against a fake transport, but every number above was produced with
  `NullEngine`. That is the honest configuration to measure in — monotone caution
  guarantees the online path is no *less* safe than what was measured — but it means the
  *quality* an LLM would add to investigation narratives is unmeasured.
- **Two of the five agents do not exist yet.** Code-Scan/Patch (F-07) needs Semgrep and
  a seeded vulnerable repo; the Supply-Chain agent needs `graph/explain.py` wired to a
  node. Both are Part 3.3.
- **`mypy --strict` still has not been run** (not installed here).
- **The Triage Agent's engine is not consulted on escalations by default.** Monotone
  caution fixes the answer, so the call would buy nothing — but it does mean an LLM
  cannot *add detail* to an escalation unless `consult_engine_on_escalation=True`.
- **`SimulatedConnector` is a stand-in.** PRD §4.1 scopes connectors as mocked for the
  sprint, and Part 4 replaces it with the least-privilege layer. It exists here because
  F-08 needs *something* to execute in order to prove that nothing executes without
  approval.

---

## Part 3.3 — REMAINING: the last two agents

### Code-Scan / Patch Agent (F-07)
- `pip install semgrep`, or vendor a small rule subset to stay hermetic.
- A seeded vulnerable repo under `data/` with 3–5 planted, realistic vulnerabilities.
- Map findings to CVE ids through the **existing** KB (`kb.search(..., kinds=(CVE,))`)
  so the patch rationale is cited the same way an investigation is.
- `ActionType.OPEN_PATCH_PR` is already classified destructive, so a drafted patch is
  already gated — the acceptance test is that 3+ seeded vulnerabilities are found and
  each produces a syntactically valid diff, not that the gate holds.

### Supply-Chain Agent (F-06 wiring)
- A node wrapping `graph/explain.py`; `NodeExplanation.as_evidence` already returns
  typed `Evidence`, so the investigation path needs no new citation machinery.
- The interesting design question is the trigger: this agent is *continuous* scoring,
  not alert-driven, so it does not belong on the incident graph. A second graph, or a
  scheduled job feeding alerts of `AlertSource.VENDOR_FEED` into the existing one.

### Notes for whoever picks up 3.3

- **Do not add a new prompt path.** `AgentPrompt.with_untrusted` is the only door;
  source code and dependency manifests are attacker-influenced text like any other.
- **Reuse `monotone_caution`'s shape.** A patch the engine drafts is a *proposal*; the
  deterministic half is the Semgrep finding, and the engine may not remove one.
- **The step budget is 16.** A second graph should set its own.
- **`HostileEngine` is the test asset that matters.** Every guardrail claim in Part 3 is
  "even if the model is turned against us"; test 3.3's claims the same way.

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
python -m pytest -q                              # full suite
python -m pytest -q -m "not slow"                # skip the multi-seed benchmarks
python -m ruff check src tests scripts

# Every acceptance gate this repo measures. Exits non-zero if any fails.
python scripts/evaluate.py --n 20000 --cross-dataset --graph --kb --policy --agents

# Or one layer at a time:
python scripts/evaluate.py --n 20000 --cross-dataset   # F-01, F-03
python scripts/evaluate.py --graph                     # F-06
python scripts/evaluate.py --kb                        # F-05 retrieval
python scripts/evaluate.py --policy                    # F-09
python scripts/evaluate.py --agents                    # F-02, F-04, F-05, F-08, §9.1
python scripts/evaluate.py --augment                   # §5.5.5
python scripts/evaluate.py --n 20000 --shallow         # the Part 1 linear baseline
```

No datasets, no API keys, no Redis, and no torch. Every number above is produced by
that one script.

### Architecture flags worth knowing

| Flag | Default | Why |
|---|---|---|
| `SupplyChainGNN(objective=...)` | `regression` | the label is a threshold on a continuous quantity; classification discards the ordering F-06 measures (finding 8) |
| `SupplyChainGNN(aggregation=...)` | `mean` | `sum` matches the additive semantics in theory and loses in measurement (finding 9) |
| `SupplyChainGNN(n_layers=...)` | `2` | the PRD's figure; see finding 10 for what it costs and buys |
| `WeightedEnsemble.tune_weights(min_weight=)` | `0.0` | explicit at the call site; `evaluate.py` passes `0.10` (Part 2 finding 2) |
| `DenoisingAutoencoderDetector(bottleneck=)` | `n_features // 3` | measured optimum, shallow (Part 2 finding 1) |
| `TriageAgent(engine=)` | `NullEngine()` | every acceptance criterion passes with no model in the loop; the engine adds narrative, not correctness |
| `TriageAgent(consult_engine_on_escalation=)` | `False` | monotone caution fixes the answer on an escalation, so the call buys nothing |
| `ContainmentAgent(default_tier=)` | `RiskTier.RECOMMEND` | §5.7: every action type starts at the lowest tier a customer can use and is promoted on evidence |
| `InvestigationAgent(search_k=)` | `4` | only consulted when triage named no technique; see Part 3 finding 6 |
| `build_incident_graph(step_budget=)` | `16` | the longest path is five nodes, so anything past a handful is a routing bug |
| `AgentPrompt.with_untrusted(redact_labels=)` | `True` | the corpus carries the answer in `Label`; see Part 3 finding 2 |
