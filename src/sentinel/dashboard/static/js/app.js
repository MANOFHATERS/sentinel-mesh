// The Analyst Copilot: routing, views, and the approval flow.
//
// Everything here is rendered through h()/s() from dom.js, so every server string
// is a Text node. Views are functions of fetched JSON; the server is the source of
// truth and nothing is cached across navigations except the session.

import { ApiError, createClient, makeStore } from "./api.js";
import { append, clear, h, replace } from "./dom.js";
import {
  diffLines,
  fmtAgo,
  fmtNumber,
  fmtPercent,
  fmtSeconds,
  fmtTime,
  humanize,
  parseRoute,
  safeHref,
  shortId,
  toneOf,
} from "./format.js";
import { barChart, lineChart } from "./charts.js";
import { renderGraph } from "./graph.js";

const root = document.getElementById("app");
let session = null;
let refreshTimer = null;
let renderToken = 0;

function storageOrNull() {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

const memoryStorage = new Map();
const api = createClient({
  fetchImpl: (...args) => window.fetch(...args),
  store: makeStore(
    storageOrNull() || {
      getItem: (k) => memoryStorage.get(k) ?? null,
      setItem: (k, v) => memoryStorage.set(k, v),
      removeItem: (k) => memoryStorage.delete(k),
    },
  ),
  onUnauthorized: () => {
    session = null;
    api.signOut();
    renderLogin("Your session was not accepted. Paste a token to sign in again.");
  },
});

// --------------------------------------------------------------------------- //
// Small components
// --------------------------------------------------------------------------- //

function badge(value, label) {
  return h("span", { class: `badge tone-${toneOf(value)}` }, label ?? humanize(value));
}

function kindBadge(kind) {
  const labels = { incident: "Incident", code_scan: "Code scan", supply_chain: "Supply chain" };
  return h("span", { class: `badge kind kind-${kind}` }, labels[kind] || humanize(kind));
}

function link(route, ...children) {
  return h("a", { href: route }, ...children);
}

function section(title, ...children) {
  return h("section", { class: "panel" }, h("h2", {}, title), ...children);
}

function kv(rows) {
  return h(
    "dl",
    { class: "kv" },
    rows
      .filter((row) => row)
      .map(([key, value]) => [h("dt", {}, key), h("dd", {}, value ?? "—")]),
  );
}

function empty(text) {
  return h("p", { class: "empty" }, text);
}

function untrusted(label, text) {
  return h(
    "figure",
    { class: "untrusted" },
    h("figcaption", {}, h("span", { class: "badge tone-bad" }, "untrusted"), " ", label),
    h("pre", {}, text),
  );
}

function assetLabel(asset) {
  if (!asset) return "—";
  const parts = [asset.address];
  if (asset.hostname) parts.push(` (${asset.hostname})`);
  return h("span", {}, parts.join(""), asset.protected ? [" ", badge("failed", "protected")] : null);
}

function errorBox(error) {
  const status = error instanceof ApiError ? `HTTP ${error.status}: ` : "";
  return h("div", { class: "alert tone-bad", role: "alert" }, status + (error.message || String(error)));
}

function table(headers, rows, { className = "" } = {}) {
  if (!rows.length) return empty("Nothing here yet.");
  return h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      { class: className },
      h("thead", {}, h("tr", {}, headers.map((head) => h("th", {}, head)))),
      h("tbody", {}, rows),
    ),
  );
}

// Verified, empty (a fresh workspace: nothing has run), or broken. An empty chain in
// a workspace that has runs is reported as broken by the server, not here.
function chainBadge(audit, ungated) {
  if (ungated.length) return badge("failed", `${ungated.length} UNGATED execution(s)`);
  if (audit.status === "verified") return badge("completed", `chain verified · ${audit.rows} rows`);
  if (audit.status === "empty") return badge("info", "chain empty · nothing recorded yet");
  return badge("failed", "chain BROKEN");
}

function chainTone(audit) {
  return audit.status === "verified" ? "good" : audit.status === "empty" ? "neutral" : "bad";
}

function statTile(label, value, note, tone = "neutral") {
  return h(
    "div",
    { class: `tile tone-${tone}` },
    h("div", { class: "tile-label" }, label),
    h("div", { class: "tile-value" }, value),
    note ? h("div", { class: "tile-note" }, note) : null,
  );
}

// --------------------------------------------------------------------------- //
// Shell
// --------------------------------------------------------------------------- //

const NAV = [
  ["overview", "Overview"],
  ["scenarios", "Scenarios"],
  ["queue", "Approval queue"],
  ["incidents", "Incidents"],
  ["supply-chain", "Supply chain"],
  ["code-scan", "Code scan"],
  ["wire", "Wire & guardrails"],
  ["audit", "Audit log"],
  ["models", "Models"],
  ["evaluation", "Evaluation"],
];

let mainEl = null;
let navEls = {};
let queueCountEl = null;
let chainEl = null;

function renderShell() {
  clear(root);
  navEls = {};
  queueCountEl = h("span", { class: "count", "aria-label": "waiting for approval" }, "");
  chainEl = h("span", { class: "chain" }, "");
  const nav = h(
    "nav",
    { class: "nav", "aria-label": "Sections" },
    NAV.map(([name, label]) => {
      const item = h("a", { href: `#/${name}`, class: "nav-item" }, label, name === "queue" ? queueCountEl : null);
      navEls[name] = item;
      return item;
    }),
  );
  mainEl = h("main", {
    class: "main",
    id: "main",
    tabindex: "-1",
    onpointerenter: () => (pointerInMain = true),
    onpointerleave: () => (pointerInMain = false),
  });
  append(root, [
    h(
      "header",
      { class: "topbar" },
      h("div", { class: "brand" }, h("span", { class: "logo", "aria-hidden": "true" }, "S"), "Sentinel Mesh", h("span", { class: "sub" }, "Analyst Copilot")),
      h(
        "div",
        { class: "who" },
        chainEl,
        h("span", { class: "tenant" }, `tenant ${session.tenant_id}`),
        h("span", {}, `${session.principal} · ${session.role}`),
        h("button", { class: "btn ghost", type: "button", onclick: signOut }, "Sign out"),
      ),
    ),
    session.can_act
      ? null
      : h("div", { class: "readonly-banner", role: "note" }, `Read-only view: you are signed in as a ${session.role}. Launching, approving and rejecting are available to analysts only.`),
    h("div", { class: "layout" }, nav, mainEl),
  ]);
}

function signOut() {
  api.signOut();
  session = null;
  // The next person to sign in starts at the Overview, not wherever the last one left off.
  window.history.replaceState(null, "", window.location.pathname + window.location.search);
  renderLogin();
}

function renderLogin(message) {
  stopRefresh();
  clear(root);
  const input = h("input", {
    id: "token",
    name: "token",
    type: "password",
    autocomplete: "off",
    required: true,
    placeholder: "Bearer token printed by python -m sentinel.dashboard",
  });
  const status = h("p", { class: "login-status", role: "status" }, message || "");
  const form = h(
    "form",
    {
      class: "login",
      onsubmit: async (event) => {
        event.preventDefault();
        const value = input.value.trim();
        if (!value) return;
        api.signIn(value);
        replace(status, "Signing in…");
        try {
          session = await api.get("/api/session");
          start();
        } catch (error) {
          api.signOut();
          replace(status, error.status === 401 ? "That token was not accepted." : error.message);
        }
      },
    },
    h("span", { class: "logo", "aria-hidden": "true" }, "S"),
    h("h1", {}, "Sentinel Mesh"),
    h("p", { class: "sub" }, "Analyst Copilot · sign in with your dashboard token"),
    h("label", { for: "token" }, "Token"),
    input,
    h("button", { class: "btn primary", type: "submit" }, "Sign in"),
    status,
  );
  append(root, h("div", { class: "login-wrap" }, form));
  input.focus();
}

function stopRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = null;
}

async function refreshChrome() {
  try {
    const overview = await api.get("/api/overview");
    replace(queueCountEl, overview.queue ? String(overview.queue) : "");
    replace(chainEl, chainBadge(overview.audit, overview.ungated_executions));
    return overview;
  } catch {
    return null;
  }
}

// --------------------------------------------------------------------------- //
// Routing
// --------------------------------------------------------------------------- //

const AUTO_REFRESH = new Set(["overview", "queue", "incidents", "scenarios", "wire", "models"]);

async function route() {
  if (!session) return;
  const { name, arg } = parseRoute(window.location.hash);
  for (const [key, el] of Object.entries(navEls)) {
    el.classList.toggle("active", key === name || (name === "incident" && key === "incidents"));
    if (key === name) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  }
  const view = VIEWS[name] || VIEWS.overview;
  const token = ++renderToken;
  stopRefresh();
  replace(mainEl, h("p", { class: "loading" }, "Loading…"));
  await draw(view, arg, token);
  if (AUTO_REFRESH.has(name)) {
    refreshTimer = setInterval(() => {
      // Never redraw under an analyst who is typing, has a control focused, or has
      // the pointer over the content: a button that is replaced between hover and
      // click turns an intended click into a missed one — or into a different one.
      const active = document.activeElement;
      if (active && mainEl.contains(active) && active !== mainEl) return;
      if (pointerInMain || document.hidden) return;
      draw(view, arg, token, { quiet: true });
    }, 4000);
  }
}

let pointerInMain = false;

async function draw(view, arg, token, { quiet = false } = {}) {
  try {
    const content = await view(arg);
    if (token !== renderToken) return;
    // A background refresh that changed nothing leaves the DOM alone.
    if (quiet && content.textContent === mainEl.textContent) return;
    replace(mainEl, content);
  } catch (error) {
    if (token !== renderToken) return;
    if (!quiet) replace(mainEl, errorBox(error));
  }
  refreshChrome();
}

function go(hash) {
  if (window.location.hash === hash) route();
  else window.location.hash = hash;
}

// --------------------------------------------------------------------------- //
// Views
// --------------------------------------------------------------------------- //

async function viewOverview() {
  const overview = await api.get("/api/overview");
  const tiles = h(
    "div",
    { class: "tiles" },
    statTile("Waiting for approval", String(overview.queue), "across all three graphs", overview.queue ? "warn" : "neutral"),
    statTile("Incidents", String(overview.incidents), `${overview.alerts_ingested} alerts ingested`),
    statTile("Alert reduction", fmtPercent(overview.alert_reduction), `${overview.alerts_reaching_human} reached a human`, "info"),
    statTile("MTTD", fmtSeconds(overview.mttd_seconds), "ingestion → triage (§9.1: < 30 s)"),
    statTile("MTTC", fmtSeconds(overview.mttc_seconds), "triage → approved containment (§9.1: < 3 min)"),
    statTile("Audit chain", overview.audit.status, `${overview.audit.rows} rows · ${fmtNumber(overview.audit.duration_ms, 1)} ms`, chainTone(overview.audit)),
    statTile("Ungated executions", String(overview.ungated_executions.length), "F-08 read back from the log", overview.ungated_executions.length ? "bad" : "good"),
    statTile("Failed actions", String(overview.failed_actions), `${overview.refusals} guardrail refusal(s)`, overview.failed_actions ? "bad" : "neutral"),
  );
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Overview"),
    tiles,
    section("Demo scenarios", h("div", { class: "cards" }, overview.scenarios.map((sc) => scenarioCard(sc, { compact: true })))),
    replayPanel(overview.feed_remaining),
  );
}

function scenarioProgress(sc) {
  if (!sc.launched) return badge("pending", "not started");
  if (sc.complete) return badge("completed", "complete");
  if (sc.waiting) return badge("awaiting_approval", `${sc.waiting} waiting for you`);
  return badge("running", "in progress");
}

function scenarioCard(sc, { compact = false } = {}) {
  const launch = h(
    "button",
    {
      class: "btn primary",
      type: "button",
      disabled: sc.launched,
      onclick: async (event) => {
        event.currentTarget.disabled = true;
        try {
          const result = await api.post(`/api/scenarios/${encodeURIComponent(sc.name)}/launch`);
          go(result.incidents.length ? `#/scenarios` : "#/queue");
        } catch (error) {
          alertError(error);
          route();
        }
      },
    },
    sc.launched ? "Launched" : "Launch scenario",
  );
  // A control the viewer role can never use is not rendered (least privilege in the UI);
  // the API still answers 403, which is the actual enforcement.
  return h(
    "article",
    { class: `card scenario ${sc.complete ? "done" : ""}` },
    h("div", { class: "card-head" }, h("h3", {}, sc.title), scenarioProgress(sc)),
    h("p", {}, sc.summary),
    h("p", { class: "meta" }, `Agents: ${sc.agents.join(", ")}`),
    sc.launched
      ? h(
          "p",
          { class: "meta" },
          `Launched ${fmtAgo(sc.launched_at)} by ${sc.launched_by} · ${sc.finished}/${sc.total} runs finished · ${sc.decisions} decision(s)`,
          sc.failed_actions ? [" · ", badge("failed", `${sc.failed_actions} failed action(s)`)] : null,
        )
      : null,
    compact
      ? null
      : h("ol", { class: "walkthrough" }, sc.walkthrough.map((step) => h("li", {}, step))),
    sc.threads.length
      ? h(
          "ul",
          { class: "threads" },
          sc.threads.map((t) =>
            h("li", {}, kindBadge(t.kind), " ", link(`#/incident/${encodeURIComponent(t.incident_id)}`, t.caption), " ", badge(t.status)),
          ),
        )
      : null,
    session.can_act ? h("div", { class: "card-actions" }, launch) : null,
  );
}

function replayPanel(remaining) {
  const count = h("input", { type: "number", min: "1", max: "500", value: "50", id: "replay-count", class: "narrow" });
  const out = h("p", { class: "meta", role: "status" }, `${remaining} held-out flows remain in the background feed.`);
  return section(
    "Background feed",
    h("p", {}, "Replay held-out flows through the same incident graph. Most are dismissed by triage; the ones that escalate join the approval queue — the alert-volume reduction above is measured on exactly this."),
    !session.can_act
      ? null
      : h(
      "div",
      { class: "row" },
      h("label", { for: "replay-count" }, "Flows"),
      count,
      h(
        "button",
        {
          class: "btn",
          type: "button",
          disabled: remaining === 0,
          onclick: async (event) => {
            const button = event.currentTarget;
            button.disabled = true;
            try {
              const result = await api.post("/api/feed/replay", { count: Number(count.value) });
              replace(out, `Ingested ${result.ingested}: ${result.dismissed} dismissed, ${result.handled} handled without a human, ${result.gated} gated for approval, ${result.failed} failed. ${result.remaining} remain.`);
            } catch (error) {
              replace(out, errorBox(error));
            } finally {
              button.disabled = false;
            }
          },
        },
        "Replay",
      ),
    ),
    out,
  );
}

function alertError(error) {
  const box = errorBox(error);
  mainEl.prepend(box);
}

async function viewScenarios() {
  const scenarios = await api.get("/api/scenarios");
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Scenarios"),
    h("p", { class: "lede" }, "PRD F-10: each scenario is completable end-to-end from this UI alone. Launch one, then work the approval queue."),
    h("div", { class: "cards" }, scenarios.map((sc) => scenarioCard(sc))),
  );
}

async function viewQueue() {
  const queue = await api.get("/api/queue");
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Approval queue"),
    h("p", { class: "lede" }, "Every run paused at the Human Approval Gate, across the incident, code-scan and supply-chain graphs. Newest first."),
    queue.length
      ? h("div", { class: "cards" }, queue.map(queueCard))
      : empty("Nothing is waiting for a decision."),
  );
}

function queueCard(item) {
  const action = item.pending_action;
  return h(
    "article",
    { class: "card queue-item" },
    h("div", { class: "card-head" }, h("div", {}, kindBadge(item.kind), " ", badge(item.severity || "info", item.severity || "—")), h("span", { class: "meta" }, `requested ${fmtAgo(action.created_at)}`)),
    h("h3", {}, humanize(action.action_type), " → ", action.target_asset ? assetLabel(action.target_asset) : action.target),
    h("p", { class: "meta" }, item.caption),
    h("p", { class: "clamp" }, action.rationale),
    h("div", { class: "card-actions" }, link(`#/incident/${encodeURIComponent(item.incident_id)}`, h("span", { class: "btn primary" }, "Review and decide"))),
  );
}

async function viewIncidents() {
  const params = new URLSearchParams(window.location.hash.split("?")[1] || "");
  const status = params.get("status") || "";
  const kind = params.get("kind") || "";
  const query = new URLSearchParams();
  if (status) query.set("status", status);
  if (kind) query.set("kind", kind);
  const data = await api.get(`/api/incidents?${query}`);
  const statusSelect = h(
    "select",
    { id: "f-status", onchange: (e) => filter("status", e.target.value) },
    ["", "awaiting_approval", "running", "completed", "dismissed", "failed"].map((v) =>
      h("option", { value: v, selected: v === status }, v ? humanize(v) : "any status"),
    ),
  );
  const kindSelect = h(
    "select",
    { id: "f-kind", onchange: (e) => filter("kind", e.target.value) },
    ["", "incident", "code_scan", "supply_chain"].map((v) => h("option", { value: v, selected: v === kind }, v ? humanize(v) : "any graph")),
  );
  function filter(key, value) {
    const next = new URLSearchParams({ status, kind });
    next.set(key, value);
    for (const [k, v] of [...next]) if (!v) next.delete(k);
    go(`#/incidents${next.toString() ? `?${next}` : ""}`);
  }
  const rows = data.items.map((item) =>
    h(
      "tr",
      { class: item.failed_actions ? "row-bad" : "" },
      h("td", {}, link(`#/incident/${encodeURIComponent(item.incident_id)}`, shortId(item.incident_id))),
      h("td", {}, kindBadge(item.kind)),
      h("td", {}, item.caption),
      h("td", {}, badge(item.status)),
      h("td", {}, item.decision ? badge(item.decision) : "—"),
      h("td", {}, item.severity ? badge(item.severity) : "—"),
      h("td", {}, item.asset ? assetLabel(item.asset) : "—"),
      h("td", {}, item.actions.map((a) => badge(a.status, `${humanize(a.action_type)}: ${a.status}`))),
      h("td", { class: "num" }, fmtSeconds(item.detect_seconds)),
      h("td", {}, fmtAgo(item.updated_at)),
    ),
  );
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Incidents"),
    h("div", { class: "row filters" }, h("label", { for: "f-status" }, "Status"), statusSelect, h("label", { for: "f-kind" }, "Graph"), kindSelect, h("span", { class: "meta" }, `${data.total} total`)),
    table(["Id", "Graph", "What", "Status", "Triage", "Severity", "Asset", "Actions", "MTTD", "Updated"], rows),
  );
}

// --- incident detail --------------------------------------------------------- //

async function viewIncident(id) {
  const detail = await api.get(`/api/incidents/${encodeURIComponent(id)}`);
  return renderIncident(detail);
}

function renderIncident(d) {
  const failed = d.actions_detail.filter((a) => a.status === "failed");
  return h(
    "div",
    { class: "stack" },
    h("p", { class: "crumbs" }, link("#/incidents", "Incidents"), " / ", shortId(d.incident_id)),
    h("div", { class: "title-row" }, h("h1", {}, d.caption), h("div", {}, kindBadge(d.kind), " ", badge(d.status), d.scenario ? [" ", h("span", { class: "badge" }, d.scenario)] : null)),
    failed.length
      ? h(
          "div",
          { class: "alert tone-bad", role: "alert" },
          h("strong", {}, `${failed.length} action(s) failed inside this ${d.status} run. `),
          failed.map((a) => h("div", {}, `${humanize(a.action_type)} → ${a.target}: ${a.failure_reason}`)),
        )
      : null,
    d.error ? h("div", { class: "alert tone-bad", role: "alert" }, `Run failed: ${d.error}`) : null,
    d.interrupt ? decisionPanel(d) : null,
    d.stalled ? recoverPanel(d) : null,
    h(
      "div",
      { class: "grid-2" },
      alertPanel(d),
      h("div", { class: "stack" }, triagePanel(d), timingPanel(d)),
    ),
    reportPanel(d.report),
    policyPanel(d.policy),
    actionsPanel(d),
    wirePanel(d.wire),
    historyPanel(d),
    timelinePanel(d.timeline),
  );
}

function decisionPanel(d) {
  const action = d.actions_detail.find((a) => a.action_id === d.interrupt.subject_id);
  const note = h("textarea", { id: "decision-note", rows: "2", maxlength: "2000", placeholder: "Optional note, recorded in the audit chain with your decision" });
  const status = h("p", { class: "meta", role: "status" }, "");
  const buttons = [];
  async function decide(approved) {
    buttons.forEach((b) => (b.disabled = true));
    replace(status, approved ? "Approving…" : "Rejecting…");
    try {
      const updated = await api.post(`/api/incidents/${encodeURIComponent(d.incident_id)}/decision`, {
        action_id: action.action_id,
        approved,
        note: note.value,
      });
      replace(mainEl, renderIncident(updated));
      refreshChrome();
    } catch (error) {
      replace(status, errorBox(error));
      buttons.forEach((b) => (b.disabled = false));
    }
  }
  const approve = h("button", { class: "btn approve", type: "button", onclick: () => decide(true) }, "Approve");
  const reject = h("button", { class: "btn reject", type: "button", onclick: () => decide(false) }, "Reject");
  buttons.push(approve, reject);
  return h(
    "section",
    { class: "panel decision", "aria-labelledby": "decision-title" },
    h("h2", { id: "decision-title" }, "Human Approval Gate"),
    h("p", { class: "decision-what" }, humanize(action.action_type), " → ", action.target_asset ? assetLabel(action.target_asset) : action.target),
    kv([
      ["Reason", d.interrupt.reason],
      ["Proposed by", humanize(action.proposed_by)],
      ["Trust tier", humanize(action.risk_tier)],
      ["Destructive", action.destructive ? "yes" : "no"],
      ["Requested", `${fmtTime(d.interrupt.requested_at)} (${fmtAgo(d.interrupt.requested_at)})`],
    ]),
    action.evidence.length ? evidenceList(action.evidence, "Evidence attached to this action") : null,
    ...(session.can_act
      ? [
          h("label", { for: "decision-note" }, "Note"),
          note,
          h("div", { class: "row" }, approve, reject, h("span", { class: "meta" }, `Recorded as ${session.principal}.`)),
        ]
      : [h("p", { class: "meta" }, "Waiting for an analyst to decide. Viewers can read the evidence but cannot approve or reject.")]),
    status,
  );
}

function recoverPanel(d) {
  const status = h("p", { class: "meta", role: "status" }, "");
  const recover = h(
    "button",
    {
      class: "btn primary",
      type: "button",
      onclick: async (event) => {
        event.currentTarget.disabled = true;
        try {
          replace(mainEl, renderIncident(await api.post(`/api/incidents/${encodeURIComponent(d.incident_id)}/recover`)));
        } catch (error) {
          replace(status, errorBox(error));
        }
      },
    },
    "Recover run",
  );
  return h(
    "section",
    { class: "panel alert tone-warn" },
    h("h2", {}, "Stalled run"),
    h("p", {}, `This run stopped at node "${d.history.length ? d.history[d.history.length - 1].node : "?"}" without finishing or pausing — a process died mid-node. Recovery re-runs from the last checkpoint; the execution journal replays any connector call that already completed instead of repeating it.`),
    session.can_act ? recover : null,
    status,
  );
}

function alertPanel(d) {
  const a = d.alert;
  return section(
    "Alert",
    kv([
      ["Source", humanize(a.source)],
      ["Asset", assetLabel(a.asset)],
      ["Flow", `${a.src_ip ?? "?"}:${a.src_port ?? "?"} → ${a.dst_ip ?? "?"}:${a.dst_port ?? "?"} ${a.protocol ?? ""}`],
      ["Signature", a.signature || "—"],
      ["Occurred", fmtTime(a.timestamp)],
      ["Ingested", fmtTime(a.ingested_at)],
      ["Injection scan", h("span", {}, badge(a.injection.verdict), " ", a.injection.summary)],
    ]),
    untrusted("raw payload — shown as inert text, never interpreted", a.raw_payload),
  );
}

function triagePanel(d) {
  const t = d.triage;
  if (!t) return section("Triage", empty("Not triaged."));
  return section(
    "Triage",
    kv([
      ["Decision", badge(t.decision)],
      ["Severity", badge(t.severity)],
      ["Confidence", fmtNumber(t.confidence)],
      ["Anomaly score", fmtNumber(t.anomaly_score)],
      ["Technique", t.technique_id || "—"],
      ["Supporting fields", t.supporting_fields.join(", ") || "—"],
      ["Model", t.model_version],
    ]),
    h("p", { class: "rationale" }, t.rationale),
  );
}

function timingPanel(d) {
  return section(
    "Timing and integrity",
    kv([
      d.timings ? ["Detect (MTTD)", fmtSeconds(d.timings.detect_seconds)] : null,
      d.timings ? ["Contain (MTTC)", fmtSeconds(d.timings.contain_seconds)] : null,
      d.timings ? ["Waiting on a human", fmtSeconds(d.timings.approval_wait_seconds)] : null,
      ["Trust tier", humanize(d.trust_tier)],
      ["Checkpoints", h("span", {}, `${d.checkpoints.count} · `, badge(d.checkpoints.verified ? "completed" : "failed", d.checkpoints.verified ? "chain verified" : "chain BROKEN"))],
      d.checkpoints.error ? ["Checkpoint error", d.checkpoints.error] : null,
    ]),
  );
}

function evidenceList(items, title) {
  return h(
    "details",
    { class: "evidence", open: items.length <= 6 },
    h("summary", {}, `${title} (${items.length})`),
    h(
      "ol",
      {},
      items.map((e) =>
        h(
          "li",
          { id: `ev-${e.ref}` },
          h("div", { class: "ev-head" }, h("code", {}, e.ref), " ", h("span", { class: "badge" }, humanize(e.kind)), " ", h("span", { class: "meta" }, `relevance ${fmtNumber(e.relevance, 2)}`)),
          h("pre", { class: "excerpt" }, e.excerpt),
        ),
      ),
    ),
  );
}

function reportPanel(report) {
  if (!report) return section("Investigation", empty("No report yet."));
  return section(
    `Investigation · ${humanize(report.agent)}`,
    h("div", { class: "row" }, badge(report.severity), h("span", { class: "meta" }, `confidence ${fmtNumber(report.confidence)} · techniques ${report.techniques.join(", ") || "—"} · ${report.grounded ? "every claim cited" : "no claims"}`)),
    h("p", { class: "summary" }, report.summary),
    report.claims.length
      ? h(
          "ol",
          { class: "claims" },
          report.claims.map((claim) =>
            h("li", {}, claim.statement, h("span", { class: "refs" }, claim.refs.map((ref) => h("code", { class: "ref" }, ref)))),
          ),
        )
      : null,
    evidenceList(report.evidence, "Evidence"),
  );
}

function actionsPanel(d) {
  const rows = d.actions_detail.map((a) =>
    h(
      "tr",
      { class: a.status === "failed" ? "row-bad" : "" },
      h("td", {}, humanize(a.action_type)),
      h("td", {}, a.target_asset ? assetLabel(a.target_asset) : a.target),
      h("td", {}, badge(a.status)),
      h("td", {}, a.requires_human_approval ? "gated" : "unattended"),
      h("td", {}, a.approved_by || "—"),
      h("td", {}, fmtTime(a.decided_at)),
      h("td", {}, fmtTime(a.executed_at)),
      h("td", {}, a.failure_reason || ""),
    ),
  );
  return section("Actions", table(["Action", "Target", "Status", "Gate", "Decided by", "Decided", "Executed", "Failure"], rows));
}

function wirePanel(wire) {
  const executions = wire.executions.map((x) =>
    h(
      "tr",
      {},
      h("td", {}, x.connector),
      h("td", {}, humanize(x.action_type)),
      h("td", {}, x.target),
      h("td", {}, badge(x.succeeded ? "executed" : "failed", x.succeeded ? "succeeded" : "failed")),
      h("td", {}, x.detail),
      h("td", {}, x.reference ? referenceLink(x.reference) : "—"),
      h("td", {}, x.replayed ? "replayed from journal" : ""),
    ),
  );
  const refusals = wire.refusals.map((r) => h("tr", { class: "row-bad" }, h("td", {}, r.actor), h("td", {}, r.guardrail || "—"), h("td", {}, r.reason)));
  const calls = wire.calls.map((c) =>
    h("tr", {}, h("td", { class: "num" }, String(c.seq)), h("td", {}, c.payload.connector), h("td", {}, `${c.payload.method} ${c.payload.route}`), h("td", { class: "num" }, String(c.payload.status ?? "—")), h("td", { class: "num" }, String(c.payload.attempt)), h("td", { class: "num" }, `${fmtNumber(c.payload.duration_ms, 1)} ms`)),
  );
  if (!executions.length && !refusals.length && !calls.length) {
    return section("On the wire", empty("No connector was called for this incident."));
  }
  return section(
    "On the wire",
    refusals.length ? [h("h3", {}, "Guardrail refusals"), table(["Actor", "Guardrail", "Reason"], refusals)] : null,
    executions.length ? [h("h3", {}, "Executions"), table(["Connector", "Action", "Target", "Outcome", "Detail", "Reference", ""], executions)] : null,
    calls.length ? [h("h3", {}, "HTTP attempts (connector_called rows)"), table(["Seq", "Connector", "Route", "Status", "Attempt", "Duration"], calls)] : null,
  );
}

function referenceLink(reference) {
  const href = safeHref(reference);
  return href ? h("a", { href, rel: "noopener noreferrer", target: "_blank" }, reference) : reference;
}

function historyPanel(d) {
  return section(
    "Graph steps",
    h(
      "ol",
      { class: "steps" },
      d.history.map((step) =>
        h("li", { class: `step ${step.outcome}` }, h("strong", {}, step.node), " ", badge(step.outcome === "ok" ? "completed" : step.outcome === "error" ? "failed" : "awaiting_approval", step.outcome), " ", h("span", { class: "meta" }, `${fmtTime(step.started_at)} · ${fmtNumber(step.duration_ms, 1)} ms`), step.detail ? h("div", { class: "meta" }, step.detail) : null),
      ),
    ),
  );
}

function timelinePanel(records) {
  return section(
    "Audit timeline",
    h("p", { class: "meta" }, "Rows from the hash-chained audit log whose subject is this alert, incident or one of its actions."),
    records.length
      ? h(
          "ol",
          { class: "timeline" },
          records.map((r) =>
            h(
              "li",
              {},
              h("div", { class: "tl-head" }, h("code", {}, `#${r.seq}`), " ", h("strong", {}, humanize(r.event_type)), " ", h("span", { class: "meta" }, `${r.actor} · ${fmtTime(r.recorded_at)}`)),
              h("details", {}, h("summary", {}, `row ${shortId(r.row_hash, 16)}…`), h("pre", {}, JSON.stringify(r.payload, null, 2))),
            ),
          ),
        )
      : empty("No audit rows."),
  );
}

// --- supply chain ------------------------------------------------------------ //

async function viewSupplyChain(arg) {
  const scope = arg || "top";
  const data = await api.get(`/api/supply-chain/graph?scope=${encodeURIComponent(scope)}`);
  const detail = h("aside", { class: "panel node-detail", "aria-live": "polite" }, empty("Select a node to see the paths behind its score."));
  const mapHolder = h("div", { class: "graph-holder" });
  async function select(nodeId) {
    replace(mapHolder, renderGraph(data, { onSelect: select, selected: nodeId }));
    replace(detail, h("p", { class: "loading" }, "Explaining…"));
    try {
      const within = data.advisory ? `?advisory=${encodeURIComponent(data.advisory.advisory_id)}` : "";
      const node = await api.get(`/api/supply-chain/nodes/${encodeURIComponent(nodeId)}${within}`);
      replace(detail, nodePanel(node));
    } catch (error) {
      replace(detail, errorBox(error));
    }
  }
  replace(mapHolder, renderGraph(data, { onSelect: select }));
  const scopes = [["top", "Top risk (whole graph)"], ["all", "Entire graph"], ...data.advisories.map((a) => [`advisory:${a.advisory_id}`, `${a.advisory_id} — ${a.package_id}`])];
  const selectEl = h(
    "select",
    { id: "scope", onchange: (e) => go(`#/supply-chain/${encodeURIComponent(e.target.value)}`) },
    scopes.map(([value, label]) => h("option", { value, selected: value === scope }, label)),
  );
  const flagged = data.flagged.map((f) =>
    h(
      "tr",
      {},
      h("td", {}, h("button", { class: "linkish", type: "button", onclick: () => select(f.node_id) }, f.node_id)),
      h("td", {}, humanize(f.kind)),
      h("td", { class: "num" }, fmtNumber(f.risk)),
      h("td", {}, f.intrinsic ? badge("failed", "risk source") : "inherited"),
      h("td", {}, humanize(f.driver)),
      h("td", { class: "num" }, String(f.paths)),
    ),
  );
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Supply-chain map"),
    h("div", { class: "row filters" }, h("label", { for: "scope" }, "Scope"), selectEl, h("span", { class: "meta" }, `${data.nodes.length} of ${data.counts.nodes} nodes · ${data.counts.risk_sources} risk sources in the graph`)),
    data.advisory
      ? h("div", { class: "alert tone-warn" }, h("strong", {}, `${data.advisory.advisory_id}: `), data.advisory.title, h("p", {}, data.advisory.summary), data.advisory.cve_ids.length ? h("p", { class: "meta" }, `CVEs: ${data.advisory.cve_ids.join(", ")}`) : null)
      : null,
    h(
      "div",
      { class: "graph-layout" },
      h("div", { class: "panel graph-panel" }, mapHolder, legend()),
      detail,
    ),
    section("Flagged nodes", table(["Node", "Kind", "Risk", "Source", "Model driver", "Paths"], flagged)),
  );
}

function legend() {
  return h(
    "ul",
    { class: "legend" },
    h("li", {}, h("span", { class: "dot organization" }), "organisation"),
    h("li", {}, h("span", { class: "dot vendor" }), "vendor"),
    h("li", {}, h("span", { class: "dot package" }), "package"),
    h("li", {}, h("span", { class: "dot intrinsic" }), "risk source (≥4 CVEs, unmaintained)"),
    h("li", {}, h("span", { class: "swatch hot" }), "exposure path driving a flag"),
    h("li", { class: "meta" }, "Size grows with risk. Edges point the way risk flows: dependency → dependent."),
  );
}

function nodePanel(node) {
  return h(
    "div",
    { class: "stack tight" },
    h("h2", {}, node.node_id),
    h("div", {}, badge(node.intrinsic ? "failed" : "info", node.intrinsic ? "risk source" : humanize(node.kind))),
    kv([
      ["Risk", fmtNumber(node.risk)],
      ["Model keyed on", `${humanize(node.driver)} (own ${fmtPercent(node.own_feature_share, 0)} / chain ${fmtPercent(node.neighbourhood_share, 0)})`],
      ["CVEs", String(node.features.cve_exposure_count)],
      ["Days since update", fmtNumber(node.features.days_since_last_update, 0)],
      ["SBOM depth", String(node.features.sbom_depth)],
      ["Breach history", String(node.features.breach_history)],
      ["Unmaintained", node.features.unmaintained ? "yes" : "no"],
    ]),
    h("h3", {}, `Exposure paths (${node.paths.length})${node.advisory_id ? ` within ${node.advisory_id}` : ""}`),
    node.paths.length
      ? h("ol", { class: "paths" }, node.paths.map((p) => h("li", {}, h("code", {}, p.nodes.join(" → ")), h("div", { class: "meta" }, `${p.hops} hop(s) · source ${p.source_cves} CVEs · contribution ${fmtNumber(p.contribution)}`))))
      : empty("No exposure path within four hops."),
  );
}

// --- code scan --------------------------------------------------------------- //

async function viewCodeScan() {
  const data = await api.get("/api/code-scan");
  if (!data.available) {
    return h("div", { class: "stack" }, h("h1", {}, "Code scan"), empty("No scan has run yet. Launch the vendor-dependency CVE scenario to scan acme/billing."));
  }
  const scan = data.scan;
  const findings = scan.findings.map((f) =>
    h(
      "tr",
      {},
      h("td", {}, badge(f.severity)),
      h("td", {}, h("code", {}, `${f.path}:${f.line}`)),
      h("td", {}, f.rule_id),
      h("td", {}, f.cwe),
      h("td", {}, f.title),
      h("td", {}, humanize(f.confidence)),
      h("td", {}, h("code", { class: "excerpt-inline" }, f.excerpt)),
    ),
  );
  const draft = scan.draft;
  return h(
    "div",
    { class: "stack" },
    h("div", { class: "title-row" }, h("h1", {}, `Code scan · ${scan.repository}`), badge(scan.status)),
    h("p", {}, `${scan.files_scanned} files, ${scan.lines_scanned} lines, ${scan.findings.length} findings. `, link(`#/incident/${encodeURIComponent(scan.incident_id)}`, "Open the approval →")),
    scan.execution
      ? h("div", { class: `alert tone-${scan.execution.succeeded ? "good" : "bad"}` }, `Draft PR ${scan.execution.succeeded ? "opened" : "failed"} by ${scan.execution.approved_by || "?"}: `, scan.execution.reference ? referenceLink(scan.execution.reference) : scan.execution.detail)
      : null,
    scan.security_findings.length ? h("div", { class: "alert tone-warn" }, scan.security_findings.map((x) => h("div", {}, x))) : null,
    section("Findings", table(["Severity", "Where", "Rule", "CWE", "Title", "Confidence", "Excerpt (untrusted)"], findings)),
    draft
      ? section(
          "Draft pull request",
          kv([
            ["Title", draft.title],
            ["Branch", h("code", {}, draft.branch)],
            ["Files", draft.files.join(", ")],
            ["Left for a human", draft.unpatched_refs.join(", ") || "—"],
            ["Diff SHA-256", h("code", {}, draft.diff_sha256)],
            ["Bound to approval", badge(scan.digest_matches ? "completed" : "failed", scan.digest_matches ? "identical to the digest the approval binds" : "MISMATCH")],
          ]),
          h("details", {}, h("summary", {}, "PR body"), h("pre", { class: "prose" }, draft.body)),
          draft.patches.map((p) =>
            h("details", { class: "patch" }, h("summary", {}, `${p.path} · ${p.rule_id}`), h("pre", { class: "diff" }, diffLines(p.diff).map((line) => h("span", { class: `d-${line.kind}` }, line.text + "\n"))), p.checks.length ? h("p", { class: "meta" }, `checks passed: ${p.checks.join(", ")}`) : null),
          ),
        )
      : section("Draft pull request", empty("No validated patch; nothing to open.")),
  );
}

// --- wire -------------------------------------------------------------------- //

async function viewWire() {
  const w = await api.get("/api/wire");
  const refusals = w.refusals.map((r) => h("tr", { class: "row-bad" }, h("td", { class: "num" }, String(r.seq)), h("td", {}, fmtTime(r.recorded_at)), h("td", {}, r.actor), h("td", {}, humanize(r.action_type)), h("td", {}, r.guardrail || "—"), h("td", {}, r.reason)));
  const failed = w.failed_actions.map((a) =>
    h("tr", { class: "row-bad" }, h("td", {}, link(`#/incident/${encodeURIComponent(a.incident_id)}`, a.caption)), h("td", {}, humanize(a.action_type)), h("td", {}, a.target), h("td", {}, a.approved_by || "—"), h("td", {}, a.failure_reason)),
  );
  const executions = w.executions.map((x) =>
    h("tr", {}, h("td", {}, x.connector), h("td", {}, humanize(x.action_type)), h("td", {}, x.target), h("td", {}, badge(x.succeeded ? "executed" : "failed", x.succeeded ? "ok" : "failed")), h("td", {}, x.approved_by || "unattended"), h("td", {}, x.reference ? referenceLink(x.reference) : x.detail), h("td", {}, x.replayed ? "journal replay" : "")),
  );
  const r = w.remote;
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Wire & guardrails"),
    h("p", { class: "lede" }, "What actually left the process. A guardrail that fires silently is one nobody tunes, so every refusal is listed with its reason."),
    section("Guardrail refusals", table(["Seq", "When", "Actor", "Action", "Guardrail", "Reason"], refusals)),
    section("Failed actions inside runs", table(["Incident", "Action", "Target", "Approved by", "Reason"], failed)),
    section("Executions", table(["Connector", "Action", "Target", "Outcome", "Approved by", "Reference", ""], executions)),
    h(
      "div",
      { class: "grid-3" },
      section("Wazuh (EDR / firewall)", kv([["Isolated hosts", r.wazuh.isolated_hosts.join(", ") || "none"], ["Blocked addresses", r.wazuh.blocked_addresses.join(", ") || "none"], ["Commands", String(r.wazuh.commands)]])),
      section(
        "GitHub",
        kv([["Draft PRs", String(r.github.pulls.length)], ["Issues", String(r.github.issues.length)], ["Merges", String(r.github.merges)]]),
        h("ul", { class: "plain" }, r.github.pulls.map((p) => h("li", {}, `#${p.number} `, p.draft ? badge("info", "draft") : badge("failed", "NOT draft"), " ", p.title)), r.github.issues.map((i) => h("li", {}, `#${i.number} issue `, i.title))),
      ),
      section("Slack", kv([["Messages", String(r.slack.messages)]]), h("ul", { class: "plain" }, r.slack.recent.map((m) => h("li", { class: "clamp" }, m)))),
    ),
    section(
      "Router configuration",
      kv([
        ["Tenant", w.guardrails.tenant_id],
        ["Protected networks", w.guardrails.protected_networks.join(", ") || "none"],
        ["Protected hosts", w.guardrails.protected_hosts.join(", ") || "none"],
        ["Blast radius", `${w.guardrails.blast_radius_per_hour ?? "∞"} destructive actions / hour`],
        ["Capabilities", w.guardrails.capabilities.join(", ")],
        ["HTTP attempts", `${w.calls.total} (${Object.entries(w.calls.by_connector).map(([k, v]) => `${k}: ${v}`).join(", ") || "none"})`],
      ]),
    ),
  );
}

// --- audit ------------------------------------------------------------------- //

async function viewAudit(arg) {
  const after = Number(arg || 0) || 0;
  const [page, verify] = await Promise.all([api.get(`/api/audit?after=${after}&limit=100`), api.get("/api/audit/verify")]);
  const rows = page.items.map((r) =>
    h(
      "tr",
      {},
      h("td", { class: "num" }, String(r.seq)),
      h("td", {}, fmtTime(r.recorded_at)),
      h("td", {}, humanize(r.event_type)),
      h("td", {}, r.actor),
      h("td", {}, h("code", {}, shortId(r.subject_id, 12))),
      h("td", {}, h("details", {}, h("summary", {}, "payload"), h("pre", {}, JSON.stringify(r.payload, null, 2)))),
      h("td", {}, h("code", { title: r.row_hash }, shortId(r.row_hash, 12))),
    ),
  );
  const last = page.items.length ? page.items[page.items.length - 1].seq : after;
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Audit log"),
    h(
      "div",
      { class: `alert tone-${verify.ungated_executions.length ? "bad" : chainTone(verify)}` },
      h("strong", {}, { verified: "Chain verified. ", empty: "Chain empty. ", broken: "Chain BROKEN. " }[verify.status]),
      verify.status === "empty" ? "Nothing has been recorded in this workspace yet." : verify.summary,
      ` Ungated executions: ${verify.ungated_executions.length}.`,
      verify.findings.map((f) => h("div", {}, f)),
    ),
    table(["Seq", "Recorded", "Event", "Actor", "Subject", "Payload", "Row hash"], rows),
    h(
      "div",
      { class: "row" },
      after > 0 ? link(`#/audit/${Math.max(0, after - 100)}`, h("span", { class: "btn" }, "← Earlier")) : null,
      page.items.length === 100 ? link(`#/audit/${last}`, h("span", { class: "btn" }, "Later →")) : null,
      h("span", { class: "meta" }, `${page.total} rows for this tenant`),
    ),
  );
}

// --- evaluation -------------------------------------------------------------- //

async function viewEvaluation() {
  const e = await api.get("/api/evaluation");
  if (!e.available) {
    return h("div", { class: "stack" }, h("h1", {}, "Evaluation"), empty(`No evaluation artifact at ${e.path}. Run scripts/evaluate.py.`));
  }
  const gates = e.gates.map((g) => h("tr", { class: g.passed ? "" : "row-bad" }, h("td", {}, h("code", {}, g.gate)), h("td", {}, badge(g.passed ? "completed" : "failed", g.passed ? "pass" : "FAIL"))));
  const hd = e.headline;
  const families = Object.entries(hd.per_family_recall || {}).map(([family, recall]) => h("tr", {}, h("td", {}, family), h("td", { class: "num" }, fmtNumber(recall))));
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Evaluation report (F-12)"),
    h("p", { class: "lede" }, `Read verbatim from ${e.path}, written by scripts/evaluate.py — the one evaluation pipeline (PRD §9.3). ${e.passed}/${e.gates.length} gates pass.`),
    h(
      "div",
      { class: "tiles" },
      statTile("ROC-AUC", fmtNumber(hd.roc_auc), "F-03 needs ≥ 0.90", "good"),
      statTile("PR-AUC", fmtNumber(hd.pr_auc)),
      statTile("Recall", fmtNumber(hd.recall)),
      statTile("Precision", fmtNumber(hd.precision)),
      statTile("Alert reduction", fmtPercent(hd.alert_reduction), `${fmtPercent(hd.alert_reduction_at_soc_base_rate)} at a 1% attack rate`),
      statTile("Regret ratio (bandit)", fmtNumber(hd.response_policy_regret_ratio)),
    ),
    h("div", { class: "grid-2" }, alertReductionPanel(hd), regretPanel(hd.regret_curves)),
    h("div", { class: "grid-2" }, section("Gates", table(["Gate", "Result"], gates)), section("Per-family recall", table(["Family", "Recall"], families))),
  );
}

// F-12: "FP-reduction chart". Share of alerts that reach a human, raw feed vs after triage.
function alertReductionPanel(hd) {
  if (!Number.isFinite(hd.alert_reduction)) return section("Alert reduction (F-12)", empty("Not in this report."));
  const categories = ["Raw feed", "After triage (test split)"];
  const values = [1, 1 - hd.alert_reduction];
  if (Number.isFinite(hd.alert_reduction_at_soc_base_rate)) {
    categories.push("After triage, 1% attack rate");
    values.push(1 - hd.alert_reduction_at_soc_base_rate);
  }
  return section(
    "Alert reduction (F-12, §9.1)",
    barChart({
      title: "Alerts reaching a human",
      categories,
      series: [{ name: "Share of alerts reaching a human", values }],
      yMax: 1,
      percent: true,
      describe: "§9.1 target: at least 60% fewer than the raw feed",
    }),
    h("p", { class: "meta" }, "The test split is one-third attacks, so every attack caught is an alert a human must see; the 1%-rate bar projects the measured false-positive rate and recall onto a realistic feed."),
  );
}

// §9.1: "cumulative regret curve vs. an oracle policy". The oracle's regret is zero by definition.
function regretPanel(curves) {
  if (!curves) return section("Response-policy regret (§9.1)", empty("This report was written without --policy, or before curves were recorded; re-run scripts/evaluate.py --policy."));
  return section(
    "Response-policy regret (F-09, §9.1)",
    lineChart({
      title: "Cumulative regret vs. the oracle",
      x: curves.episode,
      series: [
        { name: "Learned policy", values: curves.policy },
        { name: "No learning", values: curves.no_learning },
      ],
      xLabel: "episode",
      yLabel: "cumulative regret",
      describe: "mean over the held-out seeds; the oracle is the zero line",
    }),
  );
}

// --- models -------------------------------------------------------------------- //

async function viewModels() {
  const m = await api.get("/api/models");
  const ae = m.autoencoder;
  const gnn = m.gnn;
  const pol = m.policy;
  const dif = m.diffusion;
  const epochs = (n) => Array.from({ length: n }, (_, i) => i + 1);
  return h(
    "div",
    { class: "stack" },
    h("h1", {}, "Models"),
    h(
      "p",
      { class: "lede" },
      `Every model below was trained by this server when it started (seed ${m.seed}), on the synthetic corpus. These are live training records; the acceptance numbers come from the one evaluation pipeline and are on the Evaluation page.`,
    ),
    h(
      "div",
      { class: "tiles" },
      statTile("Autoencoder", `${ae.epochs_run ?? "—"} epochs`, `${ae.n_parameters ?? "—"} parameters${ae.stopped_early ? " · stopped early" : ""}`),
      statTile("GNN top-10 precision", fmtNumber(gnn.evaluation.gnn_top_k_precision, 2), `features-only baseline ${fmtNumber(gnn.evaluation.features_only_top_k_precision, 2)} · target 0.80`, gnn.evaluation.gnn_top_k_precision >= 0.8 ? "good" : "warn"),
      statTile("Policy optimal-action rate", fmtPercent(pol.optimal_action_rate), `${pol.episodes} training episodes`),
      statTile("Policy decisions served", String(pol.live_decisions), "incidents the Containment Agent sent to the policy"),
      statTile("Diffusion study", dif.status, `${dif.levels_done}/${dif.levels_total} scarcity levels`, dif.status === "failed" ? "bad" : "neutral"),
    ),
    section(
      "Anomaly detection — denoising autoencoder (§5.5.2)",
      h("p", { class: "meta" }, `Trained on benign traffic only; paired with an Isolation Forest. Ensemble weights: ${ae.detectors.map((d) => `${d.name} ${fmtNumber(d.weight, 2)}`).join(", ")}.`),
      ae.available
        ? lineChart({
            title: "Reconstruction loss per epoch",
            x: epochs(ae.train_loss.length),
            series: [
              { name: "Training", values: ae.train_loss },
              { name: "Validation", values: ae.validation_loss },
            ].filter((sr) => sr.values.length === ae.train_loss.length),
            xLabel: "epoch",
            yLabel: "MSE",
            describe: "early stopping keeps the best validation epoch",
          })
        : empty("No training history recorded."),
    ),
    section(
      "Supply-chain risk — 2-layer GraphSAGE (§5.5.3, F-06)",
      h("p", { class: "meta" }, `${gnn.objective} objective, ${gnn.aggregation} aggregation, ${gnn.n_parameters} parameters, best epoch ${gnn.best_epoch}. ${gnn.evaluation.note}`),
      lineChart({
        title: "GNN loss per epoch",
        x: epochs(gnn.train_loss.length),
        series: [
          { name: "Training", values: gnn.train_loss },
          { name: "Validation", values: gnn.validation_loss },
        ].filter((sr) => sr.values.length === gnn.train_loss.length),
        xLabel: "epoch",
        yLabel: "loss",
      }),
      barChart({
        title: `Top-${gnn.evaluation.k} precision on the ${gnn.evaluation.test_nodes}-node test split`,
        categories: ["GNN (uses the graph)", "Features only (no graph)"],
        series: [{ name: "Top-10 precision", values: [gnn.evaluation.gnn_top_k_precision, gnn.evaluation.features_only_top_k_precision] }],
        yMax: 1,
        describe: "the gap is what the graph contributes",
      }),
    ),
    section(
      "Response policy — contextual bandit (§5.5.4, F-09)",
      h("p", { class: "meta" }, `Thompson sampling, trained on ${pol.episodes} simulated incidents in ${fmtSeconds(pol.seconds)}. Served ${pol.serving}. Action mix while learning: ${Object.entries(pol.action_counts).map(([k, v]) => `${humanize(k)} ${v}`).join(", ")}.`),
      lineChart({
        title: "Cumulative regret while learning",
        x: pol.curve.episode,
        series: [
          { name: "Learned policy", values: pol.curve.policy },
          { name: "No learning", values: pol.curve.no_learning },
        ],
        xLabel: "episode",
        yLabel: "cumulative regret",
        describe: `regret ratio ${fmtNumber(pol.regret_ratio, 3)} of the no-learning policy`,
      }),
    ),
    diffusionPanel(dif),
  );
}

function diffusionPanel(dif) {
  const intro = h("p", { class: "meta" }, "A class-conditional tabular diffusion model generates extra rare-attack rows; a classifier is trained with and without them. Reported whatever the sign: on this corpus augmentation is about neutral with full data and hurts when data is scarce, as Part 2 also measured.");
  if (dif.status === "failed") return section("Diffusion augmentation (§5.5.5)", intro, errorBox(new Error(dif.error)));
  if (!dif.results.length) {
    return section("Diffusion augmentation (§5.5.5)", intro, h("p", { class: "loading" }, dif.status === "pending" ? "Not started (runs in the background when the dashboard starts)." : "Training in the background… this page refreshes when you reopen it."));
  }
  return section(
    "Diffusion augmentation (§5.5.5)",
    intro,
    barChart({
      title: "Rare-attack recall, mean of four families",
      categories: dif.results.map((r) => r.level),
      series: [
        { name: "Real data only", values: dif.results.map((r) => r.before_macro) },
        { name: "With diffusion rows", values: dif.results.map((r) => r.after_macro) },
      ],
      yMax: 1,
      describe: `${dif.levels_done}/${dif.levels_total} levels done`,
    }),
    dif.loss_curve.length
      ? lineChart({
          title: "Diffusion training loss (all-data level)",
          x: Array.from({ length: dif.loss_curve.length }, (_, i) => i + 1),
          series: [{ name: "Training loss", values: dif.loss_curve }],
          xLabel: "epoch (sampled)",
          yLabel: "loss",
        })
      : null,
  );
}

function policyPanel(policy) {
  if (!policy) return null;
  if (!policy.fitted) {
    return section("Response policy (§5.5.4)", h("p", { class: "meta" }, `No policy fitted; the response follows triage: ${humanize(policy.response)}.`));
  }
  const options = Object.entries(policy.expected).sort((a, b) => b[1] - a[1]);
  return section(
    "Response policy (§5.5.4)",
    h(
      "p",
      { class: "decision-what" },
      `Policy chose ${humanize(policy.choice)}`,
      policy.floored ? ` → raised to ${humanize(policy.response)} by the triage floor` : "",
    ),
    kv([
      ["Confidence", fmtNumber(policy.confidence, 3)],
      ["Explored?", policy.exploratory ? "yes" : "no — greedy (posterior mean)"],
      ["Triage floor applied", policy.floored ? "yes: the policy may raise attention, never lower it" : "no"],
    ]),
    table(
      ["Option", "Expected reward"],
      options.map(([name, value]) => h("tr", { class: name === policy.choice ? "row-good" : "" }, h("td", {}, humanize(name)), h("td", { class: "num" }, fmtNumber(value, 3)))),
    ),
  );
}

const VIEWS = {
  overview: viewOverview,
  scenarios: viewScenarios,
  queue: viewQueue,
  incidents: viewIncidents,
  incident: viewIncident,
  "supply-chain": viewSupplyChain,
  "code-scan": viewCodeScan,
  wire: viewWire,
  audit: viewAudit,
  evaluation: viewEvaluation,
  models: viewModels,
};

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //

function start() {
  renderShell();
  route();
}

window.addEventListener("hashchange", route);

(async function boot() {
  if (!api.hasToken()) {
    renderLogin();
    return;
  }
  try {
    session = await api.get("/api/session");
    start();
  } catch {
    renderLogin();
  }
})();
