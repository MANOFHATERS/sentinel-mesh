// Pure formatting helpers. No DOM, so `node --test` covers them directly.

export function fmtPercent(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Number(value).toFixed(digits);
}

export function fmtSeconds(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  if (value < 120) return `${value.toFixed(1)} s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value - minutes * 60);
  if (minutes < 120) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export function fmtTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toISOString().replace("T", " ").slice(0, 19) + "Z";
}

export function fmtAgo(iso, now = Date.now()) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const seconds = Math.max(0, (now - then) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

export function shortId(id, length = 8) {
  if (!id) return "—";
  return String(id).slice(0, length);
}

export function humanize(value) {
  if (value === null || value === undefined) return "—";
  return String(value).replaceAll("_", " ");
}

// Status -> tone used by badges. Anything unknown is neutral, never "good".
const TONES = {
  awaiting_approval: "warn",
  running: "info",
  completed: "good",
  dismissed: "muted",
  failed: "bad",
  pending: "warn",
  approved: "info",
  executed: "good",
  rejected: "muted",
  expired: "muted",
  escalate: "warn",
  monitor: "info",
  auto_dismiss: "muted",
  critical: "bad",
  high: "bad",
  medium: "warn",
  low: "info",
  info: "muted",
  likely_injection: "bad",
  suspicious: "warn",
  clean: "muted",
};

export function toneOf(value) {
  return TONES[value] || "neutral";
}

// Only these URL schemes may become an href. A reference URL comes from a remote
// system, and "javascript:" in an anchor is script execution on click.
export function safeHref(url) {
  if (typeof url !== "string") return null;
  const trimmed = url.trim();
  if (trimmed.startsWith("#/")) return trimmed;
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : null;
  } catch {
    return null;
  }
}

// Classify unified-diff lines for colouring. Returns [{kind, text}].
export function diffLines(diff) {
  return String(diff || "")
    .split("\n")
    .filter((line, index, all) => !(line === "" && index === all.length - 1))
    .map((text) => {
      let kind = "ctx";
      if (text.startsWith("+++") || text.startsWith("---")) kind = "file";
      else if (text.startsWith("@@")) kind = "hunk";
      else if (text.startsWith("+")) kind = "add";
      else if (text.startsWith("-")) kind = "del";
      return { kind, text };
    });
}

// Parse the hash route: "#/incident/abc" -> {name: "incident", arg: "abc"}. A query
// ("#/incidents?status=failed") is not part of the route; views read it themselves.
export function parseRoute(hash) {
  const raw = String(hash || "").replace(/^#\/?/, "").split("?")[0];
  const [name, ...rest] = raw.split("/");
  const arg = rest.length ? decodeURIComponent(rest.join("/")) : null;
  return { name: name || "overview", arg };
}
