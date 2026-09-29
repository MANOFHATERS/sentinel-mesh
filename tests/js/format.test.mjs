import assert from "node:assert/strict";
import { test } from "node:test";

import {
  diffLines,
  fmtAgo,
  fmtPercent,
  fmtSeconds,
  fmtTime,
  humanize,
  parseRoute,
  safeHref,
  toneOf,
} from "../../src/sentinel/dashboard/static/js/format.js";

test("percent and missing values", () => {
  assert.equal(fmtPercent(0.6083), "60.8%");
  assert.equal(fmtPercent(null), "—");
  assert.equal(fmtPercent(Number.NaN), "—");
  assert.equal(fmtPercent(Infinity), "—");
});

test("seconds pick a readable unit at every scale", () => {
  assert.equal(fmtSeconds(0.0421), "42 ms");
  assert.equal(fmtSeconds(12.34), "12.3 s");
  assert.equal(fmtSeconds(185), "3m 05s");
  assert.equal(fmtSeconds(7322), "2h 02m");
  assert.equal(fmtSeconds(undefined), "—");
});

test("times render as UTC and bad input degrades to a dash", () => {
  assert.equal(fmtTime("2026-09-29T09:00:05+00:00"), "2026-09-29 09:00:05Z");
  assert.equal(fmtTime("not a date"), "—");
  assert.equal(fmtAgo("2026-09-29T09:00:00Z", Date.parse("2026-09-29T09:05:00Z")), "5m ago");
  assert.equal(fmtAgo("2026-09-29T09:00:00Z", Date.parse("2026-09-29T08:00:00Z")), "0s ago");
});

test("only http(s) and in-app routes become links", () => {
  assert.equal(
    safeHref("https://github.example/acme/billing/pull/1"),
    "https://github.example/acme/billing/pull/1",
  );
  assert.equal(safeHref("#/incident/abc"), "#/incident/abc");
  assert.equal(safeHref("javascript:alert(1)"), null);
  assert.equal(safeHref(" JAVASCRIPT:alert(1)"), null);
  assert.equal(safeHref("data:text/html,<script>alert(1)</script>"), null);
  assert.equal(safeHref("vbscript:x"), null);
  assert.equal(safeHref(42), null);
});

test("unknown statuses are neutral, never good", () => {
  assert.equal(toneOf("completed"), "good");
  assert.equal(toneOf("failed"), "bad");
  assert.equal(toneOf("likely_injection"), "bad");
  assert.equal(toneOf("something-new"), "neutral");
  assert.equal(humanize("awaiting_approval"), "awaiting approval");
});

test("diff lines are classified the way a reviewer reads them", () => {
  const lines = diffLines("--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n ctx\n-old\n+new\n");
  assert.deepEqual(
    lines.map((l) => l.kind),
    ["file", "file", "hunk", "ctx", "del", "add"],
  );
});

test("routes parse with and without arguments or queries", () => {
  assert.deepEqual(parseRoute(""), { name: "overview", arg: null });
  assert.deepEqual(parseRoute("#/queue"), { name: "queue", arg: null });
  assert.deepEqual(parseRoute("#/incident/abc-123"), { name: "incident", arg: "abc-123" });
  assert.deepEqual(parseRoute("#/incidents?status=failed"), { name: "incidents", arg: null });
  assert.deepEqual(parseRoute("#/supply-chain/advisory%3ACVE-2026-41822"), {
    name: "supply-chain",
    arg: "advisory:CVE-2026-41822",
  });
});
