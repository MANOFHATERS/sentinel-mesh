import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

import { installFakeDom } from "./fakedom.mjs";

installFakeDom();
const { barChart, endLabelPositions, formatTick, formatValue, lineChart, nearestIndex, niceTicks } = await import(
  "../../src/sentinel/dashboard/static/js/charts.js"
);

beforeEach(() => installFakeDom());

function find(node, predicate, out = []) {
  if (predicate(node)) out.push(node);
  for (const child of node.childNodes || []) find(child, predicate, out);
  return out;
}

test("nice ticks cover the range with 1-2-5 steps", () => {
  assert.deepEqual(niceTicks(0, 1), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  const ticks = niceTicks(3, 287);
  assert.ok(ticks[0] <= 3 && ticks[ticks.length - 1] >= 287);
  const step = ticks[1] - ticks[0];
  assert.ok([1, 2, 5].some((m) => Number((step / 10 ** Math.floor(Math.log10(step))).toFixed(6)) === m));
  assert.deepEqual(niceTicks(5, 5).length > 1, true, "a flat series still gets an axis");
  assert.deepEqual(niceTicks(Number.NaN, 1), [0, 1]);
});

test("nearest index finds the closest x", () => {
  const xs = [1, 2, 4, 8, 16];
  assert.equal(nearestIndex(xs, 0), 0);
  assert.equal(nearestIndex(xs, 5.9), 2);
  assert.equal(nearestIndex(xs, 6.1), 3);
  assert.equal(nearestIndex(xs, 100), 4);
  assert.equal(nearestIndex([], 3), -1);
});

test("values format by magnitude, and missing values are a dash", () => {
  assert.equal(formatValue(0.12345), "0.123");
  assert.equal(formatValue(123.456), "123.5");
  assert.equal(formatValue(2152.4), "2152");
  assert.equal(formatValue(null), "—");
  assert.equal(formatValue(Number.NaN), "—");
});

test("a two-series line chart has a legend, two lines, end labels and a table", () => {
  const figure = lineChart({
    title: "Regret",
    x: [1, 2, 3, 4],
    series: [
      { name: "Learned policy", values: [0, 1, 1.5, 1.7] },
      { name: "No learning", values: [0, 2, 4, 6] },
    ],
    xLabel: "episode",
  });
  const lines = find(figure, (n) => n.tagName === "PATH" && /line/.test(n.getAttribute?.("class") || ""));
  assert.equal(lines.length, 2);
  for (const path of lines) assert.ok(!/NaN/.test(path.getAttribute("d")), "no NaN in geometry");
  assert.equal(find(figure, (n) => (n.getAttribute?.("class") || "") === "chart-legend").length, 1);
  assert.ok(figure.textContent.includes("Learned policy"));
  const rows = find(figure, (n) => n.tagName === "TR");
  assert.equal(rows.length, 1 + 4, "header plus one row per point");
  // Identity is never colour-alone: the series name is text next to its line.
  assert.equal(find(figure, (n) => (n.getAttribute?.("class") || "") === "end-label").length, 2);
});

test("a single series needs no legend box", () => {
  const figure = lineChart({ title: "Loss", x: [1, 2, 3], series: [{ name: "Training", values: [3, 2, 1] }] });
  assert.equal(find(figure, (n) => (n.getAttribute?.("class") || "") === "chart-legend").length, 0);
});

test("bars are drawn per category and series, and missing values are skipped", () => {
  const figure = barChart({
    title: "Recall",
    categories: ["all data", "25 per family"],
    series: [
      { name: "Real data only", values: [0.9, 0.8] },
      { name: "With diffusion rows", values: [0.91, Number.NaN] },
    ],
    yMax: 1,
  });
  const bars = find(figure, (n) => n.tagName === "RECT" && /bar/.test(n.getAttribute?.("class") || ""));
  assert.equal(bars.length, 3);
  for (const bar of bars) assert.ok(Number(bar.getAttribute("height")) > 0);
  assert.ok(figure.textContent.includes("—"), "the missing value shows as a dash in the table");
});

test("percent bars label as percentages", () => {
  const figure = barChart({ title: "Share", categories: ["Raw feed"], series: [{ name: "Share", values: [1] }], percent: true, yMax: 1 });
  assert.ok(figure.textContent.includes("100.0%"));
});

test("tick labels are clean at every magnitude", () => {
  assert.equal(formatTick(140), "140");
  assert.equal(formatTick(0.6000000000000001), "0.6");
  assert.equal(formatTick(Number.NaN), "—");
});

test("end labels that would collide are pushed apart, keeping their series", () => {
  const placed = endLabelPositions([100, 104, 20], 13);
  const byIndex = Object.fromEntries(placed.map((p) => [p.index, p.y]));
  assert.equal(byIndex[2], 20);
  assert.equal(byIndex[0], 100);
  assert.equal(byIndex[1], 113);
  const ys = placed.map((p) => p.y).sort((a, b) => a - b);
  for (let i = 1; i < ys.length; i += 1) assert.ok(ys[i] - ys[i - 1] >= 13);
});
