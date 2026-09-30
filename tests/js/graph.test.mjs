import assert from "node:assert/strict";
import { test } from "node:test";

import { installFakeDom } from "./fakedom.mjs";

installFakeDom();
const { buildDuration, revealOrder } = await import("../../src/sentinel/dashboard/static/js/graph.js");
const { barChart, lineChart } = await import("../../src/sentinel/dashboard/static/js/charts.js");

const links = (pairs) => pairs.map(([source, target]) => ({ source, target }));

test("the build starts at the highest-risk node and every node appears exactly once", () => {
  const nodes = [
    { id: "a", risk: 0.2 },
    { id: "b", risk: 0.9 },
    { id: "c", risk: 0.5 },
    { id: "d", risk: 0.1 },
  ];
  const order = revealOrder(nodes, links([["a", "b"], ["b", "c"], ["c", "d"]]));
  assert.equal(order[0], "b");
  assert.deepEqual([...order].sort(), ["a", "b", "c", "d"]);
});

test("each node after the first of its component appears next to one already on screen", () => {
  const nodes = Array.from({ length: 30 }, (_, i) => ({ id: `n${i}`, risk: ((i * 7) % 10) / 10 }));
  const pairs = [];
  for (let i = 1; i < 30; i += 1) pairs.push([`n${Math.floor((i - 1) / 2)}`, `n${i}`]);
  const order = revealOrder(nodes, links(pairs));
  const seen = new Set([order[0]]);
  for (const id of order.slice(1)) {
    const touching = pairs.some(([a, b]) => (a === id && seen.has(b)) || (b === id && seen.has(a)));
    assert.ok(touching, `${id} appeared with no visible neighbour`);
    seen.add(id);
  }
});

test("disconnected pieces are all shown, highest-risk component first", () => {
  const nodes = [
    { id: "x1", risk: 0.1 },
    { id: "x2", risk: 0.2 },
    { id: "y1", risk: 0.95 },
    { id: "y2", risk: 0.3 },
    { id: "lone", risk: 0.5 },
  ];
  const order = revealOrder(nodes, links([["x1", "x2"], ["y1", "y2"]]));
  assert.equal(order.length, 5);
  assert.deepEqual(order.slice(0, 2), ["y1", "y2"]);
  assert.ok(order.includes("lone"));
});

test("the order is deterministic, ignores unknown ends and self links, and tolerates missing risk", () => {
  const nodes = [{ id: "a" }, { id: "b", risk: Number.NaN }, { id: "c", risk: 0.4 }];
  const l = links([["a", "b"], ["b", "b"], ["b", "ghost"], ["c", "a"]]);
  assert.deepEqual(revealOrder(nodes, l), revealOrder(nodes, l));
  assert.equal(revealOrder(nodes, l).length, 3);
  assert.deepEqual(revealOrder([], []), []);
});

test("the build lasts long enough to watch a small graph and no longer than six seconds", () => {
  assert.equal(buildDuration(7), 2500);
  assert.equal(buildDuration(31), 31 * 260 > 6000 ? 6000 : 31 * 260);
  assert.equal(buildDuration(500), 6000);
  for (const n of [1, 10, 100, 1000]) {
    assert.ok(buildDuration(n) >= 2500 && buildDuration(n) <= 6000);
  }
});

// --- charts draw themselves ------------------------------------------------------ //

function find(node, predicate, out = []) {
  if (predicate(node)) out.push(node);
  for (const child of node.childNodes || []) find(child, predicate, out);
  return out;
}
const cls = (n) => n.getAttribute?.("class") || "";

test("bars carry a stagger class and the figure is revealed at once where there is no observer", () => {
  const figure = barChart({
    title: "T",
    categories: ["a", "b", "c"],
    series: [{ name: "s", values: [1, 2, 3] }],
  });
  assert.ok(/\bin\b/.test(cls(figure)), "no IntersectionObserver: shown immediately");
  const bars = find(figure, (n) => /\bbar\b/.test(cls(n)));
  assert.equal(bars.length, 3);
  assert.deepEqual(bars.map((b) => /\bd(\d+)\b/.exec(cls(b))[1]), ["0", "1", "2"]);
});

test("stagger classes stop at 15 so a large chart does not wait forever", () => {
  const categories = Array.from({ length: 30 }, (_, i) => `c${i}`);
  const figure = barChart({ title: "T", categories, series: [{ name: "s", values: categories.map((_, i) => i + 1) }] });
  const last = find(figure, (n) => /\bbar\b/.test(cls(n))).pop();
  assert.ok(/\bd15\b/.test(cls(last)));
});

test("lines are drawn with a unit path length so one CSS animation fits any curve", () => {
  const figure = lineChart({ title: "T", x: [1, 2, 3], series: [{ name: "s", values: [1, 3, 2] }] });
  const paths = find(figure, (n) => n.tagName === "PATH" && /\bline\b/.test(cls(n)));
  assert.equal(paths.length, 1);
  assert.equal(paths[0].getAttribute("pathLength"), "1");
});
