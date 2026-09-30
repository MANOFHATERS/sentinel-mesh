import assert from "node:assert/strict";
import { test } from "node:test";

import { createSimulation } from "../../src/sentinel/dashboard/static/js/force.js";
import {
  MAX_SCALE,
  MIN_SCALE,
  clampBox,
  fitBox,
  panBox,
  scaleOf,
  viewBoxString,
  zoomBox,
} from "../../src/sentinel/dashboard/static/js/viewport.js";

const base = { x: 0, y: 0, width: 800, height: 600 };
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("zooming in shrinks the box and keeps the point under the pointer fixed", () => {
  const zoomed = zoomBox(base, 2, 200, 150);
  close(zoomed.width, 400);
  close(zoomed.height, 300);
  // (200,150) sat 25% across and 25% down the view; it still does.
  close((200 - zoomed.x) / zoomed.width, (200 - base.x) / base.width);
  close((150 - zoomed.y) / zoomed.height, (150 - base.y) / base.height);
});

test("zoom in then out by the same factor is the identity", () => {
  const there = zoomBox(base, 1.4, 500, 100);
  const back = zoomBox(there, 1 / 1.4, 500, 100);
  for (const key of ["x", "y", "width", "height"]) close(back[key], base[key]);
});

test("zoom is clamped to the allowed range about the same centre", () => {
  const tooFar = clampBox(zoomBox(base, 100, 400, 300), base);
  close(scaleOf(tooFar, base), MAX_SCALE);
  const tooWide = clampBox(zoomBox(base, 0.001, 400, 300), base);
  close(scaleOf(tooWide, base), MIN_SCALE);
  const centre = { x: tooFar.x + tooFar.width / 2, y: tooFar.y + tooFar.height / 2 };
  assert.ok(Math.abs(centre.x - 400) < 1 && Math.abs(centre.y - 300) < 1);
  // inside the range nothing changes
  assert.equal(clampBox(base, base), base);
});

test("pan moves the view opposite to the drag and keeps its size", () => {
  const moved = panBox(base, 30, -10);
  assert.deepEqual(moved, { x: -30, y: 10, width: 800, height: 600 });
});

test("fit never shrinks the canvas below a minimum, and centres a small drawing", () => {
  const tiny = fitBox({ x: 100, y: 100, width: 100, height: 50 });
  assert.equal(tiny.width, 560);
  assert.equal(tiny.height, 380);
  close(tiny.x + tiny.width / 2, 150);
  close(tiny.y + tiny.height / 2, 125);
  const big = fitBox({ x: 0, y: 0, width: 900, height: 700 });
  assert.deepEqual(big, { x: 0, y: 0, width: 900, height: 700 });
});

test("viewBox string is four fixed-point numbers", () => {
  assert.equal(viewBoxString({ x: 1.25, y: -2, width: 800, height: 600 }), "1.3 -2.0 800.0 600.0");
});

// --- the animated build must land on the synchronous layout -------------------- //

function sample() {
  const nodes = Array.from({ length: 40 }, (_, i) => ({ id: `n${i}` }));
  const links = [];
  for (let i = 1; i < 40; i += 1) links.push({ source: `n${Math.floor((i - 1) / 2)}`, target: `n${i}` });
  return { nodes, links };
}

test("stepping tick() until settled gives exactly the picture run() gives", () => {
  const { nodes, links } = sample();
  const whole = createSimulation(nodes, links).run(300);
  const stepped = createSimulation(nodes, links);
  let guard = 0;
  while (!stepped.settled(300) && guard < 1000) {
    for (let k = 0; k < 7 && !stepped.settled(300); k += 1) stepped.tick(); // a few per "frame"
    guard += 1;
  }
  assert.ok(stepped.settled(300));
  assert.deepEqual(stepped.nodes.map((n) => [n.x, n.y]), whole.nodes.map((n) => [n.x, n.y]));
});

test("reset returns to the starting arrangement, so a rebuild ends on the same picture", () => {
  const { nodes, links } = sample();
  const sim = createSimulation(nodes, links).run(300);
  const first = sim.nodes.map((n) => [n.x, n.y]);
  sim.reset();
  assert.equal(sim.ticks, 0);
  assert.equal(sim.alpha, 1);
  assert.ok(!sim.settled(300));
  sim.run(300);
  assert.deepEqual(sim.nodes.map((n) => [n.x, n.y]), first);
});
