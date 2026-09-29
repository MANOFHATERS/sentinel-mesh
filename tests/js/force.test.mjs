import assert from "node:assert/strict";
import { test } from "node:test";

import { createSimulation } from "../../src/sentinel/dashboard/static/js/force.js";

function ring(n) {
  const nodes = Array.from({ length: n }, (_, i) => ({ id: `n${i}` }));
  const links = nodes.map((node, i) => ({ source: node.id, target: nodes[(i + 1) % n].id }));
  return { nodes, links };
}

function twoClusters() {
  const nodes = [];
  const links = [];
  for (const side of ["a", "b"]) {
    for (let i = 0; i < 10; i += 1) nodes.push({ id: `${side}${i}` });
    for (let i = 1; i < 10; i += 1) links.push({ source: `${side}0`, target: `${side}${i}` });
  }
  links.push({ source: "a0", target: "b0" });
  return { nodes, links };
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

test("layout is deterministic: same input, same picture", () => {
  const { nodes, links } = twoClusters();
  const one = createSimulation(nodes, links).run(300);
  const two = createSimulation(nodes, links).run(300);
  assert.deepEqual(
    one.nodes.map((n) => [n.x, n.y]),
    two.nodes.map((n) => [n.x, n.y]),
  );
});

test("positions stay finite and the simulation cools", () => {
  const { nodes, links } = twoClusters();
  const sim = createSimulation(nodes, links);
  sim.tick();
  const hot = sim.kineticEnergy();
  sim.run(400);
  for (const node of sim.nodes) {
    assert.ok(Number.isFinite(node.x) && Number.isFinite(node.y));
  }
  assert.ok(sim.alpha < 0.001 + 1e-9, `alpha ${sim.alpha}`);
  assert.ok(sim.kineticEnergy() < hot * 1e-2, "energy decays as alpha cools");
  assert.ok(sim.ticks <= 300, "the default schedule is about 300 ticks, as in d3");
});

test("linked nodes end closer than unlinked ones", () => {
  const { nodes, links } = twoClusters();
  const sim = createSimulation(nodes, links, { linkDistance: 30 }).run();
  const byId = new Map(sim.nodes.map((n) => [n.id, n]));
  const linked = links.map((l) => dist(byId.get(l.source), byId.get(l.target)));
  const all = [];
  for (let i = 0; i < sim.nodes.length; i += 1) {
    for (let j = i + 1; j < sim.nodes.length; j += 1) all.push(dist(sim.nodes[i], sim.nodes[j]));
  }
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  assert.ok(mean(linked) < mean(all), `${mean(linked)} vs ${mean(all)}`);
  // Two hubs joined by one edge: each cluster's leaves sit nearer their own hub.
  const a0 = byId.get("a0");
  const b0 = byId.get("b0");
  for (let i = 1; i < 10; i += 1) {
    assert.ok(dist(byId.get(`a${i}`), a0) < dist(byId.get(`a${i}`), b0));
    assert.ok(dist(byId.get(`b${i}`), b0) < dist(byId.get(`b${i}`), a0));
  }
});

test("the layout is centred on the canvas", () => {
  const { nodes, links } = ring(24);
  const sim = createSimulation(nodes, links, { width: 400, height: 300 }).run();
  const mx = sim.nodes.reduce((a, n) => a + n.x, 0) / sim.nodes.length;
  const my = sim.nodes.reduce((a, n) => a + n.y, 0) / sim.nodes.length;
  assert.ok(Math.abs(mx - 200) < 1e-6 && Math.abs(my - 150) < 1e-6);
});

test("collision keeps nodes from overlapping", () => {
  const nodes = Array.from({ length: 40 }, (_, i) => ({ id: `n${i}` }));
  const links = nodes.slice(1).map((n) => ({ source: "n0", target: n.id }));
  const radius = () => 6;
  const sim = createSimulation(nodes, links, {
    linkDistance: 1,
    charge: -1,
    radius,
    collidePadding: 1,
  }).run();
  let closest = Infinity;
  for (let i = 0; i < sim.nodes.length; i += 1) {
    for (let j = i + 1; j < sim.nodes.length; j += 1) {
      closest = Math.min(closest, dist(sim.nodes[i], sim.nodes[j]));
    }
  }
  // Springs pull every leaf onto the hub; only collision keeps them apart. Radii of
  // 6 plus padding 1 each make 14; allow the residual overlap of a cooled solver.
  assert.ok(closest > 10, `closest pair ${closest}`);
});

test("a link to an unknown node is an error, a self-loop is ignored", () => {
  assert.throws(
    () => createSimulation([{ id: "a" }], [{ source: "a", target: "ghost" }]),
    /unknown node/,
  );
  const sim = createSimulation([{ id: "a" }, { id: "b" }], [{ source: "a", target: "a" }]);
  assert.equal(sim.links.length, 0);
});

test("the input objects are not mutated", () => {
  const nodes = [{ id: "a" }, { id: "b" }];
  createSimulation(nodes, [{ source: "a", target: "b" }]).run(10);
  assert.deepEqual(nodes, [{ id: "a" }, { id: "b" }]);
});

test("a 500-node graph lays out quickly", () => {
  const nodes = Array.from({ length: 500 }, (_, i) => ({ id: `n${i}` }));
  const links = [];
  for (let i = 1; i < 500; i += 1) {
    links.push({ source: `n${Math.floor((i - 1) / 3)}`, target: `n${i}` });
  }
  const started = performance.now();
  const sim = createSimulation(nodes, links, { linkDistance: 22, charge: -26 }).run(300);
  const elapsed = performance.now() - started;
  assert.ok(sim.nodes.every((n) => Number.isFinite(n.x)));
  assert.ok(elapsed < 5000, `${elapsed.toFixed(0)} ms`);
});
