// A force-directed layout with d3-force's semantics, in ~150 lines.
//
// PRD Section 5.6 names d3-force. This is a re-implementation of the parts the
// supply-chain map uses, for the same reason the NN engine is NumPy rather than
// PyTorch: the whole repository runs offline in one command, and a front end that
// needs an npm install and a bundler breaks that. The semantics follow d3-force v3
// so a swap to the library is mechanical:
//
//   * alpha cools from 1 toward alphaTarget by alphaDecay = 1 - alphaMin^(1/300),
//     so a default simulation runs ~300 ticks;
//   * velocityDecay 0.4 is friction applied after the forces each tick;
//   * link force: spring to `distance`, strength 1/min(degree), and the correction
//     split between endpoints by degree (the "bias"), exactly as forceLink does;
//   * many-body: charge `strength` (negative repels) with distanceMin / distanceMax,
//     computed pairwise — exact rather than Barnes–Hut, which is fine at 500 nodes;
//   * collide: pairwise separation to a per-node radius;
//   * center: translates the mean position onto the centre without adding energy.
//
// Initial positions use d3's phyllotaxis arrangement, so the layout is fully
// deterministic: same nodes and links in, same picture out. `node --test` checks
// that, and that the result is finite, centred, collision-free and pulls linked
// nodes together.

const INITIAL_RADIUS = 10;
const INITIAL_ANGLE = Math.PI * (3 - Math.sqrt(5));

export function createSimulation(inputNodes, inputLinks, options = {}) {
  const {
    width = 800,
    height = 600,
    linkDistance = 40,
    charge = -60,
    distanceMin = 1,
    distanceMax = Infinity,
    radius = () => 6,
    collidePadding = 1,
    alphaMin = 0.001,
    velocityDecay = 0.4,
  } = options;
  const alphaDecay = 1 - Math.pow(alphaMin, 1 / 300);
  const cx = width / 2;
  const cy = height / 2;

  const nodes = inputNodes.map((node, index) => {
    const r = INITIAL_RADIUS * Math.sqrt(0.5 + index);
    const angle = index * INITIAL_ANGLE;
    return { ...node, index, x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), vx: 0, vy: 0 };
  });
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const links = [];
  for (const link of inputLinks) {
    const source = byId.get(link.source);
    const target = byId.get(link.target);
    if (!source || !target) throw new Error(`link references unknown node: ${link.source} -> ${link.target}`);
    if (source === target) continue;
    links.push({ ...link, source, target });
  }
  const degree = new Map(nodes.map((node) => [node, 0]));
  for (const link of links) {
    degree.set(link.source, degree.get(link.source) + 1);
    degree.set(link.target, degree.get(link.target) + 1);
  }
  const linkStrength = links.map((l) => 1 / Math.min(degree.get(l.source), degree.get(l.target)));
  const linkBias = links.map((l) => degree.get(l.source) / (degree.get(l.source) + degree.get(l.target)));
  const radii = nodes.map((node) => radius(node) + collidePadding);
  const minSq = distanceMin * distanceMin;
  const maxSq = distanceMax * distanceMax;

  let alpha = 1;
  let ticks = 0;

  // A tiny deterministic jiggle for coincident points, as d3's jiggle() but seeded.
  let seed = 0x9e3779b9;
  function jiggle() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed / 0x100000000 - 0.5) * 1e-6;
  }

  function applyLinks() {
    for (let i = 0; i < links.length; i += 1) {
      const { source, target } = links[i];
      let dx = target.x + target.vx - source.x - source.vx || jiggle();
      let dy = target.y + target.vy - source.y - source.vy || jiggle();
      let length = Math.sqrt(dx * dx + dy * dy);
      length = ((length - linkDistance) / length) * alpha * linkStrength[i];
      dx *= length;
      dy *= length;
      const bias = linkBias[i];
      target.vx -= dx * bias;
      target.vy -= dy * bias;
      source.vx += dx * (1 - bias);
      source.vy += dy * (1 - bias);
    }
  }

  function applyCharge() {
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j];
        let dx = b.x - a.x || jiggle();
        let dy = b.y - a.y || jiggle();
        let l = dx * dx + dy * dy;
        if (l >= maxSq) continue;
        if (l < minSq) l = Math.sqrt(minSq * l);
        const w = (charge * alpha) / l;
        a.vx += dx * w;
        a.vy += dy * w;
        b.vx -= dx * w;
        b.vy -= dy * w;
      }
    }
  }

  function applyCollide() {
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j];
        const r = radii[i] + radii[j];
        let dx = a.x + a.vx - b.x - b.vx || jiggle();
        let dy = a.y + a.vy - b.y - b.vy || jiggle();
        const l2 = dx * dx + dy * dy;
        if (l2 >= r * r) continue;
        const l = Math.sqrt(l2);
        const push = ((r - l) / l) * 0.7;
        dx *= push;
        dy *= push;
        const share = (radii[j] * radii[j]) / (radii[i] * radii[i] + radii[j] * radii[j]);
        a.vx += dx * share;
        a.vy += dy * share;
        b.vx -= dx * (1 - share);
        b.vy -= dy * (1 - share);
      }
    }
  }

  function applyCenter() {
    if (!nodes.length) return;
    let sx = 0;
    let sy = 0;
    for (const node of nodes) {
      sx += node.x;
      sy += node.y;
    }
    sx = sx / nodes.length - cx;
    sy = sy / nodes.length - cy;
    for (const node of nodes) {
      node.x -= sx;
      node.y -= sy;
    }
  }

  function tick() {
    alpha += (0 - alpha) * alphaDecay;
    applyLinks();
    applyCharge();
    applyCollide();
    for (const node of nodes) {
      node.vx *= 1 - velocityDecay;
      node.vy *= 1 - velocityDecay;
      node.x += node.vx;
      node.y += node.vy;
    }
    applyCenter();
    ticks += 1;
    return alpha;
  }

  function run(maxTicks = 300) {
    while (alpha >= alphaMin && ticks < maxTicks) tick();
    return api;
  }

  function kineticEnergy() {
    return nodes.reduce((sum, n) => sum + n.vx * n.vx + n.vy * n.vy, 0);
  }

  function bounds(pad = 0) {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    nodes.forEach((node, i) => {
      x0 = Math.min(x0, node.x - radii[i]);
      y0 = Math.min(y0, node.y - radii[i]);
      x1 = Math.max(x1, node.x + radii[i]);
      y1 = Math.max(y1, node.y + radii[i]);
    });
    if (!nodes.length) return { x: 0, y: 0, width, height };
    return { x: x0 - pad, y: y0 - pad, width: x1 - x0 + 2 * pad, height: y1 - y0 + 2 * pad };
  }

  // Back to the deterministic starting arrangement, so a rebuild ends on the same picture.
  function reset() {
    nodes.forEach((node, index) => {
      const r = INITIAL_RADIUS * Math.sqrt(0.5 + index);
      const angle = index * INITIAL_ANGLE;
      node.x = cx + r * Math.cos(angle);
      node.y = cy + r * Math.sin(angle);
      node.vx = 0;
      node.vy = 0;
    });
    alpha = 1;
    ticks = 0;
    seed = 0x9e3779b9;
    return api;
  }

  const api = {
    nodes,
    links,
    tick,
    run,
    reset,
    bounds,
    kineticEnergy,
    // True once run(maxTicks) would stop: cooled below alphaMin, or out of ticks. Lets a
    // caller animate the layout by calling tick() a few times per frame until it settles,
    // and land on exactly the picture run() produces.
    settled(maxTicks = 300) {
      return alpha < alphaMin || ticks >= maxTicks;
    },
    get alpha() {
      return alpha;
    },
    get ticks() {
      return ticks;
    },
  };
  return api;
}
