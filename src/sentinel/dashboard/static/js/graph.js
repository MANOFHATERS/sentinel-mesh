// Renders the supply-chain map as SVG from the force layout.
//
// The layout is computed first, then the map is *grown* in front of the reader: it starts
// at the highest-risk node and adds one node at a time along the graph's own connections
// (each new node is linked to one already on screen), so the exposure spreads outward the
// way the risk does. Each node pops in and each new link flashes as it appears. It can be
// zoomed (buttons, mouse wheel, + / - keys) and panned (drag), and hovering or focusing a
// node lights up its connections and dims the rest.
//
// Nothing here sets inline styles or markup: attributes and classes only, as the
// Content-Security-Policy and dom.js require.

import { h, s } from "./dom.js";
import { createSimulation } from "./force.js";
import { MAX_SCALE, MIN_SCALE, boxCentre, clampBox, fitBox, panBox, scaleOf, viewBoxString, zoomBox } from "./viewport.js";

export function nodeRadius(node) {
  const risk = Number.isFinite(node.risk) ? node.risk : 0;
  const base = node.kind === "organization" ? 7 : node.kind === "vendor" ? 5.5 : 4;
  return base + risk * 7;
}

// Edge keys on the highlighted exposure paths, "source->target".
export function pathEdges(paths) {
  const keys = new Set();
  for (const path of paths || []) {
    for (let i = 0; i + 1 < path.length; i += 1) keys.add(`${path[i]}->${path[i + 1]}`);
  }
  return keys;
}

// The order in which to grow the map: breadth-first from the highest-risk node, so every
// node after the first of its component appears next to one already shown. Components are
// started highest-risk first. Deterministic: ties break on id. links: [{source, target}] ids.
export function revealOrder(nodes, links) {
  const risk = new Map(nodes.map((n) => [n.id, Number.isFinite(n.risk) ? n.risk : 0]));
  const adjacent = new Map(nodes.map((n) => [n.id, []]));
  for (const { source, target } of links) {
    if (!adjacent.has(source) || !adjacent.has(target) || source === target) continue;
    adjacent.get(source).push(target);
    adjacent.get(target).push(source);
  }
  const byRisk = (a, b) => risk.get(b) - risk.get(a) || (a < b ? -1 : a > b ? 1 : 0);
  const seeds = nodes.map((n) => n.id).sort(byRisk);
  const seen = new Set();
  const order = [];
  for (const seed of seeds) {
    if (seen.has(seed)) continue;
    seen.add(seed);
    const queue = [seed];
    for (let head = 0; head < queue.length; head += 1) {
      const id = queue[head];
      order.push(id);
      for (const next of [...new Set(adjacent.get(id))].sort(byRisk)) {
        if (!seen.has(next)) {
          seen.add(next);
          queue.push(next);
        }
      }
    }
  }
  return order;
}

// How long the build takes: long enough to watch a small graph grow, bounded for a big one.
export function buildDuration(nodeCount) {
  return Math.min(6000, Math.max(2500, nodeCount * 260));
}

const WIDTH = 900;
const HEIGHT = 620;
const MAX_TICKS = 300;
const ZOOM_STEP = 1.4;
const LABELS_ALL_AT = 2.2;

function prefersReducedMotion() {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

// Returns {element, setSelected, rebuild, destroy}. ``element`` holds the controls and the SVG.
export function renderGraph(data, { onSelect, selected, animate = true } = {}) {
  const many = data.nodes.length > 150;
  const sim = createSimulation(data.nodes, data.edges, {
    width: WIDTH,
    height: HEIGHT,
    linkDistance: many ? 22 : 46,
    charge: many ? -26 : -90,
    radius: nodeRadius,
  }).run(MAX_TICKS);
  const highlighted = pathEdges(data.highlight);
  const onPath = new Set((data.highlight || []).flat());
  const advisoryPackage = data.advisory ? data.advisory.package_id : null;

  const svg = s("svg", {
    class: `graph-svg${many ? " many" : ""}`,
    role: "img",
    tabindex: "0",
    "aria-label": `Supply-chain graph: ${data.nodes.length} nodes, ${data.edges.length} edges. Use plus and minus to zoom, drag to pan.`,
  });
  const defs = s("defs", {});
  for (const [id, cls] of [["arrow", "arrow"], ["arrow-hot", "arrow hot"]]) {
    defs.appendChild(
      s(
        "marker",
        { id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse" },
        s("path", { d: "M 0 0 L 10 5 L 0 10 z", class: cls }),
      ),
    );
  }
  svg.appendChild(defs);

  // --- elements, created once and repositioned as the layout runs ---------------- //
  const edgeLayer = s("g", { class: "edges" });
  const hotLayer = s("g", { class: "edges hot" });
  const adjacency = new Map(sim.nodes.map((n) => [n.id, { edges: [], neighbours: new Set() }]));
  const edgeEls = sim.links.map((link, index) => {
    const key = `${link.source.id}->${link.target.id}`;
    const hot = highlighted.has(key);
    const line = s("line", {
      class: `edge ${link.kind}${hot ? " hot" : ""}`,
      "marker-end": hot ? "url(#arrow-hot)" : many ? null : "url(#arrow)",
    });
    (hot ? hotLayer : edgeLayer).appendChild(line);
    adjacency.get(link.source.id).edges.push(index);
    adjacency.get(link.target.id).edges.push(index);
    adjacency.get(link.source.id).neighbours.add(link.target.id);
    adjacency.get(link.target.id).neighbours.add(link.source.id);
    return line;
  });
  svg.appendChild(edgeLayer);
  svg.appendChild(hotLayer);

  const nodeLayer = s("g", { class: "nodes" });
  let hoverId = null;
  let selectedId = selected || null;
  let dragMoved = false;

  const nodeEls = sim.nodes.map((node) => {
    const classes = ["node", node.kind];
    if (node.intrinsic) classes.push("intrinsic");
    if (onPath.has(node.id)) classes.push("on-path");
    if (node.id === advisoryPackage) classes.push("advisory");
    const group = s(
      "g",
      {
        class: classes.join(" "),
        tabindex: "0",
        role: "button",
        "aria-label": `${node.id}, ${node.kind}, risk ${Number(node.risk).toFixed(3)}`,
        onclick: () => {
          if (dragMoved) return; // the end of a pan, not a click
          if (onSelect) onSelect(node.id);
        },
        onpointerenter: () => focusOn(node.id, true),
        onpointerleave: () => focusOn(null, true),
        onfocus: () => focusOn(node.id, true),
        onblur: () => focusOn(null, true),
        onkeydown: (event) => {
          if ((event.key === "Enter" || event.key === " ") && onSelect) {
            event.preventDefault();
            onSelect(node.id);
          }
        },
      },
      s("title", {}, `${node.id} (${node.kind}) risk ${Number(node.risk).toFixed(3)}${node.intrinsic ? " · risk source" : ""}`),
      s("circle", { r: nodeRadius(node).toFixed(2) }),
    );
    const always = !many || onPath.has(node.id) || node.kind === "organization";
    group.appendChild(s("text", { class: always ? "" : "minor", x: (nodeRadius(node) + 3).toFixed(1), y: "3.5" }, node.id));
    nodeLayer.appendChild(group);
    return group;
  });
  svg.appendChild(nodeLayer);

  // --- connections light up ------------------------------------------------------ //
  function focusOn(id, isHover) {
    if (isHover) hoverId = id;
    const target = hoverId || selectedId;
    svg.classList.toggle("focus-mode", Boolean(target));
    const around = target ? adjacency.get(target) : null;
    const incident = new Set(around ? around.edges : []);
    edgeEls.forEach((line, i) => line.classList.toggle("incident", incident.has(i)));
    sim.nodes.forEach((node, i) => {
      nodeEls[i].classList.toggle("nbr", Boolean(around) && (around.neighbours.has(node.id) || node.id === target));
      nodeEls[i].classList.toggle("selected", node.id === selectedId);
    });
  }

  // --- drawing the current layout -------------------------------------------------- //
  function paint() {
    sim.links.forEach((link, i) => {
      const dx = link.target.x - link.source.x;
      const dy = link.target.y - link.source.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const rt = nodeRadius(link.target) + 2; // the arrowhead sits on the target's rim
      const line = edgeEls[i];
      line.setAttribute("x1", link.source.x.toFixed(2));
      line.setAttribute("y1", link.source.y.toFixed(2));
      line.setAttribute("x2", (link.target.x - (dx / len) * rt).toFixed(2));
      line.setAttribute("y2", (link.target.y - (dy / len) * rt).toFixed(2));
    });
    sim.nodes.forEach((node, i) => nodeEls[i].setAttribute("transform", `translate(${node.x.toFixed(2)},${node.y.toFixed(2)})`));
  }

  // --- view: zoom and pan --------------------------------------------------------- //
  let base = fitBox(sim.bounds(24));
  let box = base;
  let autoFit = true;

  function applyView() {
    svg.setAttribute("viewBox", viewBoxString(box));
    const scale = scaleOf(box, base);
    svg.classList.toggle("zoomed", scale >= LABELS_ALL_AT);
    readout.textContent = `${Math.round(scale * 100)}%`;
    zoomInBtn.disabled = scale >= MAX_SCALE - 1e-6;
    zoomOutBtn.disabled = scale <= MIN_SCALE + 1e-6;
  }

  function toGraph(clientX, clientY) {
    const matrix = svg.getScreenCTM && svg.getScreenCTM();
    if (!matrix) return boxCentre(box);
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const p = point.matrixTransform(matrix.inverse());
    return { x: p.x, y: p.y };
  }

  function zoom(factor, about) {
    autoFit = false;
    const at = about || boxCentre(box);
    box = clampBox(zoomBox(box, factor, at.x, at.y), base);
    applyView();
  }

  function fit() {
    autoFit = false;
    base = fitBox(sim.bounds(24));
    box = base;
    applyView();
  }

  svg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      zoom(Math.exp(-event.deltaY * 0.0016), toGraph(event.clientX, event.clientY));
    },
    { passive: false },
  );

  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    autoFit = false;
    dragMoved = false;
    let last = { x: event.clientX, y: event.clientY };
    const start = { ...last };
    const move = (e) => {
      if (!dragMoved && Math.hypot(e.clientX - start.x, e.clientY - start.y) < 4) return;
      dragMoved = true;
      svg.classList.add("panning");
      const matrix = svg.getScreenCTM();
      const scale = matrix ? matrix.a : 1; // screen pixels per graph unit
      box = panBox(box, (e.clientX - last.x) / scale, (e.clientY - last.y) / scale);
      last = { x: e.clientX, y: e.clientY };
      applyView();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      svg.classList.remove("panning");
      // dragMoved stays true through the click that follows a drag, then clears.
      setTimeout(() => (dragMoved = false), 0);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  });

  svg.addEventListener("keydown", (event) => {
    if (event.target !== svg) return;
    if (event.key === "+" || event.key === "=") zoom(ZOOM_STEP);
    else if (event.key === "-" || event.key === "_") zoom(1 / ZOOM_STEP);
    else if (event.key === "0") fit();
    else return;
    event.preventDefault();
  });

  // --- controls ------------------------------------------------------------------- //
  const zoomInBtn = h("button", { class: "map-btn", type: "button", "aria-label": "Zoom in", title: "Zoom in", onclick: () => zoom(ZOOM_STEP) }, "+");
  const zoomOutBtn = h("button", { class: "map-btn", type: "button", "aria-label": "Zoom out", title: "Zoom out", onclick: () => zoom(1 / ZOOM_STEP) }, "−");
  const fitBtn = h("button", { class: "map-btn wide", type: "button", "aria-label": "Fit the whole graph", title: "Fit to view", onclick: fit }, "Fit");
  const rebuildBtn = h("button", { class: "map-btn wide", type: "button", "aria-label": "Rebuild the graph step by step", title: "Watch the graph build again", onclick: () => rebuild() }, "↻ Rebuild");
  const skipBtn = h("button", { class: "map-btn wide", type: "button", "aria-label": "Skip the build animation", title: "Skip to the finished graph", onclick: () => finishBuild(), hidden: true }, "Skip ⏭");
  const readout = h("span", { class: "map-readout", "aria-live": "off" }, "100%");
  const status = h("span", { class: "map-status", role: "status" }, "");
  const controls = h("div", { class: "map-controls" }, skipBtn, zoomInBtn, zoomOutBtn, fitBtn, rebuildBtn, readout);
  const element = h("div", { class: "graph-map" }, controls, status, svg);

  // --- the step-by-step build ------------------------------------------------------ //
  const order = revealOrder(data.nodes, sim.links.map((l) => ({ source: l.source.id, target: l.target.id })));
  const rank = new Map(order.map((id, i) => [id, i]));
  const nodeIndex = new Map(sim.nodes.map((n, i) => [n.id, i]));
  // A link appears with whichever of its two ends comes later in the order.
  const edgesAtStep = order.map(() => []);
  sim.links.forEach((l, i) => edgesAtStep[Math.max(rank.get(l.source.id), rank.get(l.target.id))].push(i));

  let frame = 0;
  let started = null;
  let orphanFrames = 0;
  let building = false;
  let shownNodes = 0;
  let shownLinks = 0;
  const duration = buildDuration(order.length);

  function hideAll() {
    nodeEls.forEach((n) => {
      n.classList.add("pending");
      n.classList.remove("enter");
    });
    edgeEls.forEach((e) => {
      e.classList.add("pending");
      e.classList.remove("enter");
    });
    shownNodes = 0;
    shownLinks = 0;
  }

  function showUpTo(count, animated) {
    for (; shownNodes < count; shownNodes += 1) {
      const node = nodeEls[nodeIndex.get(order[shownNodes])];
      node.classList.remove("pending");
      if (animated) node.classList.add("enter");
      for (const e of edgesAtStep[shownNodes]) {
        edgeEls[e].classList.remove("pending");
        if (animated) edgeEls[e].classList.add("enter");
        shownLinks += 1;
      }
    }
  }

  function finishBuild() {
    building = false;
    cancelAnimationFrame(frame);
    showUpTo(order.length, false);
    status.textContent = "";
    skipBtn.hidden = true;
  }

  function step() {
    if (!building) return;
    if (!svg.isConnected) {
      // Not attached yet (the page inserts the view a moment after building it), or
      // replaced by another view. Wait a little, then stop working for nothing.
      orphanFrames += 1;
      if (orphanFrames > 30) building = false;
      else frame = requestAnimationFrame(step);
      return;
    }
    orphanFrames = 0;
    const now = performance.now();
    if (started === null) started = now;
    const target = Math.min(order.length, Math.ceil(Math.min(1, (now - started) / duration) * order.length));
    showUpTo(target, true);
    status.textContent = `Building the graph — ${shownNodes} of ${order.length} nodes · ${shownLinks} links`;
    if (shownNodes >= order.length) finishBuild();
    else frame = requestAnimationFrame(step);
  }

  function startBuild() {
    cancelAnimationFrame(frame);
    hideAll();
    started = null;
    if (!animate || prefersReducedMotion() || !order.length) {
      showUpTo(order.length, false);
      return;
    }
    building = true;
    skipBtn.hidden = false;
    status.textContent = `Building the graph — 0 of ${order.length} nodes`;
    frame = requestAnimationFrame(step);
  }

  function rebuild() {
    building = false;
    startBuild();
  }

  paint();
  applyView();
  focusOn(null, false);
  startBuild();

  return {
    element,
    setSelected(id) {
      selectedId = id;
      focusOn(null, false);
    },
    rebuild,
    destroy() {
      building = false;
      cancelAnimationFrame(frame);
    },
  };
}
