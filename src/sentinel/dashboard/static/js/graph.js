// Renders the supply-chain map as SVG from the force layout.

import { s } from "./dom.js";
import { createSimulation } from "./force.js";

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

export function renderGraph(data, { onSelect, selected } = {}) {
  const width = 900;
  const height = 620;
  const many = data.nodes.length > 150;
  const sim = createSimulation(data.nodes, data.edges, {
    width,
    height,
    linkDistance: many ? 22 : 46,
    charge: many ? -26 : -90,
    radius: nodeRadius,
  }).run(300);
  // Fit the drawing, but never zoom in past a minimum canvas: a seven-node advisory
  // scope fitted edge to edge renders nodes the size of the viewport.
  const fitted = sim.bounds(24);
  const minWidth = 560;
  const minHeight = 380;
  const box = {
    x: fitted.x - Math.max(0, minWidth - fitted.width) / 2,
    y: fitted.y - Math.max(0, minHeight - fitted.height) / 2,
    width: Math.max(fitted.width, minWidth),
    height: Math.max(fitted.height, minHeight),
  };
  const highlighted = pathEdges(data.highlight);
  const onPath = new Set((data.highlight || []).flat());
  const advisoryPackage = data.advisory ? data.advisory.package_id : null;

  const svg = s("svg", {
    class: "graph-svg",
    viewBox: `${box.x.toFixed(1)} ${box.y.toFixed(1)} ${box.width.toFixed(1)} ${box.height.toFixed(1)}`,
    role: "img",
    "aria-label": `Supply-chain graph: ${data.nodes.length} nodes, ${data.edges.length} edges`,
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

  const edgeLayer = s("g", { class: "edges" });
  const hotLayer = s("g", { class: "edges hot" });
  for (const link of sim.links) {
    const key = `${link.source.id}->${link.target.id}`;
    const hot = highlighted.has(key);
    // Shorten the line so the arrowhead sits on the target's rim.
    const dx = link.target.x - link.source.x;
    const dy = link.target.y - link.source.y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const rt = nodeRadius(link.target) + 2;
    const line = s("line", {
      x1: link.source.x.toFixed(2),
      y1: link.source.y.toFixed(2),
      x2: (link.target.x - (dx / len) * rt).toFixed(2),
      y2: (link.target.y - (dy / len) * rt).toFixed(2),
      class: `edge ${link.kind}${hot ? " hot" : ""}`,
      "marker-end": hot ? "url(#arrow-hot)" : many ? null : "url(#arrow)",
    });
    (hot ? hotLayer : edgeLayer).appendChild(line);
  }
  svg.appendChild(edgeLayer);
  svg.appendChild(hotLayer);

  const nodeLayer = s("g", { class: "nodes" });
  for (const node of sim.nodes) {
    const classes = ["node", node.kind];
    if (node.intrinsic) classes.push("intrinsic");
    if (onPath.has(node.id)) classes.push("on-path");
    if (node.id === advisoryPackage) classes.push("advisory");
    if (node.id === selected) classes.push("selected");
    const group = s(
      "g",
      {
        class: classes.join(" "),
        transform: `translate(${node.x.toFixed(2)},${node.y.toFixed(2)})`,
        tabindex: "0",
        role: "button",
        "aria-label": `${node.id}, ${node.kind}, risk ${Number(node.risk).toFixed(3)}`,
        onclick: () => onSelect && onSelect(node.id),
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
    if (!many || onPath.has(node.id) || node.kind === "organization") {
      group.appendChild(s("text", { x: (nodeRadius(node) + 3).toFixed(1), y: "3.5" }, node.id));
    }
    nodeLayer.appendChild(group);
  }
  svg.appendChild(nodeLayer);
  return svg;
}
