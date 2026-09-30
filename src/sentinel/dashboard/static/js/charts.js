// Small, dependency-free SVG charts: a line chart and a bar chart.
//
// Built to one method rather than by taste: at most two series per chart (the
// reference palette's first two slots, validated for both themes), 2px lines,
// recessive grid, one y-axis, a legend whenever there are two series, a hover
// layer on every chart (crosshair + tooltip on lines, per-bar tooltips on bars), and
// a table view so no value is carried by colour alone. Colours are CSS tokens
// (--series-1, --series-2, --ref) so light and dark themes each use their own steps.
//
// The maths is exported separately (niceTicks, nearestIndex) so node --test can
// check it without a DOM.

import { h, s } from "./dom.js";

// "Nice" axis ticks covering [min, max]: steps of 1, 2 or 5 x 10^k.
export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) {
    const pad = Math.abs(min) || 1;
    min -= pad / 2;
    max += pad / 2;
  }
  const span = max - min;
  const raw = span / Math.max(1, count - 1);
  // d3's tickIncrement rule: round the raw step to the *nearest* 1-2-5 step on a
  // log scale (thresholds sqrt(50), sqrt(10), sqrt(2)), not the next one up, so a
  // 0-1 axis gets 0.2 steps rather than three ticks.
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const error = raw / magnitude;
  const factor = error >= Math.sqrt(50) ? 10 : error >= Math.sqrt(10) ? 5 : error >= Math.sqrt(2) ? 2 : 1;
  const step = factor * magnitude;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

// Index of the value in the sorted array `xs` closest to `x`.
export function nearestIndex(xs, x) {
  if (!xs.length) return -1;
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
}

// Label y-positions for line ends: sorted top to bottom and pushed apart so no two are
// closer than `gap`. Returns [{index, y}] in the input's index space.
export function endLabelPositions(ys, gap) {
  const order = ys.map((y, index) => ({ index, y })).sort((a, b) => a.y - b.y);
  for (let k = 1; k < order.length; k += 1) {
    if (order[k].y - order[k - 1].y < gap) order[k].y = order[k - 1].y + gap;
  }
  return order;
}

export function formatValue(v, digits = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(digits);
}

// Axis ticks: integers as integers at any magnitude ("140", not "140.0"), and
// fractional steps without float noise ("0.6", not "0.6000000000000001").
export function formatTick(t) {
  if (!Number.isFinite(t)) return "—";
  return Number.isInteger(t) ? String(t) : String(Number(t.toFixed(4)));
}

// Charts draw themselves the first time they scroll into view: bars grow from the
// baseline and lines draw left to right (CSS animations, gated on the ``in`` class). Where
// IntersectionObserver is missing they are simply shown.
function revealWhenVisible(figure) {
  if (typeof IntersectionObserver === "undefined") {
    figure.classList.add("in");
    return figure;
  }
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      figure.classList.add("in");
      observer.disconnect();
    }
  });
  observer.observe(figure);
  return figure;
}

const W = 640;
const H = 250;
const M = { top: 16, right: 92, bottom: 34, left: 52 };

function tableView(headers, rows) {
  return h(
    "details",
    { class: "chart-table" },
    h("summary", {}, "Show as table"),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, headers.map((x) => h("th", {}, x)))),
        h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", { class: "num" }, c))))),
      ),
    ),
  );
}

function legend(series) {
  if (series.length < 2) return null;
  return h(
    "ul",
    { class: "chart-legend" },
    series.map((sr, i) => h("li", {}, h("span", { class: `key series-${i + 1}${sr.dashed ? " dashed" : ""}` }), sr.name)),
  );
}

function axes(svg, xTicks, yTicks, sx, sy, xLabel, yLabel) {
  const grid = s("g", { class: "chart-grid" });
  for (const t of yTicks) {
    const y = sy(t);
    grid.appendChild(s("line", { x1: M.left, x2: W - M.right, y1: y, y2: y }));
    grid.appendChild(s("text", { x: M.left - 8, y: y + 4, class: "tick", "text-anchor": "end" }, formatTick(t)));
  }
  for (const t of xTicks) {
    grid.appendChild(s("text", { x: sx(t), y: H - M.bottom + 16, class: "tick", "text-anchor": "middle" }, formatTick(t)));
  }
  grid.appendChild(s("line", { x1: M.left, x2: W - M.right, y1: H - M.bottom, y2: H - M.bottom, class: "baseline" }));
  if (xLabel) grid.appendChild(s("text", { x: (M.left + W - M.right) / 2, y: H - 4, class: "axis-label", "text-anchor": "middle" }, xLabel));
  if (yLabel) grid.appendChild(s("text", { x: 12, y: M.top - 4, class: "axis-label" }, yLabel));
  svg.appendChild(grid);
}

// series: [{name, values: number[], dashed?}] sharing the x array.
export function lineChart({ title, x, series, xLabel, yLabel, yMin = null, describe = "" }) {
  const clean = series.filter((sr) => sr.values && sr.values.length);
  const all = clean.flatMap((sr) => sr.values).filter(Number.isFinite);
  const yTicks = niceTicks(yMin ?? Math.min(0, ...all), Math.max(...all));
  const xTicks = niceTicks(x[0], x[x.length - 1], 6).filter((t) => t >= x[0] && t <= x[x.length - 1]);
  const sx = (v) => M.left + ((v - x[0]) / (x[x.length - 1] - x[0] || 1)) * (W - M.left - M.right);
  const sy = (v) => H - M.bottom - ((v - yTicks[0]) / (yTicks[yTicks.length - 1] - yTicks[0] || 1)) * (H - M.top - M.bottom);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": `${title}. ${describe}` });
  axes(svg, xTicks, yTicks, sx, sy, xLabel, yLabel);
  clean.forEach((sr, i) => {
    const d = sr.values.map((v, j) => `${j ? "L" : "M"}${sx(x[j]).toFixed(1)},${sy(v).toFixed(1)}`).join("");
    svg.appendChild(s("path", { d, pathLength: "1", class: `line series-${i + 1}${sr.dashed ? " dashed" : ""}` }));
  });
  // Selective direct labels: each series' name at its end, in text ink (not the series
  // colour), pushed apart when two lines finish close together.
  for (const label of endLabelPositions(clean.map((sr) => sy(sr.values[sr.values.length - 1])), 13)) {
    const sr = clean[label.index];
    svg.appendChild(s("text", { x: sx(x[sr.values.length - 1]) + 6, y: label.y + 4, class: "end-label" }, sr.name));
  }

  const cross = s("line", { class: "crosshair", y1: M.top, y2: H - M.bottom, x1: -10, x2: -10 });
  const dots = clean.map((_, i) => s("circle", { r: "4", class: `dot series-${i + 1}`, cx: -10, cy: -10 }));
  svg.appendChild(cross);
  dots.forEach((dt) => svg.appendChild(dt));
  const hit = s("rect", { x: M.left, y: M.top, width: W - M.left - M.right, height: H - M.top - M.bottom, class: "hit" });
  svg.appendChild(hit);

  const tip = h("div", { class: "chart-tip", role: "status", "aria-live": "polite" });
  const holder = h("div", { class: "chart-holder" }, svg, tip);

  function show(event) {
    const box = svg.getBoundingClientRect();
    const vx = ((event.clientX - box.left) / box.width) * W;
    const xv = x[0] + ((vx - M.left) / (W - M.left - M.right)) * (x[x.length - 1] - x[0]);
    const i = nearestIndex(x, xv);
    if (i < 0) return;
    cross.setAttribute("x1", sx(x[i]));
    cross.setAttribute("x2", sx(x[i]));
    dots.forEach((dt, k) => {
      dt.setAttribute("cx", sx(x[i]));
      dt.setAttribute("cy", sy(clean[k].values[i]));
    });
    tip.replaceChildren(
      h("strong", {}, `${xLabel || "x"} ${formatValue(x[i], 0)}`),
      ...clean.map((sr, k) => h("div", {}, h("span", { class: `key series-${k + 1}` }), `${sr.name}: ${formatValue(sr.values[i])}`)),
    );
    tip.classList.add("visible");
    holder.classList.add("hovering");
    const px = (sx(x[i]) / W) * box.width;
    tip.style.left = `${Math.min(Math.max(px + 12, 0), box.width - 170)}px`;
    tip.style.top = "8px";
  }
  function hide() {
    tip.classList.remove("visible");
    holder.classList.remove("hovering");
  }
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerleave", hide);

  const step = Math.max(1, Math.floor(x.length / 25));
  const rows = [];
  for (let i = 0; i < x.length; i += step) rows.push([formatValue(x[i], 0), ...clean.map((sr) => formatValue(sr.values[i]))]);
  if ((x.length - 1) % step) rows.push([formatValue(x[x.length - 1], 0), ...clean.map((sr) => formatValue(sr.values[x.length - 1]))]);

  return revealWhenVisible(
    h(
      "figure",
      { class: "chart" },
      h("figcaption", {}, h("strong", {}, title), describe ? h("span", { class: "meta" }, ` — ${describe}`) : null),
      legend(clean),
      holder,
      tableView([xLabel || "x", ...clean.map((sr) => sr.name)], rows),
    ),
  );
}

// Grouped vertical bars: categories on x, up to two series.
export function barChart({ title, categories, series, yLabel, yMax = null, describe = "", percent = false }) {
  const all = series.flatMap((sr) => sr.values).filter(Number.isFinite);
  const yTicks = niceTicks(0, yMax ?? Math.max(...all, 0.0001), 5);
  const plotW = W - M.left - 20;
  const band = plotW / categories.length;
  const barW = Math.min(34, (band - 18) / series.length);
  const sy = (v) => H - M.bottom - (v / yTicks[yTicks.length - 1]) * (H - M.top - M.bottom);
  const fmt = (v) => (percent ? `${(v * 100).toFixed(1)}%` : formatValue(v));

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": `${title}. ${describe}` });
  const grid = s("g", { class: "chart-grid" });
  for (const t of yTicks) {
    grid.appendChild(s("line", { x1: M.left, x2: W - 20, y1: sy(t), y2: sy(t) }));
    grid.appendChild(s("text", { x: M.left - 8, y: sy(t) + 4, class: "tick", "text-anchor": "end" }, percent ? `${Math.round(t * 100)}%` : formatTick(t)));
  }
  grid.appendChild(s("line", { x1: M.left, x2: W - 20, y1: H - M.bottom, y2: H - M.bottom, class: "baseline" }));
  if (yLabel) grid.appendChild(s("text", { x: 12, y: M.top - 4, class: "axis-label" }, yLabel));
  svg.appendChild(grid);

  const tip = h("div", { class: "chart-tip", role: "status", "aria-live": "polite" });
  categories.forEach((cat, c) => {
    const center = M.left + band * c + band / 2;
    const groupW = barW * series.length + 2 * (series.length - 1);
    series.forEach((sr, k) => {
      const v = sr.values[c];
      if (!Number.isFinite(v)) return;
      const x0 = center - groupW / 2 + k * (barW + 2);
      const top = sy(Math.max(v, 0));
      const height = Math.max(H - M.bottom - top, 1);
      const bar = s("rect", { x: x0.toFixed(1), y: top.toFixed(1), width: barW.toFixed(1), height: height.toFixed(1), rx: "4", class: `bar series-${k + 1} d${Math.min(c * series.length + k, 15)}`, tabindex: "0" });
      const showTip = () => {
        tip.replaceChildren(h("strong", {}, cat), h("div", {}, h("span", { class: `key series-${k + 1}` }), `${sr.name}: ${fmt(v)}`));
        tip.classList.add("visible");
        const box = svg.getBoundingClientRect();
        tip.style.left = `${Math.min(Math.max(((x0 + barW) / W) * box.width + 8, 0), box.width - 170)}px`;
        tip.style.top = "8px";
      };
      bar.addEventListener("pointerenter", showTip);
      bar.addEventListener("focus", showTip);
      bar.addEventListener("pointerleave", () => tip.classList.remove("visible"));
      bar.addEventListener("blur", () => tip.classList.remove("visible"));
      svg.appendChild(bar);
    });
    svg.appendChild(s("text", { x: center, y: H - M.bottom + 16, class: "tick", "text-anchor": "middle" }, cat));
  });

  return revealWhenVisible(
    h(
      "figure",
      { class: "chart" },
      h("figcaption", {}, h("strong", {}, title), describe ? h("span", { class: "meta" }, ` — ${describe}`) : null),
      legend(series),
      h("div", { class: "chart-holder" }, svg, tip),
      tableView(["", ...series.map((sr) => sr.name)], categories.map((cat, c) => [cat, ...series.map((sr) => fmt(sr.values[c]))])),
    ),
  );
}
