// The maths behind zoom and pan on the supply-chain map, kept free of the DOM so it can
// be tested. A view is an SVG viewBox: {x, y, width, height} in graph units. Zooming in
// makes the box smaller; the point under the pointer stays where it is on screen.

export const MIN_SCALE = 0.5; // zoomed out to half the fitted drawing
export const MAX_SCALE = 12; // twelve times in

export function scaleOf(box, base) {
  return base.width / box.width;
}

// Zoom by ``factor`` (> 1 in, < 1 out) about the graph-space point (px, py).
export function zoomBox(box, factor, px, py) {
  return {
    x: px - (px - box.x) / factor,
    y: py - (py - box.y) / factor,
    width: box.width / factor,
    height: box.height / factor,
  };
}

// Keep the zoom inside [MIN_SCALE, MAX_SCALE] relative to ``base``, about the same centre.
export function clampBox(box, base) {
  const scale = scaleOf(box, base);
  if (scale >= MIN_SCALE && scale <= MAX_SCALE) return box;
  const target = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
  const width = base.width / target;
  const height = base.height / target;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  return { x: cx - width / 2, y: cy - height / 2, width, height };
}

// Move the view by (dx, dy) graph units.
export function panBox(box, dx, dy) {
  return { ...box, x: box.x - dx, y: box.y - dy };
}

export function boxCentre(box) {
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

// Fit ``bounds`` (the drawing's extent) into a canvas of at least minWidth x minHeight, so a
// seven-node advisory scope is not blown up to fill the viewport.
export function fitBox(bounds, { minWidth = 560, minHeight = 380 } = {}) {
  const width = Math.max(bounds.width, minWidth);
  const height = Math.max(bounds.height, minHeight);
  return {
    x: bounds.x - (width - bounds.width) / 2,
    y: bounds.y - (height - bounds.height) / 2,
    width,
    height,
  };
}

export function viewBoxString(box) {
  return `${box.x.toFixed(1)} ${box.y.toFixed(1)} ${box.width.toFixed(1)} ${box.height.toFixed(1)}`;
}
