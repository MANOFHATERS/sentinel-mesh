// The only way this front end builds DOM.
//
// Every string becomes a Text node. There is no innerHTML anywhere in the app and
// no way to ask this helper for one: alert payloads, evidence excerpts and code
// excerpts are attacker-controlled, and rendering one as markup would hand the
// attacker the analyst's session. The CSP (no 'unsafe-inline') is the second line;
// this is the first.

import { safeHref } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function setAttributes(el, attrs) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") {
      el.setAttribute("class", String(value));
    } else if (key.startsWith("on")) {
      if (typeof value !== "function") {
        throw new Error(`refusing string event handler for ${key}`);
      }
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") {
      for (const [name, item] of Object.entries(value)) el.dataset[name] = String(item);
    } else if (key === "html" || key === "innerHTML" || key === "srcdoc") {
      throw new Error("raw HTML is not supported");
    } else if (key === "href" || key === "src" || key === "action") {
      const safe = safeHref(String(value));
      if (safe !== null) el.setAttribute(key, safe);
    } else if (key === "value" && "value" in el) {
      el.value = String(value);
    } else {
      el.setAttribute(key, value === true ? "" : String(value));
    }
  }
}

export function append(el, children) {
  for (const child of [children].flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    if (typeof child === "object" && child.nodeType) el.appendChild(child);
    else el.appendChild(document.createTextNode(String(child)));
  }
  return el;
}

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  setAttributes(el, attrs);
  return append(el, children);
}

export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  setAttributes(el, attrs);
  return append(el, children);
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function replace(el, ...children) {
  return append(clear(el), children);
}
