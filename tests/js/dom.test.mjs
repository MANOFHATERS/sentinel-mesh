import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

import { installFakeDom } from "./fakedom.mjs";

installFakeDom();
const { h, replace, s } = await import("../../src/sentinel/dashboard/static/js/dom.js");

beforeEach(() => installFakeDom());

const PAYLOAD = '<img src=x onerror="alert(document.cookie)"><script>alert(1)</script>';

test("attacker text becomes a text node, never markup", () => {
  const el = h("pre", {}, PAYLOAD);
  assert.equal(el.childNodes.length, 1);
  assert.equal(el.childNodes[0].nodeType, 3, "a Text node");
  assert.equal(el.textContent, PAYLOAD);
});

test("raw HTML cannot be requested at all", () => {
  assert.throws(() => h("div", { html: PAYLOAD }), /raw HTML/);
  assert.throws(() => h("div", { innerHTML: PAYLOAD }), /raw HTML/);
  assert.throws(() => h("iframe", { srcdoc: PAYLOAD }), /raw HTML/);
});

test("string event handlers are refused; function handlers are bound", () => {
  assert.throws(() => h("button", { onclick: "alert(1)" }), /string event handler/);
  let clicked = 0;
  const button = h("button", { onclick: () => (clicked += 1) }, "go");
  button.dispatch("click");
  assert.equal(clicked, 1);
});

test("dangerous URLs never reach an href", () => {
  assert.equal(h("a", { href: "javascript:alert(1)" }).getAttribute("href"), null);
  assert.equal(
    h("a", { href: "https://github.example/x" }).getAttribute("href"),
    "https://github.example/x",
  );
  assert.equal(h("a", { href: "#/queue" }).getAttribute("href"), "#/queue");
});

test("null, false and nested children are handled", () => {
  const el = h("ul", {}, [h("li", {}, "a"), null, false, [h("li", {}, "b"), undefined]], 3);
  assert.equal(el.childNodes.length, 3);
  assert.equal(el.textContent, "ab3");
});

test("boolean attributes are present or absent, never 'false'", () => {
  assert.equal(h("button", { disabled: true }).getAttribute("disabled"), "");
  assert.equal(h("button", { disabled: false }).getAttribute("disabled"), null);
});

test("svg elements use the SVG namespace", () => {
  const circle = s("circle", { r: "4" });
  assert.equal(circle.namespaceURI, "http://www.w3.org/2000/svg");
  assert.equal(circle.getAttribute("r"), "4");
});

test("replace swaps content", () => {
  const el = h("div", {}, "old");
  replace(el, "new", h("b", {}, "!"));
  assert.equal(el.textContent, "new!");
});
