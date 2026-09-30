// A minimal DOM for node --test: enough of Document/Element/Text for dom.js.
// Deliberately has NO innerHTML parser, so a test that passes cannot be relying on
// markup being parsed.

class FakeNode {
  constructor(nodeType) {
    this.nodeType = nodeType;
    this.childNodes = [];
    this.parentNode = null;
  }
  appendChild(child) {
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  removeChild(child) {
    this.childNodes = this.childNodes.filter((c) => c !== child);
    child.parentNode = null;
    return child;
  }
  get firstChild() {
    return this.childNodes[0] || null;
  }
  get textContent() {
    return this.childNodes.map((c) => c.textContent).join("");
  }
}

class FakeText extends FakeNode {
  constructor(data) {
    super(3);
    this.data = data;
  }
  get textContent() {
    return this.data;
  }
}

class FakeElement extends FakeNode {
  constructor(tag, namespace = null) {
    super(1);
    this.tagName = tag.toUpperCase();
    this.namespaceURI = namespace;
    this.attributes = new Map();
    this.listeners = new Map();
    this.dataset = {};
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  // Enough of DOMTokenList for the code under test: it edits the class attribute.
  get classList() {
    const element = this;
    const read = () => (element.getAttribute("class") || "").split(/\s+/).filter(Boolean);
    const write = (names) => element.setAttribute("class", names.join(" "));
    return {
      add: (name) => write([...new Set([...read(), name])]),
      remove: (name) => write(read().filter((n) => n !== name)),
      contains: (name) => read().includes(name),
      toggle(name, force) {
        const on = force === undefined ? !read().includes(name) : Boolean(force);
        if (on) this.add(name);
        else this.remove(name);
        return on;
      },
    };
  }
  addEventListener(type, fn) {
    this.listeners.set(type, fn);
  }
  dispatch(type, event = {}) {
    const fn = this.listeners.get(type);
    if (fn) fn(event);
  }
  set innerHTML(_value) {
    throw new Error("innerHTML must never be used");
  }
}

export function installFakeDom() {
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (ns, tag) => new FakeElement(tag, ns),
    createTextNode: (data) => new FakeText(data),
  };
  return globalThis.document;
}
