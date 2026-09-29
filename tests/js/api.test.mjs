import assert from "node:assert/strict";
import { test } from "node:test";

import { ApiError, createClient, makeStore } from "../../src/sentinel/dashboard/static/js/api.js";

function memoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => map.set(k, v),
    removeItem: (k) => map.delete(k),
  };
}

function fakeFetch(responses) {
  const calls = [];
  const impl = async (path, init) => {
    calls.push({ path, init });
    const [status, body] = responses.shift();
    return {
      ok: status >= 200 && status < 300,
      status,
      text: async () => (body === undefined ? "" : JSON.stringify(body)),
    };
  };
  return { impl, calls };
}

const TOKEN = "t".repeat(40);

test("the token rides in the Authorization header, never in the URL or a cookie", async () => {
  const { impl, calls } = fakeFetch([[200, { ok: true }]]);
  const client = createClient({ fetchImpl: impl, store: makeStore(memoryStorage()) });
  client.signIn(TOKEN);
  await client.get("/api/session");
  assert.equal(calls[0].init.headers.Authorization, `Bearer ${TOKEN}`);
  assert.equal(calls[0].init.credentials, "omit");
  assert.ok(!calls[0].path.includes(TOKEN));
});

test("a POST sends JSON and returns the parsed body", async () => {
  const { impl, calls } = fakeFetch([[200, { approved: true }]]);
  const client = createClient({ fetchImpl: impl, store: makeStore(memoryStorage()) });
  const result = await client.post("/api/incidents/x/decision", { action_id: "a", approved: true });
  assert.deepEqual(result, { approved: true });
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0].init.body), { action_id: "a", approved: true });
});

test("errors carry the status and the server's detail", async () => {
  const { impl } = fakeFetch([[409, { error: "conflict", detail: "reload before deciding" }]]);
  const client = createClient({ fetchImpl: impl, store: makeStore(memoryStorage()) });
  await assert.rejects(
    client.get("/api/x"),
    (error) => error instanceof ApiError && error.status === 409 && /reload/.test(error.message),
  );
});

test("422 validation lists are flattened into one message", async () => {
  const { impl } = fakeFetch([[422, { detail: [{ msg: "Extra inputs are not permitted" }] }]]);
  const client = createClient({ fetchImpl: impl, store: makeStore(memoryStorage()) });
  await assert.rejects(client.post("/api/x", { approver: "mallory" }), /Extra inputs/);
});

test("a 401 signs the tab out", async () => {
  let kicked = 0;
  const { impl } = fakeFetch([[401, { error: "unauthorized" }]]);
  const client = createClient({
    fetchImpl: impl,
    store: makeStore(memoryStorage()),
    onUnauthorized: () => (kicked += 1),
  });
  client.signIn(TOKEN);
  await assert.rejects(client.get("/api/overview"));
  assert.equal(kicked, 1);
});

test("storage failures degrade to an in-memory token", async () => {
  const denied = () => {
    throw new Error("denied");
  };
  const broken = { getItem: denied, setItem: denied, removeItem: denied };
  const { impl, calls } = fakeFetch([[200, {}]]);
  const client = createClient({ fetchImpl: impl, store: makeStore(broken) });
  client.signIn(TOKEN);
  assert.ok(client.hasToken());
  await client.get("/api/session");
  assert.equal(calls[0].init.headers.Authorization, `Bearer ${TOKEN}`);
  client.signOut();
  assert.ok(!client.hasToken());
});
