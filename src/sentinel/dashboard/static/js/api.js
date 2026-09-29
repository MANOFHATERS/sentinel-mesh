// A thin client for /api. The token lives in sessionStorage: per tab, gone when the
// tab closes, never a cookie (see auth.py for why a header and not a cookie).

const KEY = "sentinel.token";

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

export function makeStore(storage) {
  return {
    get() {
      try {
        return storage.getItem(KEY);
      } catch {
        return null;
      }
    },
    set(token) {
      try {
        storage.setItem(KEY, token);
      } catch {
        /* storage unavailable: the token lives only in memory for this page */
      }
    },
    clear() {
      try {
        storage.removeItem(KEY);
      } catch {
        /* nothing to clear */
      }
    },
  };
}

export function createClient({ fetchImpl, store, onUnauthorized }) {
  let memoryToken = null;

  function token() {
    return store.get() || memoryToken;
  }

  async function request(method, path, body) {
    const headers = { Accept: "application/json" };
    const current = token();
    if (current) headers.Authorization = `Bearer ${current}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetchImpl(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "omit",
      cache: "no-store",
    });
    let payload = null;
    const text = await response.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { detail: text };
      }
    }
    if (!response.ok) {
      const detail = payload && (payload.detail || payload.error);
      const message = Array.isArray(detail)
        ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : detail;
      if (response.status === 401 && onUnauthorized) onUnauthorized();
      throw new ApiError(response.status, message);
    }
    return payload;
  }

  return {
    signIn(value) {
      memoryToken = value;
      store.set(value);
    },
    signOut() {
      memoryToken = null;
      store.clear();
    },
    hasToken() {
      return Boolean(token());
    },
    get: (path) => request("GET", path),
    post: (path, body) => request("POST", path, body === undefined ? {} : body),
  };
}
