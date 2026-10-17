// Small shared helpers: number formatting, DOM, toasts, debounce, API calls.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Fixed-decimal number with unit; null/NaN → "n/a". */
export function fmt(v, decimals = 2, unit = "") {
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  const s = v.toFixed(decimals).replace(/^-/, "−");
  return unit ? `${s} ${unit}` : s;
}

export function signed(v, decimals = 2) {
  if (!Number.isFinite(v)) return "n/a";
  return (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(decimals);
}

export function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

export function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) n.append(c);
  return n;
}

export function toast(message, kind = "info", ms = 4500) {
  const t = el("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status", text: message });
  $("#toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

/** fetch → JSON, throwing Error(server message) on non-2xx. */
export async function api(url, opts = {}) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new Error("cannot reach the NoiseLab server — is `flask run` still running?");
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    const err = new Error(body?.error || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

export const postJSON = (url, data, signal) =>
  api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data), signal });

/** Decode the server's base64 int16 matrix (value × scale) into an array of row arrays. */
export function decodeMatrix({ shape, scale, b64 }) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const q = new Int16Array(bytes.buffer);
  const [rows, cols] = shape;
  const out = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = new Array(cols);
    const o = r * cols;
    for (let c = 0; c < cols; c++) row[c] = q[o + c] / scale;
    out[r] = row;
  }
  return out;
}
