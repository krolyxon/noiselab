// Plain-language wording shared with the report figures (server: app/glossary.py, /api/glossary).

import { el } from "./util.js";

let G = { metrics: {}, terms: {}, families: {}, params: {}, algorithms: {} };

export function setGlossary(g) { G = g; }

/** Tooltip text: "Speech clarity gain (ΔSI-SDR), higher is better. How much closer…". */
export function tip(key, extra = "") {
  const t = G.metrics[key] || G.terms[key];
  if (!t) return extra;
  const better = t.better ? `, ${t.better} is better` : "";
  return `${t.short} (${t.tech})${better}.\n${t.explain}${extra ? `\n${extra}` : ""}`;
}

export const plainAlgorithm = (id) => G.algorithms?.[id] || "";

/** Fill the Glossary dialog: every term, plain name first. */
export function renderGlossary(root, names = {}) {
  const row = (t) => el("tr", {},
    el("td", {}, el("strong", { text: t.short })),
    el("td", { class: "mono muted", text: t.tech }),
    el("td", { text: t.better === "higher" ? "↑ higher" : t.better === "lower" ? "↓ lower" : "" }),
    el("td", {}, t.explain, t.analogy ? el("div", { class: "muted small", text: `e.g. ${t.analogy}` }) : null));
  const table = (title, items, head) => [el("h3", { text: title }),
    el("table", { class: "gloss" },
      el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))),
      el("tbody", {}, items.map(row)))];
  const ordered = (group) => (G.order?.[group] || Object.keys(G[group])).map((k) => G[group][k]).filter(Boolean);
  const fam = ordered("families");
  const algos = (G.order?.algorithms || Object.keys(G.algorithms || {})).map((id) =>
    ({ short: names[id] || id, tech: "", better: "", explain: G.algorithms[id] }));
  root.replaceChildren(
    ...table("What the scores mean", ordered("metrics"), ["Plain name", "Technical name", "Better", "Meaning"]),
    ...table("Other words", ordered("terms"), ["Plain name", "Technical name", "", "Meaning"]),
    ...table("Types of method", fam, ["Type", "Technical name", "", "Meaning"]),
    ...table("The methods", algos, ["Method", "", "", "What it does"]),
  );
}
