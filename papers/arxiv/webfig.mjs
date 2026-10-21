#!/usr/bin/env node
// Figures of main.tex, exported from ../index.html so the arXiv version draws exactly what the page draws.
//
//   node webfig.mjs
//
// Needs node 22+ (the built-in WebSocket) and a Chromium from Playwright's cache, or CHROME=/path/to/chrome.
// Each figure is the page's own SVG, its styles inlined, printed to a vector PDF at the page's aspect ratio.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

const HERE = new URL(".", import.meta.url).pathname;
const PAGE = "file://" + resolve(HERE, "../index.html");
const OUT = join(HERE, "figures");
const USER_DIR = "/tmp/memorytax-webfig";

// The page's figures; the last two are fig-cumulative with the x axis switched to turns and to tokens.
const FIGURES = [
  { file: "frontier", panels: [{ id: "fig-frontier" }], inchW: 6.5, fontScale: 1.2 },
  { file: "cumulative", panels: [{ id: "fig-cumulative", metric: "cost" }], inchW: 6.5, fontScale: 1.2 },
  { file: "cross", panels: [{ id: "fig-cross" }], inchW: 6.5, fontScale: 1.2 },
  { file: "usage", panels: [{ id: "fig-cumulative", metric: "turns" }, { id: "fig-cumulative", metric: "tokens" }],
    inchW: 6.5, gap: 0.12, fontScale: 1.2 },
];

// The page's own colours, read off a figure card, so the SVG keeps them once it leaves the page.
const VARS = ["--card", "--border", "--grid", "--axis", "--tick", "--ink", "--ink2", "--muted", "--rule", "--soft",
  "--link", "--figure-bg", "--pill", "--m-mini", "--m-luna", "--m-luna-high", "--m-opus"];

const SNIPPET = `(id, metric, vars, scale) => {
  const rules = [];
  const walk = list => { for (const r of list) {
    if (r.selectorText) rules.push(r.cssText.replace(/#fig-[a-z]+ /g, "")
      .replace(/(font-size:\s*)([0-9.]+)px/g, (_, a, n) => a + (+n * scale).toFixed(2) + "px"));  // the root is the svg now, not its card
    else if (r.cssRules && r.conditionText && !/dark/.test(r.conditionText)) walk(r.cssRules);
  } };
  for (const sheet of document.styleSheets) walk(sheet.cssRules);
  const fig = document.getElementById(id);
  if (metric) fig.querySelector('.pills[data-dim="metric"] button[data-key="' + metric + '"]').click();
  const svg = fig.querySelector("svg.chart");
  const cs = getComputedStyle(fig);
  const decl = vars.map(v => { const x = cs.getPropertyValue(v).trim(); return x ? v + ":" + x : ""; })
    .filter(Boolean).join(";");
  const clone = svg.cloneNode(true);
  const box = svg.getAttribute("viewBox").trim().split(/ +/).map(Number);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", box[2]);
  clone.setAttribute("height", box[3]);
  clone.setAttribute("style", decl + ";background:transparent");
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = rules.join("\\n");
  clone.insertBefore(style, clone.firstChild);
  // The pills above the plot are the figure's legend: the model colours and the method marks. The x-axis
  // pills only switch the page, so they stay behind.
  const rows = fig.querySelector(".chart-filters .filter-rows").cloneNode(true);
  const xaxis = rows.querySelector('[data-dim="metric"]');
  if (xaxis) { xaxis.previousElementSibling?.remove(); xaxis.remove(); }
  for (const b of rows.querySelectorAll("button")) b.removeAttribute("aria-pressed");
  return { svg: clone.outerHTML, w: box[2], h: box[3], legend: rows.outerHTML, rules: rules.join("\\n"), decl };
}`;

const sleep = ms => new Promise(r => setTimeout(r, ms));

function chrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const cache = join(homedir(), ".cache/ms-playwright");
  if (!existsSync(cache)) throw new Error("no Chromium in " + cache + "; set CHROME");
  for (const d of readdirSync(cache).filter(d => d.startsWith("chromium-")).sort().reverse())
    for (const sub of ["chrome-linux64/chrome", "chrome-linux/chrome"])
      if (existsSync(join(cache, d, sub))) return join(cache, d, sub);
  throw new Error("no Chrome binary under " + cache);
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const browser = spawn(chrome(), ["--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
    "--force-color-profile=srgb", "--disable-lcd-text", "--font-render-hinting=none", "--remote-allow-origins=*",
    "--remote-debugging-port=9336", "--user-data-dir=" + USER_DIR, "about:blank"], { stdio: "ignore" });

  let socket;
  for (let i = 0; i < 100 && !socket; i++) {
    try {
      const r = await fetch("http://127.0.0.1:9336/json/version");
      if (r.ok) socket = (await r.json()).webSocketDebuggerUrl;
    } catch {}
    if (!socket) await sleep(100);
  }
  if (!socket) throw new Error("the browser did not open a debugger port");

  const ws = new WebSocket(socket);
  const waiting = new Map();
  let next = 0;
  ws.addEventListener("message", e => {
    const m = JSON.parse(e.data), w = waiting.get(m.id);
    if (!w) return;
    waiting.delete(m.id);
    m.error ? w.reject(new Error(m.error.message)) : w.resolve(m.result);
  });
  await new Promise(r => ws.addEventListener("open", r));
  const send = (method, params = {}, session) => new Promise((resolve, reject) => {
    const msg = { id: ++next, method, params };
    if (session) msg.sessionId = session;
    waiting.set(msg.id, { resolve, reject });
    ws.send(JSON.stringify(msg));
  });

  try {
    const { targetId } = await send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
    const page = (m, p) => send(m, p, sessionId);
    await page("Page.enable");
    await page("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });

    for (const f of FIGURES) {
      const parts = [];
      for (const panel of f.panels) {
        await page("Page.navigate", { url: PAGE });
        await sleep(5000);  // the charts draw on load and move their labels during the intro
        const { result } = await page("Runtime.evaluate", {
          expression: `(${SNIPPET})(${JSON.stringify(panel.id)}, ${JSON.stringify(panel.metric ?? null)}, ` +
            `${JSON.stringify(VARS)}, ${f.fontScale ?? 1})`,
          returnByValue: true });
        const { svg, w, h, legend, decl } = result.value;
        parts.push({ svg, legend: legend ?? "", rules: result.value.rules ?? "", decl,
          inchH: h / w * f.inchW });
      }
      const gap = f.gap ?? 0;
      const inchH = parts.reduce((sum, p) => sum + p.inchH, 0) + gap * (parts.length - 1);
      // One legend above the first panel; the panels of one figure share it.
      const card = parts.map((p, i) => `<div class="fig" style="${p.decl}">` +
        (i ? "" : `<div class="legend">${p.legend}</div>`) +
        `<div class="p" style="height:${p.inchH.toFixed(3)}in">${p.svg}</div></div>`).join("");
      const px = f.inchW * 96;
      await page("Page.navigate", { url: "data:text/html;charset=utf-8," + encodeURIComponent(
        `<!doctype html><meta charset="utf-8"><style>` +
        `*{box-sizing:border-box}body{margin:0;padding:0;background:#fff}` +
        `.fig{width:${px.toFixed(1)}px;padding:0}` +
        `</style><style>${parts[0].rules}</style><style>` +
        `.chart-card, .legend { --pad: 0 }` +
        `.legend{display:block;padding:2px 0 8px;` +
        `font:${(12.5 * (f.fontScale ?? 1)).toFixed(2)}px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}` +
        `.legend .filter-rows{display:grid;grid-template-columns:auto 1fr;gap:5px 18px;align-items:center}` +
        `.legend .filter-label{color:var(--tick);font-size:${(10 * (f.fontScale ?? 1)).toFixed(2)}px;font-weight:650;` +
        `letter-spacing:.08em;text-transform:uppercase}` +
        `.legend .pills{display:flex;flex-wrap:wrap;align-items:center;gap:4px 14px}` +
        `.legend .pills button{display:inline-flex;align-items:center;gap:5px;color:var(--ink2);background:none;` +
        `border:0;padding:0;margin:0;font:inherit;cursor:default}` +
        `.legend .pills .dot{width:10px;height:10px;border-radius:50%;flex:none}` +
        `.legend .pills svg{width:16px;height:16px;flex:none}` +
        `.legend .pills svg.dash{width:18px;margin-left:2px}` +
        `.p+.p{margin-top:${gap}in}` +
        `svg{display:block;width:100%;height:auto}` +
        `</style>${card}`) });
      await sleep(600);
      const { result: size } = await page("Runtime.evaluate",
        { expression: "document.documentElement.scrollHeight", returnByValue: true });
      const pageH = Math.max(size.value, 1) / 96;
      const { data } = await page("Page.printToPDF", { printBackground: true, paperWidth: f.inchW, paperHeight: pageH,
        marginTop: 0, marginBottom: 0, marginLeft: 0, marginRight: 0, scale: 1 });
      writeFileSync(join(OUT, f.file + ".pdf"), Buffer.from(data, "base64"));
      console.log(`${f.file}.pdf  ${f.inchW.toFixed(2)} x ${pageH.toFixed(2)} in`);
    }
  } finally {
    ws.close();
    browser.kill();
  }
}

main().catch(e => { console.error(e.message); process.exit(1); });
