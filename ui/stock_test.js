/* stock_test.js -- the Stock screen (HYPERUN-STOCK).
 *
 * WHAT IT PINS.
 *   1. The board shows a region only when it SELLS what the filter asks for, and marks it in
 *      stock only when that shape is available there; vendors and regions with stock come first.
 *   2. The GPU chips count regions with stock at the chosen count, keep a GPU with none at 0
 *      (folded behind one chip) rather than dropping it, and say which chip is pressed.
 *   3. A filter arrives from the address (#/stock/gpu=H100&count=8, the Slack link) and anything
 *      unreadable there is "any", never an error.
 *   4. The alert rows say in words whether the GPU is in stock and whether the DM went, and the
 *      form sends "" as null, the way POST /v1/stock/alerts reads "any".
 *   5. Every id app.js reaches for on this screen exists in index.html.
 *
 * HOW IT RUNS THE REAL CODE. Each function is extracted from the shipped app.js by name and
 * evaluated, as servings_test.js does, so a pass here is a pass of the deployed code.
 *
 * Run: node ui/stock_test.js      (no browser, no network, no login)
 */
"use strict";

const fs = require("fs");
const path = require("path");

const SRC = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const HTML = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const failures = [];

function check(condition, description) {
  console.log((condition ? "  ok    " : "  FAIL  ") + description);
  if (!condition) failures.push(description);
}

function extract(name) {
  const at = SRC.search(new RegExp(`(^|\\n)(async )?function ${name}\\(`));
  if (at < 0) throw new Error(`app.js has no function ${name} -- was it renamed?`);
  let depth = 0;
  for (let j = SRC.indexOf("{", at); j < SRC.length; j++) {
    if (SRC[j] === "{") depth++;
    else if (SRC[j] === "}" && --depth === 0) return SRC.slice(at, j + 1).trim();
  }
  throw new Error(`unbalanced braces in ${name}`);
}

/* A top-level `const NAME = ...;`, one line or a block that ends at a line that is "};". */
function declaration(name) {
  const lines = SRC.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`const ${name} =`));
  if (start < 0) throw new Error(`app.js has no const ${name}`);
  const block = lines[start].trimEnd().endsWith("{");
  const out = [];
  for (let i = start; i < lines.length; i++) {
    out.push(lines[i]);
    if (block ? lines[i] === "};" : lines[i].trimEnd().endsWith(";")) break;
  }
  return out.join("\n");
}

const define = (src) => (0, eval)(src);
["esc", "when", "card", "STOCK_GPU_NAMES", "STOCK_VENDOR_NAMES", "stockGpuName", "stockVendorName",
 "STOCK_EMPTY_ICON"].forEach((name) => define(declaration(name).replace(/^const /, "globalThis.")));
["stockFilterFrom", "stockFits", "stockGpuCounts", "stockView", "stockShapeWord", "stockChipsHtml",
 "stockCardsHtml", "stockFreshHtml", "stockShapesHtml", "stockBoardHtml", "stockAlertRowsHtml",
 "readStockAlertForm"].forEach((name) => define(`globalThis.${name} = (${extract(name)});`));

const fields = {};
globalThis.document = {
  getElementById: (id) => {
    if (!fields[id]) fields[id] = { value: "", innerHTML: "", textContent: "", hidden: false };
    return fields[id];
  },
};
globalThis.$ = (id) => document.getElementById(id);

/* Shaped like GET /v1/stock: two vendors, H100 x8 in one Montreal region only. */
const data = {
  source: "shadeform", fetched_at: "2026-10-24T07:50:01Z", refresh_minutes: 10, stale: false,
  vendors: [
    { vendor: "lambdalabs", regions: [
      { region: "dulles-usa-1", name: "US, Dulles, VA", in_stock: true, offers: [
        { gpu: "H100", gpus: 1, available: true }, { gpu: "A10", gpus: 1, available: false }] },
      { region: "phoenix-usa-2", name: "US, Phoenix, AZ", in_stock: false, offers: [
        { gpu: "H100", gpus: 8, available: false }] } ] },
    { vendor: "hyperstack", regions: [
      { region: "montreal-canada-2", name: "CA, Montreal", in_stock: true, offers: [
        { gpu: "H100", gpus: 8, available: true }, { gpu: "A100_80G", gpus: 1, available: true },
        { gpu: "A100_80G", gpus: 8, available: true }] },
      { region: "oslo-norway-1", name: "NO, Oslo", in_stock: false, offers: [
        { gpu: "H200", gpus: 8, available: false }] } ] },
  ],
};

console.log("1. what the board shows");
{
  const any = stockView(data, { gpu: "", count: 0 });
  check(any.regionsIn === 2 && any.regionsSold === 4 && any.vendorsIn === 2,
        `any GPU: 2 of 4 regions in stock, both vendors (got ${any.regionsIn}/${any.regionsSold}/${any.vendorsIn})`);
  const h8 = stockView(data, { gpu: "H100", count: 8 });
  check(h8.regionsSold === 2 && h8.regionsIn === 1, "H100 x8 is sold in two regions and in stock in one");
  check(h8.vendors[0].vendor === "hyperstack", "the vendor with stock comes first");
  const lambda = h8.vendors.find((v) => v.vendor === "lambdalabs");
  check(lambda && lambda.regions.length === 1 && lambda.regions[0].region.region === "phoenix-usa-2",
        "a region that does not sell the shape is not on the board for it");

  const onlyIn = stockBoardHtml(h8, { gpu: "H100", count: 8 }, true, false);
  check(onlyIn.includes("CA, Montreal") && !onlyIn.includes("US, Phoenix"),
        "with 'only regions with stock' the out-of-stock region is left out");
  const all = stockBoardHtml(h8, { gpu: "H100", count: 8 }, false, false);
  check(all.includes("US, Phoenix") && all.includes(">No stock<"), "without it, the region says No stock in words");
  check(all.includes('class="vstrip"') && all.includes('<i class="ok"'), "each vendor draws its region strip");

  const shapes = stockShapesHtml(stockView(data, { gpu: "A100_80G", count: 0 }).vendors[0].regions[0].hits);
  check(shapes.includes("A100 80GB<i>×1 ×8</i>"), `one chip per GPU with its counts (got ${shapes})`);

  const none = stockBoardHtml(stockView(data, { gpu: "H200", count: 8 }), { gpu: "H200", count: 8 }, true, true);
  check(none.includes("No region has H200 ×8 in stock right now.") && none.includes("Set an alert"),
        "an empty filter says so and points at the alerts when they are on");
  const noneOff = stockBoardHtml(stockView(data, { gpu: "H200", count: 8 }), { gpu: "H200", count: 8 }, true, false);
  check(!noneOff.includes("Set an alert"), "and does not mention alerts to an account without them");
}

console.log("2. the GPU chips");
{
  const counts = stockGpuCounts(data, 0);
  const asMap = Object.fromEntries(counts);
  check(asMap.H100 === 2 && asMap.H200 === 0 && asMap.A10 === 0, "H100 in 2 regions, H200 and A10 listed at 0");
  check(counts[0][0] === "H100", "most regions first");
  const folded = stockChipsHtml(data, { gpu: "", count: 0 }, false);
  check(!folded.includes(">H200 <") && folded.includes("+2 without stock"), "GPUs with no stock fold behind one chip");
  const open = stockChipsHtml(data, { gpu: "", count: 0 }, true);
  check(open.includes("H200") && open.includes("Hide GPUs without stock"), "and unfold");
  const pressed = stockChipsHtml(data, { gpu: "H100", count: 8 }, false);
  check(pressed.includes('data-gpu="H100" aria-pressed="true"') && pressed.includes('data-gpu="" aria-pressed="false"'),
        "the chosen chip is pressed, Any is not");
  const at8 = Object.fromEntries(stockGpuCounts(data, 8));
  check(at8.H100 === 1 && at8.A10 === undefined, "at 8 GPUs per machine only shapes sold at 8 count");
}

console.log("3. the filter in the address");
{
  const f = stockFilterFrom("gpu=H100&count=8");
  check(f.gpu === "H100" && f.count === 8, "gpu=H100&count=8 reads back");
  const bad = stockFilterFrom("count=lots");
  check(bad.gpu === "" && bad.count === 0, "an unreadable count is any");
  check(stockFilterFrom("").count === 0, "no query is any");
  check(stockShapeWord({ gpu: "A100_80G", count: 1 }) === "A100 80GB ×1", "the filter in words");
}

console.log("4. freshness, cards and alerts");
{
  const at = Date.parse("2026-10-24T08:02:01Z");
  const fresh = stockFreshHtml(data, at);
  check(fresh.includes("Updated <b>07:50 UTC</b> · 12 min ago") && !fresh.includes("stale"), `fresh: ${fresh}`);
  const stale = stockFreshHtml({ ...data, stale: true }, at + 3 * 3600 * 1000);
  check(stale.includes("live-dot stale") && stale.includes("Stale · last read"), "a stale catalog says so and the dot stops");
  const cards = stockCardsHtml(data, stockView(data, { gpu: "H100", count: 8 }), { gpu: "H100", count: 8 });
  check(cards.includes("2 of 4") && cards.includes("H100 ×8 in stock") && cards.includes("1 of 2 regions"),
        "the cards: totals, and the filter's own count");

  const rows = stockAlertRowsHtml([
    { id: "a1", gpu: "H100", gpus: 8, vendor: null, created_at: "2026-10-24T07:00:00Z",
      state: { in_stock: true, notified_at: "2026-10-24T07:52:00Z" } },
    { id: "a2", gpu: "A100_80G", gpus: null, vendor: "hyperstack", created_at: "2026-10-24T07:10:00Z",
      state: { in_stock: false, checked_at: "2026-10-24T07:52:00Z" } },
    { id: "a3", gpu: "B200", gpus: 8, vendor: null, created_at: "2026-10-24T07:20:00Z",
      state: { in_stock: null } },
    { id: "a4", gpu: "L40S", gpus: 1, vendor: null, created_at: "2026-10-24T07:30:00Z",
      state: { in_stock: true, notified_at: null } },
  ]);
  check(rows.includes("DM sent 10-24 07:52 UTC"), "in stock and told");
  check(rows.includes("no stock at 10-24 07:52 UTC") && rows.includes("Hyperstack"), "waiting, with the vendor's name");
  check(rows.includes("Not checked yet") && rows.includes("first check within 10 minutes"), "a new alert");
  check(rows.includes("DM not sent yet"), "in stock but the DM has not gone");
  check(rows.indexOf("L40S") < rows.indexOf("H100"), "newest first");
  check(stockAlertRowsHtml([]).includes("No alert set."), "no alerts");

  fields["stock-a-gpu"] = { value: "H100" };
  fields["stock-a-count"] = { value: "" };
  fields["stock-a-vendor"] = { value: "" };
  const body = readStockAlertForm();
  check(body.gpu === "H100" && body.gpus === null && body.vendor === null && body.region === null,
        `"" is sent as null (got ${JSON.stringify(body)})`);
  fields["stock-a-count"].value = "8";
  check(readStockAlertForm().gpus === 8, "a count is sent as a number");
}

console.log("5. ids");
{
  const ids = new Set([...SRC.matchAll(/\$\("(stock-[a-z-]+)"\)/g)].map((m) => m[1]));
  const missing = [...ids].filter((id) => !HTML.includes(`id="${id}"`));
  check(ids.size >= 10 && missing.length === 0, `every stock-* id app.js uses is in index.html (missing: ${missing.join(", ") || "none"})`);
  check(HTML.includes('data-view="stock"') && HTML.includes('id="view-stock"'), "the nav item and the section exist");
}

if (failures.length) {
  console.error(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nall stock checks passed");
