/* servings_test.js -- the Servings screen (HYPERUN-SERVING, HYPERUN-SERVING-KEY).
 *
 * WHAT IT PINS.
 *   1. readServingForm sends exactly what the server's POST /v1/servings takes: one model source,
 *      the revision only with an HF id, empty token boxes LEFT OUT (the server's 763 / 232
 *      apply), an empty budget as null (so the server names it), the token NAME only.
 *   2. servingCostHtml shows the hourly and daily ceiling before anything is created, and the
 *      total cap when one is set.
 *   3. servingRowsHtml, servingFactsHtml, servingChosenHtml and servingPlansHtml draw fixture
 *      data shaped like GET /v1/servings/{name}: the chosen plan's machines with layers and TP,
 *      the ranked plans with score, $/h, req/s, VMs and whether each can be rented and why not,
 *      the catalog, and rank 1 never called "cheapest".
 *   4. The API key is never written to storage, is cleared on every route change, and every id
 *      app.js reaches for on this screen exists in index.html.
 *
 * HOW IT RUNS THE REAL CODE. As serving_plan_test.js does: each function is extracted from the
 * shipped app.js by name and evaluated, so a pass here is a pass of the deployed code.
 *
 * Run: node ui/servings_test.js      (no browser, no network, no login)
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

/* A top-level `const NAME = ...;`. A one-liner ends at the first ";"; an arrow function with a
 * block body (`badge`) ends at the line that is exactly "};". */
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
["esc", "fact", "when", "badge"].forEach((name) =>
  define(declaration(name).replace(/^const /, "globalThis.")));
["statusOf", "empty", "readServingForm", "servingCostHtml", "servingRowsHtml", "servingFactsHtml",
 "servingChosenHtml", "whyNotRented", "servingPlansHtml", "modelConfigHtml", "hideServingKey"]
  .forEach((name) => define(`globalThis.${name} = (${extract(name)});`));

const fields = {};
globalThis.document = {
  getElementById: (id) => {
    if (!fields[id]) fields[id] = { value: "", innerHTML: "", textContent: "", hidden: false };
    return fields[id];
  },
};
globalThis.$ = (id) => document.getElementById(id);
globalThis.note = (kind, text) => `[${kind}] ${text}`;

function setForm(o) {
  Object.keys(fields).forEach((id) => { fields[id].value = ""; });
  $("sv-name").value = o.name ?? "chat-8b";
  $("sv-source").value = o.source ?? "preset";
  $("sv-preset").value = o.preset ?? "llama-3.1-8b";
  $("sv-hf").value = o.hf ?? "";
  $("sv-url").value = o.url ?? "";
  $("sv-revision").value = o.revision ?? "";
  $("sv-in").value = o.input ?? "";
  $("sv-out").value = o.output ?? "";
  $("sv-budget").value = o.budget ?? "";
  $("sv-max").value = o.max ?? "";
  $("sv-image").value = o.image ?? "";
  $("sv-token").value = o.token ?? "";
}

// ---------------------------------------------------------------------------------------------
console.log("readServingForm: the POST /v1/servings body");

setForm({ budget: "3" });
check(JSON.stringify(readServingForm()) === JSON.stringify({
  name: "chat-8b", model: { preset: "llama-3.1-8b" }, hourly_budget: 3 }),
  "a preset with only a budget sends the name, the preset and the budget, nothing else");

setForm({ source: "hf", hf: " Qwen/Qwen3-8B ", revision: "main", input: "512", output: "128",
          budget: "2.5", max: "40", image: "vllm/vllm-openai:v0.10.0", token: "HF_TOKEN",
          preset: "llama-3.1-8b" });
let body = readServingForm();
check(JSON.stringify(body) === JSON.stringify({
  name: "chat-8b", model: { name: "Qwen/Qwen3-8B", revision: "main" }, avg_input_tokens: 512,
  avg_output_tokens: 128, hourly_budget: 2.5, max_spend_usd: 40,
  image: "vllm/vllm-openai:v0.10.0", hf_token_secret: "HF_TOKEN" }),
  "an HF id sends the id trimmed, its revision, every filled box, and the token's NAME");
check(!("preset" in body.model), "only the chosen source is sent, not the preset still selected");

setForm({ source: "url", url: "https://models.example.com/config.json", revision: "v2" });
body = readServingForm();
check(JSON.stringify(body.model) === JSON.stringify({
  config_url: "https://models.example.com/config.json" }),
  "a URL sends no revision: the URL names its own");
check(body.hourly_budget === null && !("avg_input_tokens" in body) && !("avg_output_tokens" in body),
  "an empty budget is null for the server to name; empty token boxes are left to the server's defaults");

// ---------------------------------------------------------------------------------------------
console.log("servingCostHtml: what it costs before it exists");

check(servingCostHtml(3, NaN) === "[info] Once a plan is rented it bills at most $3.00 an hour, " +
  "$72.00 a day, until you delete the serving. There is no total cap unless you set one.",
  "a $3/h budget is $72.00 a day, and the note says nothing stops it but a delete");
check(servingCostHtml(1.35, 40).endsWith("or $40.00 in total is spent."),
  "a total cap is named when it is set");
check(servingCostHtml(NaN, NaN) === "" && servingCostHtml(0, 5) === "",
  "no budget, no cost note");

// ---------------------------------------------------------------------------------------------
// Fixture: GET /v1/servings/chat-8b, shaped as servings.ServingView returns it.
// ---------------------------------------------------------------------------------------------
const VIEW = {
  name: "chat-8b", namespace: "lab-alice", phase: "Serving", message: "",
  created_at: "2026-10-20T01:00:00Z",
  model: { name: "meta-llama/Llama-3.1-8B-Instruct", preset: "llama-3.1-8b", revision: "",
           config_source: "preset:llama-3.1-8b", config: {} },
  avg_input_tokens: 763, avg_output_tokens: 232, hourly_budget: "3.0", max_spend_usd: null,
  serving_plans: [
    { rank: 2, score: "0.308", cost_per_hour: "8.0700", throughput_rps: "2.488",
      latency_seconds: "26.021", nodes: 5, executable: false,
      pipelines: [{ domain: "hyperstack/montreal-canada-2", vendor: "shadeform" },
                  { domain: "massedcompute/kansascity-usa-1", vendor: "shadeform" }] },
    { rank: 1, score: "12.558", cost_per_hour: "1.3500", throughput_rps: "16.953",
      latency_seconds: "1.760", nodes: 1, executable: true,
      pipelines: [{ domain: "hyperstack/montreal-canada-2", vendor: "shadeform" }] },
    { rank: 3, score: "0.900", cost_per_hour: "1.2000", throughput_rps: "1.080", nodes: 1,
      executable: false, pipelines: [{ domain: "aws/us-west-2", vendor: "aws" }] },
  ],
  serving_plan_chosen: { rank: 1, score: "12.558", cost_per_hour: "1.3500",
                         throughput_rps: "16.953", latency_seconds: "1.760", attempt: 1,
                         stages: [{ cloud: "hyperstack", region: "montreal-canada-2",
                                    instance_type: "A100_80G", layers: 32, tp: 1 }] },
  serving_catalog: "Shadeform catalog fetched 2026-10-20T02:39:32Z (live ConfigMap " +
                   "hyperun-shadeform-catalog)",
  endpoint: { path: "/v1/servings/lab-alice/chat-8b/v1", ready: true },
  billing: "Renting $1.35/h now (rank 1), $32.40 a day. There is no total cap: it runs until " +
           "you delete it.",
};

console.log("the list and one serving's page");

let html = servingRowsHtml([VIEW]);
check(html.includes('<tr class="click" data-serving="chat-8b">') &&
      html.includes("<td>ready</td>") && html.includes('<td class="num">$1.3500/h</td>'),
  "a list row links to the serving and says ready and what it bills now");
check(servingRowsHtml([]).includes("No servings yet"), "an empty list says how to start one");

html = servingFactsHtml(VIEW);
check(html.includes("meta-llama/Llama-3.1-8B-Instruct") && html.includes("preset llama-3.1-8b") &&
      html.includes("763 tokens in, 232 out") && html.includes("$3.0/h"),
  "the facts carry the model, where its numbers came from, the traffic and the budget");

let view = servingChosenHtml(VIEW);
check(view.summary === "rank 1, $1.3500/h, attempt 1",
  "the chosen plan's summary names its rank, price and attempt");
check(view.html.includes('<td class="num">0</td><td class="mono">A100_80G</td>' +
                         "<td>hyperstack / montreal-canada-2</td>" +
                         '<td class="num">32</td><td class="num">1</td>'),
  "each machine of the chosen plan is a row: rank, type, where, layers, TP");
check(view.html.includes("12.558 req/s per $/h"), "the chosen plan's score is shown");
check(servingChosenHtml({ ...VIEW, serving_plan_chosen: null }).summary === "none yet",
  "no plan chosen yet says so");

view = servingPlansHtml(VIEW);
check(view.summary === "3 plans, rank 1 scores 12.558 at $1.3500/h",
  "the summary names rank 1 by its score even when it is not the first row");
check(!view.summary.includes("cheapest") && !view.html.includes("cheapest"),
  "nothing calls a plan the cheapest: rank 3 here costs less per hour than rank 1");
check(view.html.includes("Ranked on: Shadeform catalog fetched 2026-10-20T02:39:32Z"),
  "the catalog the plans were ranked on is named");
check(/<th>Rank<\/th><th title="[^"]*">Score<\/th><th>\$\/h<\/th><th>req\/s<\/th><th>Latency<\/th><th>VMs<\/th>/
        .test(view.html),
  "the columns are rank, score, $/h, req/s, latency, VMs, then whether it can be rented");
check(view.html.includes('<tr class="chosen"><td class="num">1 <span class="tiny"><b>chosen</b></span></td>'),
  "the chosen plan's row is marked in words");
check(view.html.includes('no <span class="dim tiny">2 pipelines, 5 machines</span>') &&
      view.html.includes('no <span class="dim tiny">priced from AWS&#39;s table</span>') &&
      view.html.includes("<td>yes</td>"),
  "a plan that cannot be rented says what it is: its pipelines, machines or AWS pricing");
check(whyNotRented({ executable: true, nodes: 5 }) === "" &&
      whyNotRented({ executable: undefined }) === "",
  "a rentable plan, or one the operator said nothing about, gets no reason");

view = servingPlansHtml({ serving_plans: [], serving_catalog: "" });
check(view.summary === "no plans yet" && view.html.includes("has not answered yet"),
  "a serving whose planner has not answered says so");

html = servingRowsHtml([{ ...VIEW, name: "<script>x</script>" }]);
check(!html.includes("<script>") && html.includes("&lt;script&gt;"),
  "a value from the server is escaped before it reaches innerHTML");

html = modelConfigHtml({ name: "Qwen/Qwen3-8B", family: "Qwen3 dense",
  config: { architecture: "Qwen3ForCausalLM", num_layers: 36, head_dim: 128 },
  notes: ["tie_word_embeddings is true"] });
check(html.startsWith("[info] Qwen/Qwen3-8B: Qwen3 dense.") && html.includes("Layers") &&
      html.includes("36") && html.includes("tie_word_embeddings is true"),
  "a lookup shows the family, the numbers and the notes on what was derived");

// ---------------------------------------------------------------------------------------------
console.log("the key, and the wiring");

$("sv-d-key-value").textContent = "sk-serving-test";
$("sv-d-key").hidden = false;
hideServingKey();
check($("sv-d-key-value").textContent === "" && $("sv-d-key").hidden === true &&
      $("sv-d-key-toggle").textContent === "Show API key",
  "hiding the key clears its text, not only its visibility");
check(/poll\.stop\(\);[\s\S]{0,900}hideServingKey\(\);/.test(extract("route")),
  "every route change clears a shown key (route calls hideServingKey)");
check(/function drawServing\(name\) \{\s*hideServingKey\(\);/.test(SRC),
  "opening any serving page starts with no key on it");
// The Servings code, from its banner to the next one. The login flow's sessionStorage (the PKCE
// verifier) is elsewhere and legitimate; nothing in THIS block may touch storage at all.
const SERVINGS_BLOCK = SRC.slice(SRC.indexOf("3c. Servings */"), SRC.indexOf("3b. Scripts */"));
check(SERVINGS_BLOCK.length > 5000 && !/Storage|document\.cookie/.test(SERVINGS_BLOCK),
  "nothing on the Servings screen writes to browser storage, so the key cannot outlive the page");
check(/else if \(head === "servings"\) \{ show\("servings"\); drawServings\(arg \|\| ""\); \}/.test(SRC),
  "#/servings and #/servings/<name> both reach drawServings");
check(/<button data-view="servings">(<i data-lucide="[a-z0-9-]+"><\/i>)?Servings<\/button>/.test(HTML) &&
      HTML.includes('<section id="view-servings" hidden>'),
  "the nav has a Servings button and the screen exists, so show() finds it");

const ids = [...SRC.matchAll(/\$\("(sv-[a-z-]*)"\)/g)].map((m) => m[1]);
const missing = [...new Set(ids)].filter((id) => !HTML.includes(`id="${id}"`));
check(ids.length > 30 && missing.length === 0,
  `every Servings id app.js reaches for exists in index.html${missing.length ? ": missing " + missing : ""}`);

if (failures.length) {
  console.log(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nall checks passed");
