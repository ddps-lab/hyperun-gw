/* serving_plan_test.js -- the serving plan on the New job screen and on the job page.
 *
 * WHAT IT PINS (HYPERUN-SERVING-PLAN, HYPERUN-SERVING-CHECK).
 *   1. readForm sends a `serving_plan` block exactly when a model is chosen, with the two token
 *      counts as integers, the two rates as decimals, and an empty box as null -- so the
 *      server's validate names the missing field instead of this screen guessing one.
 *   2. servingRules forces what a serving plan leaves no choice about (on-demand, no vendor, no
 *      region, no GPU), gives it all back when the model is cleared, and says whether the job
 *      will rent anything.
 *   3. servingPlanHtml draws the ranked plans from fixture data shaped like GET /v1/jobs/{id}:
 *      one row per pipeline, the plan's numbers spanning them, the rented plan marked with its
 *      attempt, and a one-line note in compare mode that nothing was rented.
 *   4. servingCheckHtml lays the planner's estimate beside what vLLM measured, from a fixture
 *      shaped like the serving check's `serving-check-result.json` (hyperun.serving-check/v1),
 *      and prints null as "not measured", never as 0.
 *   5. The wiring: the panels are hidden on every open, drawn by the poll and by drawArtifacts,
 *      and every id app.js reaches for exists in index.html.
 *
 * HOW IT RUNS THE REAL CODE. As submit_form_test.js does: each function under test is extracted
 * from the shipped app.js by name and evaluated, so a pass here is a pass of the deployed code.
 * The one-line helpers those functions close over (esc, fact, when, TERMINAL) are taken from
 * app.js the same way, by their declaration line, rather than re-written here.
 *
 * Run: node ui/serving_plan_test.js      (no browser, no network, no login)
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

/* Pull one function out of app.js by name, brace-counting to its end. A rename throws rather
 * than silently testing nothing. */
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

/* A top-level `const NAME = ...;` declaration, from its line to the first line ending in ";". */
function declaration(name) {
  const lines = SRC.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`const ${name} =`));
  if (start < 0) throw new Error(`app.js has no const ${name}`);
  const out = [];
  for (let i = start; i < lines.length; i++) {
    out.push(lines[i]);
    if (lines[i].trimEnd().endsWith(";")) break;
  }
  return out.join("\n");
}

/* Evaluate at global scope (indirect eval), so the extracted functions find each other and the
 * stubs below as globals, exactly as they do in the page. */
const define = (src) => (0, eval)(src);
["esc", "fact", "when", "TERMINAL", "SERVING_CHECK_FILE"].forEach((name) =>
  define(declaration(name).replace(/^const /, "globalThis.")));
["readForm", "vendorRules", "servingRules", "servingPlanHtml", "servingCheckHtml"]
  .forEach((name) => define(`globalThis.${name} = (${extract(name)});`));

// ---------------------------------------------------------------------------------------------
// The stub document: fields by id, the vendor boxes, the selects' options, the summary note.
// ---------------------------------------------------------------------------------------------
const fields = {};
const capacityOptions = [{ value: "on-demand" }, { value: "spot" }];
const modeOptions = [{ value: "cheapest" }, { value: "" }, { value: "compare" }];
const summaryNote = { textContent: "" };
const vendorBoxes = ["aws", "runpod", "shadeform"].map((v) => ({ dataset: { vendor: v } }))
  .concat(["gcp", "azure", "lambda", "nebius"].map((v) => ({ dataset: { vendor: v, priced: "" } })));

globalThis.document = {
  getElementById: (id) => {
    if (!fields[id]) {
      fields[id] = { value: "", innerHTML: "", textContent: "", disabled: false };
      if (id === "f-capacity") fields[id].options = capacityOptions;
      if (id === "f-mode") fields[id].options = modeOptions;
      if (id === "f-serving") fields[id].querySelector = () => summaryNote;
    }
    return fields[id];
  },
  querySelectorAll: (selector) =>
    selector.includes("data-priced") ? vendorBoxes.filter((b) => "priced" in b.dataset) : vendorBoxes,
};
globalThis.$ = (id) => document.getElementById(id);
globalThis.note = (kind, text) => `[${kind}] ${text}`;

function setForm(o) {
  Object.keys(fields).forEach((id) => { fields[id].value = ""; fields[id].disabled = false; });
  $("f-image").value = "vllm/vllm-openai:v0.10.0";
  $("f-capacity").value = o.capacity ?? "on-demand";
  $("f-mode").value = o.mode ?? "cheapest";
  $("f-gpu").value = o.gpu ?? "";
  $("f-regions").value = o.regions ?? "";
  $("f-serve-model").value = o.model ?? "";
  $("f-serve-in").value = o.input ?? "";
  $("f-serve-out").value = o.output ?? "";
  $("f-serve-rps").value = o.rps ?? "";
  $("f-serve-budget").value = o.budget ?? "";
  vendorBoxes.forEach((b) => { b.checked = (o.vendors || []).includes(b.dataset.vendor); b.disabled = false; });
  capacityOptions.concat(modeOptions).forEach((opt) => { opt.disabled = false; });
}

// ---------------------------------------------------------------------------------------------
console.log("readForm: the serving_plan block");

setForm({});
check(!("serving_plan" in readForm()), "no model chosen sends no serving_plan at all");

setForm({ model: "llama-3.1-8b", input: "763", output: "232", rps: "2.5", budget: "20.8333" });
let body = readForm();
check(JSON.stringify(body.serving_plan) === JSON.stringify({
  model: "llama-3.1-8b", avg_input_tokens: 763, avg_output_tokens: 232,
  target_rps: 2.5, hourly_budget: 20.8333,
}), "a chosen model sends the five fields, tokens as integers and the two rates as decimals");

setForm({ model: "llama-3.1-70b", input: "763" });
body = readForm();
check(body.serving_plan.avg_output_tokens === null && body.serving_plan.target_rps === null &&
      body.serving_plan.hourly_budget === null,
  "an empty box goes out as null, so validate names it; the placeholder is not a default");

// ---------------------------------------------------------------------------------------------
console.log("servingRules: what choosing a model does to the rest of the form");

setForm({ capacity: "spot", gpu: "L40S", regions: "aws/us-east-1", vendors: ["aws", "gcp"] });
$("f-serve-model").value = "llama-3.1-8b";
servingRules();
check($("f-capacity").value === "on-demand" && capacityOptions[1].disabled,
  "capacity is on-demand and spot cannot be picked: Shadeform sells nothing else");
check(vendorBoxes.every((b) => !b.checked && b.disabled),
  "every vendor box is emptied and disabled: only the placement word shadeform is answered");
check($("f-regions").value === "" && $("f-regions").disabled &&
      $("f-gpu").value === "" && $("f-gpu").disabled,
  "regions and GPU are emptied and disabled: the planner picks the machines");
check(modeOptions.every((o) => !o.disabled),
  "the gcp box it unchecked no longer pins the mode to compare (vendorRules ran)");
check(summaryNote.textContent === "llama-3.1-8b, sent with this job",
  "the folded section's summary says a plan is still being sent");
check(/rents the cheapest plan that runs on one machine/.test($("f-serve-note").innerHTML) &&
      /Choose compare/.test($("f-serve-note").innerHTML),
  "under cheapest the note says the job will rent, and how to only rank instead");

$("f-mode").value = "compare";
servingRules();
check(/nothing is rented/.test($("f-serve-note").innerHTML),
  "under compare the note says nothing is rented");

$("f-serve-model").value = "";
servingRules();
check(!capacityOptions[1].disabled && vendorBoxes.every((b) => !b.disabled) &&
      !$("f-regions").disabled && !$("f-gpu").disabled && $("f-serve-note").innerHTML === "",
  "clearing the model gives every box back and drops the note");

// ---------------------------------------------------------------------------------------------
// Fixture: GET /v1/jobs/{id}, built from the real planner output PACSrun keeps as a fixture
// (rank_shadeform_2026-10-19.json): rank 1 is two pipelines on five machines, rank 3 one
// 8-GPU machine.
// ---------------------------------------------------------------------------------------------
const PLANS = [
  { rank: 1, price_basis: "on_demand", cost_per_hour: "8.0700", throughput_rps: "2.488",
    latency_seconds: "26.021", nodes: 5, executable: false, pipelines: [
      { domain: "hyperstack/montreal-canada-2", throughput_rps: "1.274", cost_per_hour: "4.0500",
        stages: [
          { instance_type: "hyperstack_A100_80Gx2", zone: "montreal-canada-2", layers: 53, tp: 2,
            price_per_hour: "2.7000" },
          { instance_type: "hyperstack_A100_80G", zone: "montreal-canada-2", layers: 27, tp: 1,
            price_per_hour: "1.3500" }] },
      { domain: "massedcompute/kansascity-usa-1", throughput_rps: "1.214", cost_per_hour: "4.0200",
        stages: [
          { instance_type: "massedcompute_L40S", zone: "kansascity-usa-1", layers: 16, tp: 1,
            price_per_hour: "0.8800" }] }] },
  { rank: 3, price_basis: "on_demand", cost_per_hour: "10.8000", throughput_rps: "12.283",
    latency_seconds: "2.042", nodes: 1, executable: true, pipelines: [
      { domain: "excesssupply/omaha-usa-2", throughput_rps: "12.283", cost_per_hour: "10.8000",
        stages: [{ instance_type: "excesssupply_A100_sxm4_80Gx8_2", zone: "omaha-usa-2",
                   layers: 80, tp: 8, price_per_hour: "10.8000" }] }] },
];
const ASKED = { planner: "shuntserve", model: "llama-3.1-70b", avgInputTokens: 763,
                avgOutputTokens: 232, targetRPS: "2.0", hourlyBudget: "20.8333" };
const CHOSEN = { rank: 3, instance_type: "excesssupply_A100_sxm4_80Gx8_2", cloud: "excesssupply",
                 region: "omaha-usa-2", cost_per_hour: "10.8000", throughput_rps: "12.283",
                 latency_seconds: "2.042", attempt: 2 };

console.log("servingPlanHtml: the ranked table");

let view = servingPlanHtml({ phase: "Compared", placement_mode: "compare", serving_plans: PLANS,
                             serving_plan_chosen: null }, ASKED);
check(view.html.startsWith("[info] Compare mode: these plans were priced and nothing was rented."),
  "compare mode leads with the one line that nothing was rented");
check(view.summary === "2 plans, cheapest $8.0700/h, nothing rented",
  "the folded summary carries the count, the cheapest price and that nothing was rented");
check(view.html.includes("Asked for llama-3.1-70b, 763 tokens in and 232 out, 2.0 req/s within $20.8333/h."),
  "the ask is restated from the spec, budget as dollars per hour");
check((view.html.match(/<tr>|<tr class/g) || []).length === 1 + 3,
  "one header row and one row per PIPELINE: rank 1 has two, rank 3 has one");
check((view.html.match(/rowspan="2"/g) || []).length === 5,
  "rank 1's rank, req/s, $/h, latency and executable span its two pipeline rows");
check(view.html.includes('hyperstack_A100_80Gx2 <span class="dim">53 layers, TP 2</span><br>' +
                         'hyperstack_A100_80G <span class="dim">27 layers, TP 1</span>'),
  "a pipeline names each machine with the layers it holds and its TP, in stage order");
check(view.html.includes("1.274 req/s, $4.0500/h"),
  "a plan of several pipelines shows each pipeline's own share under its domain");
check(view.html.includes("$8.0700") && view.html.includes("26.021 s") && view.html.includes("2.488"),
  "the plan's $/h, latency and req/s are the server's strings, unrounded");
check(/<td class="span-top" rowspan="2">no<\/td>/.test(view.html) &&
      /<td class="span-top">yes<\/td>/.test(view.html),
  "executable reads no for the five-machine plan and yes for the one-machine plan");
check(!view.html.includes('class="chosen"'), "compare mode marks no plan as chosen");

view = servingPlanHtml({ phase: "Running", placement_mode: null, serving_plans: PLANS,
                         serving_plan_chosen: CHOSEN }, ASKED);
check(!view.html.includes("Compare mode"), "a renting job gets no compare note");
check((view.html.match(/<tr class="chosen">/g) || []).length === 1 &&
      view.html.includes("<b>chosen</b>, attempt 2"),
  "the rented plan's row is marked, in words, with its attempt number");
check(view.html.includes("rank 3, attempt 2") && view.html.includes("excesssupply / omaha-usa-2") &&
      view.html.includes("$10.8000/h"),
  "the facts above the table name the rented plan, where it is and what it costs");
check(view.summary === "2 plans, cheapest $8.0700/h, rented rank 3",
  "the summary says which rank was rented");

const older = PLANS.map((p) => ({ ...p, executable: undefined, latency_seconds: undefined }));
view = servingPlanHtml({ phase: "Compared", placement_mode: "compare", serving_plans: older }, null);
check(/<td class="span-top" rowspan="2">-<\/td>/.test(view.html) && !view.html.includes("undefined"),
  "an operator that writes no executable or latency reads as -, never as no or undefined");

view = servingPlanHtml({ phase: "Pending", serving_plans: [] }, ASKED);
check(view.html.includes("The planner has not answered yet") && view.summary === "no plans yet",
  "a job whose planner is still working says so, with how long one run took");
view = servingPlanHtml({ phase: "Compared", placement_mode: "compare", serving_plans: [] }, ASKED);
check(view.html.includes("No plan came back"), "a finished job with no plans says none came back");

view = servingPlanHtml({ phase: "Compared", serving_plans: [{ ...PLANS[1], pipelines: [
  { domain: "<script>x</script>", stages: [] }] }] }, null);
check(!view.html.includes("<script>") && view.html.includes("&lt;script&gt;"),
  "a value from the server is escaped before it reaches innerHTML");

// ---------------------------------------------------------------------------------------------
// Fixture: GET /v1/jobs/{id}/serving-check, the file's fields as serve_check.sh writes them.
// ---------------------------------------------------------------------------------------------
const CHECK = {
  schema: "hyperun.serving-check/v1", status: "ok", failed_phase: null,
  message: "all three runs completed", written_at: "2026-10-19T18:40:00Z",
  model: { id: "meta-llama/Llama-3.1-8B-Instruct" },
  software: { vllm_version: "0.10.0" },
  gpu: { count: 1, cards: [{ index: 0, name: "NVIDIA A100 80GB PCIe", memory_total_mib: 81920 }] },
  plan: { cloud: "hyperstack", region: "montreal-canada-2", shade_instance_type: "A100_80G",
          usd_per_hour: 1.35, throughput_rps: 16.953, single_request_latency_ms: 1760 },
  workload: { input_tokens: 763, output_tokens: 232, target_rate_rps: 4.67 },
  runs: { target: { request_throughput_rps: 4.61, completed: 1400, num_prompts: 1400 } },
  comparison: { max_rps_measured: 14.2, max_rps_planned: 16.953, max_rps_ratio: 0.838,
                single_latency_s_measured: 1.9, single_latency_s_planned: 1.76,
                single_latency_ratio: 1.08, target_rate_sustained: true },
};

console.log("servingCheckHtml: estimate beside measurement");

view = servingCheckHtml({ found: true, name: "serving-check-result.json", check: CHECK });
check(view.html.includes("<th>Planner estimate</th><th>Measured</th><th>Measured / estimate</th>"),
  "the columns are estimate, measurement and their ratio, side by side");
check(view.html.includes('<td>Throughput at full load</td><td class="num">16.953 req/s</td>' +
                         '<td class="num">14.2 req/s</td><td class="num">0.838</td>'),
  "throughput: the planner's 16.953 req/s beside vLLM's 14.2, ratio as the script computed it");
check(view.html.includes('<td>Latency of one request alone</td><td class="num">1.76 s</td>' +
                         '<td class="num">1.9 s</td><td class="num">1.08</td>'),
  "latency: the planner's 1.76 s beside the measured 1.9 s");
check(view.html.includes('<td class="num">4.67 req/s</td><td class="num">4.61 req/s, held</td>'),
  "the target-rate run says what it reached and whether the rate was held");
check(view.html.includes("NVIDIA A100 80GB PCIe × 1") &&
      view.html.includes("hyperstack / montreal-canada-2 A100_80G, $1.35/h"),
  "the facts name the card that was measured and the plan row it came from");
check(view.summary === "measured 14.2 req/s against 16.953 estimated",
  "the folded summary carries the headline comparison");

const failed = { ...CHECK, status: "failed", failed_phase: "bench-2/3", message: "server died",
  comparison: { ...CHECK.comparison, max_rps_measured: null, max_rps_ratio: null,
                target_rate_sustained: null } };
view = servingCheckHtml({ found: true, name: "serving-check-result.json", check: failed });
check(view.html.startsWith("[err] The check failed in bench-2/3: server died."),
  "a failed check leads with the phase it failed in and its reason");
check(view.html.includes('<td class="num">not measured</td>') && !/>0 req\/s</.test(view.html),
  "a null measurement prints as not measured, never as 0");
check(view.html.includes('<td class="num">4.61 req/s, not measured</td>'),
  "a target run with no verdict on whether the rate held says not measured, rather than held or not held");
check(view.summary === "failed in bench-2/3", "the summary says it failed and where");

view = servingCheckHtml({ found: false, note: "serving-check-result.json is not in this job's result path (yet)." });
check(view.html === "[info] serving-check-result.json is not in this job's result path (yet).",
  "nothing found shows the server's own reason");

// ---------------------------------------------------------------------------------------------
console.log("wiring");

check(/drawCompare\(job\);\s*drawServingPlan\(job\);/.test(SRC),
  "the 5-second poll draws the Serving plan panel next to the comparison");
check(/\$\("d-serving-panel"\)\.hidden = true;\s*\$\("d-servecheck-panel"\)\.hidden = true;/.test(SRC),
  "both panels are hidden on every open, so a previous job's plans cannot show (HYPERUN-UI-STALE-PANELS)");
check(/a\.files\.some\(\(f\) => f\.name === SERVING_CHECK_FILE\)\) drawServingCheck\(jobId\)/.test(SRC),
  "drawArtifacts asks for the serving check only when the file is in the listing");
check(SERVING_CHECK_FILE === "serving-check-result.json",
  "the file name is the one serve_check.sh writes");
check(/if \(detailJob\) drawServingPlan\(detailJob\);/.test(SRC),
  "the spec's late arrival redraws the panel, because a finished job's poll has already stopped");
check(/\$\("f-mode"\)\.onchange = servingRules;/.test(SRC) &&
      /\$\("f-serve-model"\)\.onchange = servingRules;/.test(SRC),
  "the mode select and the model select both keep the serving note true");

const ids = [...SRC.matchAll(/\$\("((?:f-serve|d-serv|f-serving)[a-z-]*)"\)/g)].map((m) => m[1]);
const missing = [...new Set(ids)].filter((id) => !HTML.includes(`id="${id}"`));
check(ids.length > 10 && missing.length === 0,
  `every serving id app.js reaches for exists in index.html${missing.length ? ": missing " + missing : ""}`);
check(HTML.includes('<option value="llama-3.1-8b">') && HTML.includes('<option value="llama-3.1-70b">'),
  "the model select offers exactly the planner's two presets");
check(HTML.includes('id="d-log-panel"') && HTML.includes('id="d-terminal-panel"'),
  "the Logs and Terminal panels are still on the job page");

if (failures.length) {
  console.log(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nall checks passed");
