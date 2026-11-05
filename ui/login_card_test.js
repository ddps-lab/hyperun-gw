/* login_card_test.js -- what the sign-in card offers, from what GET /v1/login-config said
 * (HYPERUN-UI-LOGIN-DOWN).
 *
 * WHY THIS FILE EXISTS. On 2026-11-05 the gateway was scaled to zero on purpose, and
 * ingress-nginx answered /v1/login-config with its 503 page. The page read that the same way as
 * a deployment with no user pool, so the card showed only "Sign in with a token", and it looked
 * as if Google sign-in had been removed. A token could not have worked either: the API was down.
 * This file holds the three answers apart:
 *
 *   1. 200, enabled: true    -> the Sign in button, and no token box.
 *   2. 200, enabled: false   -> the token box, the only way into a deployment with no user pool.
 *   3. no answer             -> neither box. A note that names the server and what came back (a
 *                               status other than 200, or a request that failed outright), and a
 *                               Try again button that runs the same startup again.
 *
 * In case 3 a stored sign-in must also survive. Probing a server that is not answering can only
 * fail, and the probe's failure path signs the person out.
 *
 * HOW IT RUNS THE REAL CODE. start() itself is extracted from the shipped app.js by name, with
 * fetchLoginConfig, setLoginStage, finishCognitoLogin, note, esc and the credential store, and
 * runs against a stub document and a stub fetch. What start() hands over to once the card is
 * decided (probeAccess, showApp, showNewcomer, signOut) is stubbed and only recorded. The card's
 * elements start hidden or shown as index.html ships them.
 *
 * Run: node ui/login_card_test.js      (no browser, no network, no login)
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

/* Blunt on purpose: a rename in app.js makes this throw rather than silently test nothing. The
 * `async` in front comes along, or the extracted text fails to parse on its own `await`. */
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

// --------------------------------------------------------------------------------- the stubs

const API = "https://api.example.com";

let nodes = {};
globalThis.document = {
  getElementById: (id) => (nodes[id] = nodes[id] || {
    id, hidden: false, innerHTML: "", textContent: "", value: "", onclick: null,
  }),
};
globalThis.$ = (id) => document.getElementById(id);

/* A new page, with each box of the card hidden or shown the way index.html ships it. */
const CARD = ["login-wait", "cognito-box", "token-box", "newcomer", "login-down"];
function freshPage() {
  nodes = {};
  for (const id of CARD) {
    const tag = HTML.match(new RegExp(`<[^>]*\\bid="${id}"[^>]*>`));
    if (!tag) throw new Error(`index.html has no #${id}`);
    $(id).hidden = /\shidden[\s>]/.test(tag[0]);
  }
}
const shown = (id) => !$(id).hidden;

// The credential store is app.js's own, over a localStorage that lives in this Map.
const saved = new Map();
globalThis.localStorage = {
  getItem: (k) => (saved.has(k) ? saved.get(k) : null),
  setItem: (k, v) => saved.set(k, String(v)),
  removeItem: (k) => saved.delete(k),
};
globalThis.location = { search: "", origin: "https://console.example.com", pathname: "/" };

// What start() hands over to once the card is decided. Recorded, not run.
const handedOver = [];
globalThis.probeAccess = async () => { handedOver.push("probeAccess"); return "out"; };
globalThis.showApp = (on) => { handedOver.push(`showApp(${on})`); };
globalThis.showNewcomer = () => { handedOver.push("showNewcomer"); };
globalThis.signOut = () => { handedOver.push("signOut"); };
globalThis.history = { replaceState: () => { handedOver.push("replaceState"); } };
globalThis.checkStale = () => {};

// The two `let`s start() writes.
globalThis.apiBase = "";
globalThis.loginConfig = null;

/* Shaped like the server's own two answers (`login_config` in the server's main.py). */
const COGNITO = {
  enabled: true, client_id: "example-client-id", issuer: "https://issuer.example.com",
  login_domain: "https://login.example.com", scopes: ["openid", "email"],
  registration_requests: true,
};
const NO_POOL = { enabled: false, registration_requests: false };

// A fresh copy per reply: start() adds `scopes` to the object it is handed.
const reply = (status, body) => ({
  ok: status >= 200 && status < 300, status, json: async () => JSON.parse(JSON.stringify(body)),
});
// An HTML error page, such as ingress-nginx's 503 while the gateway is scaled to zero.
const page = (status) => ({
  ok: status >= 200 && status < 300, status,
  json: async () => { throw new SyntaxError("Unexpected token '<'"); },
});
const answers = {
  cognito: () => reply(200, COGNITO),
  noPool: () => reply(200, NO_POOL),
  ingress503: () => page(503),
  gateway502: () => page(502),
  refused: () => { throw new TypeError("Failed to fetch"); },
  htmlAt200: () => page(200),
};

let loginAnswer = null;   // what /v1/login-config does in the case under test
globalThis.fetch = async (url) => {
  if (url === "config.json") return reply(200, { api_base: API });
  if (url === `${API}/v1/login-config`) return loginAnswer();
  throw new Error(`unexpected fetch ${url}`);
};

const define = (src) => (0, eval)(src);
["esc", "store"].forEach((name) => define(declaration(name).replace(/^const /, "globalThis.")));
["note", "setLoginStage", "fetchLoginConfig", "finishCognitoLogin", "start"]
  .forEach((name) => define(`globalThis.${name} = (${extract(name)});`));

// The Try again button's wiring, taken from app.js as it ships rather than written again here.
const RETRY_WIRING = (SRC.match(/^\$\("login-retry"\)\.onclick = .*;$/m) || [""])[0];

/* One page load: a fresh card, the button wired, the store as given, then start(). */
async function load(answer, { stored = null, search = "" } = {}) {
  freshPage();
  if (RETRY_WIRING) define(RETRY_WIRING);
  saved.clear();
  if (stored) store.set(stored.server, stored.token);
  location.search = search;
  loginAnswer = answer;
  handedOver.length = 0;
  await start();
}

const noteText = () => $("login-down-note").innerHTML;

(async () => {
  // -------------------------------------------------------------------------------------------
  console.log("fetchLoginConfig: an answer passes through, anything else is marked `down`");

  loginAnswer = answers.cognito;
  check(JSON.stringify(await fetchLoginConfig(API)) === JSON.stringify(COGNITO),
    "200 with the settings comes back as the server sent it");
  loginAnswer = answers.noPool;
  const noPool = await fetchLoginConfig(API);
  check(JSON.stringify(noPool) === JSON.stringify(NO_POOL) && !("down" in noPool),
    "200 with enabled: false is an answer too, and carries no `down`");
  loginAnswer = answers.ingress503;
  check(JSON.stringify(await fetchLoginConfig(API)) ===
    JSON.stringify({ enabled: false, down: "status 503" }),
    "a 503 is not an answer: `down` says which status came back");
  loginAnswer = answers.refused;
  check(JSON.stringify(await fetchLoginConfig(API)) ===
    JSON.stringify({ enabled: false, down: "unreachable" }),
    "a request that fails outright is `down: unreachable`");

  // -------------------------------------------------------------------------------------------
  console.log("1. 200 with enabled: true -- the Sign in button, and no token box");

  await load(answers.cognito);
  check(shown("cognito-box"), "the Sign in button is shown");
  check(!shown("token-box"), "the token box is not: with Cognito on, sign-in is Google-only");
  check(!shown("login-down"), "no note that the server did not answer");
  check(!shown("login-wait"), "the 'Checking sign-in…' line is gone");
  check(handedOver.join() === "showApp(false)",
    "with nothing stored, the card stays up and nothing is probed");

  // -------------------------------------------------------------------------------------------
  console.log("2. 200 with enabled: false -- no user pool, so a token is the only way in");

  await load(answers.noPool);
  check(shown("token-box"), "the token box is shown");
  check(!shown("cognito-box"), "the Sign in button is not");
  check(!shown("login-down"),
    "no outage note: the server answered, and its answer is that there is no user pool");
  check(handedOver.join() === "showApp(false)", "the card stays up");

  // -------------------------------------------------------------------------------------------
  console.log("3. no answer -- neither box, and a note that the server did not answer");

  await load(answers.ingress503);
  check(!shown("token-box"),
    "a 503 does not draw the token box. This is the 2026-11-05 defect: the token box read as " +
    "Google sign-in having been removed, and a token could not work against a server that is down");
  check(!shown("cognito-box"), "nor the Sign in button");
  check(shown("login-down"), "the outage box is shown, and the Try again button with it");
  check(!shown("login-wait"), "the 'Checking sign-in…' line is gone");
  check(noteText().includes('class="note err"'),
    "the note is drawn with note(\"err\"), like the other sign-in errors");
  check(noteText().includes(`The server at ${API} did not answer (status 503)`),
    "it names the server and the status that came back");
  check(noteText().includes("sign-in is unavailable right now") &&
    noteText().includes("Try again later, or ask an operator"),
    "it says sign-in is unavailable for now, and what to do about it");
  check(handedOver.length === 0, "nothing is handed over: no probe, no showApp, no sign-out");

  await load(answers.refused);
  check(!shown("token-box") && !shown("cognito-box") && shown("login-down"),
    "a request that fails outright (dead host, DNS, refused CORS) gets the same card");
  check(noteText().includes(`The server at ${API} did not answer (unreachable)`),
    "and its note says the server was unreachable");

  await load(answers.htmlAt200);
  check(!shown("token-box") && shown("login-down"),
    "a 200 whose body is not the settings (an HTML page) is not an answer either");
  check(noteText().includes("(status 200, not its sign-in settings)"),
    "and its note says what came back");

  // -------------------------------------------------------------------------------------------
  console.log("a stored sign-in and a pending sign-in both survive the outage");

  await load(answers.ingress503, { stored: { server: API, token: "a-stored-token" } });
  check(shown("login-down") && !shown("token-box"),
    "a returning visitor gets the same note (the early, parallel request is the one that failed)");
  check(store.server === API && store.token === "a-stored-token",
    "the stored address and token are still there");
  check(!handedOver.includes("probeAccess") && !handedOver.includes("signOut"),
    "the server is not probed, so its failure does not sign the person out");

  await load(answers.ingress503, { search: "?code=abc" });
  check(shown("login-down") && !shown("token-box"),
    "back from Google into an outage: the same note, not the token box");
  check(!handedOver.includes("replaceState"),
    "the ?code= stays in the address unspent, so Try again can finish that sign-in");

  // -------------------------------------------------------------------------------------------
  console.log("Try again runs the same startup again");

  check(RETRY_WIRING.includes("start()"), "app.js wires the Try again button to start()");
  await load(answers.ingress503);
  loginAnswer = answers.gateway502;
  await $("login-retry").onclick();
  check(shown("login-down") && noteText().includes("(status 502)"),
    "a press that meets the outage again draws the note again, with what came back this time");

  let release = null;
  loginAnswer = () => new Promise((resolve) => { release = () => resolve(reply(200, COGNITO)); });
  const pressed = $("login-retry").onclick();
  check(shown("login-wait") && !shown("login-down"),
    "pressing it hides the note at once and shows 'Checking sign-in…' while it asks");
  await new Promise((resolve) => setImmediate(resolve));
  check(typeof release === "function", "the press asked /v1/login-config again");
  if (release) release();
  await pressed;
  check(shown("cognito-box") && !shown("login-down") && !shown("login-wait"),
    "once the server answers, the card offers the Sign in button");

  // -------------------------------------------------------------------------------------------
  console.log("wiring");

  check(/^start\(\);$/m.test(SRC), "start() runs on load");
  for (const id of ["login-down", "login-down-note", "login-retry"]) {
    check(HTML.includes(`id="${id}"`), `index.html has #${id}`);
  }
  check(/<[^>]*\bid="login-down"[^>]*\shidden[\s>]/.test(HTML),
    "the outage box starts hidden, so a normal load never flashes it");

  if (failures.length) {
    console.log(`\n${failures.length} check(s) failed`);
    process.exit(1);
  }
  console.log("\nall checks passed");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
