/* sign_out_test.js -- Sign out forgets the sign-in (HYPERUN-UI-SIGN-OUT).
 *
 * WHY THIS FILE EXISTS. signOut() removed every localStorage key that starts with "hyperun.",
 * and the two keys that sign a browser in are not among them: `store` keeps the API address and
 * the id_token under the keys they had before the rename, so that a browser signed in then stayed
 * signed in. From the rename until 2026-11-05, Sign out therefore removed the refresh token and
 * the theme and left the id_token, with two results. Run against that code, this file fails on
 * both:
 *
 *   1. With Cognito on, the page came back from Cognito's /logout, start() found the id_token,
 *      the probe passed, and the app opened again. Sign out looked like it had done nothing. The
 *      newcomer card's Sign out came back to the same newcomer card, so someone who had signed in
 *      with the wrong account could not try another one.
 *   2. Once the id_token's hour was up, refreshIfExpired() found no refresh token (Sign out had
 *      removed it), or one Cognito refused, and signed out: /logout, back, the same expired
 *      id_token, /logout again, on every load until the site's storage was cleared by hand.
 *
 * WHAT IT PINS.
 *   1. signOut() leaves nothing that signs anyone in (no id_token, no address, no refresh token,
 *      no PKCE verifier) and keeps the theme, which is not a credential.
 *   2. Signed in, Sign out, back from /logout: the sign-in card, and the page stays on it. The
 *      same from the newcomer card.
 *   3. An expired id_token with no refresh token, or with one Cognito refuses: one trip to
 *      /logout, then the sign-in card.
 *   4. An expired id_token with a refresh token Cognito accepts still opens the app, with no trip.
 *
 * HOW IT RUNS THE REAL CODE. start(), the sign-in functions it reaches (finishCognitoLogin,
 * exchange, probeAccess, call, refreshIfExpired, signOut) and the credential store are extracted
 * from the shipped app.js by name and evaluated over stubs. Where start() hands the screen over
 * (showApp, showNewcomer) is recorded, not drawn. localStorage lives across page loads, as the
 * browser's does, so one page load is one run of start() and a redirect is the next run.
 *   - The stub API accepts an id_token until its exp, as the server's JWKS check does. Cognito's
 *     /logout ends Cognito's own session; an id_token already issued stays as good as it was.
 *   - The stub Cognito sends /logout straight back to logout_uri, as the real one does when
 *     logout_uri is registered.
 * Nothing here names a storage key from before the rename (it reads `store.server` and
 * `store.token`), so the old-name ceiling does not move.
 *
 * Run: node ui/sign_out_test.js      (no browser, no network, no login)
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

/* A function by name, its `async` included. Not anchored to the start of a line, because start()
 * can be declared inside the parentheses that call it. The body is found after the parameter
 * list, which can hold a `{}` of its own (`call(path, options = {})`). A rename makes this throw
 * rather than silently test nothing. */
function extract(name) {
  const at = SRC.search(new RegExp(`(async )?function ${name}\\(`));
  if (at < 0) throw new Error(`app.js has no function ${name} -- was it renamed?`);
  let j = SRC.indexOf("(", at);
  for (let parens = 0; j < SRC.length; j++) {
    if (SRC[j] === "(") parens++;
    else if (SRC[j] === ")" && --parens === 0) break;
  }
  let depth = 0;
  for (j = SRC.indexOf("{", j); j < SRC.length; j++) {
    if (SRC[j] === "{") depth++;
    else if (SRC[j] === "}" && --depth === 0) return SRC.slice(at, j + 1);
  }
  throw new Error(`unbalanced braces in ${name}`);
}

/* A top-level `const NAME = ...`: one line, whose trailing comment comes along harmlessly, or a
 * block that ends at a line that is exactly "};". */
function declaration(name) {
  const lines = SRC.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`const ${name} =`));
  if (start < 0) throw new Error(`app.js has no const ${name} -- was it renamed?`);
  if (!lines[start].trimEnd().endsWith("{")) return lines[start];
  return lines.slice(start, lines.indexOf("};", start) + 1).join("\n");
}

const define = (src) => (0, eval)(src);
["store", "LOGIN_KEY", "REFRESH_KEY", "THEME_KEY", "redirectUri"].forEach((name) =>
  define(declaration(name).replace(/^const /, "globalThis.")));
["call", "probeAccess", "exchange", "finishCognitoLogin", "refreshIfExpired", "signOut", "start"]
  .forEach((name) => define(`globalThis.${name} = (${extract(name)});`));
// start() reads /v1/login-config through fetchLoginConfig. When that is a function of its own
// rather than an arrow inside start(), it comes along too.
if (/(^|\n)async function fetchLoginConfig\(/.test(SRC)) {
  define(`globalThis.fetchLoginConfig = (${extract("fetchLoginConfig")});`);
}

// --------------------------------------------------------------------------------- the stubs

const CONSOLE = "https://console.example.com";   // where the page is served
const API = "https://api.example.com";           // config.json's api_base
const LOGIN = "https://login.example.com";       // Cognito's login domain
const NEWCOMER = "new@example.com";              // signs in fine, is registered nowhere

/* Storage as a browser has it, down to Object.keys(localStorage) listing the stored keys, which
 * signOut() relies on. */
function storage() {
  const data = new Map();
  const api = {
    getItem: (k) => (data.has(String(k)) ? data.get(String(k)) : null),
    setItem: (k, v) => { data.set(String(k), String(v)); },
    removeItem: (k) => { data.delete(String(k)); },
    clear: () => data.clear(),
  };
  return new Proxy(api, {
    ownKeys: () => [...data.keys()],
    getOwnPropertyDescriptor: (_, k) =>
      (data.has(k) ? { value: data.get(k), enumerable: true, configurable: true } : undefined),
  });
}
globalThis.localStorage = storage();
globalThis.sessionStorage = storage();

let nodes = {};
globalThis.document = {
  getElementById: (id) => (nodes[id] = nodes[id] || {
    id, hidden: true, innerHTML: "", textContent: "", value: "",
  }),
};
globalThis.$ = (id) => document.getElementById(id);
globalThis.note = (kind, text) => `[${kind}] ${text}`;
globalThis.poll = { stop() {} };
globalThis.checkStale = () => {};
// What the card shows while start() works is not this file's subject.
globalThis.setLoginStage = () => {};

// Where start() hands the screen over. Recorded, not drawn.
let page = null;
globalThis.showApp = (on) => { page.screen = on ? "app" : "sign-in card"; };
globalThis.showNewcomer = () => { page.screen = "newcomer card"; };

/* An id_token shaped like Cognito's. Only the payload matters: the page reads `exp` and `email`
 * out of it, and the stub API checks `exp`. */
const b64url = (text) => Buffer.from(text).toString("base64url");
function idToken(email, secondsLeft) {
  const exp = Math.floor(Date.now() / 1000) + secondsLeft;
  return [b64url('{"alg":"RS256","kid":"k1"}'),
    b64url(JSON.stringify({ email, exp, token_use: "id" })), "signature"].join(".");
}
function claims(token) {
  try { return JSON.parse(Buffer.from(String(token).split(".")[1], "base64url").toString()); }
  catch { return {}; }
}
const unexpired = (token) => claims(token).exp > Date.now() / 1000;

/* Shaped like the server's login-config answer for a deployment with a user pool. */
const COGNITO = {
  enabled: true, client_id: "example-client-id", login_domain: LOGIN,
  scopes: ["openid", "email"], registration_requests: true,
};

const reply = (status, body) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => JSON.parse(JSON.stringify(body)),   // a fresh copy: start() writes into it
  text: async () => (body === null ? "" : JSON.stringify(body)),
});

/* Cognito's token endpoint: each code once, and the refresh tokens it issued and still honours. */
const codes = new Map();
const refreshTokens = new Set();
function tokenEndpoint(form) {
  if (form.get("grant_type") === "authorization_code") {
    const email = codes.get(form.get("code"));
    codes.delete(form.get("code"));
    if (!email || !form.get("code_verifier")) return reply(400, { error: "invalid_grant" });
    refreshTokens.add(`refresh:${email}`);
    return reply(200, { id_token: idToken(email, 3600), refresh_token: `refresh:${email}` });
  }
  const refresh = form.get("refresh_token") || "";
  if (form.get("grant_type") === "refresh_token" && refreshTokens.has(refresh)) {
    return reply(200, { id_token: idToken(refresh.slice("refresh:".length), 3600) });
  }
  return reply(400, { error: "invalid_grant", error_description: "Refresh Token has been revoked" });
}

/* GET /v1/namespaces, the probe. An id_token is good until its exp, and an address nobody
 * registered gets the server's 403 sentence (auth.py), which probeAccess matches. */
function namespaces(token) {
  if (!unexpired(token)) return reply(401, { detail: "The id_token has expired." });
  const { email } = claims(token);
  if (email === NEWCOMER) {
    return reply(403, { detail: `${email} signed in successfully but is not registered with this `
      + "service. An operator has to create a namespace for you and add you to the token file." });
  }
  return reply(200, { own: "team-a", namespaces: ["team-a"] });
}

globalThis.fetch = async (url, options = {}) => {
  const bearer = ((options.headers || {}).Authorization || "").replace(/^Bearer /, "");
  switch (String(url)) {
    case "config.json": return reply(200, { api_base: API });
    case `${API}/v1/login-config`: return reply(200, COGNITO);
    case `${API}/v1/namespaces`: return namespaces(bearer);
    case `${LOGIN}/oauth2/token`: return tokenEndpoint(new URLSearchParams(String(options.body)));
    default: return reply(404, null);   // a request sent after the address was forgotten
  }
};

/* A new page at CONSOLE + "/" + search. The module-level state starts over and the storage does
 * not. location.assign() records where the page went: the browser leaves once the script yields,
 * so the last address asked for is the one it goes to. */
function newPage(search) {
  page = { screen: "nothing", leftFor: null };
  nodes = {};
  globalThis.location = {
    origin: CONSOLE, pathname: "/", search,
    assign: (url) => { page.leftFor = url; },
  };
  globalThis.history = { replaceState: () => { location.search = ""; } };
  globalThis.loginConfig = null;
  globalThis.apiBase = "";
  globalThis.knownTeam = "";
  globalThis.nsView = { loaded: false, list: [], own: "", selectable: false, current: "" };
  return page;
}

async function load(search = "") {
  newPage(search);
  await start();
  return page;
}

/* Follow the page each time it goes to Cognito's /logout. Cognito ends its session there and
 * sends the browser straight back to logout_uri, which loads the page again. Stops at the first
 * load that stays, or after `limit` loads, so a loop shows up as a count. */
async function follow(from, limit = 5) {
  const loads = [];
  for (let at = from; at.leftFor && loads.length < limit;) {
    const to = new URL(at.leftFor);
    if (to.origin + to.pathname !== `${LOGIN}/logout`) break;
    at = await load(new URL(to.searchParams.get("logout_uri")).search);
    loads.push(at);
  }
  return loads;
}
const trips = (loads) => loads.map((p) => (p.leftFor ? "/logout" : p.screen)).join(" -> ");

/* A browser that holds what the arguments say, plus a theme its owner chose. */
function browser({ token = null, refresh = null } = {}) {
  localStorage.clear();
  sessionStorage.clear();
  localStorage.setItem(THEME_KEY, "dark");
  if (token) store.set(API, token);
  if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
}

/* Sign in was pressed and Cognito sent the browser back: startCognitoLogin left its verifier in
 * sessionStorage, and start() spends the code. */
async function signIn(email) {
  sessionStorage.setItem(LOGIN_KEY, "a-verifier");
  codes.set("a-code", email);
  return load("?code=a-code");
}

// --------------------------------------------------------------------------------- the checks

(async () => {
  console.log("signOut() with no user pool: nothing that signs anyone in is left");
  browser({ token: idToken("alice@example.com", 3600), refresh: "refresh:alice@example.com" });
  sessionStorage.setItem(LOGIN_KEY, "a-verifier");
  newPage("");   // loginConfig stays null: there is no Cognito session to end
  signOut();
  check(store.token === "", "the id_token is gone");
  check(store.server === "", "the API address is gone");
  check(localStorage.getItem(REFRESH_KEY) === null, "the refresh token is gone");
  check(sessionStorage.getItem(LOGIN_KEY) === null, "the PKCE verifier is gone");
  const left = Object.keys(localStorage);
  check(left.length === 1 && left[0] === THEME_KEY,
    `the theme is the only key left in localStorage (left: ${left.join(", ")})`);
  check(page.screen === "sign-in card" && page.leftFor === null,
    "the sign-in card is drawn in place");

  console.log("signOut() with Cognito on: it is all gone before the page leaves for /logout");
  browser({ token: idToken("alice@example.com", 3600), refresh: "refresh:alice@example.com" });
  newPage("");
  globalThis.loginConfig = { ...COGNITO };
  signOut();
  const to = page.leftFor ? new URL(page.leftFor) : null;
  check(Boolean(to) && to.origin + to.pathname === `${LOGIN}/logout`
      && to.searchParams.get("client_id") === COGNITO.client_id
      && to.searchParams.get("logout_uri") === `${CONSOLE}/`,
    "the page goes to Cognito's /logout, naming the client and this page as logout_uri");
  check(store.token === "" && store.server === "" && localStorage.getItem(REFRESH_KEY) === null,
    "and leaves no id_token, address or refresh token behind");

  console.log("Signed in, Sign out, back from /logout");
  browser();
  const signedIn = await signIn("alice@example.com");
  check(signedIn.screen === "app" && unexpired(store.token), "signed in through Cognito, the app opens");
  signOut();   // the header's Sign out, on that page
  const back = await follow(page);
  check(back.length === 1 && back[0].screen === "sign-in card",
    `back from /logout, the sign-in card and not the app (${trips(back)})`);
  check(back.length === 1 && back[0].leftFor === null, "and the page stays on it");
  check(store.token === "", "no id_token is stored");
  check(localStorage.getItem(THEME_KEY) === "dark", "the theme its owner chose is still set");

  console.log("The newcomer card's Sign out, to try another account");
  browser();
  const newcomer = await signIn(NEWCOMER);
  check(newcomer.screen === "newcomer card", "an address nobody registered gets the newcomer card");
  signOut();   // the newcomer card's Sign out
  const backFromNewcomer = await follow(page);
  check(backFromNewcomer.length === 1 && backFromNewcomer[0].screen === "sign-in card",
    `back from /logout, the sign-in card, where another account can sign in (${trips(backFromNewcomer)})`);

  console.log("The id_token's hour is up and there is no refresh token");
  browser({ token: idToken("alice@example.com", -120) });
  const noRefresh = [await load("")];
  noRefresh.push(...(await follow(noRefresh[0])));
  check(noRefresh.length === 2 && Boolean(noRefresh[0].leftFor)
      && noRefresh[1].screen === "sign-in card" && noRefresh[1].leftFor === null,
    `one trip to /logout, then the sign-in card, not a loop (${trips(noRefresh)})`);

  console.log("The id_token's hour is up and Cognito refuses the refresh token");
  browser({ token: idToken("alice@example.com", -120), refresh: "refresh:revoked" });
  const refused = [await load("")];
  refused.push(...(await follow(refused[0])));
  check(refused.length === 2 && Boolean(refused[0].leftFor)
      && refused[1].screen === "sign-in card" && refused[1].leftFor === null,
    `one trip to /logout, then the sign-in card, not a loop (${trips(refused)})`);
  check(localStorage.getItem(REFRESH_KEY) === null, "the refused refresh token is not kept");

  console.log("The id_token's hour is up and the refresh token is good");
  refreshTokens.add("refresh:alice@example.com");
  browser({ token: idToken("alice@example.com", -120), refresh: "refresh:alice@example.com" });
  const refreshed = await load("");
  check(refreshed.screen === "app" && refreshed.leftFor === null && unexpired(store.token),
    "the app opens on a new id_token, with no trip to Cognito");

  console.log("Wiring");
  check(/\$\("logout"\)\.onclick = signOut;/.test(SRC), "the header's Sign out button calls signOut");
  check(/\$\("nc-out"\)\.onclick = signOut;/.test(SRC), "the newcomer card's Sign out button calls signOut");
  check(HTML.includes(`localStorage.getItem("${THEME_KEY}")`),
    "the theme key signOut() keeps is the one index.html's boot script reads before the first paint");

  if (failures.length) {
    console.log(`\n${failures.length} check(s) failed`);
    process.exit(1);
  }
  console.log("\nall checks passed");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
