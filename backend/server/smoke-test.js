// backend/server/smoke-test.js
// 🎯  Zero‑dependency smoke‑test utility for the ScoreGenius NFL API.
//      • Node 18+ (global fetch + console.table) required
//      • Run:  node smoke-test.js
//        or:  BASE_URL=https://api.my‑stage.com CRON_KEY=super node smoke-test.js

import { exit } from "node:process";
import { setTimeout as delay } from "node:timers/promises";

const BASE = process.env.BASE_URL || "http://localhost:10000";
const CRON = process.env.CRON_KEY || ""; // x‑cron‑key (if your route needs it)
const TODAY = new Date().toISOString().slice(0, 10);
const SEASON = 2024;
const TEAMS = "1,2";
const SNAPS = "2024090701,2024090702"; // sample gameIds (regular‑season Wk 1)

// Each entry: { name, path, opts? }.
// `opts` can be a RequestInit; it will be merged with timeout + headers.
const endpoints = [
  { name: "Health", path: `/health` },
  { name: "Swagger UI", path: `/api-docs` },
  {
    name: "Team Full",
    path: `/api/v1/nfl/teams/${SEASON}/full?teamId=${TEAMS}`,
  },
  { name: "Team Reg‑Only", path: `/api/v1/nfl/teams/${SEASON}/regonly` },
  { name: "SOS", path: `/api/v1/nfl/teams/${SEASON}/sos?teamIds=${TEAMS}` }, // incl. teams to dodge 400
  { name: "SRS lite", path: `/api/v1/nfl/teams/${SEASON}/srs` },
  {
    name: "Dashboard",
    path: `/api/v1/nfl/teams/${SEASON}/dashboard?teamIds=${TEAMS}`,
  },
  { name: "Schedule (today)", path: `/api/v1/nfl/schedule?date=${TODAY}` },
  { name: "Snapshots", path: `/api/v1/nfl/snapshots?gameIds=${SNAPS}` },
  {
    name: "Cron health",
    path: `/api/v1/nfl/health/cron`,
    opts: CRON && { headers: { "x-cron-key": CRON } },
  },
  { name: "Validate health", path: `/api/v1/nfl/health/validate` },
];

console.log(`\n🏃  Smoke‑test against  ${BASE}\n`);

const results = [];

for (const ep of endpoints) {
  const url = BASE + ep.path;
  const timeout = 2500; // ms
  let ok = false;
  let status = null;
  let note = "";

  try {
    // Abort if server hangs
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    const res = await fetch(url, {
      signal: controller.signal,
      headers: { accept: "application/json", ...(ep.opts?.headers || {}) },
      ...ep.opts,
    });

    clearTimeout(timer);
    status = res.status;
    ok = res.ok;

    if (!ok) {
      // capture up to 200 B of the body for quick diagnostics
      const text = await res.text().catch(() => "");
      note = text.slice(0, 200).replace(/\s+/g, " ");
    }
  } catch (err) {
    status = err.name === "AbortError" ? "TIMEOUT" : "ERR";
    note = err.message ?? "";
  }

  console.log(
    `${ok ? "✅" : "❌"}  ${ep.name.padEnd(15)}  ${ep.path.padEnd(
      50
    )} → ${status}${note ? `  (${note})` : ""}`
  );

  results.push({ Endpoint: ep.name, Status: ok ? "OK" : "FAIL" });
  // brief stagger to avoid hammering the local server
  await delay(50);
}

console.log("\nSummary:");
console.table(results);

const failures = results.filter((r) => r.Status !== "OK").length;
console.log(
  failures
    ? `\n✗ ${failures} / ${results.length} endpoints FAILED`
    : `\n✓ All ${results.length} endpoints OK ✅`
);

exit(failures ? 1 : 0);
