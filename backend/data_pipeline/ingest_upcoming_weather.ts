// backend/data_pipeline/ingest_upcoming_weather.ts

// ────────── IMPORTS ──────────
import path from "path";
import fs from "fs";
import { DateTime } from "luxon";
import axios from "axios";
import { createClient } from "@supabase/supabase-js";

// ────────── LOGGING + CRASH GUARDS ──────────
const log = (...a: any[]) => console.log("[ingest]", ...a);
const warn = (...a: any[]) => console.warn("[ingest]", ...a);
const err = (...a: any[]) => console.error("[ingest]", ...a);

process.on("unhandledRejection", (e) => {
  err("unhandledRejection:", e);
  process.exitCode = 1;
});
process.on("uncaughtException", (e) => {
  err("uncaughtException:", e);
  process.exit(1);
});

// ────────── PATHS ──────────
const HERE = __dirname; // .../backend/data_pipeline
const BACKEND_DIR = path.resolve(HERE, ".."); // .../backend
const REPO_ROOT = path.resolve(BACKEND_DIR, "..");

// ────────── ENV LOAD (OPTIONAL .env LOCALLY) ──────────
(() => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const dotenv = require("dotenv");
    const candidates = [
      path.join(BACKEND_DIR, ".env"),
      path.join(REPO_ROOT, ".env"),
    ].filter((p) => fs.existsSync(p));
    if (candidates.length) {
      dotenv.config({ path: candidates[0] });
      log(`env loaded: ${candidates[0]}`);
    } else {
      log("no .env file found; using process env (CI mode)");
    }
  } catch {
    log("dotenv not installed; skipping .env load");
  }
})();

// ────────── ENV VARS + SUPABASE CLIENT ──────────
const SUPABASE_URL =
  process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || "";

const SUPABASE_SERVICE_KEY =
  process.env.SUPABASE_SERVICE_KEY ||
  process.env.SUPABASE_SERVICE_ROLE_KEY ||
  process.env.SUPABASE_SERVICE_ROLE ||
  "";

const LOOKAHEAD_DAYS = Number(process.env.WX_LOOKAHEAD_DAYS ?? 14);
const LOOKBACK_DAYS = Number(process.env.WX_LOOKBACK_DAYS ?? 0);
const START_ISO = process.env.WX_START_ISO || ""; // e.g. 2024-12-01T00:00:00Z
const END_ISO = process.env.WX_END_ISO || ""; // e.g. 2024-12-31T23:59:59Z

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  err("Missing Supabase env. SUPABASE_URL and SUPABASE_SERVICE_KEY required.");
  process.exit(1);
}

const sb = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
log("Supabase client created");

// ────────── TYPES + HELPERS ──────────
type VenueInfo = { lat: number; lon: number; is_indoor: boolean };
type MeteoRow = {
  time: string;
  temp?: number;
  rhum?: number;
  wspd?: number;
  wdir?: number;
};

const norm = (s: string | null | undefined) =>
  String(s ?? "")
    .trim()
    .toLowerCase();

const VENUE_ALIASES: Record<string, string> = {
  "huntington bank field": "cleveland browns stadium",
};

function stadiumJsonPath() {
  return path.resolve(BACKEND_DIR, "data", "stadium_data.json");
}

function loadVenueDirectory(): Record<string, VenueInfo> {
  const file = stadiumJsonPath();
  if (!fs.existsSync(file)) throw new Error(`Missing stadium JSON: ${file}`);
  const raw = JSON.parse(fs.readFileSync(file, "utf-8"));
  const nfl = raw?.NFL ?? {};
  const byStadium: Record<string, VenueInfo> = {};

  for (const team of Object.keys(nfl)) {
    const rec = nfl[team];
    const key = norm(rec?.stadium);
    if (!key) continue;
    const lat = Number(rec?.latitude ?? rec?.lat);
    const lon = Number(rec?.longitude ?? rec?.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    // Retractables treated as outdoor unless you wire real-time roof status
    const isIndoor =
      rec?.is_indoor === true ||
      rec?.dome === true ||
      (typeof rec?.roof === "string" &&
        rec.roof.toLowerCase().includes("closed"));

    byStadium[key] = { lat, lon, is_indoor: !!isIndoor };
  }

  // alias fill
  for (const [alias, canonical] of Object.entries(VENUE_ALIASES)) {
    const v = byStadium[norm(canonical)];
    if (v) byStadium[norm(alias)] = v;
  }
  return byStadium;
}

function pickNearestByTime<T extends { time: string }>(
  rows: T[],
  iso: string
): T | null {
  if (!rows.length) return null;
  const target = new Date(iso).getTime();
  let best: T | null = null;
  let bestDiff = Infinity;
  for (const r of rows) {
    const t = new Date(
      (r.time.includes("T") ? r.time : r.time + "T00:00") + "Z"
    ).getTime();
    const d = Math.abs(t - target);
    if (d < bestDiff) {
      best = r;
      bestDiff = d;
    }
  }
  return best;
}

async function fetchOpenMeteoArchiveHourly(
  lat: number,
  lon: number,
  iso: string
): Promise<MeteoRow[]> {
  const start = DateTime.fromISO(iso, { zone: "utc" })
    .minus({ hours: 1 })
    .toISODate();
  const end = DateTime.fromISO(iso, { zone: "utc" })
    .plus({ hours: 1 })
    .toISODate();

  const { data } = await axios.get(
    "https://archive-api.open-meteo.com/v1/archive",
    {
      params: {
        latitude: lat,
        longitude: lon,
        start_date: start,
        end_date: end,
        hourly:
          "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        temperature_unit: "fahrenheit",
        windspeed_unit: "mph",
        timezone: "UTC",
      },
      timeout: 15000,
    }
  );

  const t = data?.hourly?.time ?? [];
  const temp = data?.hourly?.temperature_2m ?? [];
  const rh = data?.hourly?.relative_humidity_2m ?? [];
  const wspd = data?.hourly?.wind_speed_10m ?? [];
  const wdir = data?.hourly?.wind_direction_10m ?? [];

  return t.map((time: string, i: number) => ({
    time,
    temp: temp[i],
    rhum: rh[i],
    wspd: wspd[i],
    wdir: wdir[i],
  }));
}

function buildSnapshot(opts: {
  teamName: string;
  venueNorm: string;
  info: VenueInfo;
  hour: MeteoRow | null;
  source: "open-meteo" | "indoor";
  kickoffUtcISO: string;
}) {
  const { teamName, venueNorm, info, hour, source, kickoffUtcISO } = opts;
  const isIndoor = source === "indoor" ? true : info.is_indoor;

  return {
    sport: "NFL",
    team_name: teamName,
    city: null,
    stadium: venueNorm,
    latitude: info.lat,
    longitude: info.lon,
    is_indoor: isIndoor,
    orientation_deg: null,
    temperature_f: isIndoor ? null : hour?.temp ?? null,
    feels_like_f: null,
    humidity_pct: isIndoor ? null : hour?.rhum ?? null,
    wind_speed_mph: isIndoor ? null : hour?.wspd ?? null,
    wind_deg: isIndoor ? null : hour?.wdir ?? null,
    description: null,
    icon: null,
    precip_prob_pct: null,
    source,
    raw: hour,
    captured_at: kickoffUtcISO,
  };
}

// ────────── MAIN ──────────
async function main() {
  log("START");
  log("load venues from:", stadiumJsonPath());
  const venues = loadVenueDirectory();
  log("venues loaded:", Object.keys(venues).length);

  // Game selection: priority = fixed range > lookahead > lookback
  type GameRow = {
    game_id: number;
    scheduled_time: string; // ISO UTC
    home_team: string;
    venue: string | null;
  };

  const toIso = (dt: DateTime) =>
    dt.toISO({ suppressMilliseconds: true, includeOffset: false }) ??
    new Date().toISOString();

  async function fetchGames(
    startIso: string,
    endIso: string
  ): Promise<{ games: GameRow[]; error: unknown }> {
    const { data, error } = await sb
      .from("nfl_game_schedule")
      .select("game_id, scheduled_time, home_team, venue")
      .gte("scheduled_time", startIso)
      .lte("scheduled_time", endIso)
      .order("scheduled_time", { ascending: true });

    // Coerce: Supabase returns GameRow[] | null; normalize to [] for typing
    return { games: (data ?? []) as GameRow[], error };
  }

  let games: GameRow[] = [];
  let lastError: unknown = null;

  if (START_ISO && END_ISO) {
    log(`fixed range [${START_ISO} .. ${END_ISO}]`);
    const res = await fetchGames(START_ISO, END_ISO);
    games = res.games;
    lastError = res.error ?? null;
  } else {
    const nowIso = toIso(DateTime.utc());
    const endIso = toIso(DateTime.utc().plus({ days: LOOKAHEAD_DAYS }));

    log(`window (lookahead) = ${LOOKAHEAD_DAYS}d  [${nowIso} .. ${endIso}]`);
    let res = await fetchGames(nowIso, endIso);
    games = res.games;
    lastError = res.error ?? null;

    if (games.length === 0 && LOOKBACK_DAYS > 0) {
      const startIso = toIso(DateTime.utc().minus({ days: LOOKBACK_DAYS }));
      log(
        `fallback (lookback) = ${LOOKBACK_DAYS}d  [${startIso} .. ${nowIso}]`
      );
      res = await fetchGames(startIso, nowIso);
      games = res.games;
      lastError = res.error ?? lastError;
    }
  }

  if (lastError) throw lastError;
  log("games found:", games.length);

  if (!games?.length) {
    const dv = await sb
      .from("nfl_game_schedule")
      .select("venue")
      .not("venue", "is", null)
      .order("venue", { ascending: true });

    const venuesDistinct = Array.from(
      new Set((dv.data ?? []).map((r: any) => norm(r.venue)).filter(Boolean))
    ).slice(0, 50);

    log("distinct venue samples:", venuesDistinct);
  }

  let inserted = 0;
  let skippedNoVenue = 0;
  let indoorCount = 0;

  for (const g of games ?? []) {
    const venueNorm = norm(g.venue);
    const info = venues[venueNorm];
    if (!info) {
      skippedNoVenue++;
      warn(
        `skip game_id=${g.game_id} unknown venue="${venueNorm}" raw="${g.venue}"`
      );
      continue;
    }

    let snapshot: any;

    if (info.is_indoor) {
      indoorCount++;
      snapshot = buildSnapshot({
        teamName: g.home_team,
        venueNorm,
        info,
        hour: null,
        source: "indoor",
        kickoffUtcISO: g.scheduled_time,
      });
    } else {
      const rows = await fetchOpenMeteoArchiveHourly(
        info.lat,
        info.lon,
        g.scheduled_time
      );
      const nearest = pickNearestByTime(rows, g.scheduled_time);
      snapshot = buildSnapshot({
        teamName: g.home_team,
        venueNorm,
        info,
        hour: nearest,
        source: "open-meteo",
        kickoffUtcISO: g.scheduled_time,
      });
    }

    const { error: insErr } = await sb
      .from("weather_forecast_snapshots")
      .insert([{ ...snapshot }]);

    if (insErr) {
      warn(`insert failed game_id=${g.game_id}: ${insErr.message}`);
      continue;
    }
    inserted++;
  }

  log("refresh MVs…");
  await sb.rpc("refresh_mv_nfl_latest_forecast_per_game");
  await sb.rpc("refresh_nfl_weather_climatology");

  console.log(
    JSON.stringify(
      {
        found: games?.length ?? 0,
        inserted,
        skipped_no_venue: skippedNoVenue,
        indoor_snapshots: indoorCount,
        lookahead_days: LOOKAHEAD_DAYS,
        start_iso: START_ISO || null,
        end_iso: END_ISO || null,
        lookback_days: LOOKBACK_DAYS || null,
      },
      null,
      2
    )
  );

  log("DONE");
}

// run if invoked directly
if (require.main === module) {
  main().catch((e) => {
    err("fatal:", e);
    process.exit(1);
  });
}
