// backend/data_pipeline/ingest_upcoming_weather.ts

// ────────── IMPORTS ──────────
import path from "path";
import fs from "fs";
import { DateTime } from "luxon";
import axios, { AxiosRequestConfig } from "axios";
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

// Tunables
const OPEN_METEO_TIMEOUT_MS = Number(
  process.env.OPEN_METEO_TIMEOUT_MS ?? 15000
);
const OPEN_METEO_RETRIES = Math.max(
  1,
  Number(process.env.OPEN_METEO_RETRIES ?? 3)
);
const FORECAST_HORIZON_DAYS = 16; // open-meteo forecast supports ~16 days

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

type GameRow = {
  game_id: number;
  scheduled_time: string; // ISO UTC
  home_team: string;
  venue: string | null;
};

const norm = (s: string | null | undefined) =>
  String(s ?? "")
    .trim()
    .toLowerCase();

const VENUE_ALIASES: Record<string, string> = {
  "huntington bank field": "cleveland browns stadium",
};

// Simple retry with exp backoff + jitter
async function withRetry<T>(fn: () => Promise<T>, label: string): Promise<T> {
  let lastErr: any;
  for (let attempt = 1; attempt <= OPEN_METEO_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (e: any) {
      lastErr = e;
      const delayMs =
        Math.min(3000, 250 * Math.pow(2, attempt)) +
        Math.floor(Math.random() * 200);
      warn(
        `${label} attempt ${attempt}/${OPEN_METEO_RETRIES} failed: ${
          e?.message || e
        }. retry in ${delayMs}ms`
      );
      await new Promise((r) => setTimeout(r, delayMs));
    }
  }
  throw lastErr;
}

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

    // Retractables treated as outdoor unless wired to real-time roof status
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
  const target = DateTime.fromISO(iso, { zone: "utc" }).toMillis();
  let best: T | null = null;
  let bestDiff = Infinity;
  for (const r of rows) {
    const tStr = r.time.includes("T") ? r.time : `${r.time}T00:00`;
    const t = DateTime.fromISO(tStr, { zone: "utc" }).toMillis();
    const d = Math.abs(t - target);
    if (d < bestDiff) {
      best = r;
      bestDiff = d;
    }
  }
  return best;
}

// ────────── OPEN-METEO CALLS ──────────
async function fetchOpenMeteoArchiveHourly(
  lat: number,
  lon: number,
  kickoffIso: string
): Promise<MeteoRow[]> {
  // Archive API does NOT support future dates. Clamp target to yesterday end-of-day.
  const nowUtc = DateTime.utc();
  const maxArchive = nowUtc.minus({ days: 1 }).endOf("day");
  const kickoffUTC = DateTime.fromISO(kickoffIso, { zone: "utc" });
  const target = kickoffUTC > maxArchive ? maxArchive : kickoffUTC;

  // Small window around kickoff for better nearest selection
  const start = target.minus({ hours: 2 }).toISODate();
  const end = target.plus({ hours: 2 }).toISODate();

  const config: AxiosRequestConfig = {
    url: "https://archive-api.open-meteo.com/v1/archive",
    method: "GET",
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
    timeout: OPEN_METEO_TIMEOUT_MS,
  };

  const { data } = await withRetry(
    () => axios.request(config),
    "open-meteo-archive"
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

async function fetchOpenMeteoForecastHourly(
  lat: number,
  lon: number,
  kickoffIso: string
): Promise<MeteoRow[]> {
  // Forecast API supports ~16 days ahead; request enough to cover kickoff
  const nowStart = DateTime.utc().startOf("day");
  const kickoff = DateTime.fromISO(kickoffIso, { zone: "utc" });
  const daysAhead = Math.max(0, Math.ceil(kickoff.diff(nowStart, "days").days));
  const forecastDays = Math.min(
    FORECAST_HORIZON_DAYS,
    Math.max(1, daysAhead + 1)
  );

  const config: AxiosRequestConfig = {
    url: "https://api.open-meteo.com/v1/forecast",
    method: "GET",
    params: {
      latitude: lat,
      longitude: lon,
      hourly:
        "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
      temperature_unit: "fahrenheit",
      windspeed_unit: "mph",
      timezone: "UTC",
      forecast_days: forecastDays,
    },
    timeout: OPEN_METEO_TIMEOUT_MS,
  };

  const { data } = await withRetry(
    () => axios.request(config),
    "open-meteo-forecast"
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

// ────────── SNAPSHOT BUILDER ──────────
function buildSnapshot(opts: {
  teamName: string;
  venueNorm: string;
  info: VenueInfo;
  hour: MeteoRow | null;
  source: "open-meteo-forecast" | "open-meteo-archive" | "indoor";
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

    return { games: (data ?? []) as GameRow[], error };
  }

  // Determine window: fixed > lookahead > lookback
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

  // Counters
  let inserted = 0;
  let skippedNoVenue = 0;
  let indoorCount = 0;
  let skippedBeyondForecast = 0;
  let usedForecast = 0;
  let usedArchive = 0;
  let nullNearest = 0;

  const now = DateTime.utc();

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

    const kickoff = DateTime.fromISO(g.scheduled_time, { zone: "utc" });

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
      // Decide forecast vs archive:
      // - Past (kickoff strictly before now) → archive
      // - Future or later today (kickoff >= now) → forecast (if within horizon)
      let rows: MeteoRow[] = [];
      let source: "open-meteo-forecast" | "open-meteo-archive" =
        "open-meteo-forecast";

      if (kickoff < now) {
        // ARCHIVE path
        source = "open-meteo-archive";
        try {
          rows = await fetchOpenMeteoArchiveHourly(
            info.lat,
            info.lon,
            g.scheduled_time
          );
          usedArchive++;
        } catch (e: any) {
          warn(
            `archive fetch failed game_id=${g.game_id} venue="${venueNorm}": ${
              e?.message || e
            }`
          );
          rows = [];
        }
      } else {
        // FORECAST path (enforce horizon)
        const daysAhead = Math.ceil(
          kickoff.diff(now.startOf("day"), "days").days
        );
        if (daysAhead > FORECAST_HORIZON_DAYS) {
          skippedBeyondForecast++;
          warn(
            `skip game_id=${g.game_id}: kickoff ${g.scheduled_time} beyond ${FORECAST_HORIZON_DAYS}-day forecast horizon`
          );
          continue;
        }

        try {
          rows = await fetchOpenMeteoForecastHourly(
            info.lat,
            info.lon,
            g.scheduled_time
          );
          usedForecast++;
        } catch (e: any) {
          warn(
            `forecast fetch failed game_id=${g.game_id} venue="${venueNorm}": ${
              e?.message || e
            }`
          );
          rows = [];
        }
      }

      const nearest = pickNearestByTime(rows, g.scheduled_time);
      if (!nearest) {
        nullNearest++;
        warn(
          `no nearest hour found game_id=${g.game_id} kickoff=${g.scheduled_time} rows=${rows.length}`
        );
      }

      snapshot = buildSnapshot({
        teamName: g.home_team,
        venueNorm,
        info,
        hour: nearest,
        source,
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
  try {
    await sb.rpc("refresh_mv_nfl_latest_forecast_per_game");
  } catch (e: any) {
    warn(`refresh_mv_nfl_latest_forecast_per_game failed: ${e?.message || e}`);
  }

  try {
    await sb.rpc("refresh_nfl_weather_climatology");
  } catch (e: any) {
    warn(`refresh_nfl_weather_climatology failed: ${e?.message || e}`);
  }

  const summary = {
    found: games?.length ?? 0,
    inserted,
    skipped_no_venue: skippedNoVenue,
    indoor_snapshots: indoorCount,
    forecast_used: usedForecast,
    archive_used: usedArchive,
    skipped_beyond_forecast_horizon: skippedBeyondForecast,
    null_nearest_rows: nullNearest,
    lookahead_days: LOOKAHEAD_DAYS,
    start_iso: START_ISO || null,
    end_iso: END_ISO || null,
    lookback_days: LOOKBACK_DAYS || null,
    horizon_days: FORECAST_HORIZON_DAYS,
  };

  console.log(JSON.stringify(summary, null, 2));
  log("DONE");
}

// run if invoked directly
if (require.main === module) {
  main().catch((e) => {
    err("fatal:", e);
    process.exit(1);
  });
}
