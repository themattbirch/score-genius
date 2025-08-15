// backfill_missing_weather.ts  — clean, CJS-friendly, drop-in
import fs from "fs";
import path from "path";
import axios from "axios";
import { DateTime } from "luxon";
import { createClient } from "@supabase/supabase-js";
import * as dotenv from "dotenv";

// ---------- ENV ----------
dotenv.config({ path: path.resolve(process.cwd(), ".env") });
const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY!;
const RAPIDAPI_KEY = process.env.RAPIDAPI_KEY || ""; // optional (fallback exists)

const sb = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// ---------- Types ----------
type VenueInfo = { lat: number; lon: number; is_indoor: boolean };
type MeteoRow = {
  time: string;
  temp?: number;
  rhum?: number;
  wspd?: number;
  wdir?: number;
};

// ---------- Utils ----------
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const norm = (s: string | null | undefined) =>
  String(s ?? "")
    .trim()
    .toLowerCase();

// Some schedule names differ from your JSON’s stadium names
const VENUE_ALIASES: Record<string, string> = {
  "huntington bank field": "cleveland browns stadium",
};

function loadVenueDirectory(): Record<string, VenueInfo> {
  const file = path.resolve(process.cwd(), "data", "stadium_data.json"); // backend/data/stadium_data.json
  if (!fs.existsSync(file)) {
    console.warn(
      `No stadium_data.json at ${file}. Unknown venues will be skipped.`
    );
    return {};
  }
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
    const isIndoor =
      !!rec?.is_indoor ||
      !!rec?.dome ||
      (typeof rec?.roof === "string" &&
        rec.roof.toLowerCase().includes("closed"));
    byStadium[key] = { lat, lon, is_indoor: isIndoor };
  }
  for (const [alias, canonical] of Object.entries(VENUE_ALIASES)) {
    const v = byStadium[norm(canonical)];
    if (v) byStadium[norm(alias)] = v;
  }
  return byStadium;
}

const VENUE_DIR = loadVenueDirectory();

function pickNearestByTime<T extends { time: string }>(
  rows: T[],
  kickoffUtcISO: string
): T | null {
  if (!rows?.length) return null;
  const target = new Date(kickoffUtcISO).getTime();
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

// ---------- Weather sources ----------
async function fetchMeteostatPointHourly(
  lat: number,
  lon: number,
  kickoffUtcISO: string
) {
  const start = DateTime.fromISO(kickoffUtcISO, { zone: "utc" })
    .minus({ hours: 1 })
    .toISODate();
  const end = DateTime.fromISO(kickoffUtcISO, { zone: "utc" })
    .plus({ hours: 1 })
    .toISODate();

  const { data } = await axios.get(
    "https://meteostat.p.rapidapi.com/point/hourly",
    {
      params: { lat, lon, start, end, tz: "UTC", units: "imperial" },
      headers: {
        "x-rapidapi-host": "meteostat.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
      },
      timeout: 15_000,
    }
  );

  // { meta, data: [{ time:'YYYY-MM-DD HH:mm', temp, rhum, wspd, wdir }, ...] }
  return (data?.data ?? []) as MeteoRow[];
}

async function fetchMeteostatWithRetry(
  lat: number,
  lon: number,
  kickoffUtcISO: string,
  tries = 4
) {
  let attempt = 0,
    delay = 800;
  for (;;) {
    try {
      return await fetchMeteostatPointHourly(lat, lon, kickoffUtcISO);
    } catch (e: any) {
      const status = e?.response?.status;
      attempt++;
      const retriable = status === 429 || status >= 500 || !status;
      if (!retriable || attempt >= tries) throw e;
      await sleep(delay);
      delay *= 2;
    }
  }
}

async function fetchOpenMeteoArchiveHourly(
  lat: number,
  lon: number,
  kickoffUtcISO: string
) {
  const start = DateTime.fromISO(kickoffUtcISO, { zone: "utc" })
    .minus({ hours: 1 })
    .toISODate();
  const end = DateTime.fromISO(kickoffUtcISO, { zone: "utc" })
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
        hourly: [
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "wind_direction_10m",
        ].join(","),
        temperature_unit: "fahrenheit",
        windspeed_unit: "mph",
        timezone: "UTC",
      },
      timeout: 15_000,
    }
  );

  const t = data?.hourly?.time || [];
  const temp = data?.hourly?.temperature_2m || [];
  const rh = data?.hourly?.relative_humidity_2m || [];
  const wspd = data?.hourly?.wind_speed_10m || [];
  const wdir = data?.hourly?.wind_direction_10m || [];

  const rows: MeteoRow[] = t.map((time: string, i: number) => ({
    time, // 'YYYY-MM-DDTHH:00'
    temp: temp[i] ?? null,
    rhum: rh[i] ?? null,
    wspd: wspd[i] ?? null,
    wdir: wdir[i] ?? null,
  }));
  return rows;
}

// ---------- Snapshot builder ----------
// add param 'kickoffUtcISO' to buildSnapshot signature & calls
function buildSnapshot({
  teamName,
  venueNorm,
  info,
  hour,
  source,
  kickoffUtcISO,
}: {
  teamName: string;
  venueNorm: string;
  info: { lat: number; lon: number; is_indoor: boolean };
  hour: MeteoRow | null;
  source: "meteostat" | "open-meteo" | "indoor";
  kickoffUtcISO: string;
}) {
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
    // >>> critical for MV selection:
    captured_at: kickoffUtcISO, // align snapshot time to the game time
  };
}

// ---------- Main ----------
async function main() {
  // Pull candidates from MV (missing weather), oldest first
  const { data: mvRows, error: mvErr } = await sb
    .from("weather_nfl_latest_forecast_per_game")
    .select(
      "game_id, venue_norm, scheduled_time, temperature_f, wind_mph, humidity_pct"
    )
    .order("scheduled_time", { ascending: true });
  if (mvErr) throw mvErr;

  const missing = (mvRows ?? []).filter(
    (r) =>
      r.temperature_f == null || r.wind_mph == null || r.humidity_pct == null
  );

  for (const r of missing) {
    const venue = norm(r.venue_norm);
    const info = VENUE_DIR[venue];
    if (!info) {
      console.warn(
        `Skipping unknown venue coords: "${venue}" (game_id ${r.game_id})`
      );
      continue;
    }

    // Get home team
    const { data: sched, error: schedErr } = await sb
      .from("nfl_game_schedule")
      .select("home_team")
      .eq("game_id", r.game_id)
      .single();
    if (schedErr) {
      console.warn(`No schedule for game_id ${r.game_id}: ${schedErr.message}`);
      continue;
    }
    const homeTeam: string = sched?.home_team ?? "unknown";

    let snapshot: any;

    if (info.is_indoor) {
      // Indoor: no hourly rows, explicit "indoor" source, pass kickoff time
      snapshot = buildSnapshot({
        teamName: homeTeam,
        venueNorm: venue,
        info,
        hour: null,
        source: "indoor",
        kickoffUtcISO: r.scheduled_time,
      });
    } else {
      // Outdoor: try Meteostat with retry, fall back to Open-Meteo
      let rows: MeteoRow[] = [];
      let used: "meteostat" | "open-meteo" = "meteostat";

      try {
        if (!RAPIDAPI_KEY) throw new Error("No RAPIDAPI_KEY");
        rows = await fetchMeteostatWithRetry(
          info.lat,
          info.lon,
          r.scheduled_time
        );
      } catch (e: any) {
        const status = e?.response?.status;
        if (status === 403 || status === 429 || !RAPIDAPI_KEY) {
          console.warn(
            `Meteostat ${
              status ?? "no-key"
            }; falling back to Open-Meteo for ${venue} (game ${r.game_id})`
          );
          used = "open-meteo";
          rows = await fetchOpenMeteoArchiveHourly(
            info.lat,
            info.lon,
            r.scheduled_time
          );
        } else {
          throw e;
        }
      }

      const nearest = pickNearestByTime(rows, r.scheduled_time);

      snapshot = buildSnapshot({
        teamName: homeTeam,
        venueNorm: venue,
        info,
        hour: nearest,
        source: used,
        kickoffUtcISO: r.scheduled_time, // <-- required by buildSnapshot signature
      });
    }

    const { error: insErr } = await sb
      .from("weather_forecast_snapshots")
      .insert([{ ...snapshot }]); // snapshot already includes raw & captured_at

    if (insErr) {
      console.warn(`Insert failed for game ${r.game_id}: ${insErr.message}`);
    } else {
      console.log(
        `Inserted snapshot for game ${r.game_id} @ ${venue} [is_indoor=${snapshot.is_indoor}, source=${snapshot.source}]`
      );
    }

    // Gentle pacing to avoid rate limits when many outdoor games
    await sleep(300);
  }

  // Refresh MVs (assumes you created these functions once)
  await sb.rpc("refresh_mv_nfl_latest_forecast_per_game");
  await sb.rpc("refresh_nfl_weather_climatology");
  console.log("Refreshed MVs.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
