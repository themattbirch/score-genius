// backend/server/services/nfl_service.js

import supabase from "../utils/supabase_client.js";
import cache from "../utils/cache.js";
import { DateTime } from "luxon";

const SCHEDULE_TABLE = "nfl_preseason_schedule";
const HIST_STATS_TABLE = "nfl_historical_game_stats";
const SNAPSHOT_TABLE = "nfl_snapshots";
const FULL_VIEW = "v_nfl_team_season_full";
const REG_VIEW = "v_nfl_team_season_regonly";
const CACHE_TTL = Number(process.env.NFL_TEAM_CACHE_TTL || 43200);

const NFL_SOS_VIEW = "v_nfl_team_sos";
const NFL_SRS_VIEW = "v_nfl_team_srs_lite";
const ADVANCED_STATS_VIEW = "mv_nfl_advanced_team_stats";
const DEFAULT_CACHE_TTL = Number(process.env.NFL_VIEW_CACHE_TTL || 3600);

// (This function doesn't need to change)
function buildViewKey(view, { season, teamIds, conference, division }) {
  return [
    view,
    season,
    teamIds?.join(",") ?? "all",
    conference ?? "*",
    division ?? "*",
  ].join(":");
}

// (This function doesn't need to change)
export function buildCacheHeader() {
  return {
    "Cache-Control":
      "public, max-age=43200, stale-while-revalidate=3600, stale-if-error=86400",
  };
}

export function mapScheduleRow(raw, isPast) {
  const base = {
    id: String(raw.game_id),
    game_date: raw.game_date,
    gameTimeUTC: raw.scheduled_time || raw.game_timestamp, // Use timestamp for past games
    status: raw.status,
    homeTeamId: raw.home_team_id,
    awayTeamId: raw.away_team_id,
    dataType: isPast ? "historical" : "schedule",
    // 👇 This now handles both tables
    homeTeamName: raw.home_team_name ?? raw.home_team,
    awayTeamName: raw.away_team_name ?? raw.away_team,
  };

  if (isPast) {
    return {
      ...base,
      home_final_score: raw.home_score,
      away_final_score: raw.away_score,
    };
  }

  return {
    ...base,
    spreadLine: raw.spread_clean
      ? parseFloat((raw.spread_clean.match(/-?\d+(\.\d+)?/) || ["0"])[0])
      : null,
    totalLine: raw.total_clean
      ? parseFloat((raw.total_clean.match(/\d+(\.\d+)?/) || ["0"])[0])
      : null,
    predictedHomeScore: raw.predicted_home_score,
    predictedAwayScore: raw.predicted_away_score,
  };
}

// Numeric helpers
const toNum = (v) => {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  const n = Number.parseFloat(v);
  return Number.isFinite(n) ? n : null;
};
function safePct(num, den) {
  return den ? num / den : null;
}
function recomputeYardsPerPlay(yards, plays, ypp) {
  if (!plays) return toNum(ypp);
  const est = yards / plays;
  const rep = toNum(ypp);
  if (rep == null) return est;
  return Math.abs(rep - est) > 1.5 ? est : rep;
}

// Full mapping for team-season rows
export function mapNflRow(raw, includeRaw = false) {
  const wins = raw.won ?? raw.wins_total ?? 0;
  const losses = raw.lost ?? raw.losses_total ?? 0;
  const ties = raw.ties ?? raw.ties_total ?? 0;
  const gp = raw.games_played ?? wins + losses + ties;

  const pointsFor = raw.points_for ?? raw.points_for_total ?? 0;
  const pointsAgainst = raw.points_against ?? raw.points_against_total ?? 0;
  const pointsDiff =
    raw.points_difference ?? raw.points_diff_total ?? pointsFor - pointsAgainst;

  const yards = raw.yards_total ?? null;
  const plays = raw.plays_total ?? null;
  const ypp = recomputeYardsPerPlay(yards, plays, raw.yards_per_play_avg);

  const obj = {
    season: raw.season,
    teamId: raw.team_id,
    teamName: raw.team_name,
    teamLogo: raw.team_logo,
    conference: raw.conference,
    division: raw.division,
    wins,
    losses,
    ties,
    winPct: safePct(wins, wins + losses + ties) ?? toNum(raw.win_pct) ?? 0,
    pointsFor,
    pointsAgainst,
    pointsDiff,
    gamesPlayed: gp,
    plays,
    yards,
    yardsPerPlay: ypp,
    turnoversPerGame: toNum(raw.turnovers_per_game_avg),
    thirdDownPct: toNum(raw.third_down_pct_avg),
    fourthDownPct: toNum(raw.fourth_down_pct_avg),
    redZonePct: toNum(raw.red_zone_pct_avg),
    drivesPerGame: toNum(raw.drives_per_game_avg),
    possessionTimeAvgSec: toNum(raw.possession_time_avg_sec),
    passYards: raw.pass_yards_total ?? null,
    passYardsPerPass: toNum(raw.pass_yards_per_pass_avg),
    rushYards: raw.rush_yards_total ?? null,
    rushYardsPerRush: toNum(raw.rush_yards_per_rush_avg),
    sackRatePerPlay: toNum(raw.sack_rate_per_play),
    sackRatePerDropback: toNum(raw.sack_rate_per_dropback),
    penaltiesPerPlay: toNum(raw.penalties_per_play),
    home: {
      games: raw.games_played_home ?? null,
      wins: raw.wins_home_total ?? null,
      losses: raw.losses_home_total ?? null,
      ties: raw.ties_home_total ?? null,
      winPct: toNum(raw.win_pct_home),
      pointsFor: raw.points_for_home_total ?? null,
      pointsAgainst: raw.points_against_home_total ?? null,
    },
    away: {
      games: raw.games_played_away ?? null,
      wins: raw.wins_away_total ?? null,
      losses: raw.losses_away_total ?? null,
      ties: raw.ties_away_total ?? null,
      winPct: toNum(raw.win_pct_away),
      pointsFor: raw.points_for_away_total ?? null,
      pointsAgainst: raw.points_against_away_total ?? null,
    },
    recordHome: raw.record_home ?? null,
    recordRoad: raw.record_road ?? null,
    recordConf: raw.record_conference ?? null,
    recordDiv: raw.record_division ?? null,
    streak: raw.streak ?? "",
    updatedAt: raw.standings_updated_at ?? null,
  };

  if (includeRaw) obj.raw = raw;
  return obj;
}

// Internal key builder
function buildKey(
  prefix,
  { season, teamIds, conference, division, includeRaw }
) {
  return [
    prefix,
    season,
    teamIds?.join("-") || "all",
    conference || "*",
    division || "*",
    includeRaw ? "raw" : "noRaw",
  ].join(":");
}

// Internal fetcher from views
async function fetchFromView(view, args) {
  const cacheKey = buildKey(view, args);
  const cached = cache.get(cacheKey);
  if (cached !== undefined) return cached;

  let query = supabase.from(view).select("*").eq("season", args.season);
  if (args.teamIds?.length) query = query.in("team_id", args.teamIds);
  if (args.conference) query = query.eq("conference", args.conference);
  if (args.division) query = query.eq("division", args.division);

  query = query
    .order("conference", { ascending: true })
    .order("division", { ascending: true })
    .order("team_name", { ascending: true });

  const { data, error, status } = await query;
  if (error) {
    const err = new Error(error.message || `Error fetching ${view}`);
    err.status = status || 503;
    throw err;
  }

  const mapped = Array.isArray(data)
    ? data.map((r) => mapNflRow(r, args.includeRaw))
    : [];
  cache.set(cacheKey, mapped, CACHE_TTL);
  return mapped;
}

// Public service functions
export async function fetchNflTeamSeasonFull(args) {
  return fetchFromView(FULL_VIEW, args);
}
export async function fetchNflTeamSeasonRegOnly(args) {
  return fetchFromView(REG_VIEW, args);
}
export async function fetchNflScheduleData(date) {
  const nowEt = DateTime.now().setZone("America/New_York");
  const input = DateTime.fromISO(date, { zone: "America/New_York" });
  if (!input.isValid) {
    const err = new Error(`Invalid date: ${date}`);
    err.status = 400;
    throw err;
  }

  const isPast = input.startOf("day") < nowEt.startOf("day");
  const table = isPast ? HIST_STATS_TABLE : SCHEDULE_TABLE;

  // 👇 Selects the correct columns for each table
  const cols = isPast
    ? "game_id, game_date, game_timestamp, home_team_name, away_team_name, home_team_id, away_team_id, home_score, away_score"
    : "game_id, game_date, scheduled_time, home_team, away_team, home_team_id, away_team_id, status, spread_clean, total_clean, predicted_home_score, predicted_away_score";

  const { data, error, status } = await supabase
    .from(table)
    .select(cols)
    .eq("game_date", date)
    .order(isPast ? "game_date" : "scheduled_time", { ascending: true });

  if (error) {
    const e = new Error(error.message);
    e.status = status;
    throw e;
  }

  // 👇 A more robust filter that checks for either set of team names
  return (data || [])
    .filter(
      (row) =>
        row &&
        row.game_id &&
        (row.home_team || row.home_team_name) &&
        (row.away_team || row.away_team_name)
    )
    .map((r) => mapScheduleRow(r, isPast));
}

export async function fetchNflSnapshotsByIds(ids) {
  const hits = [];
  const misses = [];
  ids.forEach((id) => {
    const c = cache.get(id);
    if (c) hits.push(c);
    else misses.push(id);
  });

  let fetched = [];
  if (misses.length) {
    const { data, error, status } = await supabase
      .from(SNAPSHOT_TABLE)
      .select("*")
      .in("game_id", misses);

    if (error) {
      if (
        error.message.includes(`relation "${SNAPSHOT_TABLE}" does not exist`)
      ) {
        return [...hits];
      }
      const e = new Error(error.message);
      e.status = status;
      throw e;
    }
    fetched = Array.isArray(data) ? data : [];
    fetched.forEach((snap) => cache.set(snap.game_id, snap));
  }
  return [...hits, ...fetched];
}
export async function fetchNflSnapshotData(gameId) {
  const cached = cache.get(gameId);
  if (cached) return cached;
  const { data, error, status } = await supabase
    .from(SNAPSHOT_TABLE)
    .select("*")
    .eq("game_id", gameId)
    .maybeSingle();
  if (error) {
    const e = new Error(error.message);
    e.status = status;
    throw e;
  }
  if (!data) {
    const e = new Error(`Snapshot for game ${gameId} not found`);
    e.status = 404;
    throw e;
  }
  cache.set(gameId, data);
  return data;
}
export async function fetchNflDashboardCards({
  season,
  teamIds,
  conference,
  division,
}) {
  // Reuse our generic fetchFromView
  return fetchFromView("v_nfl_dashboard_cards", {
    season,
    teamIds,
    conference,
    division,
    includeRaw: false,
  });
}
// Fetch Strength of Schedule
export async function fetchNflSos({ season, teamIds, conference, division }) {
  const cacheKey = buildViewKey(NFL_SOS_VIEW, {
    season,
    teamIds,
    conference,
    division,
  });
  const cached = cache.get(cacheKey);
  if (cached !== undefined) return cached;

  let q = supabase.from(NFL_SOS_VIEW).select("*").eq("season", season);
  if (teamIds?.length) q = q.in("team_id", teamIds);
  if (conference) q = q.eq("conference", conference);
  if (division) q = q.eq("division", division);

  const { data, error, status } = await q;

  if (error) {
    const e = new Error(error.message || "Error fetching SoS");
    e.status = status;
    throw e;
  }

  // Normalize field: rename sos_pct to sos so downstream code is consistent
  const result = Array.isArray(data)
    ? data.map((r) => {
        const copy = { ...r };
        if (copy.sos_pct !== undefined) {
          copy.sos = copy.sos_pct;
          delete copy.sos_pct;
        }
        return copy;
      })
    : [];

  cache.set(cacheKey, result, DEFAULT_CACHE_TTL);
  return result;
}

// Cron & Validation RPC calls ----------------------------------------------

/**
 * Checks cron job health via RPC check_nfl_cron_health()
 */
export async function checkCronHealth() {
  const { data, error, status } = await supabase.rpc("check_nfl_cron_health");
  if (error) {
    const e = new Error(error.message || "Cron health RPC failed");
    e.status = status || 503;
    throw e;
  }
  return data;
}

/**
 * Validates team aggregation via RPC validate_nfl_team_agg()
 */
export async function validateTeamAgg() {
  const { data, error, status } = await supabase.rpc("validate_nfl_team_agg");
  if (error) {
    const e = new Error(error.message || "Validation RPC failed");
    e.status = status || 503;
    throw e;
  }
  return data;
}
/**
 * Fetch Simple‑Rating‑System “lite” from v_nfl_team_srs_lite
 */
export async function fetchNflSrs({ season, teamIds, conference, division }) {
  const cacheKey = buildViewKey(NFL_SRS_VIEW, {
    season,
    teamIds,
    conference,
    division,
  });
  const cached = cache.get(cacheKey);
  if (cached !== undefined) return cached;

  let q = supabase.from(NFL_SRS_VIEW).select("*").eq("season", season);
  if (teamIds?.length) q = q.in("team_id", teamIds);
  if (conference) q = q.eq("conference", conference);
  if (division) q = q.eq("division", division);

  const { data, error, status } = await q;
  if (error) {
    const e = new Error(error.message || "Error fetching SRS");
    e.status = status;
    throw e;
  }

  const result = Array.isArray(data) ? data : [];
  cache.set(cacheKey, result, DEFAULT_CACHE_TTL);
  return result;
}
export async function fetchNflGameById(gameId) {
  // Determine if this game is historical or upcoming based on today
  const nowEt = DateTime.now().setZone("America/New_York");
  // We'll fetch from the schedule table; for past games, you may want to join the historical table instead
  const cols = [
    "game_id",
    "game_date",
    "scheduled_time",
    "home_team_id",
    "away_team_id",
    "status",
    "spread_clean",
    "total_clean",
    "predicted_home_score",
    "predicted_away_score",
  ].join(", ");

  const { data, error, status } = await supabase
    .from(SCHEDULE_TABLE)
    .select(cols)
    .eq("game_id", gameId)
    .maybeSingle();

  if (error) {
    const e = new Error(error.message || `Error fetching game ${gameId}`);
    e.status = status || 503;
    throw e;
  }

  if (!data) {
    return null;
  }

  // Map raw row to your controller shape
  // We pass `false` for `isPast` so mapScheduleRow knows to look for scheduled_time, predictions
  return mapScheduleRow(data, false);
}

/**
 * Fetches advanced team stats (SRS, SOS, etc.) from the materialized view.
 */
export async function fetchNflAdvancedStats({ season }) {
  const cacheKey = `nfl-advanced-stats:${season}`;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const { data, error, status } = await supabase
    .from(ADVANCED_STATS_VIEW)
    .select("*")
    .eq("season", season)
    .order("team_name", { ascending: true });

  if (error) {
    const e = new Error(error.message || "Error fetching advanced stats");
    e.status = status;
    throw e;
  }

  const result = Array.isArray(data) ? data : [];
  cache.set(cacheKey, result, DEFAULT_CACHE_TTL);
  return result;
}
export async function fetchNflSeasonStats({
  season,
  teamIds,
  conference,
  division,
}) {
  // Reuse generic view fetch pattern manually here
  const cacheKey = buildViewKey("mv_nfl_season_stats", {
    season,
    teamIds,
    conference,
    division,
  });
  const cached = cache.get(cacheKey);
  if (cached !== undefined) return cached;

  let query = supabase
    .from("mv_nfl_season_stats")
    .select("*")
    .eq("season", season);
  if (teamIds?.length) query = query.in("team_id", teamIds);
  if (conference) query = query.eq("conference", conference);
  if (division) query = query.eq("division", division);

  const { data, error, status } = await query;
  if (error) {
    const e = new Error(error.message || "Error fetching season stats");
    e.status = status || 503;
    throw e;
  }
  const result = Array.isArray(data) ? data : [];
  cache.set(cacheKey, result, DEFAULT_CACHE_TTL);
  return result;
}
