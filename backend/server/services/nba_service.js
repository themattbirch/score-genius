// backend/server/services/nba_service.js

// Import the shared Supabase client instance
import supabase from "../utils/supabase_client.js";
// Import Luxon for robust date/timezone handling
import { DateTime } from "luxon";
// Import your simple in‑memory or Redis cache helper
import cache from "../utils/cache.js";

// Define constants specific to this service
const NBA_SCHEDULE_TABLE = "nba_game_schedule";
const NBA_INJURIES_TABLE = "nba_injuries";
const NBA_HISTORICAL_GAMES_TABLE = "nba_historical_game_stats";
const NBA_HISTORICAL_TEAM_STATS_TABLE = "nba_historical_team_stats";
const NBA_HISTORICAL_PLAYER_STATS_TABLE = "nba_historical_player_stats";
const ET_ZONE_IDENTIFIER = "America/New_York";

// --- Helper function for dates (ensure consistent formatting YYYY-MM-DD) ---
const getUTCDateString = (date) => date.toISOString().split("T")[0];
// --- End Helper Function ---

// Fetch today & tomorrow’s schedule
export const fetchNbaScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching NBA schedule for today/tomorrow ET...");
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const todayStr = nowEt.toISODate();
  const tomorrowStr = nowEt.plus({ days: 1 }).toISODate();

  const { data, error, status } = await supabase
    .from(NBA_SCHEDULE_TABLE)
    .select(
      `
      game_id,
      game_date,
      scheduled_time,
      status,
      home_team,
      away_team,
      venue,
      predicted_home_score,
      predicted_away_score,
      moneyline_clean,
      spread_clean,
      total_clean
    `
    )
    .in("game_date", [todayStr, tomorrowStr])
    .order("scheduled_time", { ascending: true });

  if (error) {
    console.error("Supabase error fetching NBA schedule:", error);
    const dbError = new Error(error.message || "Database query failed");
    dbError.status = status || 500;
    throw dbError;
  }

  console.log(`Service: Found ${data?.length ?? 0} NBA games.`);
  return data || [];
};

// Fetch current injuries, caching for 30m
export const fetchNbaInjuries = async () => {
  const cacheKey = "nba_injuries_current";
  const ttl = 1800;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying injuries table...`);
  const { data, error } = await supabase
    .from(NBA_INJURIES_TABLE)
    .select(
      `
      injury_id, player_id, player_display_name, team_id, team_display_name,
      report_date_utc, injury_status, injury_status_abbr, injury_type,
      injury_location, injury_detail, injury_side, return_date_est,
      short_comment, long_comment, created_at, last_api_update_time
    `
    )
    .order("report_date_utc", { ascending: false });

  if (error) {
    console.error("Supabase error fetching injuries:", error);
    return null;
  }

  console.log(`Fetched ${data.length} injuries. Caching ${ttl}s.`);
  cache.set(cacheKey, data, ttl);
  return data;
};

// Fetch historical games with pagination & filters
export const fetchNbaGameHistory = async ({
  startDate,
  endDate,
  teamName,
  limit,
  page,
}) => {
  const cacheKey = `nba_game_history_${startDate || "null"}_${
    endDate || "null"
  }_${teamName || "null"}_${limit}_${page}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying historical games...`);
  let query = supabase.from(NBA_HISTORICAL_GAMES_TABLE).select(`
      game_id, game_date, home_team, away_team, home_score, away_score,
      home_q1, home_q2, home_q3, home_q4, home_ot,
      away_q1, away_q2, away_q3, away_q4, away_ot,
      home_assists, home_steals, home_blocks, home_turnovers, home_fouls, home_total_reb,
      away_assists, away_steals, away_blocks, away_turnovers, away_fouls, away_total_reb
    `);

  if (startDate) query = query.gte("game_date", startDate);
  if (endDate) query = query.lte("game_date", endDate);
  if (teamName)
    query = query.or(
      `home_team.ilike.%${teamName}%,away_team.ilike.%${teamName}%`
    );

  query = query.order("game_date", { ascending: false });
  query = query.range((page - 1) * limit, page * limit - 1);

  const { data, error } = await query;
  if (error) {
    console.error("Supabase error fetching historical games:", error);
    return null;
  }

  console.log(`Fetched ${data.length} historical games. Caching ${ttl}s.`);
  cache.set(cacheKey, data, ttl);
  return data;
};

// Fetch team stats for a given season
export const fetchNbaTeamStatsBySeason = async (teamId, seasonYearStr) => {
  const cacheKey = `nba_team_stats_${teamId}_${seasonYearStr}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  const startYear = parseInt(seasonYearStr, 10);
  if (isNaN(startYear)) {
    console.error("Invalid season year:", seasonYearStr);
    return null;
  }
  const seasonRange = `${startYear}-${startYear + 1}`;

  console.log(
    `CACHE MISS: ${cacheKey}. Querying team stats for ${seasonRange}...`
  );
  const { data, error } = await supabase
    .from(NBA_HISTORICAL_TEAM_STATS_TABLE)
    .select("*")
    .eq("team_id", teamId)
    .eq("season", seasonRange)
    .maybeSingle();

  if (error) {
    console.error("Supabase error fetching team stats:", error);
    return null;
  }

  console.log(
    data
      ? `Fetched stats for team ${teamId}, caching ${ttl}s.`
      : `No stats found for team ${teamId}. Caching null.`
  );
  cache.set(cacheKey, data, ttl);
  return data;
};

// Fetch player game history
export const fetchNbaPlayerGameHistory = async (playerId, { limit, page }) => {
  const cacheKey = `nba_player_history_${playerId}_${limit}_${page}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying player history...`);
  let query = supabase
    .from(NBA_HISTORICAL_PLAYER_STATS_TABLE)
    .select(
      `
      game_id, player_id, player_name, team_id, team_name, game_date, minutes,
      points, rebounds, assists, steals, blocks, turnovers, fouls,
      fg_made, fg_attempted, three_made, three_attempted, ft_made, ft_attempted
    `
    )
    .eq("player_id", playerId)
    .order("game_date", { ascending: false })
    .range((page - 1) * limit, page * limit - 1);

  const { data, error } = await query;
  if (error) {
    console.error("Supabase error fetching player history:", error);
    return null;
  }

  console.log(`Fetched ${data.length} player games. Caching ${ttl}s.`);
  cache.set(cacheKey, data, ttl);
  return data;
};

// Another schedule helper used by your controller
export const WorkspaceNbaScheduleForTodayAndTomorrow = async () => {
  const cacheKey = "nba_schedule_today_tomorrow";
  const ttl = 1800;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying schedule...`);
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  const todayStr = getUTCDateString(today);
  const tomorrowStr = getUTCDateString(tomorrow);

  const { data, error } = await supabase
    .from(NBA_SCHEDULE_TABLE)
    .select(
      `
      game_id, game_date, home_team, away_team, scheduled_time,
      venue, status, moneyline_clean, spread_clean, total_clean,
      predicted_home_score, predicted_away_score, updated_at
    `
    )
    .in("game_date", [todayStr, tomorrowStr])
    .order("scheduled_time", { ascending: true });

  if (error) {
    console.error("Supabase error:", error);
    return null;
  }

  console.log(`Fetched ${data.length} games. Caching ${ttl}s.`);
  cache.set(cacheKey, data, ttl);
  return data;
};

/**
 * Returns tomorrow’s or today’s schedule with model predictions & odds.
 * @param {string} date YYYY‑MM‑DD
 */
export async function getSchedule(date) {
  const { data, error } = await supabase
    .from("nba_games_schedule")
    .select(
      `
      id, home_team_abbr, away_team_abbr, tipoff_ts,
      spread, total, model_home_score, model_away_score
    `
    )
    .eq("game_date", date)
    .order("tipoff_ts");

  if (error) throw error;
  return data.map((row) => ({
    id: row.id,
    homeTeam: row.home_team_abbr,
    awayTeam: row.away_team_abbr,
    tipoff: row.tipoff_ts,
    spread: row.spread,
    total: row.total,
    predictionHome: row.model_home_score,
    predictionAway: row.model_away_score,
  }));
}
