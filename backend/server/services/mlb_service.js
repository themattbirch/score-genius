// backend/server/services/mlb_service.js

import supabase from "../utils/supabase_client.js";
import { DateTime } from "luxon";
import cache from "../utils/cache.js";

const MLB_SCHEDULE_TABLE = "mlb_game_schedule";
const MLB_HISTORICAL_GAMES_TABLE = "mlb_historical_game_stats";
const MLB_HISTORICAL_TEAM_STATS_TABLE = "mlb_historical_team_stats";
const ET_ZONE_IDENTIFIER = "America/New_York";

const getUTCDateString = (date) => {
  return date.toISOString().split("T")[0];
};

/**
 * Represents unified MLB game data.
 * @typedef {object} UnifiedMLBGameData
 * @property {string} id
 * @property {string} game_date
 * @property {string} homeTeamName
 * @property {string} awayTeamName
 * @property {string | null} [gameTimeUTC]
 * @property {string | null} [statusState]
 * @property {string | null} [homePitcher]
 * @property {string | null} [awayPitcher]
 * @property {string | null} [homePitcherHand]
 * @property {string | null} [awayPitcherHand]
 * @property {string | number | null} [moneylineHome]
 * @property {string | number | null} [moneylineAway]
 * @property {number | null} [spreadLine]
 * @property {number | null} [totalLine]
 * @property {number | null} [home_final_score]
 * @property {number | null} [away_final_score]
 * @property {'schedule' | 'historical'} dataType
 */

// --- This function seems unrelated to the specific date schedule needed ---
// Keeping it as it was, but ensure it's not being called by mistake by the controller
export const fetchMlbScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching MLB schedule for today/tomorrow ET...");
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const todayStr = nowEt.toISODate();
  const tomorrowStr = nowEt.plus({ days: 1 }).toISODate();
  console.log(`Service: Querying ${MLB_SCHEDULE_TABLE} for dates: ${todayStr}, ${tomorrowStr}`);
  try {
    const { data, error, status } = await supabase
      .from(MLB_SCHEDULE_TABLE)
      .select(`game_id, scheduled_time_utc, game_date_et, status_detail, status_state, home_team_name, away_team_name, home_probable_pitcher_name, home_probable_pitcher_handedness, away_probable_pitcher_name, away_probable_pitcher_handedness, moneyline_home_clean, moneyline_away_clean, spread_home_line_clean, spread_home_price_clean, spread_away_price_clean, total_line_clean, total_over_price_clean, total_under_price_clean`)
      .in("game_date_et", [todayStr, tomorrowStr])
      .order("scheduled_time_utc", { ascending: true });

    if (error) {
      console.error("Supabase error fetching MLB schedule (Today/Tomorrow):", error);
      throw error; // Re-throw Supabase errors
    }
    console.log(`Service: Found ${data?.length ?? 0} MLB games (Today/Tomorrow).`);
    return data || [];
  } catch (error) {
    console.error("Error in fetchMlbScheduleForTodayAndTomorrow service:", error);
    throw error; // Re-throw other errors
  }
};

// --- CORRECTED: MLB Historical Game List (Removed misplaced schedule code) ---
export const fetchMlbGameHistory = async (options) => {
  const cacheKey = `mlb_game_history_${JSON.stringify(options)}`;
  const ttl = 86400;
  const cachedData = cache.get(cacheKey);
  if (cachedData !== undefined) {
    console.log(`CACHE HIT for key: ${cacheKey}`);
    return cachedData;
  }

  console.log(`CACHE MISS: ${cacheKey}. Fetching historical MLB games:`, options);
  try {
    const selectColumns = `game_id, game_date_time_utc, season, league_id, status_long, status_short, home_team_id, home_team_name, away_team_id, away_team_name, home_score, away_score, home_hits, away_hits, home_errors, away_errors, h_inn_1, h_inn_2, h_inn_3, h_inn_4, h_inn_5, h_inn_6, h_inn_7, h_inn_8, h_inn_9, h_inn_extra, a_inn_1, a_inn_2, a_inn_3, a_inn_4, a_inn_5, a_inn_6, a_inn_7, a_inn_8, a_inn_9, a_inn_extra, updated_at`;
    let query = supabase.from(MLB_HISTORICAL_GAMES_TABLE).select(selectColumns);

    // Apply filters
    if (options.startDate) query = query.gte("game_date_time_utc", options.startDate);
    if (options.endDate) query = query.lte("game_date_time_utc", options.endDate);
    if (options.teamName) query = query.or(`home_team_name.ilike.%${options.teamName}%,away_team_name.ilike.%${options.teamName}%`);

    query = query.order("game_date_time_utc", { ascending: false });
    const offset = (options.page - 1) * options.limit;
    query = query.range(offset, offset + options.limit - 1);

    // *** Execute the HISTORICAL query ***
    const { data, error } = await query;

    if (error) {
      console.error("Supabase error fetching MLB historical games:", error.message);
      return []; // Return empty array on error, don't cache
    }

    const resultData = data || [];
    console.log(`Workspaceed ${resultData.length} MLB historical games. Caching.`);
    cache.set(cacheKey, resultData, ttl);
    return resultData;

  } catch (error) {
    console.error("Error in fetchMlbGameHistory service:", error.message);
    return []; // Return empty array on unexpected errors
  }
};

// --- MLB Historical Team Stats (Keep as is) ---
export const fetchMlbTeamStatsBySeason = async (teamId, seasonYear) => {
  // ... (code seems okay, no changes needed from previous version) ...
  // Remember to handle cache and errors appropriately
    const cacheKey = `mlb_team_stats_${teamId}_${seasonYear}`;
    const ttl = 86400;
    const cachedData = cache.get(cacheKey);
    if (cachedData !== undefined) { return cachedData; }
    console.log(`CACHE MISS: ${cacheKey}. Fetching MLB team stats...`);
    try {
        const selectColumns = `id, team_id, team_name, season, league_id, league_name, games_played_home, games_played_away, games_played_all, wins_home_total, wins_home_percentage, wins_away_total, wins_away_percentage, wins_all_total, wins_all_percentage, losses_home_total, losses_home_percentage, losses_away_total, losses_away_percentage, losses_all_total, losses_all_percentage, runs_for_total_home, runs_for_total_away, runs_for_total_all, runs_for_avg_home, runs_for_avg_away, runs_for_avg_all, runs_against_total_home, runs_against_total_away, runs_against_total_all, runs_against_avg_home, runs_against_avg_away, runs_against_avg_all, updated_at`;
        const { data, error } = await supabase
            .from(MLB_HISTORICAL_TEAM_STATS_TABLE)
            .select(selectColumns)
            .eq("team_id", teamId)
            .eq("season", seasonYear)
            .maybeSingle();
        if (error) throw error;
        console.log(`Workspaceed MLB team stats for ${teamId} ${seasonYear}. Caching.`);
        cache.set(cacheKey, data, ttl);
        return data;
    } catch (error) {
        console.error("Error in fetchMlbTeamStatsBySeason service:", error.message);
        return null; // Or throw error
    }
};

// --- This function also seems unrelated to getting schedule by a specific date ---
// Keeping it as it was, but ensure it's not called by mistake
export const WorkspaceMlbScheduleForTodayAndTomorrow = async () => {
  const cacheKey = "mlb_schedule_today_tomorrow";
  const ttl = 1800;
  const cachedData = cache.get(cacheKey);
  if (cachedData !== undefined) { return cachedData; }
  console.log(`CACHE MISS: ${cacheKey}. Fetching from Supabase...`);
  try {
    const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
    const todayStr = nowEt.toISODate();
    const tomorrowStr = nowEt.plus({ days: 1 }).toISODate();
    console.log(`Querying MLB schedule for dates: ${todayStr}, ${tomorrowStr}`);
    const { data, error } = await supabase
      .from(MLB_SCHEDULE_TABLE)
      .select(`game_id, scheduled_time_utc, game_date_et, status_detail, status_state, home_team_id, home_team_name, away_team_id, away_team_name, home_probable_pitcher_name, away_probable_pitcher_name, home_probable_pitcher_handedness, away_probable_pitcher_handedness, moneyline_home_clean, moneyline_away_clean, spread_home_line_clean, spread_home_price_clean, spread_away_price_clean, total_line_clean, total_over_price_clean, total_under_price_clean, updated_at`)
      .in("game_date_et", [todayStr, tomorrowStr])
      .order("scheduled_time_utc", { ascending: true });
    if (error) throw error;
    const resultData = data || [];
    console.log(`Workspaceed ${resultData.length} MLB games (Workspace). Caching.`);
    cache.set(cacheKey, resultData, ttl);
    return resultData;
  } catch (error) {
    console.error(`Error in WorkspaceMlbSchedule service: ${error.message}`);
    return []; // Return empty array on error
  }
};


// --- CORRECTED & REFACTORED function to get schedule/results by specific date ---
/**
 * Fetches EITHER schedule/odds data (today/future) OR historical results (past dates)
 * for MLB games on a specific date (ET).
 * @param {string} date - The date in YYYY-MM-DD format.
 * @returns {Promise<UnifiedMLBGameData[]>} - A promise resolving to an array of game data objects.
 */
export const getMlbScheduleByDate = async (date) => {
  console.log(`[mlb_service getMlbScheduleByDate] Received date: ${date}`);
  let isPastDate = false;
  try {
    const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
    const inputDateEt = DateTime.fromISO(date, { zone: ET_ZONE_IDENTIFIER });
    if (!inputDateEt.isValid) throw new Error(`Invalid date: ${date}`);
    isPastDate = inputDateEt.startOf('day') < nowEt.startOf('day');
    console.log(`[mlb_service getMlbScheduleByDate] Date ${date}. Is Past: ${isPastDate}`);
  } catch (e) {
    console.error(`[mlb_service getMlbScheduleByDate] Error parsing date: ${date}`, e);
    throw new Error("Invalid date format. Use YYYY-MM-DD.");
  }

  let query; // Will hold the Supabase query promise
  let selectColumns; // Will hold the columns string

  // Build the query based on date
  if (isPastDate) {
    console.log(`[mlb_service] Fetching historical MLB data for ${date}`);
    selectColumns = `game_id, game_date_time_utc, home_team_name, away_team_name, home_score, away_score, status_short`;
    const startOfInputDayET = DateTime.fromISO(date, { zone: ET_ZONE_IDENTIFIER }).startOf("day");
    const startOfNextDayET = startOfInputDayET.plus({ days: 1 });
    const startUTC = startOfInputDayET.toUTC().toISO();
    const endUTC = startOfNextDayET.toUTC().toISO();
    console.log(`[mlb_service] Querying UTC range: ${startUTC} to ${endUTC}`);

    query = supabase
      .from(MLB_HISTORICAL_GAMES_TABLE)
      .select(selectColumns)
      .gte("game_date_time_utc", startUTC)
      .lt("game_date_time_utc", endUTC)
      .order("game_date_time_utc", { ascending: true });

  } else { // Fetch Schedule Data
    console.log(`[mlb_service] Fetching schedule MLB data for ${date}`);
    selectColumns = `
        game_id, scheduled_time_utc, game_date_et, status_detail, status_state,
        home_team_name, away_team_name, home_probable_pitcher_name,
        home_probable_pitcher_handedness, away_probable_pitcher_name,
        away_probable_pitcher_handedness, moneyline_home_clean, moneyline_away_clean,
        spread_home_line_clean, spread_home_price_clean, spread_away_price_clean,
        total_line_clean, total_over_price_clean, total_under_price_clean
    `;
    query = supabase
      .from(MLB_SCHEDULE_TABLE)
      .select(selectColumns)
      .eq("game_date_et", date)
      .order("scheduled_time_utc", { ascending: true });
  }

  // Execute the query and handle results/errors/mapping
  try {
    const { data, error } = await query; // Execute the built query

    console.log(`[mlb_service getMlbScheduleByDate] Supabase returned ${data?.length ?? 0} rows for date ${date}. Error: ${error ? error.message : "No"}`);
    if (error) throw error;
    if (!Array.isArray(data)) {
      console.warn(`[mlb_service] Supabase data not array for ${date}. Data:`, data);
      return [];
    }

    // Map results to unified structure
    const results = data.map((row) => {
      if (isPastDate) {
        /** @type {UnifiedMLBGameData} */
        const gameData = { /* ... map historical fields ... */
            id: String(row.game_id), game_date: date, homeTeamName: row.home_team_name, awayTeamName: row.away_team_name,
            gameTimeUTC: row.game_date_time_utc, statusState: row.status_short ?? "Final", homePitcher: null, awayPitcher: null,
            homePitcherHand: null, awayPitcherHand: null, moneylineHome: null, moneylineAway: null, spreadLine: null, totalLine: null,
            home_final_score: row.home_score, away_final_score: row.away_score, dataType: "historical",
        };
        return gameData;
      } else {
        /** @type {UnifiedMLBGameData} */
        const gameData = { /* ... map schedule fields ... */
            id: String(row.game_id), game_date: row.game_date_et, homeTeamName: row.home_team_name, awayTeamName: row.away_team_name,
            gameTimeUTC: row.scheduled_time_utc, statusState: row.status_state, homePitcher: row.home_probable_pitcher_name,
            awayPitcher: row.away_probable_pitcher_name, homePitcherHand: row.home_probable_pitcher_handedness,
            awayPitcherHand: row.away_probable_pitcher_handedness, moneylineHome: row.moneyline_home_clean,
            moneylineAway: row.moneyline_away_clean, spreadLine: row.spread_home_line_clean, totalLine: row.total_line_clean,
            home_final_score: null, away_final_score: null, dataType: "schedule",
        };
        return gameData;
      }
    });
    return results;

  } catch (err) {
    console.error(`[mlb_service getMlbScheduleByDate] Error processing date ${date}:`, err);
    throw err; // Re-throw error
  }
};