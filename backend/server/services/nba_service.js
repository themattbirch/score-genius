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
const NBA_SNAPSHOT_TABLE = "nba_snapshots";

// --- Helper function for dates (ensure consistent formatting YYYY-MM-DD) ---
const getUTCDateString = (date) => date.toISOString().split("T")[0];
// --- End Helper Function ---

// --- Unified Data Structure ---
// --- JSDoc Definition for the data structure returned by getSchedule ---
/**
 * Represents unified NBA game data. Matches frontend UnifiedGame type.
 * @typedef {object} UnifiedNBAGameData
 * @property {string} id
 * @property {string} game_date
 * @property {string} homeTeamName // Standardized name
 * @property {string} awayTeamName // Standardized name
 * @property {string | null} [gameTimeUTC] // Standardized name (from scheduled_time)
 * @property {string | null} [statusState] // Add if available
 * @property {number | null} [spreadLine] // Standardized name (from spread_clean)
 * @property {number | null} [totalLine] // Standardized name (from total_clean)
 * @property {number | null} [predictionHome]
 * @property {number | null} [predictionAway]
 * @property {number | null} [home_final_score]
 * @property {number | null} [away_final_score]
 * @property {'schedule' | 'historical'} dataType
 */

// Fetch current injuries, caching for 30m
// backend/server/services/nba_service.js

// Fetch current injuries (no cache for now)
export const fetchNbaInjuries = async (date) => {
  // The 'date' parameter is no longer used
  console.log("→ [fetchNbaInjuries] querying ALL active injuries…");

  const { data, error } = await supabase
    .from(NBA_INJURIES_TABLE)
    .select(
      `
      injury_id,
      player_display_name,
      team_display_name,
      report_date_utc,
      injury_status,
      injury_type,
      injury_detail
      `
    )
    .order("report_date_utc", { ascending: false });

  if (error) {
    console.error("→ [fetchNbaInjuries] Supabase error:", error);
    return [];
  }

  console.log(`→ [fetchNbaInjuries] got ${data.length} total rows`);

  const normalized = data.map((inj) => ({
    id: String(inj.injury_id),
    player: inj.player_display_name,
    team_display_name: inj.team_display_name,
    status: inj.injury_status || "N/A",
    detail: inj.injury_detail || "",
    updated: inj.report_date_utc,
    injury_type: inj.injury_type || null,
  }));

  return normalized;
};
// Fetch historical games w/ pagination & filters
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

/* ---------------------------------------------------------
 *  ALL-TEAMS season stats (used by Stats screen)
 *  GET /team-stats?season=YYYY
 * --------------------------------------------------------*/
export const fetchNbaAllTeamStatsBySeason = async (seasonYear) => {
  const seasonRange = `${seasonYear}-${seasonYear + 1}`;
  const cacheKey = `nba_all_team_stats_${seasonRange}`;
  const ttl = 60 * 30; // 30 min

  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying all-team stats…`);
  const { data, error } = await supabase
    .from(NBA_HISTORICAL_TEAM_STATS_TABLE)
    .select(
      "team_id, team_name, season, games_played_home, games_played_away, games_played_all, wins_home_percentage, wins_away_percentage, wins_all_percentage, points_for_avg_home, points_for_avg_away, points_for_avg_all, points_against_avg_home, points_against_avg_away, points_against_avg_all, current_form"
    )
    .eq("season", seasonRange)
    // Most UIs like to see best teams first
    .order("team_name", { ascending: true });

  if (error) {
    console.error("Supabase error fetching all-team stats:", {
      message: error.message,
      hint: error.hint,
      details: error.details,
      code: error.code,
    });

    // Still throw a clean error up to the controller
    const dbError = new Error(`Supabase query failed: ${error.message}`);
    dbError.status = 500;
    throw dbError;
  }
  cache.set(cacheKey, data || [], ttl);
  return data || [];
};

// Fetch player game history
/**
 * Fetches a player’s game history for a given season (July→June),
 * optionally filtering out games below a minutes threshold.
 *
 * @param {string|number} playerId
 * @param {number} seasonYear  // e.g. 2024 for the 2024–25 season
 * @param {object} opts
 * @param {number} [opts.limit=20]
 * @param {number} [opts.page=1]
 */
export const fetchNbaPlayerGameHistory = async (
  playerId,
  seasonYear,
  { limit = 20, page = 1 } = {}
) => {
  const cacheKey = `nba_player_history_${playerId}_${seasonYear}_${limit}_${page}`;
  const ttl = 86400;
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return cached;
  }

  console.log(`CACHE MISS: ${cacheKey}. Querying player history...`);

  // Build the date-window filters for July 1 of seasonYear → July 1 of next year
  const seasonStart = `${seasonYear}-07-01`;
  const seasonEnd = `${seasonYear + 1}-07-01`;

  // Build your Supabase query
  let query = supabase
    .from("nba_historical_player_stats")
    .select(
      `
      player_id,
      player_name,
      team_name,
      game_date,
      minutes,
      points,
      rebounds,
      assists,
      steals,
      blocks,
      turnovers,
      fouls,
      fg_made,
      fg_attempted,
      three_made,
      three_attempted,
      ft_made,
      ft_attempted
    `
    )
    // filter to only this season’s games
    .gte("game_date", seasonStart)
    .lt("game_date", seasonEnd)
    // filter out sub-minMp games
    .gte("minutes")
    // pagination
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

/**
 * Fetches EITHER schedule/prediction data (today/future) OR
 * historical results (past dates) for NBA games on a specific date (ET).
 * @param {string} date - The date in YYYY-MM-DD format.
 * @returns {Promise<UnifiedNBAGameData[]>} - A promise resolving to an array of game data objects.
 */
/**
 * Fetches EITHER schedule/prediction data (today/future) OR
 * historical results (past dates) for NBA games on a specific date (ET).
 * @param {string} date - The date in YYYY-MM-DD format.
 * @returns {Promise<UnifiedNBAGameData[]>} - A promise resolving to an array of game data objects.
 */
export async function getSchedule(date) {
  console.log("[nba_service getSchedule] Received date:", date);

  /* ---------- 1. Past-or-future check -------------------- */
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const inputEt = DateTime.fromISO(date, { zone: ET_ZONE_IDENTIFIER });
  if (!inputEt.isValid) {
    const bad = new Error(`Invalid date: ${date}`);
    bad.status = 400;
    throw bad;
  }
  const isPastDate = inputEt.startOf("day") < nowEt.startOf("day");

  /* ---------- 2. Pick table & columns ------------------- */
  const TABLE = isPastDate ? NBA_HISTORICAL_GAMES_TABLE : NBA_SCHEDULE_TABLE;

  const columns = isPastDate
    ? `game_id, game_date, home_team, away_team, home_score, away_score`
    : `game_id, game_date, home_team, away_team, scheduled_time,
       spread_clean, total_clean,
       predicted_home_score, predicted_away_score`;

  /* ---------- 3. Supabase query ------------------------- */
  const { data, error, status } = await supabase
    .from(TABLE)
    .select(columns)
    .eq("game_date", date)
    .order(isPastDate ? "game_date" : "scheduled_time", { ascending: true });

  if (error) {
    const dbErr = new Error(
      error.message || "Supabase query failed (getSchedule)"
    );
    dbErr.status = status || 503; // ← key line
    throw dbErr;
  }

  /* ---------- 4. Map rows to unified structure ---------- */
  if (!Array.isArray(data)) {
    console.warn("[nba_service] Supabase returned non-array:", data);
    return [];
  }

  return data.map((row) => {
    if (isPastDate) {
      return {
        id: String(row.game_id),
        game_date: row.game_date,
        homeTeamName: row.home_team,
        awayTeamName: row.away_team,
        gameTimeUTC: null,
        statusState: "Final",
        spreadLine: null,
        totalLine: null,
        predictionHome: null,
        predictionAway: null,
        home_final_score: row.home_score,
        away_final_score: row.away_score,
        dataType: "historical",
      };
    } else {
      return {
        id: String(row.game_id),
        game_date: row.game_date,
        homeTeamName: row.home_team,
        awayTeamName: row.away_team,
        gameTimeUTC: row.scheduled_time,
        statusState: "Scheduled",
        spreadLine: row.spread_clean
          ? parseFloat(
              (String(row.spread_clean).match(/-?\d+(\.\d+)?/) || ["0"])[0]
            )
          : null,
        totalLine: row.total_clean
          ? parseFloat(
              (String(row.total_clean).match(/\d+(\.\d+)?/) || ["0"])[0]
            )
          : null,
        predictionHome: row.predicted_home_score,
        predictionAway: row.predicted_away_score,
        home_final_score: null,
        away_final_score: null,
        dataType: "schedule",
      };
    }
  });
}
/* ---------------------------------------------------------
 * ALL-PLAYERS season stats
 * GET /player-stats?season=YYYY
 * --------------------------------------------------------*/
export const fetchNbaAllPlayerStatsBySeason = async (
  seasonYear,
  { search = null } // Signature should only accept { search }
) => {
  const cacheKey = `nba_all_player_stats_agg_${seasonYear}_${search || "null"}`;
  const ttl = 1800;
  // ... cache check ...

  console.log(
    `CACHE MISS (Aggregated): ${cacheKey}. Calling RPC (no minMp)...`
  );

  // rpcParams should ONLY contain p_season_year and p_search
  const rpcParams = {
    p_season_year: seasonYear,
    p_search: search,
  };

  const { data, error } = await supabase.rpc(
    "get_nba_player_season_stats",
    rpcParams
  );

  // Handle potential errors from the RPC call
  if (error) {
    console.error("Supabase RPC error fetching aggregated player stats:", {
      message: error.message,
      details: error.details, // Might contain more SQL-specific info
      hint: error.hint,
      code: error.code, // PostgreSQL error code
    });
    // Throw a generic error to be handled by the controller
    throw new Error(
      `Database function 'get_nba_player_season_stats' failed: ${error.message}`
    );
  }

  // The RPC function is designed to return the array of aggregated player stats.
  // If 'data' is null or not an array (unexpected), return an empty array.
  const results = Array.isArray(data) ? data : [];

  console.log(
    `Workspaceed ${results.length} aggregated player season stats via RPC. Caching for ${ttl}s.`
  );
  cache.set(cacheKey, results, ttl); // Cache the results
  return results; // Return the aggregated data
};
/**
 * Fetches calculated advanced team stats (Pace, Ratings, etc.) for a given NBA season using RPC.
 * @param {number} seasonYear - The starting year of the season (e.g., 2023 for 2023-24)
 * @returns {Promise<object[]>} - Array of team objects with advanced stats.
 */
export const fetchNbaAdvancedStatsBySeason = async (seasonYear) => {
  const cacheKey = `nba_advanced_stats_${seasonYear}`;
  // Advanced stats based on full season data, cache for a decent time (e.g., 6 hours)
  const ttl = 60 * 60 * 6; // 6 hours in seconds

  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    console.log(`CACHE HIT: ${cacheKey}`);
    return Array.isArray(cached) ? cached : [];
  }

  console.log(
    `CACHE MISS: ${cacheKey}. Calling RPC get_nba_advanced_team_stats...`
  );

  // Prepare parameters for the RPC call
  const rpcParams = {
    p_season_year: seasonYear,
  };

  // Call the RPC function
  const { data, error } = await supabase.rpc(
    "get_nba_advanced_team_stats", // Name of the function in Supabase
    rpcParams
  );

  if (error) {
    console.error("Supabase RPC error fetching advanced team stats:", {
      message: error.message,
      details: error.details,
      hint: error.hint,
      code: error.code,
    });
    // Throw error for the controller to catch
    throw new Error(
      `Database function 'get_nba_advanced_team_stats' failed: ${error.message}`
    );
  }

  const results = Array.isArray(data) ? data : [];

  console.log(
    `Workspaceed ${results.length} teams with advanced stats via RPC. Caching for ${ttl}s.`
  );
  cache.set(cacheKey, results, ttl); // Cache the results
  return results;
};

/**
 * Fetches multiple NBA snapshots by game_id from the Supabase table.
 * Includes caching for already generated snapshots.
 *
 * @param {string[]} gameIds - An array of NBA game_ids.
 * @returns {Promise<object[]>} - A promise resolving to an array of snapshot data objects.
 */
export async function fetchNbaSnapshotsByIds(gameIds) {
  if (!gameIds || gameIds.length === 0) {
    return [];
  }

  const fetchedSnapshots = [];
  const unfetchedGameIds = [];

  // Check cache for each gameId first
  for (const gameId of gameIds) {
    if (cache.has(gameId)) {
      fetchedSnapshots.push(cache.get(gameId));
    } else {
      unfetchedGameIds.push(gameId);
    }
  }

  if (unfetchedGameIds.length > 0) {
    console.log(
      `Service: Fetching ${
        unfetchedGameIds.length
      } NBA snapshots from DB for IDs: ${unfetchedGameIds.join(", ")}`
    );

    // Fetch from Supabase. Select all columns Python upserts.
    // Ensure column names here match what Python upserts (headline_stats, bar_chart_data etc.)
    // And don't use the old aliases (bar_data, radar_data, pie_data)
    const { data, error, status } = await supabase
      .from(NBA_SNAPSHOT_TABLE)
      .select(
        `
        game_id,
        game_date,
        season,
        headline_stats,
        bar_chart_data,
        radar_chart_data,
        pie_chart_data,
        last_updated
        `
      )
      .in("game_id", unfetchedGameIds);

    if (error) {
      console.error(
        `Service: Supabase error fetching NBA snapshots by IDs: ${error.message}`
      );
      const dbErr = new Error(
        error.message || "Failed fetching NBA snapshot data by IDs"
      );
      dbErr.status = status || 503;
      throw dbErr;
    }

    if (data && data.length > 0) {
      for (const snapshot of data) {
        // Cache newly fetched snapshots
        cache.set(snapshot.game_id, snapshot);

        // Perform runtime shape-check for the fetched data
        function assertArray(col, name) {
          if (!Array.isArray(snapshot[col])) {
            console.warn(
              `Snapshot for game_id ${
                snapshot.game_id
              }: ${name} expected as Array, got ${typeof snapshot[col]}.`
            );
            // Optionally throw or replace with empty array if critical
            snapshot[col] = [];
          }
        }
        assertArray("headline_stats", "headline_stats");
        assertArray("bar_chart_data", "bar_chart_data");
        assertArray("radar_chart_data", "radar_chart_data");
        assertArray("pie_chart_data", "pie_chart_data");

        fetchedSnapshots.push(snapshot);
      }
    } else {
      console.log(
        `Service: No snapshots found in DB for IDs: ${unfetchedGameIds.join(
          ", "
        )}`
      );
    }
  }

  return fetchedSnapshots;
}

/**
 * Fetches a single NBA snapshot by game_id from the Supabase table.
 * @param {string} gameId - The NBA game_id.
 * @returns {Promise<object>} - A promise resolving to a single snapshot data object.
 */
export async function fetchNbaSnapshotData(gameId) {
  if (cache.has(gameId)) {
    return cache.get(gameId);
  }
  // 1) fetch + alias columns so the JS property names match your Python payload
  const { data, error, status } = await supabase
    .from(NBA_SNAPSHOT_TABLE)
    .select(
      `
      game_id,
      game_date,
      season,
      headline_stats,
      bar_chart_data,
      key_metrics_data,
      radar_chart_data,
      pie_chart_data,
      last_updated
    `
    )
    .eq("game_id", gameId)
    .maybeSingle();

  // 2) supabase error → 5xx
  if (error) {
    const err = new Error(error.message || "Failed fetching NBA snapshot data");
    err.status = status || 503;
    throw err;
  }

  // 3) no row → 404
  if (!data) {
    const err = new Error(`Snapshot for game ${gameId} not found`);
    err.status = 404;
    throw err;
  }

  // 4) runtime shape‐check
  function assertArray(col, name) {
    if (col === undefined) return; // Allow properties to be missing from the select
    if (!Array.isArray(col)) {
      const err = new Error(`${name} expected as Array, got ${typeof col}`);
      err.status = 500;
      throw err;
    }
  }
  assertArray(data.headline_stats, "headline_stats");
  assertArray(data.bar_chart_data, "bar_chart_data");
  assertArray(data.key_metrics_data, "key_metrics_data"); // Add check for new property
  assertArray(data.radar_chart_data, "radar_chart_data");
  assertArray(data.pie_chart_data, "pie_chart_data");

  cache.set(gameId, data);
  return data;
}
