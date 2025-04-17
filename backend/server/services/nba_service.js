// backend/server/services/nba_service.js
// Import the shared Supabase client instance
import supabase from "../utils/supabase_client.js";
// Import Luxon for robust date/timezone handling
import { DateTime } from "luxon"; // Make sure you did: npm install luxon

// Define constants specific to this service
const NBA_SCHEDULE_TABLE = "nba_game_schedule";
const NBA_INJURIES_TABLE = "nba_injuries";
const NBA_HISTORICAL_GAMES_TABLE = "nba_historical_game_stats";
const NBA_HISTORICAL_TEAM_STATS_TABLE = "nba_historical_team_stats";
const NBA_HISTORICAL_PLAYER_STATS_TABLE = "nba_historical_player_stats";
const ET_ZONE_IDENTIFIER = "America/New_York";

import cache from "../utils/cache.js";

// --- Helper function for dates (ensure consistent formatting YYYY-MM-DD) ---
// You might already have date helpers, use those if available.
// Using UTC dates is generally recommended to avoid timezone issues.
const getUTCDateString = (date) => {
  return date.toISOString().split("T")[0];
};
// --- End Helper Function ---

export const fetchNbaScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching NBA schedule for today/tomorrow ET...");
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const todayStr = nowEt.toISODate(); // Format: YYYY-MM-DD
  const tomorrowStr = nowEt.plus({ days: 1 }).toISODate(); // Format: YYYY-MM-DD
  console.log(
    `Service: Querying Supabase table '${NBA_SCHEDULE_TABLE}' for dates: ${todayStr}, ${tomorrowStr}`
  );

  try {
    // Select specific columns based on provided list + predictions
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
      ) // Select desired columns
      .in("game_date", [todayStr, tomorrowStr]) // Filter on correct date column
      .order("scheduled_time", { ascending: true }); // Order by correct time column

    if (error) {
      console.error("Supabase error fetching NBA schedule:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError;
    }
    console.log(`Service: Found ${data ? data.length : 0} NBA games.`);
    return data || [];
  } catch (error) {
    console.error("Error in fetchNbaSchedule service:", error);
    throw error; // Re-throw for controller
  }
};

// --- NBA Injuries ---
export const fetchNbaInjuries = async () => {
    // Cache Key: Simple, as it likely fetches all current injuries
    const cacheKey = 'nba_injuries_current';
    // TTL: 1 hour (adjust as needed based on update frequency)
    const ttl = 3600;

    const cachedData = cache.get(cacheKey);
    if (cachedData !== undefined) {
        console.log(`CACHE HIT for key: ${cacheKey}`);
        return cachedData;
    }

    console.log(`CACHE MISS for key: ${cacheKey}. Fetching from Supabase table '${NBA_INJURIES_TABLE}'...`);
    try {
        // Select relevant columns, exclude raw_api_response
        const selectColumns = `
            injury_id, player_id, player_display_name, team_id, team_display_name,
            report_date_utc, injury_status, injury_status_abbr, injury_type,
            injury_location, injury_detail, injury_side, return_date_est,
            short_comment, long_comment, created_at, last_api_update_time
        `;
        const { data, error, status } = await supabase
            .from(NBA_INJURIES_TABLE)
            .select(selectColumns)
            .order("report_date_utc", { ascending: false }); // Show newest reports first

        if (error) {
            console.error("Supabase error fetching injuries:", error.message);
            return null; // Don't cache errors
        }

        const resultData = data || [];
        console.log(`Successfully fetched ${resultData.length} NBA injury records. Caching result with TTL: ${ttl}s`);
        cache.set(cacheKey, resultData, ttl);
        return resultData;

    } catch (error) {
        console.error("Error in fetchNbaInjuries service:", error.message);
        return null; // Return null on unexpected errors
    }
};

// --- NBA Historical Game Stats ---
export const fetchNbaGameHistory = async (options) => {
    const { startDate, endDate, teamName, limit, page } = options;
    // Cache Key: Dynamic, includes all filter/pagination options
    // Use 'null' strings for potentially undefined values to ensure key consistency
    const cacheKey = `nba_game_history_${startDate || 'null'}_${endDate || 'null'}_${teamName || 'null'}_${limit}_${page}`;
    // TTL: 1 day (historical data)
    const ttl = 86400;

    const cachedData = cache.get(cacheKey);
    if (cachedData !== undefined) {
        console.log(`CACHE HIT for key: ${cacheKey}`);
        return cachedData;
    }

    console.log(`CACHE MISS for key: ${cacheKey}. Fetching historical NBA games with options:`, options);
    try {
        // Use the select list defined in your controller logic
        const selectColumns = `
            game_id, game_date, home_team, away_team, home_score, away_score,
            home_q1, home_q2, home_q3, home_q4, home_ot,
            away_q1, away_q2, away_q3, away_q4, away_ot,
            home_assists, home_steals, home_blocks, home_turnovers, home_fouls, home_total_reb,
            away_assists, away_steals, away_blocks, away_turnovers, away_fouls, away_total_reb
        `;
        let query = supabase.from(NBA_HISTORICAL_GAMES_TABLE).select(selectColumns);

        // Apply filters (ensure your DB column names match, e.g., 'game_date')
        if (startDate) query = query.gte("game_date", startDate);
        if (endDate) query = query.lte("game_date", endDate);
        if (teamName) query = query.or(`home_team.ilike.%${teamName}%,away_team.ilike.%${teamName}%`);

        query = query.order("game_date", { ascending: false });

        const offset = (page - 1) * limit;
        query = query.range(offset, offset + limit - 1);

        const { data, error, status } = await query;

        if (error) {
            console.error("Supabase error fetching historical games:", error.message);
            return null; // Don't cache errors
        }

        const resultData = data || [];
        console.log(`Successfully fetched ${resultData.length} NBA historical games. Caching result with TTL: ${ttl}s`);
        cache.set(cacheKey, resultData, ttl);
        return resultData;

    } catch (error) {
        console.error("Error in fetchNbaGameHistory service:", error.message);
        return null;
    }
};

// --- NBA Historical Team Stats ---
export const fetchNbaTeamStatsBySeason = async (teamId, seasonYear) => { // Assuming service takes year, not string range
    // Construct season string if needed by DB, otherwise use seasonYear directly
    // Example: const seasonRangeStr = `${seasonYear}-${seasonYear + 1}`;
    // Cache Key: Dynamic
    const cacheKey = `nba_team_stats_${teamId}_${seasonYear}`;
    // TTL: 1 day
    const ttl = 86400;

    const cachedData = cache.get(cacheKey);
    if (cachedData !== undefined) {
        // Special check: If null was explicitly cached, return null
        if (cachedData === null) {
             console.log(`CACHE HIT for key: ${cacheKey} (Result: Not Found)`);
             return null;
        }
        console.log(`CACHE HIT for key: ${cacheKey}`);
        return cachedData;
    }

    console.log(`CACHE MISS for key: ${cacheKey}. Fetching NBA team stats for team ${teamId}, season ${seasonYear}...`);
    try {
        // Select all columns, assuming they are all needed for team stats page
        // If table grows, specify columns
        const { data, error, status } = await supabase
            .from(NBA_HISTORICAL_TEAM_STATS_TABLE)
            .select('*')
            .eq("team_id", teamId)
            // Filter by season - ensure 'season' column format matches 'seasonYear' param
            .eq("season", seasonYear) // Or use seasonRangeStr if needed
            .maybeSingle(); // Expect only one row (or null)

        if (error) {
            console.error("Supabase error fetching historical team stats:", error.message);
            // Cache null to prevent retrying a known error state for the TTL duration? Optional.
            // cache.set(cacheKey, null, ttl); // Consider implications
            return null; // Don't cache errors by default
        }

        // Cache the result, whether it's data or null (meaning not found)
        if (data) {
            console.log(`Successfully fetched stats for team ${teamId}, season ${seasonYear}. Caching result with TTL: ${ttl}s`);
        } else {
            console.log(`No stats found for team ${teamId}, season ${seasonYear}. Caching 'null' with TTL: ${ttl}s`);
        }
        cache.set(cacheKey, data, ttl); // Cache the actual data or null
        return data;

    } catch (error) {
        console.error("Error in fetchNbaTeamStatsBySeason service:", error.message);
        return null;
    }
};

// --- NBA Historical Player Stats ---
export const fetchNbaPlayerGameHistory = async (playerId, options) => {
    const { limit, page } = options;
    // Cache Key: Dynamic
    const cacheKey = `nba_player_history_${playerId}_${limit}_${page}`;
    // TTL: 1 day
    const ttl = 86400;

    const cachedData = cache.get(cacheKey);
    if (cachedData !== undefined) {
        console.log(`CACHE HIT for key: ${cacheKey}`);
        return cachedData;
    }

    console.log(`CACHE MISS for key: ${cacheKey}. Fetching NBA player game log for player ${playerId}, limit=${limit}, page=${page}...`);
    try {
        // Use select columns specified previously
        const selectColumns = `
            game_id, player_id, player_name, team_id, team_name, game_date, minutes,
            points, rebounds, assists, steals, blocks, turnovers, fouls,
            fg_made, fg_attempted, three_made, three_attempted, ft_made, ft_attempted
        `;
        let query = supabase
            .from(NBA_HISTORICAL_PLAYER_STATS_TABLE)
            .select(selectColumns)
            .eq("player_id", playerId);

        query = query.order("game_date", { ascending: false });

        const offset = (page - 1) * limit;
        query = query.range(offset, offset + limit - 1);

        const { data, error, status } = await query;

        if (error) {
            console.error("Supabase error fetching NBA player game log:", error.message);
            return null; // Don't cache errors
        }

        const resultData = data || [];
        console.log(`Successfully fetched ${resultData.length} NBA historical games for player ${playerId}. Caching result with TTL: ${ttl}s`);
        cache.set(cacheKey, resultData, ttl);
        return resultData;

    } catch (error) {
        console.error("Error in fetchNbaPlayerGameHistory service:", error.message);
        return null;
    }
};

export const WorkspaceNbaScheduleForTodayAndTomorrow = async () => {
  const cacheKey = "nba_schedule_today_tomorrow";
  const ttl = 3600; // 60 minutes in seconds

  // 1. Check cache first
  const cachedData = cache.get(cacheKey);
  if (cachedData !== undefined) {
    console.log(`CACHE HIT for key: ${cacheKey}`);
    return cachedData;
  }

  console.log(`CACHE MISS for key: ${cacheKey}. Fetching from Supabase...`);

  // 2. If cache miss, query Supabase
  try {
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    // Get date strings in YYYY-MM-DD format
    const todayStr = getUTCDateString(today);
    const tomorrowStr = getUTCDateString(tomorrow);

    // *** Use the correct date column name from your list ***
    console.log(
      `Querying NBA schedule for dates in game_date: ${todayStr}, ${tomorrowStr}`
    );

    // --- Corrected Supabase query logic for NBA ---
    const { data, error } = await supabase
      // *** Use the CORRECT NBA table name ***
      .from("nba_game_schedule")
      // *** Use the CORRECT NBA column names based on your list ***
      // Selecting a useful subset including predictions and cleaned odds
      .select(
        `
                game_id,
                game_date,
                home_team,
                away_team,
                scheduled_time,
                venue,
                status,
                moneyline_clean,
                spread_clean,
                total_clean,
                predicted_home_score,
                predicted_away_score,
                updated_at
            `
      )
      // *** Filter using the correct date column name ***
      .in("game_date", [todayStr, tomorrowStr])
      // *** Order by the correct time column name ***
      .order("scheduled_time", { ascending: true });
    // --- End Supabase query logic ---

    if (error) {
      console.error(
        `Supabase error fetching NBA schedule from 'nba_game_schedule':`,
        error.message
      );
      return null; // Don't cache errors
    }

    // 3. Store the fetched data in cache if successful
    if (data) {
      console.log(
        `Successfully fetched ${data.length} NBA games from Supabase. Caching result with TTL: ${ttl}s`
      );
      cache.set(cacheKey, data, ttl);
    } else {
      console.log(
        "No NBA games found for today/tomorrow in Supabase. Caching empty array."
      );
      cache.set(cacheKey, [], ttl);
    }

    return data || []; // Return data or an empty array
  } catch (error) {
    console.error(
      `Unexpected error in WorkspaceNbaScheduleForTodayAndTomorrow service: ${error.message}`
    );
    return null; // Return null on unexpected errors
  }
};
