import supabase from "../utils/supabase_client.js"; // <-- IMPORT THE CLIENT
import { DateTime } from "luxon"; // Using Luxon for robust timezone handling

const MLB_SCHEDULE_TABLE = "mlb_game_schedule";
const MLB_HISTORICAL_GAMES_TABLE = "mlb_historical_game_stats";
const MLB_HISTORICAL_TEAM_STATS_TABLE = "mlb_historical_team_stats";
const ET_ZONE_IDENTIFIER = "America/New_York"; // IANA identifier

import cache from "../utils/cache.js"; // Import the cache instance

// --- Helper function for dates (ensure consistent formatting YYYY-MM-DD) ---
// Reuse or define date helpers as needed. Using UTC is recommended.
const getUTCDateString = (date) => {
  return date.toISOString().split("T")[0];
};
// --- End Helper Function ---

export const fetchMlbScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching MLB schedule for today/tomorrow ET...");
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const todayStr = nowEt.toISODate(); // YYYY-MM-DD
  const tomorrowStr = nowEt.plus({ days: 1 }).toISODate(); // YYYY-MM-DD
  console.log(
    `Service: Querying Supabase table '${MLB_SCHEDULE_TABLE}' for dates: ${todayStr}, ${tomorrowStr}`
  ); // Use MLB_SCHEDULE_TABLE const

  try {
    const { data, error, status } = await supabase
      .from(MLB_SCHEDULE_TABLE) // Use constant
      // Select specific columns based on provided list + predictions
      .select(
        `
            game_id,
            scheduled_time_utc,
            game_date_et,
            status_detail,
            status_state,
            home_team_name,
            away_team_name,
            home_probable_pitcher_name,
            home_probable_pitcher_handedness,
            away_probable_pitcher_name,
            away_probable_pitcher_handedness,
            moneyline_home_clean,
            moneyline_away_clean,
            spread_home_line_clean,
            spread_home_price_clean,
            spread_away_price_clean,
            total_line_clean,
            total_over_price_clean,
            total_under_price_clean
          `
      )
      .in("game_date_et", [todayStr, tomorrowStr]) // Filter on correct date column
      .order("scheduled_time_utc", { ascending: true }); // Order by correct time column

    if (error) {
      console.error("Supabase error fetching MLB schedule:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError;
    }
    console.log(`Service: Found ${data ? data.length : 0} MLB games.`);
    return data || [];
  } catch (error) {
    console.error("Error in fetchMlbSchedule service:", error);
    throw error;
  }
};

// --- MLB Historical Game Stats ---
export const fetchMlbGameHistory = async (options) => {
  // Cache Key: Stringify options object for simplicity handles all params
  const cacheKey = `mlb_game_history_${JSON.stringify(options)}`;
  // TTL: 1 day
  const ttl = 86400;

  const cachedData = cache.get(cacheKey);
  if (cachedData !== undefined) {
    console.log(`CACHE HIT for key: ${cacheKey}`);
    return cachedData;
  }

  console.log(
    `CACHE MISS for key: ${cacheKey}. Fetching historical MLB games with options:`,
    options
  );
  try {
    // Select relevant columns from your list, exclude raw_api_response
    const selectColumns = `
            game_id, game_date_time_utc, season, league_id, status_long, status_short,
            home_team_id, home_team_name, away_team_id, away_team_name,
            home_score, away_score, home_hits, away_hits, home_errors, away_errors,
            h_inn_1, h_inn_2, h_inn_3, h_inn_4, h_inn_5, h_inn_6, h_inn_7, h_inn_8, h_inn_9, h_inn_extra,
            a_inn_1, a_inn_2, a_inn_3, a_inn_4, a_inn_5, a_inn_6, a_inn_7, a_inn_8, a_inn_9, a_inn_extra,
            updated_at
        `;
    let query = supabase.from(MLB_HISTORICAL_GAMES_TABLE).select(selectColumns);

    // Apply filters (ensure your DB column names match)
    // Assuming you have a 'game_date' column suitable for filtering
    if (options.startDate)
      query = query.gte("game_date_time_utc", options.startDate); // Adjust column if needed
    if (options.endDate)
      query = query.lte("game_date_time_utc", options.endDate); // Adjust column if needed
    if (options.teamName)
      query = query.or(
        `home_team_name.ilike.%${options.teamName}%,away_team_name.ilike.%${options.teamName}%`
      ); // Adjust columns if needed

    // Order by date - ensure 'game_date_time_utc' is correct for sorting
    query = query.order("game_date_time_utc", { ascending: false });

    const offset = (options.page - 1) * options.limit;
    query = query.range(offset, offset + options.limit - 1);

    const { data, error, status } = await query;

    if (error) {
      console.error(
        "Supabase error fetching MLB historical games:",
        error.message
      );
      return null; // Don't cache errors
    }

    const resultData = data || [];
    console.log(
      `Successfully fetched ${resultData.length} MLB historical games. Caching result with TTL: ${ttl}s`
    );
    cache.set(cacheKey, resultData, ttl);
    return resultData;
  } catch (error) {
    console.error("Error in fetchMlbGameHistory service:", error.message);
    return null;
  }
};

// --- MLB Historical Team Stats ---
export const fetchMlbTeamStatsBySeason = async (teamId, seasonYear) => {
  // Assuming service takes year number
  // Cache Key: Dynamic
  const cacheKey = `mlb_team_stats_${teamId}_${seasonYear}`;
  // TTL: 1 day
  const ttl = 86400;

  const cachedData = cache.get(cacheKey);
  if (cachedData !== undefined) {
    if (cachedData === null) {
      console.log(`CACHE HIT for key: ${cacheKey} (Result: Not Found)`);
      return null;
    }
    console.log(`CACHE HIT for key: ${cacheKey}`);
    return cachedData;
  }

  console.log(
    `CACHE MISS for key: ${cacheKey}. Fetching MLB team stats for team ${teamId}, season ${seasonYear}...`
  );
  try {
    // Select relevant columns, exclude raw_api_response
    const selectColumns = `
            id, team_id, team_name, season, league_id, league_name,
            games_played_home, games_played_away, games_played_all,
            wins_home_total, wins_home_percentage, wins_away_total, wins_away_percentage,
            wins_all_total, wins_all_percentage, losses_home_total, losses_home_percentage,
            losses_away_total, losses_away_percentage, losses_all_total, losses_all_percentage,
            runs_for_total_home, runs_for_total_away, runs_for_total_all,
            runs_for_avg_home, runs_for_avg_away, runs_for_avg_all,
            runs_against_total_home, runs_against_total_away, runs_against_total_all,
            runs_against_avg_home, runs_against_avg_away, runs_against_avg_all,
            updated_at
        `;
    const { data, error, status } = await supabase
      .from(MLB_HISTORICAL_TEAM_STATS_TABLE)
      .select(selectColumns)
      .eq("team_id", teamId)
      // Ensure 'season' column format matches 'seasonYear' param
      .eq("season", seasonYear)
      .maybeSingle(); // Expect only one row (or null)

    if (error) {
      console.error(
        "Supabase error fetching MLB historical team stats:",
        error.message
      );
      return null; // Don't cache errors
    }

    if (data) {
      console.log(
        `Successfully fetched MLB stats for team ${teamId}, season ${seasonYear}. Caching result with TTL: ${ttl}s`
      );
    } else {
      console.log(
        `No MLB stats found for team ${teamId}, season ${seasonYear}. Caching 'null' with TTL: ${ttl}s`
      );
    }
    cache.set(cacheKey, data, ttl); // Cache the actual data or null
    return data;
  } catch (error) {
    console.error("Error in fetchMlbTeamStatsBySeason service:", error.message);
    return null;
  }
};

export const WorkspaceMlbScheduleForTodayAndTomorrow = async () => {
  const cacheKey = "mlb_schedule_today_tomorrow";
  const ttl = 1800; // 30 minutes in seconds

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

    // Assuming your game_date_et column stores dates as YYYY-MM-DD strings
    // Adjust date formatting if necessary to match your Supabase column
    const todayStr = getUTCDateString(today); // Or format as needed e.g., YYYY-MM-DD
    const tomorrowStr = getUTCDateString(tomorrow); // Or format as needed

    console.log(
      `Querying MLB schedule for dates in game_date_et: ${todayStr}, ${tomorrowStr}`
    );

    // --- Corrected Supabase query logic ---
    const { data, error } = await supabase
      // *** Use the CORRECT table name ***
      .from("mlb_game_schedule")
      // *** Use the CORRECT column names based on your list ***
      .select(
        `
                game_id,
                scheduled_time_utc,
                game_date_et,
                status_detail,
                status_state,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_probable_pitcher_name,
                away_probable_pitcher_name,
                home_probable_pitcher_handedness,
                away_probable_pitcher_handedness,
                moneyline_home_clean,
                moneyline_away_clean,
                spread_home_line_clean,
                spread_home_price_clean,
                spread_away_price_clean,
                total_line_clean,
                total_over_price_clean,
                total_under_price_clean,
                updated_at
            `
      )
      // Filter using the correct date column name
      .in("game_date_et", [todayStr, tomorrowStr])
      // Order by the correct time column name
      .order("scheduled_time_utc", { ascending: true });
    // --- End Supabase query logic ---

    if (error) {
      // Log the specific Supabase error
      console.error(
        `Supabase error fetching MLB schedule from 'mlb_game_schedule':`,
        error.message
      );
      // Don't cache errors, return null
      return null;
    }

    // 3. Store the fetched data in cache if successful
    if (data) {
      console.log(
        `Successfully fetched ${data.length} MLB games from Supabase. Caching result with TTL: ${ttl}s`
      );
      cache.set(cacheKey, data, ttl);
    } else {
      console.log(
        "No MLB games found for today/tomorrow in Supabase. Caching empty array."
      );
      cache.set(cacheKey, [], ttl);
    }

    return data || []; // Return data or an empty array
  } catch (error) {
    // Catch any other unexpected errors in the function
    console.error(
      `Unexpected error in WorkspaceMlbScheduleForTodayAndTomorrow service: ${error.message}`
    );
    return null; // Return null on unexpected errors
  }
};
