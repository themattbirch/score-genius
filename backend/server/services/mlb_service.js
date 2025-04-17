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

export const fetchMlbGameHistory = async (options) => {
  const { startDate, endDate, teamName, limit, page } = options;
  console.log(`Service: Fetching MLB historical games with options:`, options);

  // Define columns based on your 'mlb_historical_game_stats' table
  // Select core info + scores, hits, errors. Add/remove as needed for PWA.
  const selectColumns = `
    game_id, game_date_time_utc, season, status_short, status_long,
    home_team_id, home_team_name, away_team_id, away_team_name,
    home_score, away_score, home_hits, away_hits, home_errors, away_errors
  `;
  // Note: Innings columns omitted for brevity, add if needed: h_inn_1, a_inn_1, etc.

  try {
    // Start building the Supabase query
    let query = supabase.from(MLB_HISTORICAL_GAMES_TABLE).select(selectColumns);

    // Apply filters based on options provided
    if (startDate) {
      // Filter based on the UTC timestamp column
      query = query.gte("game_date_time_utc", `${startDate}T00:00:00Z`);
    }
    if (endDate) {
      query = query.lte("game_date_time_utc", `${endDate}T23:59:59Z`);
    }
    if (teamName) {
      // Use case-insensitive 'ilike' for partial name matching
      // Ensure team names in Supabase are consistent or adjust filtering logic
      query = query.or(
        `home_team_name.ilike.%${teamName}%,away_team_name.ilike.%${teamName}%`
      );
    }

    // Apply ordering - most recent games first
    query = query.order("game_date_time_utc", { ascending: false });

    // Apply pagination
    const offset = (page - 1) * limit;
    query = query.range(offset, offset + limit - 1);

    // Execute the final query
    const { data, error, status } = await query;

    if (error) {
      console.error("Supabase error fetching MLB historical games:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError;
    }

    console.log(
      `Service: Found ${data ? data.length : 0} MLB historical games.`
    );
    return data || [];
  } catch (error) {
    console.error("Error in fetchMlbGameHistory service:", error);
    throw error; // Re-throw for controller
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

        console.log(`Querying MLB schedule for dates in game_date_et: ${todayStr}, ${tomorrowStr}`);

        // --- Corrected Supabase query logic ---
        const { data, error } = await supabase
            // *** Use the CORRECT table name ***
            .from('mlb_game_schedule')
            // *** Use the CORRECT column names based on your list ***
            .select(`
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
            `)
            // Filter using the correct date column name
            .in('game_date_et', [todayStr, tomorrowStr])
            // Order by the correct time column name
            .order('scheduled_time_utc', { ascending: true });
        // --- End Supabase query logic ---

        if (error) {
            // Log the specific Supabase error
            console.error(`Supabase error fetching MLB schedule from 'mlb_game_schedule':`, error.message);
            // Don't cache errors, return null
            return null;
        }

        // 3. Store the fetched data in cache if successful
        if (data) {
            console.log(`Successfully fetched ${data.length} MLB games from Supabase. Caching result with TTL: ${ttl}s`);
            cache.set(cacheKey, data, ttl);
        } else {
            console.log("No MLB games found for today/tomorrow in Supabase. Caching empty array.");
            cache.set(cacheKey, [], ttl);
        }

        return data || []; // Return data or an empty array

    } catch (error) {
        // Catch any other unexpected errors in the function
        console.error(`Unexpected error in WorkspaceMlbScheduleForTodayAndTomorrow service: ${error.message}`);
        return null; // Return null on unexpected errors
    }
};