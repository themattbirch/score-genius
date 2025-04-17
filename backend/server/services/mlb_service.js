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

export const fetchMlbTeamStatsBySeason = async (teamId, season) => {
  // Assuming 'season' in the Supabase table is an integer (e.g., 2023)
  console.log(
    `Service: Fetching MLB historical team stats for team ${teamId}, season ${season}...`
  );
  try {
    // Select all columns based on user list for mlb_historical_team_stats
    const selectColumns = `
        team_id, team_name, season, league_id, league_name,
        games_played_home, games_played_away, games_played_all,
        wins_home_total, wins_home_percentage, wins_away_total, wins_away_percentage,
        wins_all_total, wins_all_percentage, losses_home_total, losses_home_percentage,
        losses_away_total, losses_away_percentage, losses_all_total, losses_all_percentage,
        runs_for_total_home, runs_for_total_away, runs_for_total_all,
        runs_for_avg_home, runs_for_avg_away, runs_for_avg_all,
        runs_against_total_home, runs_against_total_away, runs_against_total_all,
        runs_against_avg_home, runs_against_avg_away, runs_against_avg_all,
        updated_at
    `; // Exclude raw_api_response by default for API

    const { data, error, status } = await supabase
      .from(MLB_HISTORICAL_TEAM_STATS_TABLE)
      .select(selectColumns)
      .eq("team_id", teamId)
      .eq("season", season) // Query using the integer season year
      .maybeSingle(); // Expect only one row or null

    if (error) {
      console.error(
        "Supabase error fetching MLB historical team stats:",
        error
      );
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError;
    }

    if (data) {
      console.log(
        `Service: Found MLB stats for team ${teamId}, season ${season}.`
      );
    } else {
      console.log(
        `Service: No MLB stats found for team ${teamId}, season ${season}.`
      );
    }
    return data; // Return the single data object or null
  } catch (error) {
    console.error("Error in fetchMlbTeamStatsBySeason service:", error);
    throw error;
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

    const todayStr = getUTCDateString(today);
    const tomorrowStr = getUTCDateString(tomorrow);

    console.log(`Querying MLB schedule for dates: ${todayStr}, ${tomorrowStr}`);

    // --- Your existing Supabase query logic ---
    const { data, error } = await supabase
      .from("mlb_games_view") // *** Replace with your actual table/view name ***
      .select(
        `
                game_id,
                game_date,
                game_datetime_utc,
                status,
                home_team_id,
                home_team_name,
                home_team_score,
                away_team_id,
                away_team_name,
                away_team_score,
                home_probable_pitcher,
                away_probable_pitcher,
                odds_source,
                last_updated_odds,
                home_odds_ml,
                away_odds_ml,
                home_odds_spread,
                away_odds_spread,
                total_over_under
            `
      ) // *** Select the columns needed by the frontend ***
      .in("game_date", [todayStr, tomorrowStr])
      .order("game_datetime_utc", { ascending: true });
    // --- End Supabase query logic ---

    if (error) {
      console.error("Supabase error fetching MLB schedule:", error.message);
      // Don't cache errors, return null or throw
      return null;
    }

    // 3. Store the fetched data in cache if successful
    if (data) {
      console.log(
        `Successfully fetched ${data.length} MLB games. Caching result with TTL: ${ttl}s`
      );
      cache.set(cacheKey, data, ttl);
    } else {
      console.log(
        "No MLB games found for today/tomorrow. Caching empty array."
      );
      cache.set(cacheKey, [], ttl);
    }

    return data || []; // Return data or an empty array
  } catch (error) {
    console.error(
      `Error in WorkspaceMlbScheduleForTodayAndTomorrow service: ${error.message}`
    );
    return null; // Or throw
  }
};