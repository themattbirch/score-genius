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

import cache from "../utils/cache.js"; // Import the cache instance

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

export const fetchNbaInjuries = async () => {
  console.log(
    `Service: Fetching all current NBA injuries from table '${NBA_INJURIES_TABLE}'...`
  );
  try {
    // Fetch all columns, order by latest report date first
    const { data, error, status } = await supabase
      .from(NBA_INJURIES_TABLE)
      .select("*")
      .order("report_date_utc", { ascending: false }); // Show newest reports first

    if (error) {
      console.error("Supabase error fetching injuries:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500; // Attach status code if available
      throw dbError;
    }

    console.log(`Service: Found ${data ? data.length : 0} NBA injury records.`);
    return data || []; // Return data or empty list
  } catch (error) {
    console.error("Error in fetchNbaInjuries service:", error);
    throw error; // Re-throw for the controller
  }
};

export const fetchNbaGameHistory = async (options) => {
  const { startDate, endDate, teamName, limit, page } = options;
  console.log(`Service: Fetching NBA historical games with options:`, options);

  // Define columns to select (use actual names from your list)
  const selectColumns = `
    game_id, game_date, home_team, away_team, home_score, away_score,
    home_q1, home_q2, home_q3, home_q4, home_ot,
    away_q1, away_q2, away_q3, away_q4, away_ot,
    home_assists, home_steals, home_blocks, home_turnovers, home_fouls, home_total_reb,
    away_assists, away_steals, away_blocks, away_turnovers, away_fouls, away_total_reb
  `; // Add/remove columns as needed by frontend

  try {
    let query = supabase.from(NBA_HISTORICAL_GAMES_TABLE).select(selectColumns);

    // Apply filters
    if (startDate) {
      query = query.gte("game_date", startDate);
    }
    if (endDate) {
      query = query.lte("game_date", endDate);
    }
    if (teamName) {
      // Case-insensitive search for team name in either home or away column
      // Note: Supabase ILIKE might be less efficient than direct equals if case is known
      query = query.or(
        `home_team.ilike.%${teamName}%,away_team.ilike.%${teamName}%`
      );
      // Or for exact match (if normalize_team_name is consistent):
      // const normalizedName = normalizeTeamName(teamName); // Ensure you have this helper or pass normalized
      // query = query.or(`home_team.eq.${normalizedName},away_team.eq.${normalizedName}`);
    }

    // Apply ordering
    query = query.order("game_date", { ascending: false }); // Show most recent first

    // Apply pagination
    const offset = (page - 1) * limit;
    query = query.range(offset, offset + limit - 1);

    // Execute query
    const { data, error, status } = await query;

    if (error) {
      console.error("Supabase error fetching historical games:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError;
    }

    console.log(
      `Service: Found ${data ? data.length : 0} NBA historical games.`
    );
    return data || [];
  } catch (error) {
    console.error("Error in fetchNbaGameHistory service:", error);
    throw error;
  }
};

export const fetchNbaTeamStatsBySeason = async (teamId, seasonYearStr) => {
  // Param is starting year string like "2022"
  // Construct the season range string (e.g., "2022-2023") expected in DB
  const startYear = parseInt(seasonYearStr);
  const endYear = startYear + 1;
  const seasonRangeStr = `${startYear}-${endYear}`; // e.g., "2022-2023"

  console.log(
    `Service: Fetching NBA historical team stats for team ${teamId}, season range '${seasonRangeStr}'...`
  );
  try {
    const { data, error, status } = await supabase
      .from(NBA_HISTORICAL_TEAM_STATS_TABLE) // Use constant defined above
      .select("*") // Select all stats columns for this team/season
      .eq("team_id", teamId)
      // --- CORRECTED FILTER ---
      .eq("season", seasonRangeStr) // Use the constructed string range for the query
      // --- END CORRECTION ---
      .maybeSingle(); // Expect only one row (or null)

    if (error) {
      console.error("Supabase error fetching historical team stats:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError;
    }

    if (data) {
      console.log(
        `Service: Found stats for team ${teamId}, season ${seasonRangeStr}.`
      );
    } else {
      console.log(
        `Service: No stats found for team ${teamId}, season ${seasonRangeStr}.`
      );
    }
    return data; // Return the single data object or null
  } catch (error) {
    console.error("Error in fetchNbaTeamStatsBySeason service:", error);
    throw error; // Re-throw
  }
};

export const fetchNbaPlayerGameHistory = async (playerId, options) => {
  const { limit, page } = options;
  console.log(
    `Service: Fetching NBA historical game log for player ${playerId}, limit=${limit}, page=${page}...`
  );

  // Define columns based on your nba_historical_player_stats list
  const selectColumns = `
    game_id, player_id, player_name, team_id, team_name, game_date, minutes,
    points, rebounds, assists, steals, blocks, turnovers, fouls,
    fg_made, fg_attempted, three_made, three_attempted, ft_made, ft_attempted
  `; // Select the stats needed for a game log display

  try {
    let query = supabase
      .from(NBA_HISTORICAL_PLAYER_STATS_TABLE)
      .select(selectColumns)
      .eq("player_id", playerId); // Filter by player_id (pass string ID from controller)

    // Apply ordering - most recent games first using 'game_date'
    query = query.order("game_date", { ascending: false });

    // Apply pagination
    const offset = (page - 1) * limit;
    query = query.range(offset, offset + limit - 1);

    // Execute query
    const { data, error, status } = await query;

    if (error) {
      console.error("Supabase error fetching NBA player game log:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError;
    }

    console.log(
      `Service: Found ${
        data ? data.length : 0
      } NBA historical games for player ${playerId}.`
    );
    return data || [];
  } catch (error) {
    console.error("Error in fetchNbaPlayerGameHistory service:", error);
    throw error;
  }
};

export const WorkspaceNbaScheduleForTodayAndTomorrow = async () => {
  const cacheKey = "nba_schedule_today_tomorrow";
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
