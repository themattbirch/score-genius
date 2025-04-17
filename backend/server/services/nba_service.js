// backend/server/services/nba_service.js
// Import the shared Supabase client instance
import supabase from "../utils/supabase_client.js";
// Import Luxon for robust date/timezone handling
import { DateTime } from "luxon"; // Make sure you did: npm install luxon

// Define constants specific to this service
const NBA_SCHEDULE_TABLE = "nba_game_schedule";
const NBA_INJURIES_TABLE = "nba_injuries";
const NBA_HISTORICAL_GAMES_TABLE = "nba_historical_game_stats";
const ET_ZONE_IDENTIFIER = "America/New_York";

// Replace this function in backend/server/services/nba_service.js
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

export const fetchNbaGamesHistory = async (options) => {
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
    console.error("Error in fetchNbaGamesHistory service:", error);
    throw error;
  }
};

// Add services for predictions, team stats etc. later...
