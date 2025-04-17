// backend/server/services/nba_service.js
// Import the shared Supabase client instance
import supabase from "../utils/supabase_client.js";
// Import Luxon for robust date/timezone handling
import { DateTime } from "luxon"; // Make sure you did: npm install luxon

// Define constants specific to this service
const NBA_SCHEDULE_TABLE = "nba_game_schedule";
const NBA_INJURIES_TABLE = "nba_injuries";
const SUPABASE_TABLE_NAME = "nba_game_schedule"; // <--- Use the correct NBA table name
const ET_ZONE_IDENTIFIER = "America/New_York";

export const fetchNbaScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching NBA schedule for today/tomorrow ET...");
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const todayStr = nowEt.toISODate(); // Format: YYYY-MM-DD
  const tomorrowStr = nowEt.plus({ days: 1 }).toISODate(); // Format: YYYY-MM-DD
  console.log(
    `Service: Querying Supabase table '${SUPABASE_TABLE_NAME}' for dates: ${todayStr}, ${tomorrowStr}`
  );

  try {
    // Query Supabase nba_game_schedule table
    const { data, error, status } = await supabase
      .from(SUPABASE_TABLE_NAME)
      .select("*")
      // --- CORRECTED COLUMN NAME ---
      .in("game_date", [todayStr, tomorrowStr]) // Use 'game_date' to match table column
      // --- END CORRECTION ---
      .order("scheduled_time_utc", { ascending: true }); // Assumes scheduled_time_utc exists

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

// Add services for predictions, team stats etc. later...
