// backend/server/services/nba_service.js
// Import the shared Supabase client instance
import supabase from "../utils/supabase_client.js";
// Import Luxon for robust date/timezone handling
import { DateTime } from "luxon"; // Make sure you did: npm install luxon

// Define constants specific to this service
const SUPABASE_TABLE_NAME = "nba_game_schedule"; // <--- Use the correct NBA table name
const ET_ZONE_IDENTIFIER = "America/New_York";

export const fetchNbaScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching NBA schedule for today/tomorrow ET...");

  // Get today and tomorrow's date in ET using Luxon
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
      .select("*") // Select all columns for now, adjust as needed for PWA
      // Ensure your NBA table also has a 'game_date_et' column storing 'YYYY-MM-DD'
      .in("game_date_et", [todayStr, tomorrowStr])
      .order("scheduled_time_utc", { ascending: true }); // Order by game time

    if (error) {
      console.error("Supabase error fetching NBA schedule:", error);
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500;
      throw dbError; // Throw it for the controller to catch
    }

    console.log(`Service: Found ${data ? data.length : 0} NBA games.`);
    return data || []; // Return data or empty array if null
  } catch (error) {
    console.error("Error in fetchNbaSchedule service:", error);
    throw error; // Re-throw error for controller
  }
};

// Add services for predictions, team stats etc. later...
