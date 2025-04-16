// backend/server/services/mlb_service.js
// Ensure path to supabase client utility is correct
import supabase from '../utils/supabase_client.js';
// Luxon helps with date/time/zone handling
import { DateTime } from 'luxon';

// Ensure table name matches your Supabase table
const SUPABASE_TABLE_NAME = "mlb_game_schedule";
const ET_ZONE_IDENTIFIER = "America/New_York"; // IANA identifier

export const fetchMlbScheduleForTodayAndTomorrow = async () => {
  console.log("Service: Fetching MLB schedule for today/tomorrow ET...");

  // Get today and tomorrow's date in ET using Luxon
  const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
  const todayStr = nowEt.toISODate(); // Format: YYYY-MM-DD
  const tomorrowStr = nowEt.plus({ days: 1 }).toISODate(); // Format: YYYY-MM-DD

  console.log(`Service: Querying Supabase table '${SUPABASE_TABLE_NAME}' for dates: ${todayStr}, ${tomorrowStr}`);

  try {
    // Query Supabase table 'mlb_game_schedule'
    const { data, error, status } = await supabase
      .from(SUPABASE_TABLE_NAME)
      .select("*") // Select all columns for now
      .in("game_date_et", [todayStr, tomorrowStr]) // Filter by the correct ET date column
      .order("scheduled_time_utc", { ascending: true }); // Order by game time

    if (error) {
      console.error("Supabase error fetching schedule:", error);
      // Create a more informative error object
      const dbError = new Error(error.message || "Database query failed");
      dbError.status = status || 500; // Add HTTP status if available
      throw dbError; // Throw it for the controller to catch
    }

    console.log(`Service: Found ${data ? data.length : 0} games.`);
    return data || []; // Return the data array or an empty array if null

  } catch (error) {
    console.error("Error in fetchMlbSchedule service:", error);
    // Re-throw the error so the controller's error handler catches it
    throw error;
  }
};

// Add more service functions for MLB data here...