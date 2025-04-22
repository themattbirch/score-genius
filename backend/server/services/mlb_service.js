// backend/server/services/mlb_service.js

import supabase from "../utils/supabase_client.js";
import { DateTime } from "luxon";
import cache from "../utils/cache.js";

const MLB_SCHEDULE_TABLE = "mlb_game_schedule";
const MLB_HISTORICAL_GAMES_TABLE = "mlb_historical_game_stats";
const MLB_HISTORICAL_TEAM_STATS_TABLE = "mlb_historical_team_stats";
const ET_ZONE_IDENTIFIER = "America/New_York";

const getUTCDateString = (date) => {
  return date.toISOString().split("T")[0];
};

/**
 * Represents unified MLB game data, either from schedule or historical records.
 * @typedef {object} UnifiedMLBGameData
 * @property {string} id - The unique game identifier (stringified game_id).
 * @property {string} game_date - The original date of the game (YYYY-MM-DD ET).
 * @property {string} homeTeamName - Full name of the home team.
 * @property {string} awayTeamName - Full name of the away team.
 * @property {string | null} [gameTimeUTC] - ISO UTC timestamp for scheduled games or historical games.
 * @property {string | null} [statusState] - Game status from schedule or historical (e.g., 'Final', 'Scheduled').
 * @property {string | null} [homePitcher] - Pitcher name.
 * @property {string | null} [awayPitcher] - Pitcher name.
 * @property {string | null} [homePitcherHand] - Handedness.
 * @property {string | null} [awayPitcherHand] - Handedness.
 * @property {string | number | null} [moneylineHome] - Odds.
 * @property {string | number | null} [moneylineAway] - Odds.
 * @property {number | null} [spreadLine] - Odds.
 * @property {number | null} [totalLine] - Odds.
 * @property {number | null} [home_final_score] - Actual home score from historical.
 * @property {number | null} [away_final_score] - Actual away score from historical.
 * @property {'schedule' | 'historical'} dataType - Indicates the source of the data.
 */

// --- Function to get schedule/results for a SPECIFIC date ---
/**
 * Fetches EITHER schedule/odds data (today/future) OR
 * historical results (past dates) for MLB games on a specific date (ET).
 * @param {string} date - The date in YYYY-MM-DD format.
 * @returns {Promise<UnifiedMLBGameData[]>} - A promise resolving to an array of game data objects.
 */
export const getMlbScheduleByDate = async (date) => {
  console.log(`[mlb_service getMlbScheduleByDate] Received date: ${date}`);

  // 1. Determine if date is past or present/future in ET
  let isPastDate = false;
  let todayEtStr = "";
  try {
    const nowEt = DateTime.now().setZone(ET_ZONE_IDENTIFIER);
    todayEtStr = nowEt.toISODate();
    const inputDateEt = DateTime.fromISO(date, { zone: ET_ZONE_IDENTIFIER });

    if (!inputDateEt.isValid) {
      throw new Error(`Invalid date string provided: ${date}`);
    }
    if (inputDateEt.startOf("day") < nowEt.startOf("day")) {
      isPastDate = true;
    }
    console.log(
      `[mlb_service getMlbScheduleByDate] Date ${date} vs Today ${todayEtStr}. Is Past: ${isPastDate}`
    );
  } catch (e) {
    console.error(
      `[mlb_service getMlbScheduleByDate] Error parsing date: ${date}`,
      e
    );
    throw new Error("Invalid date format provided. Use YYYY-MM-DD.");
  }

  let data = null;
  let error = null;
  let results = [];

  // Use a single try block for the database interaction + mapping
  try {
    // --- Conditionally Fetch Data ---
    if (isPastDate) {
      console.log(
        `[mlb_service getMlbScheduleByDate] Fetching historical data for ${date} from ${MLB_HISTORICAL_GAMES_TABLE}`
      );
      const historicalColumns = `game_id, game_date_time_utc, home_team_name, away_team_name, home_score, away_score, status_short`;
      const startOfInputDayET = DateTime.fromISO(date, {
        zone: ET_ZONE_IDENTIFIER,
      }).startOf("day");
      const startOfNextDayET = startOfInputDayET.plus({ days: 1 });
      const startUTC = startOfInputDayET.toUTC().toISO();
      const endUTC = startOfNextDayET.toUTC().toISO();
      console.log(
        `[mlb_service getMlbScheduleByDate] Querying UTC range: ${startUTC} to ${endUTC}`
      );

      const response = await supabase
        .from(MLB_HISTORICAL_GAMES_TABLE)
        .select(historicalColumns)
        .gte("game_date_time_utc", startUTC)
        .lt("game_date_time_utc", endUTC)
        .order("game_date_time_utc", { ascending: true });

      data = response.data;
      error = response.error;
    } else {
      // Fetch Schedule Data
      console.log(
        `[mlb_service getMlbScheduleByDate] Fetching schedule data for ${date} from ${MLB_SCHEDULE_TABLE}`
      );
      const scheduleColumns = `
          game_id, scheduled_time_utc, game_date_et, status_detail, status_state,
          home_team_name, away_team_name, home_probable_pitcher_name,
          home_probable_pitcher_handedness, away_probable_pitcher_name,
          away_probable_pitcher_handedness, moneyline_home_clean, moneyline_away_clean,
          spread_home_line_clean, spread_home_price_clean, spread_away_price_clean,
          total_line_clean, total_over_price_clean, total_under_price_clean
      `;
      const response = await supabase
        .from(MLB_SCHEDULE_TABLE)
        .select(scheduleColumns)
        .eq("game_date_et", date)
        .order("scheduled_time_utc", { ascending: true });

      data = response.data;
      error = response.error;
    }

    // --- Handle Results ---
    console.log(
      `[mlb_service getMlbScheduleByDate] Supabase returned ${
        data?.length ?? 0
      } rows for date ${date}. Error: ${error ? error.message : "No"}`
    );

    if (error) {
      throw error; // Throw Supabase errors
    }
    if (!Array.isArray(data)) {
      console.warn(
        `[mlb_service getMlbScheduleByDate] Supabase data is not an array for date ${date}. Data:`,
        data
      );
      return []; // Return empty array if data invalid
    }

    // --- Map results to the Unified Structure ---
    results = data.map((row) => {
      if (isPastDate) {
        /** @type {UnifiedMLBGameData} */
        const gameData = {
          id: String(row.game_id),
          game_date: date,
          homeTeamName: row.home_team_name,
          awayTeamName: row.away_team_name,
          gameTimeUTC: row.game_date_time_utc,
          statusState: row.status_short ?? "Final",
          homePitcher: null,
          awayPitcher: null,
          homePitcherHand: null,
          awayPitcherHand: null,
          moneylineHome: null,
          moneylineAway: null,
          spreadLine: null,
          totalLine: null,
          home_final_score: row.home_score,
          away_final_score: row.away_score,
          dataType: "historical",
        };
        return gameData;
      } else {
        /** @type {UnifiedMLBGameData} */
        const gameData = {
          id: String(row.game_id),
          game_date: row.game_date_et,
          homeTeamName: row.home_team_name,
          awayTeamName: row.away_team_name,
          gameTimeUTC: row.scheduled_time_utc,
          statusState: row.status_state,
          homePitcher: row.home_probable_pitcher_name,
          awayPitcher: row.away_probable_pitcher_name,
          homePitcherHand: row.home_probable_pitcher_handedness,
          awayPitcherHand: row.away_probable_pitcher_handedness,
          moneylineHome: row.moneyline_home_clean,
          moneylineAway: row.moneyline_away_clean,
          spreadLine: row.spread_home_line_clean,
          totalLine: row.total_line_clean,
          home_final_score: null,
          away_final_score: null,
          dataType: "schedule",
        };
        return gameData;
      }
    });

    return results; // Return the mapped array
  } catch (err) {
    // Catch errors from Supabase or mapping
    console.error(
      `[mlb_service getMlbScheduleByDate] Error processing date ${date}:`,
      err
    );
    throw err; // Re-throw error to be handled by the controller
  }
};

// --- Other Service Functions (fetchMlbScheduleForTodayAndTomorrow, fetchMlbGameHistory, fetchMlbTeamStatsBySeason, WorkspaceMlbScheduleForTodayAndTomorrow) ---
// Keep these functions as they were, they serve different purposes
