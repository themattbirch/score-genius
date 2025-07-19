import fs from "fs";
import path from "path";
import axios from "axios";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/* ------------------------------------------------------------------ */
/*  In‑memory cache of venue JSON                                      */
/* ------------------------------------------------------------------ */

let stadiumDataCache = null;

function getStadiumData() {
  if (stadiumDataCache) return stadiumDataCache;

  try {
    const dataPath = path.join(__dirname, "../../data/stadium_data.json");
    stadiumDataCache = JSON.parse(fs.readFileSync(dataPath, "utf8"));
    console.log("Stadium data loaded and cached.");
    return stadiumDataCache;
  } catch (err) {
    console.error("FATAL: cannot read stadium_data.json", err);
    process.exit(1);
  }
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function normalizeTeamKey(teamName) {
  if (/^St\.\S/.test(teamName)) return teamName.replace(/^St\.(?=\S)/, "St. ");
  if (teamName === "Oakland Athletics") return "Athletics";
  return teamName;
}

function getWindDirection(deg) {
  if (deg == null) return "N/A";
  const dirs = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
  ];
  return dirs[Math.round(deg / 22.5) % 16];
}

function getRelativeWindInfo(windDeg, orientation, isIndoor) {
  if (isIndoor) return { text: "Indoor/N/A", angle: 0 };
  if (windDeg == null || orientation == null) return { text: "N/A", angle: 0 };

  const windTo = (windDeg + 180) % 360;
  const delta = (windTo - orientation + 360) % 360;

  let text;
  if (delta < 22.5 || delta >= 337.5) text = "Blowing Out";
  else if (delta < 67.5) text = "Out to Right";
  else if (delta < 112.5) text = "L to R";
  else if (delta < 157.5) text = "In from Left";
  else if (delta < 202.5) text = "Blowing In";
  else if (delta < 247.5) text = "In from Right";
  else if (delta < 292.5) text = "R to L";
  else text = "Out to Left";

  return { text, angle: delta };
}

/* ------------------------------------------------------------------ */
/*  Public helpers                                                     */
/* ------------------------------------------------------------------ */

/**
 * Look up and return the raw venue JSON entry for a team.
 * @param {string} sport   - "mlb" | "nfl" | ...
 * @param {string} team    - Team name as passed by the client.
 */
function getVenueInfo(sport, team) {
  const leagueData = getStadiumData()[sport.toUpperCase()];
  if (!leagueData) throw new Error(`Invalid sport: ${sport}`);

  const lookupKey = normalizeTeamKey(team);
  const venue = leagueData[lookupKey];

  if (!venue)
    throw new Error(
      `No venue info for team "${team}" (normalized "${lookupKey}") in ${sport}`
    );

  return venue; // { latitude, longitude, orientation, is_indoor?, ... }
}

/**
 * Fetch (or skip) weather for the given team.
 */
async function getWeatherDataForTeam(sport, team) {
  const venue = getVenueInfo(sport, team);

  // Skip live API call for indoor venues
  if (venue.is_indoor) {
    return {
      temperature: null,
      feels_like: null,
      humidity: null,
      description: "Indoor venue",
      icon: null,
      city: venue.city,
      stadium: venue.stadium,
      description: "Indoor venue",
      windSpeed: 0,
      windDirection: "N/A",
      ballparkWindText: "Indoor/N/A",
      ballparkWindAngle: 0,
    };
  }

  /* -------- outdoor: call OpenWeather ------- */
  const { WEATHER_API_KEY } = process.env;
  if (!WEATHER_API_KEY)
    throw new Error("WEATHER_API_KEY missing from environment");

  const url =
    `https://api.openweathermap.org/data/2.5/weather` +
    `?lat=${venue.latitude}&lon=${venue.longitude}` +
    `&units=imperial&appid=${WEATHER_API_KEY}`;

  try {
    const { data } = await axios.get(url);

    const relWind = getRelativeWindInfo(
      data.wind.deg,
      venue.orientation,
      false
    );

    return {
      temperature: Math.round(data.main.temp),
      feels_like: Math.round(data.main.feels_like),
      humidity: data.main.humidity,
      description: data.weather[0].description,
      icon: data.weather[0].icon,
      city: venue.city,
      windSpeed: Math.round(data.wind.speed),
      windDirection: getWindDirection(data.wind.deg),
      ballparkWindText: relWind.text,
      ballparkWindAngle: relWind.angle,
    };
  } catch (err) {
    console.error("OpenWeather error:", err.response?.data || err.message);
    throw new Error("Failed to retrieve weather data from external API.");
  }
}

export { getWeatherDataForTeam, getVenueInfo };
