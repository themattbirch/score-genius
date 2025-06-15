// backend/server/services/weather_service.js

import fs from "fs";
import path from "path";
import axios from "axios";
import { fileURLToPath } from "url";

// Recreate __dirname for ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Memoize the stadium data to avoid reading the file on every single request.
let stadiumDataCache = null;

function getStadiumData() {
  if (stadiumDataCache) {
    return stadiumDataCache;
  }
  try {
    const dataPath = path.join(__dirname, "../../data/stadium_data.json");
    const rawData = fs.readFileSync(dataPath, "utf8");
    stadiumDataCache = JSON.parse(rawData);
    console.log("Stadium data loaded and cached successfully.");
    return stadiumDataCache;
  } catch (error) {
    console.error("FATAL: Could not read or parse stadium_data.json", error);
    process.exit(1);
  }
}

// Load the data as soon as the service is initialized
getStadiumData();

/**
 * Converts wind degrees to a cardinal direction.
 * @param {number} degrees - Wind direction in degrees.
 * @returns {string} The cardinal direction (e.g., 'NNE').
 */
function getWindDirection(degrees) {
  if (degrees === undefined) return "N/A";
  const directions = [
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
  const index = Math.round(degrees / 22.5) % 16;
  return directions[index];
}

/**
 * Fetches weather data for a given team and sport.
 * @param {string} sport - The league ('MLB' or 'NFL').
 * @param {string} teamName - The name of the team (e.g., "Houston Astros").
 * @returns {Promise<object>} A simplified weather data object.
 */
async function getWeatherDataForTeam(sport, teamName) {
  const { WEATHER_API_KEY } = process.env;
  if (!WEATHER_API_KEY) {
    throw new Error("Weather API key is not defined in the environment.");
  }

  const leagueData = getStadiumData()[sport];
  if (!leagueData) {
    throw new Error(`Invalid sport specified: ${sport}.`);
  }

  const teamInfo = leagueData[teamName];
  if (!teamInfo) {
    throw new Error(
      `Could not find stadium information for team: ${teamName} in ${sport}.`
    );
  }

  const { latitude, longitude } = teamInfo;
  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=imperial&appid=${WEATHER_API_KEY}`;

  try {
    const response = await axios.get(url);
    const { data } = response;
    return {
      temperature: Math.round(data.main.temp),
      feels_like: Math.round(data.main.feels_like),
      humidity: data.main.humidity,
      windSpeed: Math.round(data.wind.speed),
      windDirection: getWindDirection(data.wind.deg),
      description: data.weather[0].description,
      icon: data.weather[0].icon,
      city: teamInfo.city,
    };
  } catch (error) {
    console.error(
      "Error fetching data from OpenWeatherMap:",
      error.response?.data || error.message
    );
    throw new Error("Failed to retrieve weather data from external API.");
  }
}

// Use a named export
export { getWeatherDataForTeam };
