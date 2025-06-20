// backend/server/services/weather_service.js

import fs from "fs";
import path from "path";
import axios from "axios";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let stadiumDataCache = null;

function normalizeTeamKey(teamName) {
  // Fix “St.Louis” → “St. Louis”
  if (/^St\.\S/.test(teamName)) {
    return teamName.replace(/^St\.(?=\S)/, "St. ");
  }
  // Alias “Oakland Athletics” → “Athletics”
  if (teamName === "Oakland Athletics") {
    return "Athletics";
  }
  return teamName;
}

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

getStadiumData();

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

function getRelativeWindInfo(windDegrees, stadiumOrientation) {
  // Basic guards
  if (windDegrees == null || stadiumOrientation == null) {
    return { text: "N/A", angle: 0 };
  }
  if (stadiumOrientation === 0) {
    // domes / indoor
    return { text: "Indoor/N/A", angle: 0 };
  }

  // Convert “from” ➜ “to”
  const windTo = (windDegrees + 180) % 360;

  // Δ between wind-to vector and the home-to-CF line
  const delta = (windTo - stadiumOrientation + 360) % 360;

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

async function getWeatherDataForTeam(sport, teamName) {
  const { WEATHER_API_KEY } = process.env;
  if (!WEATHER_API_KEY) {
    throw new Error("Weather API key is not defined in the environment.");
  }

  const leagueData = getStadiumData()[sport];
  if (!leagueData) {
    throw new Error(`Invalid sport specified: ${sport}.`);
  }

  const lookupKey = normalizeTeamKey(teamName);
  const teamInfo = leagueData[lookupKey];

  if (!teamInfo) {
    throw new Error(
      `Could not find stadium information for team: ${teamName} (normalized to ${lookupKey}) in ${sport}.`
    );
  }

  const { latitude, longitude, orientation } = teamInfo;
  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=imperial&appid=${WEATHER_API_KEY}`;

  try {
    const response = await axios.get(url);
    const { data } = response;

    const relativeWind = getRelativeWindInfo(data.wind.deg, orientation);

    return {
      temperature: Math.round(data.main.temp),
      feels_like: Math.round(data.main.feels_like),
      humidity: data.main.humidity,
      description: data.weather[0].description,
      icon: data.weather[0].icon,
      city: teamInfo.city,
      windSpeed: Math.round(data.wind.speed),
      windDirection: getWindDirection(data.wind.deg),
      ballparkWindText: relativeWind.text,
      ballparkWindAngle: relativeWind.angle,
    };
  } catch (error) {
    console.error(
      "Error fetching data from OpenWeatherMap:",
      error.response?.data || error.message
    );
    throw new Error("Failed to retrieve weather data from external API.");
  }
}

export { getWeatherDataForTeam };
