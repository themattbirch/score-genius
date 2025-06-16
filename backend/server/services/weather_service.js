import fs from "fs";
import path from "path";
import axios from "axios";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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

/**
 * FINAL CORRECTED VERSION: Calculates the wind's direction relative to the ballpark's layout.
 */
function getRelativeWindInfo(windDegrees, stadiumOrientation) {
  if (
    windDegrees === undefined ||
    stadiumOrientation === null ||
    stadiumOrientation === undefined
  ) {
    return { text: "N/A", angle: 0 };
  }
  if (stadiumOrientation === 0) {
    return { text: "Indoor/N/A", angle: 0 };
  }

  const rotationAngle = (windDegrees - stadiumOrientation + 360) % 360;

  let description = "Variable";
  // The logic below is now corrected to match the icon's visual rotation
  if (rotationAngle >= 337.5 || rotationAngle < 22.5)
    description = "Blowing Out";
  else if (rotationAngle >= 22.5 && rotationAngle < 67.5)
    description = "Out to Right";
  else if (rotationAngle >= 67.5 && rotationAngle < 112.5)
    description = "L to R";
  else if (rotationAngle >= 112.5 && rotationAngle < 157.5)
    description = "In from Left";
  else if (rotationAngle >= 157.5 && rotationAngle < 202.5)
    description = "Blowing In";
  else if (rotationAngle >= 202.5 && rotationAngle < 247.5)
    description = "In from Right"; // Corrected
  else if (rotationAngle >= 247.5 && rotationAngle < 292.5)
    description = "R to L";
  else if (rotationAngle >= 292.5 && rotationAngle < 337.5)
    description = "Out to Left"; // Corrected

  return {
    text: description,
    angle: rotationAngle,
  };
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

  const teamInfo = leagueData[teamName];
  if (!teamInfo) {
    throw new Error(
      `Could not find stadium information for team: ${teamName} in ${sport}.`
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
