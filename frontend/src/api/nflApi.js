// frontend/src/api/nflApi.js

// Base URL for your backend's NFL endpoints.
// This assumes your frontend dev server proxies requests to the backend.
const API_BASE_URL = "/api/v1/nfl"; // fix to match backend path

export const getScheduleByDate = async (date) => {
  if (!date) {
    console.warn("getScheduleByDate (NFL) called without a date.");
    return [];
  }

  const response = await fetch(`${API_BASE_URL}/schedule?date=${date}`);
  if (!response.ok) {
    const errorInfo = await response.json().catch(() => ({}));
    console.error("Error fetching NFL schedule from API:", errorInfo);
    throw new Error(
      errorInfo.message || `Request failed with status ${response.status}`
    );
  }
  const body = await response.json();
  return body.data || [];
};

// New: fetch team stats summary (for rankings table)
export const getTeamStatsSummary = async (season) => {
  if (!season) {
    console.warn("getTeamStatsSummary called without season.");
    return [];
  }

  const response = await fetch(
    `${API_BASE_URL}/team-stats/summary?season=${season}`
  );
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    console.error("Error fetching NFL team stats summary:", err);
    throw new Error(err.message || `Failed with status ${response.status}`);
  }
  const { data } = await response.json();
  return data || [];
};
