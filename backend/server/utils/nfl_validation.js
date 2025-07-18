// backend/server/utils/nfl_validation.js

// Allowed values
export const NFL_ALLOWED_CONFERENCES = ["AFC", "NFC"];
export const NFL_ALLOWED_DIVISIONS = ["East", "West", "North", "South"];

/**
 * Helpers for parsing & normalizing parameters
 */
export function parseSeasonParam(seasonStr) {
  if (!seasonStr || !/^\d{4}$/.test(seasonStr)) return null;
  return Number(seasonStr);
}

export function parseCsvInts(csv) {
  if (!csv) return null;
  const arr = String(csv)
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number(s))
    .filter((n) => Number.isInteger(n));
  return arr.length ? arr : null;
}

export function normConf(v) {
  if (!v) return null;
  const up = String(v).trim().toUpperCase();
  return NFL_ALLOWED_CONFERENCES.includes(up) ? up : null;
}

export function normDiv(v) {
  if (!v) return null;
  const cap = String(v).trim();
  const norm = cap.charAt(0).toUpperCase() + cap.slice(1).toLowerCase();
  return NFL_ALLOWED_DIVISIONS.includes(norm) ? norm : null;
}

export function badParam(res, msg) {
  return res.status(400).json({ message: msg });
}

/**
 * Validation helpers for routes (if you ever need to use them directly)
 */
export function isValidConference(v) {
  return v && NFL_ALLOWED_CONFERENCES.includes(String(v).toUpperCase());
}

export function isValidDivision(v) {
  const norm =
    String(v).charAt(0).toUpperCase() + String(v).slice(1).toLowerCase();
  return NFL_ALLOWED_DIVISIONS.includes(norm);
}
