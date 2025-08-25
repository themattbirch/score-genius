// src/utils/date.ts

/**
 * Format a Date as YYYY-MM-DD in the *local* browser timezone.
 */
export function getLocalYYYYMMDD(d: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);
}

/**
 * Format a Date as YYYY-MM-DD in Eastern Time (America/New_York).
 * Use this for API calls to keep backend/frontend consistent.
 */
export function getEasternYYYYMMDD(d: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);
}
