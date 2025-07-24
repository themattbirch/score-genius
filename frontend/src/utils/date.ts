// src/utils/date.ts

export function getLocalYYYYMMDD(d: Date = new Date()): string {
  // format “YYYY‑MM‑DD” in Eastern Time no matter the host timezone
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);
}
