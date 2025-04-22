// src/types/index.ts (or similar shared location)
export type Sport = "NBA" | "MLB";

export interface UnifiedGame {
  id: string;
  game_date: string; // YYYY-MM-DD (ET, based on backend logic)
  homeTeamName: string;
  awayTeamName: string;
  gameTimeUTC?: string | null; // ISO UTC timestamp for scheduled or historical
  statusState?: string | null; // Status description/short code
  // Schedule specific (might be null if historical)
  homePitcher?: string | null;
  awayPitcher?: string | null;
  homePitcherHand?: string | null;
  awayPitcherHand?: string | null;
  moneylineHome?: string | number | null;
  moneylineAway?: string | number | null;
  spreadLine?: number | null; // Check type from backend mapping
  totalLine?: number | null; // Check type from backend mapping
  predictionHome?: number | null; // NBA specific schedule
  predictionAway?: number | null; // NBA specific schedule
  spread?: number | null; // NBA specific schedule (duplicate?) -> Consolidate in backend mapping
  total?: number | null; // NBA specific schedule (duplicate?) -> Consolidate in backend mapping
  tipoff?: string | null; // NBA specific schedule (duplicate?) -> Use gameTimeUTC
  // Historical specific (might be null if schedule)
  home_final_score?: number | null;
  away_final_score?: number | null;
  // Discriminator
  dataType: "schedule" | "historical";
  // Implicit: sport (can be derived from context if needed, or added here by backend)
}
