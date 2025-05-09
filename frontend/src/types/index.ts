// src/types/index.ts
import type { LucideIcon } from "lucide-react";
export type IconType = LucideIcon;
export type Sport = "NBA" | "MLB";

export interface UnifiedGame {
  id: string;
  game_date: string; // YYYY-MM-DD (ET, based on backend logic)
  scheduled_time: string;
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
export interface UnifiedTeamStats {
  team_id: string; // or number ­– keep in sync with backend
  team_name: string;
  season: number;

  wins_all_percentage: number;
  points_for_avg_all: number;
  points_against_avg_all: number;
  current_form: string | null;
}

export interface UnifiedPlayerStats {
  player_id: string;
  player_name: string;
  team_name: string;
  games_played: number | null; // Allow null if calculation fails

  minutes: number; // mpg
  points: number; // ppg
  rebounds: number;
  assists: number;
  steals: number | null; // Will be NULL from DB
  blocks: number | null;

  fg_made: number | null; // Calculated, could be null if inputs missing
  fg_attempted: number | null; // Will be NULL from DB

  three_made: number;
  three_attempted: number;
  ft_made: number;
  ft_attempted: number;

  /* derived in the fetcher ↓ */
  three_pct: number;
  ft_pct: number;
  [key: string]: string | number | undefined | null; // Allow nulls
}

/* Re‑export everything from one place */
export * from "./";
