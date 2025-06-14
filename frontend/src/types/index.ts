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
  statusState?: string | null; // Status description + short code
  // Schedule specific (might be null if historical)
  sport: Sport;
  homePitcher?: string | null;
  awayPitcher?: string | null;
  homePitcherHand?: string | null;
  awayPitcherHand?: string | null;
  moneylineHome?: string | number | null;
  moneylineAway?: string | number | null;
  spreadLine?: number | null; // Check type from backend mapping
  totalLine?: number | null; // Check type from backend mapping

  // NBA Predictions
  predictionHome?: number | null;
  predictionAway?: number | null;

  // Added for MLB Predictions
  predicted_home_runs?: number | null;
  predicted_away_runs?: number | null;

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
  team_id: string | number; // DB sometimes returns BIGINT
  team_name: string;
  season: number;

  /* ---------- shared ----------- */
  wins_all_percentage: number;
  current_form: string | null;

  /* ---------- NBA ------------ */
  points_for_avg_all?: number | null;
  points_against_avg_all?: number | null;

  /* ---------- MLB ------------ */
  runs_for_avg_all?: number | null;
  runs_against_avg_all?: number | null;

  /* – you can extend later without breaking existing code – */
  [key: string]: string | number | undefined | null;
}

export interface UnifiedPlayerStats {
  player_id: string | number;
  player_name: string;
  team_name: string;
  games_played: number | null;

  minutes: number;
  points: number;
  rebounds: number;
  assists: number;
  steals: number | null;
  blocks: number | null;

  fg_made: number | null;
  fg_attempted: number | null;
  three_made: number;
  three_attempted: number;
  ft_made: number;
  ft_attempted: number;

  three_pct: number;
  ft_pct: number;

  [key: string]: string | number | undefined | null;
}

export interface MlbAdvancedTeamStats {
  team_id: number;
  team_name: string;

  /* core advanced metrics we read in the UI */
  win_pct: number;
  pythagorean_win_pct: number;
  run_differential: number;
  run_differential_avg: number;
  luck_factor: number;
  games_played: number;

  /* keep it open-ended for any future fields */
  [key: string]: string | number | undefined | null;
}

// Add NBA advanced stats interface
export interface NbaAdvancedTeamStats {
  team_name: string;
  pace: number;
  off_rtg: number;
  def_rtg: number;
  efg_pct: number;
  tov_pct: number;
  oreb_pct: number;
  games_played: number;

  /* allow extension */
  [key: string]: string | number | undefined | null;
}
