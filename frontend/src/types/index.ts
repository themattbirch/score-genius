// frontend/src/types/index.ts
import type { LucideIcon } from "lucide-react";

export type IconType = LucideIcon;
export type Sport = "NBA" | "MLB" | "NFL";

// Define interfaces for chart data structures, if not already present
export interface BarChartData {
  category: string;
  Home: number; // Assuming this structure from your BarChartComponent
  Away: number;
}

export interface RadarChartData {
  metric: string;
  home_raw: number | string;
  away_raw: number | string;
  home_idx: number;
  away_idx: number;
}

export interface PieChartDataItem {
  category: string;
  value: number;
  color?: string; // Color might be optional
}

export interface NbaPreGameOffenseDataItem {
  metric: string;
  Home: number;
  Away: number;
}

// Define interface for Headline Stats
export interface HeadlineStat {
  label: string;
  value: string | number;
}

export interface UnifiedGame {
  id: string;
  game_date: string; // YYYY-MM-DD (ET, based on backend logic)
  scheduled_time: string;
  scheduled_time_utc?: string;
  homeTeamName: string;
  awayTeamName: string;
  gameTimeUTC?: string | null; // ISO UTC timestamp for scheduled or historical
  statusState?: string | null; // Status description + short code
  // Schedule specific (may be null if historical)
  sport: Sport;
  homePitcher?: string | null;
  awayPitcher?: string | null;
  homePitcherHand?: string | null;
  awayPitcherHand?: string | null;
  moneylineHome?: string | number | null;
  moneylineAway?: string | number | null;
  spreadLine?: number | null; // Check type from backend mapping
  totalLine?: number | null; // Check type from backend mapping
  moneyline_clean?: string | null;
  spread_clean?: string | null;
  total_clean?: string | null;

  // NBA Predictions
  predictionHome?: number | null;
  predictionAway?: number | null;

  // Added for MLB Predictions
  predicted_home_runs?: number | null;
  predicted_away_runs?: number | null;

  //  NFL Predictions
  predicted_home_score?: number | null;
  predicted_away_score?: number | null;

  spread?: number | null; // NBA specific schedule (duplicate?) -> Consolidate in backend mapping
  total?: number | null; // NBA specific schedule (duplicate?) -> Consolidate in backend mapping
  tipoff?: string | null; // NBA specific schedule (duplicate?) -> Use gameTimeUTC
  // Historical specific (might be null if schedule)
  home_final_score?: number | null;
  away_final_score?: number | null;
  // Discriminator
  dataType: "schedule" | "historical";
  // Implicit: sport (can be derived from context if needed, or added here by backend)

  venueLocation?: VenueLocation;
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

export interface NflAdvancedTeamStats {
  team_name: string;
  games_played: number;
  srs: number; // Simple Rating System
  sos: number; // Strength of Schedule
  point_differential: number;
  turnover_differential: number;
  pythagorean_wins: number;
  luck: number; // Actual Wins - Pythagorean Wins

  /* allow extension */
  [key: string]: string | number | undefined | null;
}
interface SnapshotModalProps {
  gameId: string;
  sport: Sport;
  isOpen: boolean;
  onClose: () => void;
}
export interface SnapshotData {
  headline_stats?: HeadlineStat[];
  bar_chart_data?: BarChartData[];
  radar_chart_data?: RadarChartData[];
  key_metrics_data?: BarChartData[];
  pie_chart_data?: PieChartDataItem[] | NbaPreGameOffenseDataItem[];
  is_historical?: boolean;
  stage?: string;
}
export interface WeatherData {
  temperature: number;
  feels_like: number;
  humidity: number;
  windSpeed: number;
  windDirection: string;
  description: string;
  icon: string;
  city: string;
  ballparkWindText: string;
  ballparkWindAngle: number;
  isIndoor?: boolean;
}

/**
 * Defines the structure for a game's location, which will be
 * passed from the GameCard down to the WeatherBadge.
 */
export interface VenueLocation {
  latitude: number;
  longitude: number;
}
export interface NflTeamSummary {
  teamId: string;
  teamName: string;
  season: number;
  srs?: number;
  sos?: number;
  sosRank?: number;
  winPct?: number;
  pythagoreanWinPct?: number;
  avgThirdDownPct?: number;
  avgRedZonePct?: number;
  avgYardsPerDrive?: number;
  avgTurnoversPerGame?: number;
  avgTimeOfPossession?: string;

  [key: string]: any;
}
// ────────────────────────────────────────────────────────────
// Edge / Value types
// ────────────────────────────────────────────────────────────
export type EdgeTier = "HIGH" | "MED" | "LOW";
export type EdgeMarket = "ML" | "SPREAD";
export type EdgeSide = "HOME" | "AWAY";

export interface ValueEdge {
  market: EdgeMarket;
  side: EdgeSide;
  edgePct: number; // (modelProb - marketProb) * 100
  modelProb: number; // 0..1
  marketProb: number; // 0..1 (vig-free)
  z: number; // standardized confidence
  tier: EdgeTier;
}
