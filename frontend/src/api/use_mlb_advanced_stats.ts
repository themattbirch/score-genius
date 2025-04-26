// frontend/src/api/use_mlb_advanced_stats.ts
import { useQuery } from "@tanstack/react-query";
import type { Sport } from "@/types"; // Assuming Sport type is in '@/types'
import { apiFetch } from "@/api/client";

// Define the shape of the data returned by the MLB Advanced Stats API/RPC function
// Based on the columns from the successful curl response
export interface MlbAdvancedTeamStats {
  team_id: number;
  team_name: string;
  season: number;
  games_played: number;
  wins: number;
  runs_for: number;
  runs_against: number;
  run_differential: number;
  run_differential_avg: number;
  win_pct: number;
  pythagorean_win_pct: number;
  expected_wins: number;
  luck_factor: number;
  home_away_win_pct_split: number;
  home_away_run_diff_avg_split: number;
  // Index signature allows dynamic access if needed later (optional but can be useful)
  [key: string]: string | number | undefined | null;
}

/**
 * Fetches calculated advanced team stats for a given MLB season via RPC endpoint.
 */
async function fetchMlbAdvancedStats({
  season,
}: {
  season: number;
}): Promise<MlbAdvancedTeamStats[]> {
  const endpoint = `/api/v1/mlb/team-stats/advanced?season=${season}`;
  const res = await apiFetch(endpoint);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `Failed to fetch MLB advanced‐stats: ${res.status} ${text}`
    );
  }
  const json = (await res.json()) as {
    message: string;
    retrieved: number;
    data: MlbAdvancedTeamStats[];
  };
  return Array.isArray(json.data) ? json.data : [];
}

/**
 * Hook: load & cache advanced MLB team stats
 */
export function useMlbAdvancedStats({
  season,
  sport,
  enabled = true,
}: {
  season: number;
  sport: Sport;
  enabled?: boolean;
}) {
  return useQuery<MlbAdvancedTeamStats[]>({
    queryKey: ["mlbAdvancedStats", sport, season],
    queryFn: () => fetchMlbAdvancedStats({ season }),
    enabled: enabled && sport === "MLB",
    staleTime: 1000 * 60 * 60 * 24, // 24h
    refetchOnMount: "always",
    refetchOnWindowFocus: false,
  });
}
