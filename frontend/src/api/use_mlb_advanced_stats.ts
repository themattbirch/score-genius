// frontend/src/api/use_mlb_advanced_stats.ts
import { useQuery } from "@tanstack/react-query";
import type { Sport } from "@/types"; // Assuming Sport type is in '@/types'

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
const fetchMlbAdvancedStats = async ({
  season,
}: {
  season: number;
}): Promise<MlbAdvancedTeamStats[]> => {
  // Construct API endpoint URL for MLB Advanced Stats
  const endpoint = `/api/v1/mlb/team-stats/advanced?season=${season}`;

  const res = await fetch(endpoint);

  if (!res.ok) {
    const errorText = await res.text();
    console.error(
      `Failed to fetch MLB advanced stats for season ${season}:`,
      res.status,
      errorText
    );
    throw new Error(
      errorText || `Failed to fetch MLB advanced stats (${res.status})`
    );
  }
  const json = await res.json();

  // The backend wraps data in a 'data' property
  if (!Array.isArray(json.data)) {
    console.error(
      "API response data for MLB advanced stats is not an array:",
      json.data
    );
    return []; // Return empty array on invalid data format
  }

  // Cast the data to the specific MLB interface
  return json.data as MlbAdvancedTeamStats[];
};

/**
 * Hook to load and cache advanced MLB team stats for a specific season.
 */
export const useMlbAdvancedStats = ({
  season,
  sport, // Keep for consistency, but logic is MLB-specific
  enabled = true,
  staleTime = 1000 * 60 * 60 * 24, // Cache for 24 hours - historical stats don't change often
}: {
  season: number;
  sport: Sport;
  enabled?: boolean;
  staleTime?: number;
}) =>
  useQuery<MlbAdvancedTeamStats[]>({
    // Query key specific to MLB advanced stats
    queryKey: ["mlbAdvancedStats", sport, season],
    queryFn: () => fetchMlbAdvancedStats({ season }),
    // Ensure hook only runs when the selected sport is MLB and it's explicitly enabled
    enabled: enabled && sport === "MLB",
    staleTime,
    refetchOnWindowFocus: false, // Data is historical, less need to refetch on focus
    refetchOnMount: "always", // Consider if needed, 'always' ensures data check on mount if enabled changes
  });
