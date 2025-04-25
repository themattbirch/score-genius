// frontend/src/api/useAdvancedStats.ts
import { useQuery } from "@tanstack/react-query";
import type { Sport } from "@/types"; // Assuming Sport type is in '@/types'

// Define the shape of the data returned by the API/RPC function
// Based on the columns defined in the SQL function's RETURNS TABLE
export interface AdvancedTeamStats {
  team_name: string;
  games_played: number | null; // Use number | null for safety
  pace: number | null;
  off_rtg: number | null;
  def_rtg: number | null;
  efg_pct: number | null;
  tov_pct: number | null;
  oreb_pct: number | null;
  // Index signature allows dynamic access if needed later
  [key: string]: string | number | undefined | null;
}

/**
 * Fetches calculated advanced team stats for a given NBA season.
 */
const fetchAdvancedStats = async ({
  season,
}: {
  season: number;
}): Promise<AdvancedTeamStats[]> => {
  // Construct the API endpoint URL
  // Note: We hardcode 'nba' as this is NBA-specific for now
  const endpoint = `/api/v1/nba/advanced-stats?season=${season}`;

  const res = await fetch(endpoint);

  if (!res.ok) {
    const errorText = await res.text();
    console.error(
      `Failed to fetch advanced stats for season ${season}:`,
      res.status,
      errorText
    );
    throw new Error(
      errorText || `Failed to fetch advanced stats (${res.status})`
    );
  }
  const json = await res.json();

  if (!Array.isArray(json.data)) {
    console.error(
      "API response data for advanced stats is not an array:",
      json.data
    );
    return []; // Return empty array on invalid data format
  }

  return json.data as AdvancedTeamStats[];
};

/**
 * Hook to load and cache advanced NBA team stats for a specific season.
 */
export const useAdvancedStats = ({
  season,
  // sport parameter is kept for consistency but currently ignored (assumes NBA)
  sport,
  enabled = true,
  staleTime = 1000 * 60 * 60 * 1, // Cache for 1 hour? Advanced stats change less often
}: {
  season: number;
  sport: Sport; // Keep for key consistency, even if logic is NBA-only
  enabled?: boolean;
  staleTime?: number;
}) =>
  useQuery<AdvancedTeamStats[]>({
    // Query key includes sport for consistency, even though it's NBA only now
    queryKey: ["advancedStats", sport, season],
    queryFn: () => fetchAdvancedStats({ season }),
    enabled: enabled && sport === "NBA", // Ensure hook only runs for NBA
    staleTime,
    refetchOnMount: "always", // Refetch if enabled status changes
  });
