// frontend/src/api/useAdvancedStats.ts
import { useQuery } from "@tanstack/react-query";
import type { Sport } from "@/types";
import { apiFetch } from "@/api/client";

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
async function fetchAdvancedStats({
  season,
}: {
  season: number;
}): Promise<AdvancedTeamStats[]> {
  const res = await apiFetch(`/api/v1/nba/advanced-stats?season=${season}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `Failed to fetch NBA advanced‐stats: ${res.status} ${text}`
    );
  }
  const json = (await res.json()) as {
    message: string;
    retrieved: number;
    data: AdvancedTeamStats[];
  };
  if (!Array.isArray(json.data)) {
    console.error("Invalid NBA advanced‐stats response:", json);
    return [];
  }
  return json.data;
}

/**
 * Hook: load & cache advanced NBA team stats
 */
export function useAdvancedStats({
  sport,
  season,
  enabled = true,
}: {
  sport: Sport;
  season: number;
  enabled?: boolean;
}) {
  return useQuery<AdvancedTeamStats[]>({
    queryKey: ["advancedStats", sport, season],
    queryFn: () => fetchAdvancedStats({ season }),
    enabled: enabled && sport === "NBA",
    staleTime: 1000 * 60 * 60, // 1h
    refetchOnMount: "always",
  });
}
