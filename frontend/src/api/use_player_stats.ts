import { Sport } from "@/types";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";

/* ────────────  DATA SHAPE  ──────────── */
export interface UnifiedPlayerStats {
  player_id: string | number; // Allow number from RPC (BIGINT), or string if cast
  player_name: string;
  team_name: string;
  games_played: number; // Make sure this field (returned by RPC) is included
  minutes: number;
  points: number;
  rebounds: number;
  assists: number;
  steals: number;
  blocks: number;
  // Include turnovers/fouls if your RPC function returns them and you need them
  // turnovers?: number;
  // fouls?: number;
  fg_made: number;
  fg_attempted: number;
  three_made: number;
  three_attempted: number;
  ft_made: number;
  ft_attempted: number;
  // Percentages are provided by backend RPC now
  fg_pct: number;
  three_pct: number;
  ft_pct: number;
  // Index signature allows accessing properties dynamically if needed
  // Added undefined for potential optional fields like turnovers/fouls
  [key: string]: string | number | undefined;
}
async function fetchPlayerStats({
  sport,
  season,
  search,
}: {
  sport: Sport;
  season: number;
  search?: string;
}): Promise<UnifiedPlayerStats[]> {
  const params = new URLSearchParams({ season: String(season) });
  if (search) params.set("search", search);

  const res = await apiFetch(
    `/api/v1/${sport.toLowerCase()}/player-stats?${params.toString()}`
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `Failed to fetch ${sport} player‐stats: ${res.status} ${text}`
    );
  }

  const json = (await res.json()) as {
    message: string;
    retrieved: number;
    data: UnifiedPlayerStats[];
  };

  if (!Array.isArray(json.data)) {
    console.error("Invalid player‐stats response:", json);
    return [];
  }
  return json.data;
}

/**
 * Hook: load & cache player stats for NBA or MLB
 */
export function usePlayerStats({
  sport,
  season,
  search = "",
  enabled = true,
}: {
  sport: Sport;
  season: number;
  search?: string;
  enabled?: boolean;
}) {
  return useQuery<UnifiedPlayerStats[]>({
    queryKey: ["playerStats", sport, season, search],
    queryFn: () => fetchPlayerStats({ sport, season, search }),
    enabled, // ← add this
    staleTime: 1000 * 60 * 30,
    refetchOnMount: "always",
  });
}
