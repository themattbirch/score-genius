import { Sport } from "@/types";
import { useQuery } from "@tanstack/react-query";

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
/* ────────────  FETCHER  ──────────── */
const fetchPlayerStats = async ({
  sport, // Keep sport for API path
  season,
  search,
}: {
  sport: Sport;
  season: number;
  search: string;
}): Promise<UnifiedPlayerStats[]> => {
  const params = new URLSearchParams({
    season: String(season),
  });
  if (search) params.set("search", search);

  const res = await fetch(
    `/api/v1/${sport.toLowerCase()}/player-stats?` + params.toString()
  );

  if (!res.ok) {
    // Log more details on error
    const errorText = await res.text();
    console.error("Failed to fetch player stats:", res.status, errorText);
    throw new Error(
      errorText || `Failed to fetch player stats (${res.status})`
    );
  }
  const json = await res.json();

  // Directly return the data from the API response,
  // assuming it now contains the pre-calculated percentages from the RPC.
  // Add a check to ensure data is an array.
  if (!Array.isArray(json.data)) {
    console.error(
      "API response data for player stats is not an array:",
      json.data
    );
    // Return empty array or throw error based on how you want to handle unexpected API responses
    return [];
    // OR: throw new Error("Invalid data format received from API");
  }

  // The backend RPC now provides fg_pct, three_pct, ft_pct directly.
  // (Optional: If player_id needs to be a string consistently in frontend, map it here)
  // return json.data.map(p => ({ ...p, player_id: String(p.player_id) })) as UnifiedPlayerStats[];
  return json.data as UnifiedPlayerStats[];
};

// --- The usePlayerStats hook itself likely doesn't need changes ---
export const usePlayerStats = ({
  sport,
  season,
  search = "",
  enabled = true,
}: {
  sport: Sport;
  season: number;
  search?: string;
  enabled?: boolean;
}) =>
  useQuery<UnifiedPlayerStats[]>({
    queryKey: ["playerStats", sport, season, search],
    queryFn: () => fetchPlayerStats({ sport, season, search }),
    staleTime: 1000 * 60 * 30, // 30 minutes
    enabled,
  });

// You might also want to update the UnifiedPlayerStats interface if the RPC
// added/removed/renamed fields (e.g., added games_played)
