// frontend/src/api/use_team_stats.ts
import { Sport, UnifiedTeamStats } from "@/types";
import { useQuery } from "@tanstack/react-query";

/**
 * Fetches season-level stats for ALL teams of a given sport.
 */
const fetchTeamStats = async ({
  sport,
  season,
}: {
  sport: Sport;
  season: number;
}): Promise<UnifiedTeamStats[]> => {
  const endpoint = `/api/v1/${sport.toLowerCase()}/team-stats?season=${season}`;
  const res = await fetch(endpoint);
  if (!res.ok) throw new Error(await res.text());
  const json = await res.json();
  return json.data;
};
/**
 * Hook to load and cache team stats for *any* sport.
 * Accepts an `enabled` flag so you can skip bad queries.
 */
export const useTeamStats = ({
  sport,
  season,
  enabled = true,
  staleTime = 1000 * 60 * 30,
}: {
  sport: Sport;
  season: number;
  enabled?: boolean;
  staleTime?: number;
}) =>
  useQuery<UnifiedTeamStats[]>({
    queryKey: ["teamStats", sport, season],
    queryFn: () => fetchTeamStats({ sport, season }),
    enabled,
    staleTime,
    // Always refetch whenever this hook remounts after being disabled:
    refetchOnMount: "always",
  });

