// frontend/src/api/use_team_stats.ts

import { Sport, UnifiedTeamStats } from "@/types";
import { useQuery, QueryFunctionContext } from "@tanstack/react-query";

/** ------------------------------
 *  Low‑level fetcher used by the
 *  StatsScreen (and useTeamStats).
 *  ------------------------------*/
export const fetchTeamStats = async ({
  sport,
  season,
}: {
  sport: Sport;
  season: number;
}): Promise<UnifiedTeamStats[]> => {
  const endpoint = `/api/v1/${sport.toLowerCase()}/team-stats?season=${season}`;
  const res = await fetch(endpoint);

  if (!res.ok) {
    throw new Error(await res.text());
  }

  const json = await res.json();
  return json.data; // <-- THIS LINE IS THE KEY!
};


/** Convenience hook if other screens want it */
export const useTeamStats = ({
  sport,
  season,
  staleTime = 1000 * 60 * 30, // 30‑min cache
}: {
  sport: Sport;
  season: number;
  staleTime?: number;
}) =>
  useQuery<UnifiedTeamStats[]>({
    queryKey: ["teamStats", sport, season],
    queryFn: () => fetchTeamStats({ sport, season }),
    staleTime,
  });
