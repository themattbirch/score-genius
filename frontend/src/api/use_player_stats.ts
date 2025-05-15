import { useQuery } from "@tanstack/react-query";
import type { Sport } from "@/types";

export interface UnifiedPlayerStats {
  /* … unchanged … */
}

function isJson(res: Response) {
  return (res.headers.get("content-type") ?? "").includes("application/json");
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
  if (!navigator.onLine) throw new Error("Browser offline");

  const params = new URLSearchParams({ season: String(season) });
  if (search) params.set("search", search);

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await fetch(`/api/v1/${sport.toLowerCase()}/player-stats?${params}`, {
      signal: controller.signal,
      headers: { accept: "application/json" },
      cache: "no-store",
    });
  } finally {
    clearTimeout(tid);
  }

  if (!res.ok || !isJson(res)) {
    throw new Error(`${sport} player-stats request failed (${res.status})`);
  }

  const json = (await res.json()) as { data: UnifiedPlayerStats[] };
  return Array.isArray(json.data) ? json.data : [];
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
    enabled,
    staleTime: 1_800_000, // 30 min
    retry: (failureCount: number) => navigator.onLine && failureCount < 3,
    refetchOnMount: "always",
  });
}
