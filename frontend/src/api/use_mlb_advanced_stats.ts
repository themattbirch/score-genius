// frontend/src/api/use_mlb_advanced_stats.ts

import { useQuery } from "@tanstack/react-query";
import type { Sport } from "@/types";

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

  /* keep it open-ended */
  [key: string]: string | number | undefined | null;
}

function isJson(res: Response) {
  return (res.headers.get("content-type") ?? "").includes("application/json");
}

async function fetchMlbAdvancedStats(
  season: number
): Promise<MlbAdvancedTeamStats[]> {
  if (!navigator.onLine) throw new Error("Browser offline");

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await fetch(`/api/v1/mlb/team-stats/advanced?season=${season}`, {
      signal: controller.signal,
      headers: { accept: "application/json" },
      cache: "no-store",
    });
  } finally {
    clearTimeout(tid);
  }

  if (!res.ok || !isJson(res)) {
    throw new Error(`MLB advanced-stats request failed (${res.status})`);
  }

  const json = (await res.json()) as { data: MlbAdvancedTeamStats[] };
  return Array.isArray(json.data) ? json.data : [];
}

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
    queryFn: () => fetchMlbAdvancedStats(season),
    enabled: enabled && sport === "MLB",
    staleTime: 86_400_000, // 24 h
    retry: (failureCount: number) => navigator.onLine && failureCount < 3,
    refetchOnWindowFocus: false,
  });
}
