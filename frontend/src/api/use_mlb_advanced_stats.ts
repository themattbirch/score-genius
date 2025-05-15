// frontend/src/api/use_mlb_advanced_stats.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { Sport, MlbAdvancedTeamStats } from "@/types";

function isJson(r: Response) {
  return (r.headers.get("content-type") ?? "").includes("application/json");
}

async function fetchMlbAdvancedStats(
  season: number
): Promise<MlbAdvancedTeamStats[]> {
  if (!navigator.onLine) throw new Error("Browser offline");

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await apiFetch(`/api/v1/mlb/team-stats/advanced?season=${season}`, {
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

  const { data } = (await res.json()) as { data: MlbAdvancedTeamStats[] };
  return Array.isArray(data) ? data : [];
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
    retry: (f) => navigator.onLine && f < 3,
    refetchOnWindowFocus: false,
  });
}
