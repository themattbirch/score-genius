// frontend/src/api/use_nba_advanced_stats.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { Sport, NbaAdvancedTeamStats as AdvancedTeamStats } from "@/types";

function isJson(r: Response) {
  return (r.headers.get("content-type") ?? "").includes("application/json");
}

async function fetchAdvancedStats(
  season: number
): Promise<AdvancedTeamStats[]> {
  if (!navigator.onLine) throw new Error("Browser offline");

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await apiFetch(`/api/v1/nba/advanced-stats?season=${season}`, {
      signal: controller.signal,
      headers: { accept: "application/json" },
      cache: "no-store",
    });
  } finally {
    clearTimeout(tid);
  }

  if (!res.ok || !isJson(res)) {
    throw new Error(`NBA advanced-stats request failed (${res.status})`);
  }

  const { data } = (await res.json()) as { data: AdvancedTeamStats[] };
  return Array.isArray(data) ? data : [];
}

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
    queryFn: () => fetchAdvancedStats(season),
    enabled: enabled && sport === "NBA",
    staleTime: 3_600_000, // 1 h
    retry: (f) => navigator.onLine && f < 3,
  });
}
