// frontend/src/api/use_nfl_advanced_stats.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { Sport, NflAdvancedTeamStats } from "@/types";

function isJson(r: Response) {
  return (r.headers.get("content-type") ?? "").includes("application/json");
}

async function fetchNflAdvancedStats(
  season: number
): Promise<NflAdvancedTeamStats[]> {
  if (!navigator.onLine) throw new Error("Browser offline");

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await apiFetch(`/api/v1/nfl/team-stats/advanced?season=${season}`, {
      signal: controller.signal,
      headers: { accept: "application/json" },
      cache: "no-store",
    });
  } finally {
    clearTimeout(tid);
  }

  if (!res.ok || !isJson(res)) {
    throw new Error(`NFL advanced-stats request failed (${res.status})`);
  }

  const { data } = (await res.json()) as { data: NflAdvancedTeamStats[] };
  return Array.isArray(data) ? data : [];
}

export function useNflAdvancedStats({
  season,
  sport,
  enabled = true,
}: {
  season: number;
  sport: Sport;
  enabled?: boolean;
}) {
  return useQuery<NflAdvancedTeamStats[]>({
    queryKey: ["nflAdvancedStats", sport, season],
    queryFn: () => fetchNflAdvancedStats(season),
    enabled: enabled && sport === "NFL",
    staleTime: 86_400_000, // 24 hours
    retry: (f) => navigator.onLine && f < 3,
    refetchOnWindowFocus: false,
  });
}
