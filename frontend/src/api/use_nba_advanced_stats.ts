import { useQuery } from "@tanstack/react-query";
import type { Sport } from "@/types";

export interface AdvancedTeamStats {
  team_name: string;
  team_id: string;
}

function isJson(res: Response) {
  return (res.headers.get("content-type") ?? "").includes("application/json");
}

async function fetchAdvancedStats(
  season: number
): Promise<AdvancedTeamStats[]> {
  if (!navigator.onLine) throw new Error("Browser offline");

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await fetch(`/api/v1/nba/advanced-stats?season=${season}`, {
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

  const json = (await res.json()) as { data: AdvancedTeamStats[] };
  return Array.isArray(json.data) ? json.data : [];
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
    retry: (failureCount: number) => navigator.onLine && failureCount < 3,
  });
}
