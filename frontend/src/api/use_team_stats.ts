import { useQuery } from "@tanstack/react-query";
import type { Sport, UnifiedTeamStats } from "@/types";

function isJson(res: Response) {
  return (res.headers.get("content-type") ?? "").includes("application/json");
}

async function fetchTeamStats({
  sport,
  season,
}: {
  sport: Sport;
  season: number;
}): Promise<UnifiedTeamStats[]> {
  if (!navigator.onLine) throw new Error("Browser offline");

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await fetch(
      `/api/v1/${sport.toLowerCase()}/team-stats?season=${season}`,
      {
        signal: controller.signal,
        headers: { accept: "application/json" },
        cache: "no-store",
      }
    );
  } finally {
    clearTimeout(tid);
  }

  if (!res.ok || !isJson(res)) {
    throw new Error(`${sport} team-stats request failed (${res.status})`);
  }

  const json = (await res.json()) as { data: UnifiedTeamStats[] };
  return Array.isArray(json.data) ? json.data : [];
}

export function useTeamStats({
  sport,
  season,
  enabled = true,
  staleTime = 1_800_000,
}: {
  sport: Sport;
  season: number;
  enabled?: boolean;
  staleTime?: number;
}) {
  return useQuery<UnifiedTeamStats[]>({
    queryKey: ["teamStats", sport, season],
    queryFn: () => fetchTeamStats({ sport, season }),
    enabled,
    staleTime,
    retry: (failureCount: number) => navigator.onLine && failureCount < 3,
    refetchOnMount: "always",
  });
}
