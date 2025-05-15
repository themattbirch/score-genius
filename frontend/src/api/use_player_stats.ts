// frontend/src/api/use_player_stats.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { Sport, UnifiedPlayerStats } from "@/types";

function isJson(r: Response) {
  return (r.headers.get("content-type") ?? "").includes("application/json");
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
    res = await apiFetch(
      `/api/v1/${sport.toLowerCase()}/player-stats?${params}`,
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
    throw new Error(`${sport} player-stats request failed (${res.status})`);
  }

  const { data } = (await res.json()) as { data: UnifiedPlayerStats[] };
  return Array.isArray(data) ? data : [];
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
    retry: (f) => navigator.onLine && f < 3,
    refetchOnMount: "always",
  });
}
