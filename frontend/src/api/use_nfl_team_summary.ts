// frontend/src/api/use_nfl_team_summary.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { Sport } from "@/types";
import type { NflTeamSummary } from "@/types"; // define below if not yet
import { normalizeNflTeamSummaryRow } from "./normalize";

function isJson(r: Response) {
  return (r.headers.get("content-type") ?? "").includes("application/json");
}

async function fetchNflTeamSummary(season: number): Promise<NflTeamSummary[]> {
  if (!navigator.onLine) throw new Error("Browser offline");

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 10_000);

  let res: Response;
  try {
    res = await apiFetch(`/api/v1/nfl/team-stats/summary?season=${season}`, {
      signal: controller.signal,
      headers: { accept: "application/json" },
    });
  } finally {
    clearTimeout(tid);
  }

  if (!res.ok || !isJson(res)) {
    const errBody = await res.text().catch(() => "");
    throw new Error(
      `NFL team summary request failed (${res.status}) ${
        errBody ? "- " + errBody : ""
      }`
    );
  }

  const parsed = (await res.json()) as {
    data: Record<string, any>[];
  };
  const rawData = Array.isArray(parsed.data) ? parsed.data : [];
  const normalized = rawData.map(normalizeNflTeamSummaryRow);
  return normalized;
}

export function useNflTeamSummary({
  season,
  sport,
  enabled = true,
}: {
  season: number;
  sport: Sport;
  enabled?: boolean;
}) {
  return useQuery<NflTeamSummary[]>({
    queryKey: ["nflTeamSummary", sport, season],
    queryFn: () => fetchNflTeamSummary(season),
    enabled: enabled && sport === "NFL",
    staleTime: 1_800_000, // 30 minutes; adjust as needed
    retry: (f) => navigator.onLine && f < 3,
    refetchOnWindowFocus: false,
  });
}
