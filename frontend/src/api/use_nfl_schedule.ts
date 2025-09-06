// frontend/src/api/use_nfl_schedule.ts
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { UnifiedGame } from "@/types";

type GameWithET = UnifiedGame & { gameTimeET: string };

const toET = (iso?: string) =>
  iso
    ? new Date(iso).toLocaleTimeString("en-US", {
        timeZone: "America/New_York",
        hour: "numeric",
        minute: "2-digit",
      })
    : "";

// unwrap supports both [{...}] & { data: [{...}] }
function unwrap(json: unknown): any[] {
  if (Array.isArray(json)) return json;
  if (json && typeof json === "object" && "data" in (json as any)) {
    const d = (json as any).data;
    if (Array.isArray(d)) return d;
  }
  return [];
}

export const useNFLSchedule = (
  date: string,
  options?: Partial<UseQueryOptions<GameWithET[], Error>>
) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["nflSchedule", date],
    staleTime: 60_000,
    retry: (fails) => navigator.onLine && fails < 3,
    enabled: !!date && (options?.enabled ?? true),

    queryFn: async () => {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 10_000);
      try {
        const res = await apiFetch(`/api/v1/nfl/schedule?date=${date}`, {
          signal: controller.signal,
          cache: "no-store",
          headers: { accept: "application/json" },
        });
        if (!res.ok) {
          throw new Error(
            `Schedule request failed (${res.status} ${res.statusText})`
          );
        }

        const json = await res.json();
        const rows = unwrap(json);

        // mirror MLB mapping; do NOT gate on predictions
        return rows.map((g: any) => ({
          ...g,
          sport: "NFL",
          // keep both timestamp + derived ET string for UI
          gameTimeET: toET(
            g.scheduled_time ?? g.gameTimeUTC ?? g.game_time_utc
          ),
        }));
      } finally {
        clearTimeout(tid);
      }
    },
    ...options,
  });
