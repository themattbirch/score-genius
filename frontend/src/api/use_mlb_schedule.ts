// frontend/src/api/use_mlb_schedule.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { UnifiedGame } from "@/types";

type GameWithET = UnifiedGame & { gameTimeET: string };

export const useMLBSchedule = (date: string) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["mlbSchedule", date],
    staleTime: 60_000,
    retry: (fails) => navigator.onLine && fails < 3,
    enabled: !!date,

    queryFn: async () => {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 10_000);

      let res: Response;
      try {
        res = await apiFetch(`/api/v1/mlb/schedule?date=${date}`, {
          signal: controller.signal,
          cache: "no-store",
          headers: { accept: "application/json" },
        });
      } finally {
        clearTimeout(tid);
      }

      if (!res.ok)
        throw new Error(
          `Schedule request failed (${res.status} ${res.statusText})`
        );

      const { data } = (await res.json()) as { data: UnifiedGame[] };
      return data.map((g) => ({
        ...g,
        sport: "MLB",
        gameTimeET: new Date(g.scheduled_time).toLocaleTimeString("en-US", {
          timeZone: "America/New_York",
          hour: "numeric",
          minute: "2-digit",
        }),
      }));
    },
  });
