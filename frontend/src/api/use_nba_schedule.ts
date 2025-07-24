// frontend/src/api/use_nba_schedule.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { UnifiedGame } from "@/types";

type GameWithET = UnifiedGame & { gameTimeET: string };

export const useNBASchedule = (date: string) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["nbaSchedule", date],
    staleTime: 60_000,
    retry: (fails) => navigator.onLine && fails < 3,
    enabled: !!date,

    queryFn: async () => {
      /* abort after 10 s */
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 10_000);

      let res: Response;
      try {
        res = await apiFetch(`/api/v1/nba/schedule?date=${date}`, {
          signal: controller.signal,
          cache: "no-store",
          headers: { accept: "application/json" },
        });
      } finally {
        clearTimeout(tid);
      }

      if (!res.ok) throw new Error(`Schedule request failed (${res.status})`);

      const { data } = (await res.json()) as { data: UnifiedGame[] };
      return data.map((g) => ({
        ...g,
        sport: "NBA",
        gameTimeET: new Date(g.scheduled_time).toLocaleTimeString("en-US", {
          timeZone: "America/New_York",
          hour: "numeric",
          minute: "2-digit",
        }),
      }));
    },
  });
