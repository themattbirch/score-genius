// src/api/use_nba_schedule.ts
import { useQuery } from "@tanstack/react-query";
import type { UnifiedGame } from "@/types";

type GameWithET = UnifiedGame & { gameTimeET: string };
interface Resp {
  data: UnifiedGame[];
}

export const useNBASchedule = (date: string) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["nbaSchedule", date],

    queryFn: async () => {
      /* 1 ▸ abort after 10 s */
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 10_000);

      let res: Response;
      try {
        res = await fetch(`/api/v1/nba/schedule?date=${date}`, {
          signal: controller.signal,
          headers: { accept: "application/json" },
          cache: "no-store", // bypass any SW stale cache
        });
      } finally {
        clearTimeout(tid);
      }

      if (!res.ok) throw new Error("Network request failed");

      const json: Resp = await res.json();
      return json.data.map((g) => ({
        ...g,
        gameTimeET: new Date(g.scheduled_time).toLocaleTimeString("en-US", {
          timeZone: "America/New_York",
          hour: "numeric",
          minute: "2-digit",
        }),
      }));
    },

    staleTime: 60_000,
    retry: (failures) => navigator.onLine && failures < 3,
  });
