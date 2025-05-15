// frontend/src/api/use_mlb_schedule.ts
import { useQuery } from "@tanstack/react-query";
import type { UnifiedGame } from "@/types";

type GameWithET = UnifiedGame & { gameTimeET: string };

interface Resp {
  message: string;
  retrieved: number;
  data: UnifiedGame[];
}

export const useMLBSchedule = (date: string) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["mlbSchedule", date],
    enabled: !!date,
    staleTime: 60_000,

    queryFn: async () => {
      /* ---- abort after 10 s ---- */
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 10_000);

      let res: Response;
      try {
        res = await fetch(`/api/v1/mlb/schedule?date=${date}`, {
          signal: controller.signal,
          headers: { accept: "application/json" },
          cache: "no-store",
        });
      } finally {
        clearTimeout(tid);
      }

      if (!res.ok) {
        throw new Error(
          `Schedule request failed (${res.status} ${res.statusText})`
        );
      }

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

    retry: (failures) => navigator.onLine && failures < 3,
  });
