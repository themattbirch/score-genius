// frontend/src/api/use_nfl_schedule.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { UnifiedGame } from "@/types";

type GameWithET = UnifiedGame & { gameTimeET: string };

export const useNFLSchedule = (date: string) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["nflSchedule", date],
    staleTime: 60_000, // Refetch data considered stale after 1 minute
    retry: (fails) => navigator.onLine && fails < 3, // Retry up to 3 times if online
    enabled: !!date, // Only run the query if a date is provided

    queryFn: async () => {
      const controller = new AbortController();
      // Abort the fetch request after 10 seconds
      const tid = setTimeout(() => controller.abort(), 10_000);

      let res: Response;
      try {
        res = await apiFetch(`/api/v1/nfl/schedule?date=${date}`, {
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
      // Transform the data for frontend use
      return data.map((g) => ({
        ...g,
        sport: "NFL",
        gameTimeET: new Date(g.scheduled_time).toLocaleTimeString("en-US", {
          timeZone: "America/New_York",
          hour: "numeric",
          minute: "2-digit",
        }),
      }));
    },
  });
