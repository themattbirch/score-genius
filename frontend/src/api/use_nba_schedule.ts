// src/api/use_nba_schedule.ts
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { UnifiedGame } from "@/types";

// Extend UnifiedGame with your new field
type GameWithET = UnifiedGame & { gameTimeET: string };

interface Resp {
  data: UnifiedGame[];
  message: string;
  retrieved: number;
}

export const useNBASchedule = (date: string) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["nbaSchedule", date],
    queryFn: async () => {
      const res = await apiFetch(`/api/v1/nba/schedule?date=${date}`);
      if (!res.ok) throw new Error(await res.text());
      const json: Resp = await res.json();

      return json.data.map((game) => ({
        ...game,
        gameTimeET: new Date(game.scheduled_time).toLocaleTimeString("en-US", {
          timeZone: "America/New_York",
          hour: "numeric",
          minute: "2-digit",
        }),
      }));
    },
    staleTime: 5_000,
  });
