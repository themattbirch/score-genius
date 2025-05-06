// frontend/src/api/use_mlb_schedule.ts
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import type { UnifiedGame } from "@/types";

// Extend UnifiedGame with ET field
type GameWithET = UnifiedGame & { gameTimeET: string };

interface Resp {
  message: string;
  retrieved: number;
  data: UnifiedGame[];
}

export const useMLBSchedule = (date: string) =>
  useQuery<GameWithET[], Error>({
    queryKey: ["mlbSchedule", date],
    queryFn: async () => {
      const res = await apiFetch(`/api/v1/mlb/schedule?date=${date}`);
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
    staleTime: 0,
    enabled: !!date,
  });
