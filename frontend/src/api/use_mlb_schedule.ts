// frontend/src/api/use_mlb_schedule.ts
import { useQuery } from "@tanstack/react-query";
import { UnifiedGame } from "@/types";
import { apiFetch } from "@/api/client";

interface Resp {
  message: string;
  retrieved: number;
  data: UnifiedGame[];
}

export const useMLBSchedule = (date: string) =>
  useQuery<UnifiedGame[]>({
    queryKey: ["mlbSchedule", date],
    queryFn: async () => {
      const res = await apiFetch(`/api/v1/mlb/schedule?date=${date}`);
      if (!res.ok) throw new Error(await res.text());
      const json: Resp = await res.json();
      return json.data;
    },
    staleTime: 60_000,
    enabled: !!date,
  });
