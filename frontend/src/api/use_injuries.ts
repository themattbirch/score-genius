// frontend/src/api/use_injuries.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import type { Sport } from "@/contexts/sport_context";
import { apiFetch } from "@/api/client";

// frontend/src/api/use_injuries.ts
export interface Injury {
  id: string;
  player: string;
  team_display_name: string;
  status: "Out" | "Doubtful" | "Questionable" | "Probable" | "Day-to-Day";
  detail: string;
  updated: string;
  injury_type: string | null; // ← remove the “?” here
}

export function useInjuries(
  league: string,
  date: string,
  options: { enabled?: boolean } = {}
) {
  const { enabled = true } = options;
  return useQuery<Injury[], Error>({
    queryKey: ["injuries", league, date],
    queryFn: async () => {
      const res = await apiFetch(
        `/api/v1/${league.toLowerCase()}/injuries?date=${date}`
      );
      if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
      const json = (await res.json()) as { data: any[] };

      // map raw payload into our Injury type
      return json.data.map((i) => ({
        id: i.id,
        player: i.player,
        team_display_name: i.team_display_name,
        status: i.status,
        detail: i.detail,
        updated: i.updated,
        injury_type: i.injury_type ?? null,
      }));
    },
    enabled: enabled && Boolean(date),
  });
}
