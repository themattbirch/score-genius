// frontend/src/api/use_injuries.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import type { Sport } from "@/contexts/sport_context";
import { apiFetch } from "@/api/client";

export interface Injury {
  id: string; // Assuming injury_id is mapped to id by backend or selected as id
  player: string; // Assuming player_display_name maps to player
  // --- FIX: Use the correct property name from the DB table ---
  team_display_name: string; // <--- Changed from 'team' if it was that before
  // --- END FIX ---
  status: "Out" | "Doubtful" | "Questionable" | "Probable" | "Day‑to‑Day"; // Assuming injury_status maps to status
  detail: string; // Assuming injury_detail maps to detail
  updated: string; // Assuming report_date_utc maps to updated
  injury_type?: string | null;
  // Add other fields if your useQuery select statement includes & maps them
}
// frontend/src/api/use_injuries.ts
export function useInjuries(league: string, date: string) {
  return useQuery<Injury[], Error>({
    queryKey: ["injuries", league, date],
    queryFn: async () => {
      const res = await apiFetch(
        `/api/v1/${league.toLowerCase()}/injuries?date=${date}`
      );
      if (!res.ok) {
        throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);
      }
      const json = (await res.json()) as { data: Injury[] };
      return json.data;
    },
    enabled: Boolean(date),
  });
}
