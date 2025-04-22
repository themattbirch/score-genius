// frontend/src/api/use_injuries.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import type { Sport } from "@/contexts/sport_context";

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV ? "http://localhost:3001" : PROD_BASE;

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
export const useInjuries = (sport: Sport, date: string) =>
  useQuery<Injury[], Error>({
    queryKey: ["injuries", sport, date], // Keep key specific to sport/date

    // Only execute the query if the sport is NBA and a valid date is provided
    enabled: sport === "NBA" && !!date,
    // --- END ADDITION ---

    queryFn: async () => {
      // This function will now only run when enabled (sport === 'NBA')
      // We can keep the dynamic URL or hardcode NBA, hardcoding is clearer now:
      const url = `${BASE}/api/v1/nba/injuries`;
      console.log(`[useInjuries] Fetching NBA injuries for date: ${date}`); // Log specific fetch

      const params: Record<string, string | number> = { date };
      // Add cache‑buster only when viewing today's NBA page (logic is fine)
      if (date === new Date().toISOString().slice(0, 10)) {
        params._ = Date.now();
      }

      // Type for potential API response structures
      type InjuryResponse = Injury[] | { injuries?: Injury[]; data?: Injury[] };

      const { data: raw } = await axios.get<InjuryResponse>(url, { params });

      // Normalize different possible response shapes into a flat array
      if (Array.isArray(raw)) return raw;
      // Add null/undefined checks for safety before accessing properties
      if (raw && Array.isArray(raw.injuries)) return raw.injuries;
      if (raw && Array.isArray(raw.data)) return raw.data;
      // If response is unexpected or empty after checks, return empty array
      console.warn(
        `[useInjuries] Unexpected injury data structure received:`,
        raw
      );
      return [];
    },

    staleTime: 10 * 60_000, // 10 min

    // Keep NBA-specific refetch logic
    refetchInterval:
      sport === "NBA" && date === new Date().toISOString().slice(0, 10)
        ? 30_000 // refetch every 30s for live NBA today
        : false, // No refetch interval otherwise (including MLB)
  });
