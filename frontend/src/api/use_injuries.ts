// frontend/src/api/use_injuries.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import type { Sport } from '@/contexts/sport_context';

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV ? 'http://localhost:3001' : PROD_BASE;

export interface Injury {
  // Assume interface is correct: id, player, team, status, detail, updated
  id: string;
  player: string;
  team: string;
  status: 'Out' | 'Doubtful' | 'Questionable' | 'Probable' | 'Day‑to‑Day';
  detail: string;
  updated: string; // ISO date string
}

export const useInjuries = (sport: Sport, date: string) =>
  useQuery<Injury[], Error>({
    queryKey: ['injuries', sport, date], // Keep key specific to sport/date

    // --- ADD THIS LINE ---
    // Only execute the query if the sport is NBA and a valid date is provided
    enabled: sport === 'NBA' && !!date,
    // --- END ADDITION ---

    queryFn: async () => {
      // This function will now only run when enabled (sport === 'NBA')
      // We can keep the dynamic URL or hardcode NBA, hardcoding is clearer now:
      const url = `${BASE}/api/v1/nba/injuries`;
      console.log(`[useInjuries] Fetching NBA injuries for date: ${date}`); // Log specific fetch

      const params: Record<string, string|number> = { date };
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
      console.warn(`[useInjuries] Unexpected injury data structure received:`, raw);
      return [];
    },

    staleTime: 10 * 60_000, // 10 min

    refetchInterval: // Keep NBA-specific refetch logic
      sport === 'NBA' && date === new Date().toISOString().slice(0, 10)
        ? 30_000 // refetch every 30s for live NBA today
        : false, // No refetch interval otherwise (including MLB)
  });