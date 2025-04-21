// frontend/src/api/use_injuries.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import type { Sport } from '@/contexts/sport_context';

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV ? 'http://localhost:3001' : PROD_BASE;

export interface Injury {
  id: string;
  player: string;
  team: string;
  status: 'Out' | 'Doubtful' | 'Questionable' | 'Probable' | 'Day‑to‑Day';
  detail: string;
  updated: string; // ISO date string
}

export const useInjuries = (sport: Sport, date: string) =>
  useQuery<Injury[], Error>({
    queryKey: ['injuries', sport, date],
    queryFn: async () => {
      const url = `${BASE}/api/v1/${sport.toLowerCase()}/injuries`;

      const params: Record<string, string|number> = { date };
      // Add a cache‑buster when viewing today's NBA page
      if (sport === 'NBA' && date === new Date().toISOString().slice(0, 10)) {
        params._ = Date.now();
      }

      const { data: raw } = await axios.get<
        Injury[] | { injuries?: Injury[]; data?: Injury[] }
      >(url, { params });

      // Normalize both possible shapes into a flat array
      if (Array.isArray(raw)) return raw;
      if (Array.isArray(raw.injuries)) return raw.injuries;
      if (Array.isArray(raw.data)) return raw.data;
      return [];
    },
    staleTime: 10 * 60_000, // 10 min
    refetchInterval:
      sport === 'NBA' && date === new Date().toISOString().slice(0, 10)
        ? 30_000 // refetch every 30s for live NBA
        : false,
  });
