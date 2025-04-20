// frontend/src/api/use_injuries.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import type { Sport } from '@/contexts/sport_context';

const BASE = import.meta.env.VITE_API_BASE_URL as string;

export interface Injury {
  id: string;
  player: string;
  team: string;
  status: 'Out' | 'Doubtful' | 'Questionable' | 'Probable' | 'Day‑to‑Day';
  detail: string;
  updated: string; // ISO date
}

export const useInjuries = (sport: Sport, date: string) =>
  useQuery<Injury[], Error>({
    queryKey: ['injuries', sport, date],
    queryFn: async () => {
      const url = `${BASE}/api/v1/${sport.toLowerCase()}/injuries`;
      const response = await axios.get<Injury[] | { injuries: Injury[] }>(
        url,
        { params: { date } }
      );
      const raw = response.data;
      if (Array.isArray(raw)) {
        return raw;
      }
      if (Array.isArray((raw as any).injuries)) {
        return (raw as any).injuries;
      }
      return [];
    },
    staleTime: 10 * 60_000, // 10 minutes
  });
