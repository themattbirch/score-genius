// frontend/src/api/use_injuries.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import type { Sport } from '@/contexts/sport_context';

export interface Injury {
  id: string;
  player: string;
  team: string;
  status: "Out" | "Doubtful" | "Questionable" | "Probable" | "Day‑to‑Day";
  detail: string;
  updated: string; // ISO date string
}

export const useInjuries = (sport: Sport, date: string) =>
  useQuery<Injury[], Error>({
    queryKey: ["injuries", sport, date],
    queryFn: async () => {
      const url = `/api/v1/${sport.toLowerCase()}/injuries`;
      const { data: raw } = await axios.get<
        Injury[] | { injuries?: Injury[]; data?: Injury[] }
      >(url, { params: { date } });

      // Normalize both possible shapes into a flat array
      if (Array.isArray(raw))           return raw;
      if (Array.isArray(raw.injuries))  return raw.injuries;
      if (Array.isArray(raw.data))      return raw.data;
      return [];
    },
    staleTime: 10 * 60_000,
  });