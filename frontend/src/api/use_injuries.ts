import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import type { Sport } from '@/contexts/sport_context';

export interface Injury {
  id: string;
  player: string;
  team: string;
  status: 'Out' | 'Doubtful' | 'Questionable' | 'Probable' | 'Day‑to‑Day';
  detail: string;
  updated: string; // ISO date
}

export const useInjuries = (sport: Sport, date: string) =>
  useQuery<Injury[]>({
    queryKey: ['injuries', sport, date],
    queryFn: () =>
      axios
        .get(`/api/${sport.toLowerCase()}/injuries`, { params: { date } })
        .then((r) => {
          const raw = r.data;
          // backend might wrap in `{ injuries: [...] }`
          if (Array.isArray(raw)) return raw;
          if (Array.isArray(raw?.injuries)) return raw.injuries;
          return []; // always array fallback
        }),
    staleTime: 10 * 60_000,
  });
