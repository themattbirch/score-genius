// frontend/src/api/use_schedule.ts

import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Sport } from '@/contexts/sport_context';

export interface Game {
  id: string;
  homeTeam: string;
  awayTeam: string;
  tipoff: string; // ISO
  spread: number;
  total: number;
  predictionHome: number;
  predictionAway: number;
}

export const useSchedule = (sport: Sport, date: string) =>
  useQuery<Game[]>({
    queryKey: ['schedule', sport, date],
    queryFn: () =>
      axios
        .get(`/api/${sport.toLowerCase()}/schedule`, { params: { date } })
        .then((r) => r.data),
    staleTime: 60_000,
    // ⬇️ Refetch every 30 s on game day to update scores
    refetchInterval: sport === 'NBA' && date === new Date().toISOString().slice(0,10)
      ? 30_000
      : false,
  });
