// frontend/src/api/use_schedule.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Sport } from '@/contexts/sport_context';

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV
  ? 'http://localhost:3001'
  : PROD_BASE;

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
  useQuery<Game[], Error>({
    queryKey: ['schedule', sport, date],
    queryFn: async () => {
      const url = `${BASE}/api/v1/${sport.toLowerCase()}/schedule`;
      const response = await axios.get<Game[]>(url, { params: { date } });
      return response.data;
    },
    staleTime: 60_000,
    refetchInterval:
      sport === 'NBA' && date === new Date().toISOString().slice(0, 10)
        ? 30_000
        : false,
  });
