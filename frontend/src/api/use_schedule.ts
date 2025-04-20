// frontend/src/api/use_schedule.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { Sport } from "@/contexts/sport_context";

export interface Game {
  id: string;
  homeTeam: string;
  awayTeam: string;
  tipoff: string; // ISO string
  spread: number;
  total: number;
  predictionHome: number;
  predictionAway: number;
}

export const useSchedule = (sport: Sport, date: string) =>
  useQuery<Game[], Error>({
    queryKey: ["schedule", sport, date],
    queryFn: async () => {
      const url = `/api/v1/${sport.toLowerCase()}/schedule`;

      // Tell TS this might be Game[] OR { data: Game[] }
      const response = await axios.get<Game[] | { data: Game[] }>(url, {
        params: { date, _: Date.now() }, // cache‑buster
      });

      const payload = response.data;
      // Normalize both shapes into a flat Game[]
      return Array.isArray(payload) ? payload : payload.data ?? [];
    },
    staleTime: 60_000,
  });
