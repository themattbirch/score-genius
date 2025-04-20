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
      // Use a relative path so Vite proxy ("/api" → backend) picks it up
      const url = `/api/v1/${sport.toLowerCase()}/schedule`;
      const { data } = await axios.get<
        Game[] | { message: string; retrieved: number; data: Game[] }
      >(url, { params: { date } });

      // If it’s already an array, return it; otherwise unwrap `.data`
      return Array.isArray(data) ? data : data.data ?? [];
    },
    staleTime: 60_000,
  });
