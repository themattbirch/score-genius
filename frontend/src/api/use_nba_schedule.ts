// frontend/src/api/use_nba_schedule.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { Sport } from "@/contexts/sport_context";

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV ? "http://localhost:3001" : PROD_BASE;

// Interface for the actual API response structure from the controller
interface NBAAPIResponse {
  message: string;
  retrieved: number;
  data: Game[]; // The array is nested here
}

// Keep existing Game interface - make odds nullable based on service mapping
export interface Game {
  id: string;
  homeTeam: string;
  awayTeam: string;
  tipoff: string; // ISO
  spread?: number | null;
  total?: number | null;
  predictionHome?: number | null;
  predictionAway?: number | null;
}

// Hook definition
export const useNBASchedule = (sport: Sport, date: string) => {
  // Hook still ultimately returns Game[] or Error
  return useQuery<Game[], Error>({
    queryKey: ["schedule", sport, date], // Keep key consistent with arguments

    queryFn: async (): Promise<Game[]> => {
      // Mark Promise as Game[]
      const url = `${BASE}/api/v1/${sport.toLowerCase()}/schedule`;
      // Axios should expect the wrapper object type now
      const response = await axios.get<NBAAPIResponse>(url, {
        params: { date },
      });

      // *** FIX: Extract the nested 'data' array from the response object ***
      if (response && response.data && Array.isArray(response.data.data)) {
        return response.data.data; // Return the nested array
      } else {
        // Handle unexpected structure from API
        console.error(
          "Unexpected API response structure for NBA schedule:",
          response.data
        );
        throw new Error(
          "Invalid data structure received from NBA schedule API."
        );
        // OR return []; // Return empty array if preferred
      }
      // *** END FIX ***
    },
    staleTime: 60_000,
    // Keep conditional refetch if desired for NBA Today
    refetchInterval:
      sport === "NBA" && date === new Date().toISOString().slice(0, 10)
        ? 30_000
        : false,
  });
};
