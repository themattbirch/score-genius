import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { Sport, UnifiedGame } from "@/types"; // Import UnifiedGame

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV ? "http://localhost:3001" : PROD_BASE;

// Response wrapper from the CONTROLLER
interface NBAAPIResponse {
  message: string;
  retrieved: number;
  data: UnifiedGame[]; // Expect backend service mapped data here
}

export const useNBASchedule = (sport: Sport, date: string) => {
  return useQuery<UnifiedGame[], Error>({
    // Return UnifiedGame[]
    queryKey: ["nbaSchedule", sport, date], // Unique key

    queryFn: async (): Promise<UnifiedGame[]> => {
      // Return UnifiedGame[]
      const url = `${BASE}/api/v1/nba/schedule`;
      const response = await axios.get<NBAAPIResponse>(url, {
        params: { date },
      });

      // Extract nested data array (already mapped by backend service)
      if (response?.data?.data && Array.isArray(response.data.data)) {
        return response.data.data;
      } else {
        console.error(
          "Unexpected API response for NBA schedule:",
          response?.data
        );
        return []; // Return empty array on unexpected structure
      }
    },
    staleTime: 60_000,
    refetchInterval:
      sport === "NBA" && date === new Date().toISOString().slice(0, 10)
        ? 30_000
        : false,
    enabled: !!date && !!sport, // Keep enabled check
  });
};
