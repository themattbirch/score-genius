import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { UnifiedGame } from "@/types"; // Import UnifiedGame

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV ? "http://localhost:3001" : PROD_BASE;

// Response wrapper from the CONTROLLER
interface MLBAPIResponse {
  message: string;
  retrieved: number;
  data: UnifiedGame[]; // Expect backend service mapped data here
}

// Remove old local MLBGame interface

export const useMLBSchedule = (date: string) => {
  return useQuery<UnifiedGame[], Error>({
    // Return UnifiedGame[]
    queryKey: ["mlbSchedule", date],

    queryFn: async (): Promise<UnifiedGame[]> => {
      // Return UnifiedGame[]
      const url = `${BASE}/api/v1/mlb/schedule`;
      const response = await axios.get<MLBAPIResponse>(url, {
        params: { date },
      });

      // Extract the nested 'data' array (already mapped by backend service)
      if (response?.data?.data && Array.isArray(response.data.data)) {
        return response.data.data;
      } else {
        console.error(
          "Unexpected API response for MLB schedule:",
          response?.data
        );
        return []; // Return empty array on unexpected structure
      }
    },
    staleTime: 60_000,
    refetchInterval: false,
    enabled: !!date, // Keep enabled check
  });
};
