// frontend/src/api/use_mlb_schedule.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

const PROD_BASE = import.meta.env.VITE_API_BASE_URL as string;
const BASE = import.meta.env.DEV ? "http://localhost:3001" : PROD_BASE;

// Interface for the actual API response structure
interface MLBAPIResponse {
  message: string;
  retrieved: number;
  data: MLBGame[]; // The array is nested here
}

// Interface for the game data - adjust types based on curl output
export interface MLBGame {
  game_id: string;
  scheduled_time_utc: string;
  game_date_et: string;
  status_detail?: string | null; // Allow null
  status_state: string;
  home_team_name: string;
  away_team_name: string;
  home_probable_pitcher_name?: string | null; // Allow null
  away_probable_pitcher_name?: string | null; // Allow null
  home_probable_pitcher_handedness?: string | null; // Allow null
  away_probable_pitcher_handedness?: string | null; // Allow null

  // Odds seem to be strings or numbers, allow null too
  moneyline_home_clean?: string | number | null;
  moneyline_away_clean?: string | number | null;
  spread_home_line_clean?: number | null;
  spread_home_price_clean?: string | number | null; // Prices often strings like "+110"
  spread_away_price_clean?: string | number | null;
  total_line_clean?: number | null;
  total_over_price_clean?: string | number | null;
  total_under_price_clean?: string | number | null;
}

export const useMLBSchedule = (date: string) => {
  // useQuery hook still expects to ultimately receive MLBGame[]
  return useQuery<MLBGame[], Error>({
    queryKey: ["mlbSchedule", date],
    queryFn: async (): Promise<MLBGame[]> => { // Mark Promise as MLBGame[]
      const url = `${BASE}/api/v1/mlb/schedule`;
      // Axios should expect the wrapper object type
      const response = await axios.get<MLBAPIResponse>(url, { params: { date } });
      // *** FIX: Extract the 'data' array from the response object ***
      // Add checks in case response.data or response.data.data is missing
      if (response && response.data && Array.isArray(response.data.data)) {
          return response.data.data;
      } else {
          // Handle unexpected structure from API
          console.error("Unexpected API response structure for MLB schedule:", response.data);
          throw new Error("Invalid data structure received from MLB schedule API.");
          // OR return []; // Return empty array if preferred over throwing error
      }
    },
    staleTime: 60_000,
    refetchInterval: false,
  });
};