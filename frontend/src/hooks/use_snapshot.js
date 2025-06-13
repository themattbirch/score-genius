// frontend/src/hooks/use_snapshot.js

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";

/**
 * Custom hook to fetch game snapshot data.
 * @param {string} gameId - The ID of the game.
 * @param {'NBA' | 'MLB'} sport - The sport ('NBA' or 'MLB').
 * @returns {object} Query result object with data, error, isFetching, refetch, etc.
 */
export const use_snapshot = (gameId, sport) => {
  const isEnabled = !!gameId && !!sport;
  console.log(
    "use_snapshot hook - checking enabled status:",
    isEnabled,
    "gameId:",
    gameId,
    "typeof gameId:",
    typeof gameId,
    "sport:",
    sport,
    "typeof sport:",
    typeof sport
  );

  const query = useQuery({
    queryKey: ["snapshot", sport, gameId],
    queryFn: async () => {
      if (!isEnabled) return null;
      console.log(
        `[apiFetch] Constructed URL: ${
          import.meta.env.VITE_API_BASE_URL
        }/api/v1/${sport.toLowerCase()}/snapshots/${gameId}`
      );
      const response = await apiFetch(
        `/api/v1/${sport.toLowerCase()}/snapshots/${gameId}`
      );
      if (!response.ok) {
        const errorBody = await response.text();
        console.error(
          `API Error for /api/v1/${sport.toLowerCase()}/snapshots/${gameId}: ${
            response.status
          } - ${errorBody}`
        );
        throw new Error(`Failed to fetch snapshot: ${response.statusText}`);
      }
      return response.json();
    },
    enabled: false,
    staleTime: 120 * 1000,
    cacheTime: 5 * 60 * 1000,
    retry: 1,
  });

  return query;
};
