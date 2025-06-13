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

  const query = useQuery(
    ["snapshot", sport, gameId],
    async () => {
      if (!isEnabled) {
        return null;
      }
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
        throw new Error(
          `Failed to fetch snapshot for ${sport} game ${gameId}: ${response.statusText}`
        );
      }
      return response.json();
    },
    {
      enabled: false, // only fetch when refetch() is called
      staleTime: 120 * 1000, // 2 minutes
      cacheTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    }
  );

  return query;
};
