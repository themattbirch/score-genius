// frontend/src/hooks/use_snapshot.js

// frontend/src/hooks/use_snapshot.js

import { useQuery } from "@tanstack/react-query";

/**
 * Custom hook to fetch game snapshot data.
 * @param {string} gameId - The ID of the game.
 * @param {'NBA' | 'MLB'} sport - The sport ('NBA' or 'MLB').
 * @returns {object} Query result object with data, isLoading, isError, error.
 */
// --- CHANGE THIS LINE ---
export const use_snapshot = (gameId, sport) => {
  // Changed from useSnapshot to use_snapshot
  const isEnabled = !!gameId && !!sport;

  return useQuery({
    queryKey: ["snapshot", sport, gameId],
    queryFn: async () => {
      if (!isEnabled) {
        return null;
      }
      const response = await fetch(
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
    staleTime: 120 * 1000,
    cacheTime: 5 * 60 * 1000,
    enabled: isEnabled,
  });
};
