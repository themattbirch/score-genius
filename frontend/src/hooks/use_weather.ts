// frontend/src/hooks/use_weather.ts

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import { WeatherData, Sport } from "@/types";

/**
 * Custom hook to fetch weather data for a specific team.
 * @param {Sport | undefined} sport - The league ('MLB' or 'NFL').
 * @param {string | undefined} teamName - The name of the team (e.g., "Boston Red Sox").
 * @returns Query result object from @tanstack/react-query.
 */
export const useWeather = (sport?: Sport, teamName?: string) => {
  // The query will only run if both sport and teamName are valid strings.
  const isEnabled = !!sport && !!teamName;

  return useQuery<WeatherData, Error>({
    // React Query uses this key to cache data. It will refetch if sport or teamName changes.
    queryKey: ["weather", sport, teamName],

    queryFn: async () => {
      // The AbortController is a standard way to handle query cancellation.
      const controller = new AbortController();
      const signal = controller.signal;

      // Construct the path for apiFetch, ensuring the team name is properly URL-encoded.
      const path = `/api/weather?sport=${sport}&teamName=${encodeURIComponent(
        teamName!
      )}`;
      const response = await apiFetch(path, { signal });

      if (!response.ok) {
        const errorBody = await response.text();
        console.error(
          `API Error for ${path}: ${response.status} - ${errorBody}`
        );
        throw new Error(`Failed to fetch weather: ${response.statusText}`);
      }
      return response.json();
    },

    // This is the key: it runs automatically when the component mounts with valid props.
    enabled: isEnabled,

    // Sensible caching to avoid hitting the API on every render or quick tab-switch.
    staleTime: 1000 * 60 * 15, // Data is considered fresh for 15 minutes
    refetchOnWindowFocus: false,
    retry: 2, // Retry failed requests twice
  });
};
