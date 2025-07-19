// frontend/src/components/games/weather_badge.tsx

import React from "react";
import type { WeatherData } from "@/types";

export interface WeatherBadgeProps {
  isLoading: boolean;
  isError: boolean;
  data: WeatherData | undefined;
  isIndoor?: boolean;
  onClick: () => void;
}

const WeatherBadge: React.FC<WeatherBadgeProps> = ({
  isLoading,
  isError,
  data,
  isIndoor,
  onClick,
}) => {
  /* ─── indoor venue ─────────────────────────────────────────────── */
  if (isIndoor) {
    return (
      <button
        onClick={onClick}
        data-tour="weather-badge"
        className="rounded-full bg-amber-600/90 hover:bg-amber-700 px-4 py-2 text-sm font-semibold text-white"
      >
        Indoor Game
      </button>
    );
  }

  /* ─── loading / error states ───────────────────────────────────── */
  if (isLoading)
    return (
      <button
        disabled
        data-tour="weather-badge"
        className="rounded-full bg-slate-400 px-4 py-2 text-sm text-white"
      >
        Loading…
      </button>
    );

  if (isError || !data)
    return (
      <button
        disabled
        data-tour="weather-badge"
        className="rounded-full bg-red-500/80 px-4 py-2 text-sm text-white"
      >
        N/A
      </button>
    );

  /* ─── outdoor: render temp / wind ───────────────────────────────── */
  return (
    <button
      onClick={onClick}
      data-tour="weather-badge"
      className="rounded-full bg-green-700 hover:bg-green-800 px-4 py-2 flex items-center space-x-1 text-sm font-semibold text-white"
    >
      {/* arrow icon */}
      <span
        className="inline-block rotate-[-45deg]"
        style={{ transform: `rotate(${data.ballparkWindAngle}deg)` }}
      >
        ↑
      </span>
      <span>
        {data.temperature}°F / {data.windSpeed}mph
      </span>
    </button>
  );
};

export default WeatherBadge;
