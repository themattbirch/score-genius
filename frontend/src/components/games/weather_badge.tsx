// frontend/src/components/games/WeatherBadge.jsx

import React from "react";
import { WeatherData } from "@/types";

// A simple SVG icon for the badge.
const WeatherIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 20 20"
    fill="currentColor"
    className="w-5 h-5 mr-1.5"
  >
    <path
      fillRule="evenodd"
      d="M15.312 11.342a1.25 1.25 0 01.44 1.127l-.396 3.168a1.25 1.25 0 01-1.24.963H6.884a1.25 1.25 0 01-1.24-.963l-.396-3.168a1.25 1.25 0 01.44-1.127 8.943 8.943 0 015.824 0zM10 2a6.5 6.5 0 00-6.415 7.035.75.75 0 01-1.498-.14A8.001 8.001 0 0110 0a8.001 8.001 0 017.913 8.895.75.75 0 11-1.498.14A6.5 6.5 0 0010 2z"
      clipRule="evenodd"
    />
  </svg>
);

interface WeatherBadgeProps {
  isLoading: boolean;
  isError: boolean;
  data: WeatherData | undefined;
  onClick: () => void; // Function to open the details modal
}

const WeatherBadge: React.FC<WeatherBadgeProps> = ({
  isLoading,
  isError,
  data,
  onClick,
}) => {
  const renderContent = () => {
    if (isLoading) {
      return <span className="animate-pulse">Loading...</span>;
    }
    if (isError || !data) {
      // FR-WB-3: Shows a placeholder on error or if no data
      return <span>--°F / --mph</span>;
    }
    // FR-WB-1: Shows the live temperature and wind speed
    return (
      <span>
        {data.temperature}°F / {data.windSpeed}mph
      </span>
    );
  };

  const tooltipText = data
    ? `${data.city}: ${data.description}, ${data.temperature}°F, Wind: ${data.windSpeed}mph`
    : "Click to view ballpark weather";

  return (
    <button
      onClick={onClick}
      title={tooltipText}
      aria-label={tooltipText}
      className={`
        rounded-full bg-badge-weather text-white text-xs font-semibold
        px-3 py-1 inline-flex items-center justify-center
        min-w-[120px] h-8 transition-all duration-200 ease-in-out
        hover:bg-sky-500 focus:outline-none focus:ring-2 focus:ring-offset-2 
        focus:ring-sky-500 focus:ring-offset-gray-800
      `}
    >
      <WeatherIcon />
      <span className="truncate">{renderContent()}</span>
    </button>
  );
};

export default WeatherBadge;
